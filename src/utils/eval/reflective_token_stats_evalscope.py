#!/usr/bin/env python3
"""
Reflective Token Statistics Script for EvalScope Results

统计 evalscope 框架运行的 evaluation 结果中的 reflective token 出现数量和频次，
并将结果作为新字段写入对应的 reports 目录下的 json 文件中。

Usage:
    python reflective_token_stats_evalscope.py --input_dir /path/to/eval/results
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# 与原始脚本一致的 reflective tokens 列表
REFLECTIVE_TOKENS = [
    "wait, ", "hmm", "okay", "alternatively", "let me think"
]


def normalize_text(text: str) -> str:
    """Normalize text for token matching."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def count_reflective_tokens(text: str, tokens: List[str]) -> Dict[str, int]:
    """Count occurrences of reflective tokens in text."""
    normalized_text = normalize_text(text)
    token_counts = defaultdict(int)
    for token in tokens:
        pattern = r'\b' + re.escape(token.lower()) + r'\b'
        matches = re.findall(pattern, normalized_text)
        token_counts[token] = len(matches)
    return dict(token_counts)


def _content_to_string(content: Any) -> Optional[str]:
    """将 content 字段统一转为字符串（可能是 str 或 list of dict 如 [{"type":"...", "text":"..."}]）。"""
    if content is None:
        return None
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get('text') or block.get('content')
                if text and isinstance(text, str):
                    parts.append(text)
        return ' '.join(parts).strip() or None
    return None


def extract_content_from_evalscope_record(record: Dict[str, Any]) -> Optional[str]:
    """
    从 evalscope 单条 jsonl 记录中提取模型输出文本。
    支持 model_output.choices[0].message.content 或 model_output.messages[0].content；
    content 可为字符串或 list of dict（含 'text'/'content'）。
    """
    model_output = record.get('model_output') or {}
    choices = model_output.get('choices') or []
    if choices:
        msg = choices[0].get('message') or {}
        content = msg.get('content')
        out = _content_to_string(content)
        if out:
            return out
    messages = model_output.get('messages') or []
    if messages:
        content = messages[0].get('content')
        out = _content_to_string(content)
        if out:
            return out
    return None


def process_evalscope_jsonl(
    jsonl_path: Path,
    tokens: List[str],
) -> Dict[str, Any]:
    """
    处理单个 evalscope 的 .jsonl 预测文件，统计 reflective tokens。

    Returns:
        与原始脚本结构兼容的统计字典，用于写入 report。
    """
    total_tokens = defaultdict(int)
    total_reflective_per_response = []
    total_responses = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = extract_content_from_evalscope_record(record)
            if content is None:
                continue
            total_responses += 1
            token_counts = count_reflective_tokens(content, tokens)
            for token, count in token_counts.items():
                total_tokens[token] += count
            total_reflective_per_response.append(sum(token_counts.values()))

    avg_tokens_per_response = {}
    for token in tokens:
        total_count = total_tokens[token]
        avg_tokens_per_response[token] = total_count / total_responses if total_responses > 0 else 0

    avg_total_reflective = (
        sum(total_reflective_per_response) / len(total_reflective_per_response)
        if total_reflective_per_response else 0
    )

    return {
        'total_responses': total_responses,
        'avg_total_reflective_tokens_per_response': avg_total_reflective,
        'avg_tokens_per_response': avg_tokens_per_response,
        'total_token_counts': dict(total_tokens),
        'reflective_tokens_analyzed': tokens,
        'source': str(jsonl_path),
    }


def find_report_path_for_predictions(
    run_dir: Path,
    model_id: str,
    prediction_stem: str,
) -> Optional[Path]:
    """
    根据 predictions 下的 jsonl 文件名，找到对应的 report json 路径。
    规则：reports/<model_id>/ 下存在 <report_base>.json，且 report_base 是 prediction_stem 的前缀。
    若有多个匹配则取最长前缀对应的 report。
    """
    reports_dir = run_dir / 'reports' / model_id
    if not reports_dir.is_dir():
        return None
    candidates: List[Tuple[int, Path]] = []
    for p in reports_dir.iterdir():
        if p.suffix != '.json' or not p.is_file():
            continue
        base = p.stem
        if prediction_stem.startswith(base) or prediction_stem == base:
            candidates.append((len(base), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def find_evalscope_predictions_and_reports(root_dir: Path) -> List[Tuple[Path, Path]]:
    """
    递归扫描 root_dir，找到所有 evalscope 的 predictions（*.jsonl），
    并解析出对应的 report json 路径（若存在）。
    目录结构约定: .../run/predictions/<model_id>/<bench>.jsonl 与 .../run/reports/<model_id>/<report>.json
    """
    root_dir = root_dir.resolve()
    pairs: List[Tuple[Path, Path]] = []
    for pred_path in root_dir.rglob('*.jsonl'):
        try:
            pred_relative = pred_path.relative_to(root_dir)
            part_list = pred_relative.parts
            if 'predictions' not in part_list:
                continue
            pi = part_list.index('predictions')
            if pi + 2 >= len(part_list):
                continue
            run_dir = root_dir / Path(*part_list[:pi]) if pi > 0 else root_dir
            model_id = part_list[pi + 1]
            report_path = find_report_path_for_predictions(
                run_dir, model_id, pred_path.stem
            )
            if report_path is not None and report_path.exists():
                pairs.append((pred_path, report_path))
        except (ValueError, IndexError):
            continue
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='统计 evalscope evaluation 结果中的 reflective tokens，并写入对应 reports 的 json'
    )
    parser.add_argument(
        '--input-dir', type=str, required=True,
        help='根目录：将递归扫描其下所有 evalscope 的 predictions 与 reports',
    )
    parser.add_argument(
        '--tokens', type=str, nargs='*', default=REFLECTIVE_TOKENS,
        help='要统计的 reflective token 列表（默认使用内置列表）',
    )
    parser.add_argument(
        '--key', type=str, default='reflective_token_stats',
        help='写入 report json 时使用的键名（默认: reflective_token_stats）',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='只打印将要处理的文件，不写入 report',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误：目录不存在 {input_dir}")
        return 1

    pairs = find_evalscope_predictions_and_reports(input_dir)
    # 去重（同一 (pred, report) 可能被 rglob 重复）
    seen = set()
    unique_pairs = []
    for p, r in pairs:
        k = (str(p), str(r))
        if k not in seen:
            seen.add(k)
            unique_pairs.append((p, r))

    print(f"共找到 {len(unique_pairs)} 个 evalscope benchmark（predictions + 对应 report）")

    if not unique_pairs:
        print("未找到可处理的 predictions/report 对，请确认目录下存在 evalscope 的 predictions 与 reports 结构。")
        return 0

    success = 0
    fail = 0
    for i, (pred_path, report_path) in enumerate(unique_pairs, 1):
        print(f"[{i}/{len(unique_pairs)}] {pred_path.name} -> {report_path}")
        try:
            stats = process_evalscope_jsonl(pred_path, args.tokens)
        except Exception as e:
            print(f"  处理失败: {e}")
            fail += 1
            continue
        if args.dry_run:
            print(f"  total_responses={stats['total_responses']}, "
                  f"avg_reflective={stats['avg_total_reflective_tokens_per_response']:.4f}")
            success += 1
            continue
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            print(f"  读取 report 失败: {e}")
            fail += 1
            continue
        report[args.key] = stats
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  写入 report 失败: {e}")
            fail += 1
            continue
        print(f"  已写入 {args.key}: total_responses={stats['total_responses']}, "
              f"avg_reflective={stats['avg_total_reflective_tokens_per_response']:.4f}")
        success += 1

    print(f"\n完成: 成功 {success}, 失败 {fail}")
    return 0


if __name__ == '__main__':
    exit(main())
