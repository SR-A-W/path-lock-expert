#!/usr/bin/env python3
"""
Average Output Length Statistics Script for EvalScope Results

统计 evalscope 框架运行的 evaluation 结果中每条输出的 token 长度，
并将统计结果（avg_output_length）写入对应的 reports 目录下的 json 文件中。

输出长度：优先使用 model_output.usage.output_tokens（与 eval 的 max_length 一致），
若缺失则回退为 content 字符数。

Usage:
    python eval/scripts/avg_output_length_evalscope.py --input-dir eval/results/pl_moe/qwen_pl_moe_140k
    python eval/scripts/avg_output_length_evalscope.py --input-dir eval/results/ --dry-run
"""

import argparse
import json
from pathlib import Path


def get_output_content(record: dict) -> str:
    """从单条 prediction 记录中取出模型输出文本。"""
    try:
        choices = record.get("model_output", {}).get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""
    except (IndexError, KeyError, TypeError):
        return ""


def get_output_length(record: dict) -> int:
    """
    单条记录的 output length。
    优先使用 model_output.usage.output_tokens（与 eval 的 max_length 一致）；
    若缺失则回退为 content 的字符数。
    """
    try:
        usage = (record.get("model_output") or {}).get("usage")
        if usage is not None and usage.get("output_tokens") is not None:
            return int(usage["output_tokens"])
    except (TypeError, ValueError):
        pass
    content = get_output_content(record)
    return len(content) if content else 0


def parse_subset_from_filename(stem: str, base_name: str) -> str | None:
    """
    从 prediction 文件名 stem 解析 subset 名。
    - stem == base_name（如 aime24.jsonl）-> 'default'
    - stem == base_name_suffix（如 math_500_Level 1、aime24_default）-> 返回 suffix
    """
    if stem == base_name:
        return "default"
    prefix = base_name + "_"
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix):].strip() or None


def collect_lengths_by_subset(predictions_dir: Path, base_name: str) -> tuple[dict[str, list[int]], list[int]]:
    """
    遍历 predictions_dir 下所有与 base_name 对应的 jsonl 文件，
    按 subset 收集每条记录的 output length。
    返回 (by_subset: { "Level 1": [len1, ...], ... }, all_lengths: [])。
    """
    by_subset: dict[str, list[int]] = {}
    all_lengths: list[int] = []

    for p in predictions_dir.glob("*.jsonl"):
        stem = p.stem
        subset = parse_subset_from_filename(stem, base_name)
        if subset is None:
            continue
        lengths = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    lengths.append(get_output_length(record))
                except json.JSONDecodeError:
                    continue
        if lengths:
            by_subset[subset] = lengths
            all_lengths.extend(lengths)

    return by_subset, all_lengths


def is_run_dir(path: Path) -> bool:
    """判断 path 是否为单次 run 目录（含 predictions/ 与 reports/）。"""
    return (path / "predictions").is_dir() and (path / "reports").is_dir()


def collect_run_dirs_recursive(root: Path) -> list[Path]:
    """
    从 root 递归查找所有 run 目录（含 predictions/ 与 reports/）。
    从子目录向根检测：先进入子目录，若当前目录是 run_dir 则加入结果且不再下钻，
    否则继续递归子目录。这样 root 下任意深度的所有实验 run 都会被收集。
    """
    root = root.resolve()
    if not root.is_dir():
        return []
    result: list[Path] = []

    def scan(dir_path: Path) -> None:
        if is_run_dir(dir_path):
            result.append(dir_path)
            return
        try:
            children = sorted(dir_path.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_dir():
                scan(child)

    scan(root)
    return sorted(result)


def process_one_run(run_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """对单个 run 目录执行统计并更新 report。返回 (success, fail)。"""
    predictions_root = run_dir / "predictions"
    reports_root = run_dir / "reports"
    success = 0
    fail = 0

    for model_dir in sorted(reports_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        pred_dir = predictions_root / model_name
        if not pred_dir.is_dir():
            print(f"  跳过模型 {model_name}: 无对应 predictions")
            fail += 1
            continue

        for report_path in sorted(model_dir.glob("*.json")):
            base_name = report_path.stem
            by_subset, all_lengths = collect_lengths_by_subset(pred_dir, base_name)
            if not by_subset and not all_lengths:
                print(f"  未找到与 report 对应的 prediction 文件: {report_path}")
                fail += 1
                continue

            avg_by_subset = {s: round(sum(lens) / len(lens), 2) for s, lens in by_subset.items()}
            overall_avg = round(sum(all_lengths) / len(all_lengths), 2) if all_lengths else 0.0

            if dry_run:
                print(f"  {report_path.name}: overall={overall_avg}, "
                      f"by_subset={avg_by_subset}")
                success += 1
                continue

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            report["avg_output_length"] = {
                "by_subset": avg_by_subset,
                "overall": overall_avg,
            }

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"  已写入 avg_output_length: {report_path.name} "
                  f"overall={overall_avg}, by_subset={avg_by_subset}")
            success += 1

    return success, fail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统计 evalscope evaluation 结果中的平均输出长度，并写入对应 reports 的 json"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="根目录：将递归扫描其下所有 evalscope 的 run 目录（含 predictions/ 与 reports/）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印统计结果，不写入 report",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"错误：目录不存在 {input_dir}")
        return 1

    run_dirs = collect_run_dirs_recursive(input_dir)
    print(f"共找到 {len(run_dirs)} 个 run 目录\n")

    if not run_dirs:
        print("未找到任何 run 目录（需含 predictions/ 与 reports/）。")
        return 0

    total_success = 0
    total_fail = 0
    for run_dir in run_dirs:
        try:
            rel = run_dir.relative_to(input_dir)
        except ValueError:
            rel = run_dir
        print(f"[Run] {rel}")
        s, f = process_one_run(run_dir, dry_run=args.dry_run)
        total_success += s
        total_fail += f

    print(f"\n完成: 成功 {total_success}, 失败 {total_fail}")
    return 0


if __name__ == "__main__":
    exit(main())
