#!/usr/bin/env python3
"""
Reflective Token Statistics Script for EvalScope Results

Count the occurrences and frequency of reflective tokens in the evalscope evaluation results,
and write the result as a new field into the json files under the corresponding reports directory.

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


# Default list of reflective tokens
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
    """Normalize the content field to a string (may be a str or a list of dicts such as [{"type":"...", "text":"..."}])."""
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
    Extract the model output text from a single evalscope jsonl record.
    Supports model_output.choices[0].message.content or model_output.messages[0].content;
    content may be a string or a list of dicts (with 'text'/'content').
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
    Process one evalscope .jsonl prediction file and count reflective tokens.

    Returns:
        Statistics dict to be written into the report.
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
    Find the matching report json path from the jsonl file name under predictions.
    Rule: <report_base>.json exists under reports/<model_id>/ and report_base is a prefix of prediction_stem.
    If several match, the report with the longest prefix is used.
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
    Recursively scan root_dir to find all evalscope predictions (*.jsonl),
    and resolve the matching report json path (if it exists).
    Expected directory layout: .../run/predictions/<model_id>/<bench>.jsonl and .../run/reports/<model_id>/<report>.json
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
        description='Count reflective tokens in evalscope evaluation results and write them into the corresponding reports json'
    )
    parser.add_argument(
        '--input-dir', type=str, required=True,
        help='Root directory: all evalscope predictions and reports below it are scanned recursively',
    )
    parser.add_argument(
        '--tokens', type=str, nargs='*', default=REFLECTIVE_TOKENS,
        help='List of reflective tokens to count (defaults to the built-in list)',
    )
    parser.add_argument(
        '--key', type=str, default='reflective_token_stats',
        help='Key name used when writing into the report json (default: reflective_token_stats)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Only print the files that would be processed; do not write to the report',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: directory does not exist {input_dir}")
        return 1

    pairs = find_evalscope_predictions_and_reports(input_dir)
    # De-duplicate (the same (pred, report) pair may be found twice by rglob)
    seen = set()
    unique_pairs = []
    for p, r in pairs:
        k = (str(p), str(r))
        if k not in seen:
            seen.add(k)
            unique_pairs.append((p, r))

    print(f"Found {len(unique_pairs)} evalscope benchmarks (predictions + matching report)")

    if not unique_pairs:
        print("No predictions/report pair found; check that the directory contains the evalscope predictions and reports layout.")
        return 0

    success = 0
    fail = 0
    for i, (pred_path, report_path) in enumerate(unique_pairs, 1):
        print(f"[{i}/{len(unique_pairs)}] {pred_path.name} -> {report_path}")
        try:
            stats = process_evalscope_jsonl(pred_path, args.tokens)
        except Exception as e:
            print(f"  Processing failed: {e}")
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
            print(f"  Failed to read report: {e}")
            fail += 1
            continue
        report[args.key] = stats
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Failed to write report: {e}")
            fail += 1
            continue
        print(f"  Wrote {args.key}: total_responses={stats['total_responses']}, "
              f"avg_reflective={stats['avg_total_reflective_tokens_per_response']:.4f}")
        success += 1

    print(f"\nDone: succeeded {success}, failed {fail}")
    return 0


if __name__ == '__main__':
    exit(main())
