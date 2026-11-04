#!/usr/bin/env python3
"""
Average Output Length Statistics Script for EvalScope Results

Compute the token length of every output in the evalscope evaluation results,
and write the statistics (avg_output_length) into the json files under the corresponding reports directory.

Output length: prefers model_output.usage.output_tokens (consistent with the eval max_length),
falling back to the character count of the content if it is missing.

Usage:
    python eval/scripts/avg_output_length_evalscope.py --input-dir eval/results/pl_moe/qwen_pl_moe_140k
    python eval/scripts/avg_output_length_evalscope.py --input-dir eval/results/ --dry-run
"""

import argparse
import json
from pathlib import Path


def get_output_content(record: dict) -> str:
    """Extract the model output text from a single prediction record."""
    try:
        choices = record.get("model_output", {}).get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""
    except (IndexError, KeyError, TypeError):
        return ""


def get_output_length(record: dict) -> int:
    """
    Output length of a single record.
    Prefers model_output.usage.output_tokens (consistent with the eval max_length);
    falls back to the character count of the content if it is missing.
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
    Parse the subset name from the prediction file name stem.
    - stem == base_name(e.g. aime24.jsonl)-> 'default'
    - stem == base_name_suffix(e.g. math_500_Level 1, aime24_default) -> return the suffix
    """
    if stem == base_name:
        return "default"
    prefix = base_name + "_"
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix):].strip() or None


def collect_lengths_by_subset(predictions_dir: Path, base_name: str) -> tuple[dict[str, list[int]], list[int]]:
    """
    Iterate over all jsonl files under predictions_dir that correspond to base_name,
    collecting the output length of every record per subset.
    Returns (by_subset: { "Level 1": [len1, ...], ... }, all_lengths: [])。
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
    """Check whether path is a single run directory (containing predictions/ and reports/)."""
    return (path / "predictions").is_dir() and (path / "reports").is_dir()


def collect_run_dirs_recursive(root: Path) -> list[Path]:
    """
    Recursively find all run directories (containing predictions/ and reports/) under root.
    Walk down from the root: if the current directory is a run directory it is added and not descended into,
    otherwise recurse into its subdirectories. This collects runs at any depth under root.
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
    """Compute statistics for a single run directory and update the report. Returns (success, fail)."""
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
            print(f"  Skipping model {model_name}: no matching predictions")
            fail += 1
            continue

        for report_path in sorted(model_dir.glob("*.json")):
            base_name = report_path.stem
            by_subset, all_lengths = collect_lengths_by_subset(pred_dir, base_name)
            if not by_subset and not all_lengths:
                print(f"  No prediction file found for report: {report_path}")
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

            print(f"  Wrote avg_output_length: {report_path.name} "
                  f"overall={overall_avg}, by_subset={avg_by_subset}")
            success += 1

    return success, fail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the average output length in evalscope evaluation results and write it into the corresponding reports json"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Root directory: all evalscope run directories (containing predictions/ and reports/) below it are scanned recursively",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print the statistics; do not write to the report",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"Error: directory does not exist {input_dir}")
        return 1

    run_dirs = collect_run_dirs_recursive(input_dir)
    print(f"Found {len(run_dirs)} run directories\n")

    if not run_dirs:
        print("No run directory found (it must contain predictions/ and reports/).")
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

    print(f"\nDone: succeeded {total_success}, failed {total_fail}")
    return 0


if __name__ == "__main__":
    exit(main())
