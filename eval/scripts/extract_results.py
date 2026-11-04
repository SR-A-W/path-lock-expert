#!/usr/bin/env python3
"""
Result extraction script: consolidates the data scattered across an evalscope result directory into a standardized JSON.

Extracted from evalscope's predictions, reports and reviews:
- accuracy (from the report)
- avg_output_length (computed directly from predictions)
- reflective_token_stats (computed directly from predictions, which avoids the report overwriting multiple subsets)
- per_sample_data (predictions and reviews aligned)

Usage:
    python eval/scripts/extract_results.py \
      --result-dir eval/results/pl_moe/qwen_pl_moe_140k \
      --config eval/scripts/model_configs/qwen_pl_moe_140k.yaml \
      --output-dir eval/results_json

    # Or use CLI arguments instead of a config:
    python eval/scripts/extract_results.py \
      --result-dir eval/results/pl_moe/qwen_pl_moe_140k \
      --model-name qwen_pl_moe_140k \
      --display-name "Qwen2.5-7B PL-MoE (140k)" \
      --base-model "Qwen2.5-7B-Instruct" \
      --dataset "naive-mix-140k" \
      --output-dir eval/results_json
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Constants ──────────────────────────────────────────────────────────────────────

# Directory name -> standardized key
BENCHMARK_DIR_TO_KEY = {
    "math500": "math500",
    "aime24": "aime24",
    "mmlu_stem": "mmlu_stem",
    "gpqa_diamond": "gpqa_diamond",
}

# Standardized key -> report file stem (without .json)
BENCHMARK_REPORT_NAMES = {
    "math500": "math_500",
    "aime24": "aime24",
    "mmlu_stem": "mmlu",
    "gpqa_diamond": "gpqa_diamond",
}

MODES = ["think", "no_think"]

REFLECTIVE_TOKENS = ["wait, ", "hmm", "okay", "alternatively", "let me think"]

# ── Helper functions ──────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Normalize text for token matching."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def count_reflective_tokens(text: str, tokens: list[str] | None = None) -> dict[str, int]:
    """Count occurrences of reflective tokens in the text."""
    if tokens is None:
        tokens = REFLECTIVE_TOKENS
    normalized = normalize_text(text)
    counts = {}
    for token in tokens:
        pattern = r'\b' + re.escape(token.lower()) + r'\b'
        counts[token] = len(re.findall(pattern, normalized))
    return counts


def extract_content(record: dict) -> str | None:
    """Extract the model output text from an evalscope prediction record."""
    model_output = record.get("model_output") or {}
    # Try choices[0].message.content
    choices = model_output.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                b.get("text") or b.get("content", "")
                for b in content if isinstance(b, dict)
            ]
            joined = " ".join(parts).strip()
            if joined:
                return joined
    # Try messages[0].content
    messages = model_output.get("messages") or []
    if messages:
        content = messages[0].get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def get_output_length(record: dict) -> int:
    """Get the number of output tokens of one record (prefers usage.output_tokens, falls back to the character count)."""
    try:
        usage = (record.get("model_output") or {}).get("usage")
        if usage and usage.get("output_tokens") is not None:
            return int(usage["output_tokens"])
    except (TypeError, ValueError):
        pass
    content = extract_content(record) or ""
    return len(content)


# ── Directory detection ──────────────────────────────────────────────────────────────────


def find_latest_complete_run(benchmark_dir: Path, mode: str) -> Path | None:
    """Find the latest timestamp directory containing reports/ under benchmark_dir/mode/."""
    mode_dir = benchmark_dir / mode
    if not mode_dir.is_dir():
        return None
    candidates = []
    for d in mode_dir.iterdir():
        if d.is_dir() and (d / "reports").is_dir() and (d / "predictions").is_dir():
            candidates.append(d)
    if not candidates:
        return None
    # Timestamp directory names look like YYYYMMDD_HHMMSS, so lexicographic order is chronological order
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def auto_detect_model_id(run_dir: Path) -> str | None:
    """Auto-detect the model_id subdirectory name under predictions/."""
    pred_root = run_dir / "predictions"
    if not pred_root.is_dir():
        return None
    subdirs = [d for d in pred_root.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0].name
    if len(subdirs) > 1:
        warnings.warn(f"Multiple subdirectories under predictions/: {[d.name for d in subdirs]}; using the first one")
        return subdirs[0].name
    return None


# ── Data extraction ──────────────────────────────────────────────────────────────────


def load_report(run_dir: Path, model_id: str, report_stem: str) -> dict | None:
    """Load the report JSON file."""
    report_path = run_dir / "reports" / model_id / f"{report_stem}.json"
    if not report_path.is_file():
        # Try to find any .json under reports/model_id/
        reports_dir = run_dir / "reports" / model_id
        if reports_dir.is_dir():
            json_files = list(reports_dir.glob("*.json"))
            if len(json_files) == 1:
                report_path = json_files[0]
            else:
                warnings.warn(f"Cannot match report: {report_path}; the directory contains {[f.name for f in json_files]}")
                return None
        else:
            return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_accuracy(report: dict) -> float | None:
    """Extract the overall accuracy from the report."""
    return report.get("score")


def extract_accuracy_by_subset(report: dict) -> dict[str, float]:
    """Extract the per-subset accuracy from the report."""
    subsets = {}
    metrics = report.get("metrics") or []
    if not metrics:
        return subsets
    # Take the first metric (usually mean_acc)
    first_metric = metrics[0]
    categories = first_metric.get("categories") or []
    for cat in categories:
        for s in cat.get("subsets") or []:
            subsets[s["name"]] = s["score"]
    return subsets


def compute_stats_from_predictions(
    pred_dir: Path,
    report_stem: str,
) -> tuple[list[int], dict[str, Any]]:
    """
    Compute output_length and reflective_token_stats directly from the predictions directory.
    Returns (all_lengths, reflective_stats_dict).
    """
    all_lengths: list[int] = []
    total_token_counts: dict[str, int] = defaultdict(int)
    total_reflective_per_response: list[int] = []
    total_responses = 0

    # Collect all prediction files matching report_stem
    pred_files = []
    for p in pred_dir.glob("*.jsonl"):
        stem = p.stem
        # Matching rule: stem == report_stem, or stem starts with report_stem_
        if stem == report_stem or stem.startswith(report_stem + "_"):
            pred_files.append(p)

    for pred_file in sorted(pred_files):
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # output length
                all_lengths.append(get_output_length(record))
                # reflective tokens
                content = extract_content(record)
                if content:
                    total_responses += 1
                    counts = count_reflective_tokens(content)
                    for token, count in counts.items():
                        total_token_counts[token] += count
                    total_reflective_per_response.append(sum(counts.values()))

    # Aggregate reflective stats
    avg_reflective = (
        sum(total_reflective_per_response) / len(total_reflective_per_response)
        if total_reflective_per_response else 0.0
    )
    avg_per_token = {}
    for token in REFLECTIVE_TOKENS:
        avg_per_token[token] = (
            total_token_counts[token] / total_responses if total_responses > 0 else 0.0
        )
    # Fraction of samples containing reflective tokens
    samples_with = sum(1 for c in total_reflective_per_response if c > 0)
    frac_with = samples_with / len(total_reflective_per_response) if total_reflective_per_response else 0.0

    reflective_stats = {
        "avg_count": round(avg_reflective, 4),
        "samples_with_reflective": round(frac_with, 4),
        "total_responses": total_responses,
        "per_token_avg": {k: round(v, 4) for k, v in avg_per_token.items()},
        "total_token_counts": dict(total_token_counts),
    }

    return all_lengths, reflective_stats


def extract_per_sample_data(
    pred_dir: Path,
    review_dir: Path,
    report_stem: str,
) -> list[dict]:
    """
    Align predictions and reviews and extract per-sample data.
    Returns [{"output_length": int, "correct": bool, "reflective_count": int}, ...]
    """
    samples = []

    pred_files = sorted([
        p for p in pred_dir.glob("*.jsonl")
        if p.stem == report_stem or p.stem.startswith(report_stem + "_")
    ])

    for pred_file in pred_files:
        # The corresponding review file has the same name
        review_file = review_dir / pred_file.name
        # Read predictions
        pred_records = []
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pred_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Read reviews (if present)
        review_records = []
        if review_file.is_file():
            with open(review_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        review_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Align line by line
        for i, pred in enumerate(pred_records):
            output_length = get_output_length(pred)
            content = extract_content(pred)
            reflective_count = sum(count_reflective_tokens(content).values()) if content else 0

            correct = None
            if i < len(review_records):
                try:
                    acc = review_records[i]["sample_score"]["score"]["value"]["acc"]
                    correct = bool(acc >= 0.5)
                except (KeyError, TypeError):
                    pass

            samples.append({
                "output_length": output_length,
                "correct": correct,
                "reflective_count": reflective_count,
            })

    return samples


# ── Main flow ────────────────────────────────────────────────────────────────────


def process_model(
    result_dir: Path,
    model_name: str,
    display_name: str,
    base_model: str,
    dataset: str,
    include_per_sample: bool = True,
) -> dict:
    """Process all benchmark results of a single model and return a standardized JSON dict."""
    output = {
        "model": model_name,
        "model_display_name": display_name,
        "base_model": base_model,
        "dataset": dataset,
        "eval_date": None,
        "results": {},
    }
    if include_per_sample:
        output["per_sample_data"] = {}

    for bench_dir_name, bench_key in BENCHMARK_DIR_TO_KEY.items():
        bench_dir = result_dir / bench_dir_name
        if not bench_dir.is_dir():
            warnings.warn(f"Benchmark directory does not exist: {bench_dir}")
            continue

        report_stem = BENCHMARK_REPORT_NAMES[bench_key]
        output["results"][bench_key] = {}
        if include_per_sample:
            output["per_sample_data"][bench_key] = {}

        for mode in MODES:
            run_dir = find_latest_complete_run(bench_dir, mode)
            if run_dir is None:
                warnings.warn(f"No complete run: {bench_dir}/{mode}/")
                continue

            # Record eval_date (timestamp of the latest run)
            try:
                ts = run_dir.name  # e.g. "20260212_004651"
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%Y-%m-%d")
                if output["eval_date"] is None or date_str > output["eval_date"]:
                    output["eval_date"] = date_str
            except ValueError:
                pass

            model_id = auto_detect_model_id(run_dir)
            if model_id is None:
                warnings.warn(f"Cannot detect model_id: {run_dir}")
                continue

            pred_dir = run_dir / "predictions" / model_id
            review_dir = run_dir / "reviews" / model_id

            # Load the report to get accuracy
            report = load_report(run_dir, model_id, report_stem)
            accuracy = extract_accuracy(report) if report else None
            accuracy_by_subset = extract_accuracy_by_subset(report) if report else {}

            # Compute output_length and reflective stats directly from predictions
            all_lengths, reflective_stats = compute_stats_from_predictions(pred_dir, report_stem)
            avg_output_length = (
                round(sum(all_lengths) / len(all_lengths), 2) if all_lengths else None
            )

            result_entry = {
                "accuracy": accuracy,
                "avg_output_length": avg_output_length,
                "reflective_token_stats": reflective_stats,
            }
            if accuracy_by_subset:
                result_entry["accuracy_by_subset"] = accuracy_by_subset

            output["results"][bench_key][mode] = result_entry

            # per-sample data
            if include_per_sample and pred_dir.is_dir():
                per_sample = extract_per_sample_data(pred_dir, review_dir, report_stem)
                if per_sample:
                    output["per_sample_data"][bench_key][mode] = per_sample

            print(f"  [{bench_key}/{mode}] acc={accuracy}, avg_len={avg_output_length}, "
                  f"avg_reflective={reflective_stats['avg_count']}, run={run_dir.name}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract evalscope results into a standardized JSON"
    )
    parser.add_argument(
        "--result-dir", type=Path, required=True,
        help="evalscope result root directory (e.g. eval/results/pl_moe/qwen_pl_moe_140k)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Model metadata YAML config file",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--display-name", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval/results_json"),
        help="Output directory (default eval/results_json)",
    )
    parser.add_argument(
        "--no-per-sample", action="store_true",
        help="Do not include per_sample_data (reduces file size)",
    )
    args = parser.parse_args()

    # Get metadata from the config or the CLI
    if args.config and args.config.is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get("model", args.model_name)
        display_name = cfg.get("display_name", args.display_name)
        base_model = cfg.get("base_model", args.base_model)
        dataset = cfg.get("dataset", args.dataset)
    else:
        model_name = args.model_name
        display_name = args.display_name
        base_model = args.base_model
        dataset = args.dataset

    if not model_name:
        # Infer from the result-dir name
        model_name = args.result_dir.name
    if not display_name:
        display_name = model_name
    if not base_model:
        base_model = ""
    if not dataset:
        dataset = ""

    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Result directory does not exist: {result_dir}")

    print(f"Extracting model: {model_name}")
    print(f"Result directory: {result_dir}")

    output = process_model(
        result_dir=result_dir,
        model_name=model_name,
        display_name=display_name,
        base_model=base_model,
        dataset=dataset,
        include_per_sample=not args.no_per_sample,
    )

    # Write JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
