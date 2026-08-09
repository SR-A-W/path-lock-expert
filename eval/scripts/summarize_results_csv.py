#!/usr/bin/env python3
"""
汇总所有 evalscope 评估结果到一个 CSV 文件（兼容 Google Sheets 导入）。

递归扫描 --input-dir 下所有模型的评估结果，提取 accuracy、avg output length、
reflective token 数、"wait, " token 数，输出到一个 CSV 表格。

支持两种目录结构：
  - 带 mode:  {model}/{benchmark}/{mode}/{timestamp}/   (PL-MoE 模型)
  - 无 mode:  {model}/{benchmark}/{timestamp}/           (baseline 模型)

Usage:
    python eval/scripts/summarize_results_csv.py \
      --input-dir eval/results/pl_moe \
      --output eval/results_summary.csv

    # 扫描多个目录（如 pl_moe + baselines）:
    python eval/scripts/summarize_results_csv.py \
      --input-dir eval/results/pl_moe eval/results/baselines \
      --output eval/results_summary.csv
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


# ── Reflective token 统计 ─────────────────────────────────────────────────────

REFLECTIVE_TOKENS = ["wait, ", "hmm", "okay", "alternatively", "let me think"]


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def count_reflective_tokens(text: str) -> dict[str, int]:
    normalized = normalize_text(text)
    counts = {}
    for token in REFLECTIVE_TOKENS:
        pattern = r'\b' + re.escape(token.lower()) + r'\b'
        counts[token] = len(re.findall(pattern, normalized))
    return counts


# ── Prediction 记录解析 ───────────────────────────────────────────────────────

def extract_content(record: dict) -> str:
    """从 evalscope prediction 记录中提取模型输出文本。"""
    model_output = record.get("model_output") or {}
    choices = model_output.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def get_output_length(record: dict) -> int:
    """获取 output token 数（优先 usage.output_tokens，回退字符数）。"""
    try:
        usage = (record.get("model_output") or {}).get("usage")
        if usage and usage.get("output_tokens") is not None:
            return int(usage["output_tokens"])
    except (TypeError, ValueError):
        pass
    return len(extract_content(record))


# ── 目录探测 ──────────────────────────────────────────────────────────────────

KNOWN_BENCHMARKS = {"math500", "aime24", "mmlu_stem", "gpqa_diamond"}
KNOWN_MODES = {"think", "no_think"}

# report 文件名映射（目录名 → report stem）
BENCHMARK_REPORT_STEMS = {
    "math500": "math_500",
    "aime24": "aime24",
    "mmlu_stem": "mmlu",
    "gpqa_diamond": "gpqa_diamond",
}


def is_run_dir(path: Path) -> bool:
    """run 目录必须有 reports 目录（评估完成的标志）。predictions 可选。"""
    return (path / "reports").is_dir()


def find_latest_run(parent: Path) -> Optional[Path]:
    """在 parent 下找到最新的 timestamp run 目录。"""
    candidates = [d for d in parent.iterdir() if d.is_dir() and is_run_dir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def auto_detect_model_id(run_dir: Path) -> Optional[str]:
    """从 predictions 或 reports 子目录中检测 model_id。"""
    for subdir_name in ("predictions", "reports"):
        root = run_dir / subdir_name
        if root.is_dir():
            subdirs = [d for d in root.iterdir() if d.is_dir()]
            if subdirs:
                return subdirs[0].name
    return None


def discover_models(input_dirs: list[Path]) -> list[dict]:
    """
    扫描所有 input_dirs，发现模型及其可用的 benchmark/mode 组合。

    返回 list of:
      {"model_name": str, "benchmark": str, "mode": str|None, "run_dir": Path}
    """
    entries = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            continue
        for model_dir in sorted(input_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for bench_dir in sorted(model_dir.iterdir()):
                if not bench_dir.is_dir() or bench_dir.name not in KNOWN_BENCHMARKS:
                    continue
                benchmark = bench_dir.name

                # 检查是否有 mode 子目录
                has_modes = any(
                    (bench_dir / m).is_dir() for m in KNOWN_MODES
                )

                if has_modes:
                    for mode in KNOWN_MODES:
                        mode_dir = bench_dir / mode
                        if not mode_dir.is_dir():
                            continue
                        run_dir = find_latest_run(mode_dir)
                        if run_dir:
                            entries.append({
                                "model_name": model_name,
                                "benchmark": benchmark,
                                "mode": mode,
                                "run_dir": run_dir,
                            })
                else:
                    # 无 mode，直接 benchmark/timestamp/
                    run_dir = find_latest_run(bench_dir)
                    if run_dir:
                        entries.append({
                            "model_name": model_name,
                            "benchmark": benchmark,
                            "mode": None,
                            "run_dir": run_dir,
                        })

    return entries


# ── 数据提取 ──────────────────────────────────────────────────────────────────

def extract_metrics(run_dir: Path, benchmark: str) -> dict[str, Any]:
    """
    从单个 run 中提取 4 个指标: acc, avg_len, #reflective, #wait。
    """
    model_id = auto_detect_model_id(run_dir)
    if not model_id:
        return {}

    report_stem = BENCHMARK_REPORT_STEMS.get(benchmark, benchmark)

    # 1. 从 report JSON 读取 accuracy、avg_output_length、reflective_token_stats
    accuracy = None
    avg_len = None
    avg_reflective = None
    avg_wait = None
    report_data = None

    report_dir = run_dir / "reports" / model_id
    if report_dir.is_dir():
        report_path = report_dir / f"{report_stem}.json"
        if not report_path.is_file():
            json_files = list(report_dir.glob("*.json"))
            report_path = json_files[0] if len(json_files) == 1 else None
        if report_path and report_path.is_file():
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            accuracy = report_data.get("score")

            # avg_output_length (report 中可能有)
            aol = report_data.get("avg_output_length")
            if isinstance(aol, dict):
                avg_len = aol.get("overall")
            elif isinstance(aol, (int, float)):
                avg_len = aol

            # reflective_token_stats (report 中可能有)
            rts = report_data.get("reflective_token_stats")
            if isinstance(rts, dict):
                avg_reflective = rts.get("avg_total_reflective_tokens_per_response")
                avg_per = rts.get("avg_tokens_per_response")
                if isinstance(avg_per, dict):
                    avg_wait = avg_per.get("wait, ")

    # 2. 如果 report 中没有 avg_len / reflective，回退到从 predictions 计算
    if avg_len is None or avg_reflective is None:
        pred_dir = run_dir / "predictions" / model_id
        all_lengths = []
        total_reflective = 0
        total_wait = 0
        total_records = 0

        if pred_dir.is_dir():
            for pred_file in sorted(pred_dir.glob("*.jsonl")):
                stem = pred_file.stem
                if stem != report_stem and not stem.startswith(report_stem + "_"):
                    continue
                with open(pred_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        total_records += 1
                        all_lengths.append(get_output_length(record))
                        content = extract_content(record)
                        if content:
                            counts = count_reflective_tokens(content)
                            total_reflective += sum(counts.values())
                            total_wait += counts.get("wait, ", 0)

            if avg_len is None and all_lengths:
                avg_len = round(sum(all_lengths) / len(all_lengths), 2)
            if avg_reflective is None and total_records > 0:
                avg_reflective = round(total_reflective / total_records, 4)
            if avg_wait is None and total_records > 0:
                avg_wait = round(total_wait / total_records, 4)

    return {
        "acc": accuracy,
        "avg_len": avg_len,
        "#reflective": avg_reflective,
        "#wait": avg_wait,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

METRIC_KEYS = ["acc", "avg_len", "#reflective", "#wait"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总 evalscope 评估结果到 CSV（兼容 Google Sheets）"
    )
    parser.add_argument(
        "--input-dir", type=Path, nargs="+", required=True,
        help="结果根目录（可指定多个），如 eval/results/pl_moe eval/results/baselines",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出 CSV 文件路径（默认为第一个 --input-dir 下的 results_summary.csv）",
    )
    args = parser.parse_args()

    # 默认 output 路径：第一个 input-dir 下
    if args.output is None:
        args.output = str(args.input_dir[0] / "results_summary.csv")

    # 1. 发现所有模型/benchmark/mode
    entries = discover_models(args.input_dir)
    if not entries:
        print("未找到任何评测结果。")
        return 1

    # 2. 确定所有出现过的 benchmarks（保持固定顺序）
    benchmark_order = [b for b in ["math500", "aime24", "mmlu_stem", "gpqa_diamond"]
                       if any(e["benchmark"] == b for e in entries)]

    # 3. 按 (model, mode) 分组提取数据
    # row_key = (model_name, mode_label)
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    row_source: dict[tuple[str, str], str] = {}  # 记录每行数据的来源目录
    for entry in entries:
        model = entry["model_name"]
        mode = entry["mode"]  # None for single-mode
        mode_label = mode if mode else "-"
        benchmark = entry["benchmark"]
        run_dir = entry["run_dir"]

        row_key = (model, mode_label)
        if row_key not in rows:
            rows[row_key] = {}
            # 来源目录: run_dir 往上 3~4 级即模型目录
            # e.g. eval/results/pl_moe/model_name/benchmark/mode/timestamp
            # 取到 model_name 那一级的 parent
            model_dir = run_dir
            while model_dir.name != model and model_dir != model_dir.parent:
                model_dir = model_dir.parent
            row_source[row_key] = str(model_dir)

        print(f"  提取 {model}/{benchmark}/{mode_label} ← {run_dir.name}")
        metrics = extract_metrics(run_dir, benchmark)
        rows[row_key][benchmark] = metrics

    # 4. 构建 CSV
    # 列头: model, mode, benchmark1_acc, benchmark1_avg_len, benchmark1_#reflective, benchmark1_#wait, ...
    header = ["model", "mode"]
    for bench in benchmark_order:
        for metric in METRIC_KEYS:
            header.append(f"{bench}_{metric}")
    header.append("source_dir")

    # 排序行: 按 model 名排序，同 model 内 think 在前 no_think 在后
    mode_sort = {"think": 0, "no_think": 1, "-": 2}
    sorted_keys = sorted(rows.keys(), key=lambda k: (k[0], mode_sort.get(k[1], 9)))

    csv_rows = []
    for model, mode_label in sorted_keys:
        row = [model, mode_label]
        bench_data = rows[(model, mode_label)]
        for bench in benchmark_order:
            metrics = bench_data.get(bench, {})
            for metric in METRIC_KEYS:
                val = metrics.get(metric)
                if val is None:
                    row.append("")
                else:
                    row.append(str(val))
        row.append(row_source.get((model, mode_label), ""))
        csv_rows.append(row)

    # 5. 写入 CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"\n已写入: {output_path}")
    print(f"  模型数: {len(set(k[0] for k in sorted_keys))}")
    print(f"  总行数: {len(csv_rows)}")
    print(f"  Benchmarks: {benchmark_order}")

    # 打印预览
    print(f"\n{'─' * 40} 预览 {'─' * 40}")
    # 用固定宽度打印前几行
    col_widths = [max(len(header[i]), max((len(r[i]) for r in csv_rows), default=0)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("  ".join("─" * w for w in col_widths))
    for row in csv_rows:
        print(fmt.format(*row))

    return 0


if __name__ == "__main__":
    exit(main())
