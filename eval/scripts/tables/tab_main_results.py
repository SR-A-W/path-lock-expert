#!/usr/bin/env python3
"""
Generate the main results table (all models x 4 benchmarks x think/no_think: accuracy + average output length).

Usage:
    python eval/scripts/tables/tab_main_results.py \
      --results-dir eval/results_json \
      --output eval/tables/tab_main_results.tex
"""

import argparse
import sys
from pathlib import Path

# Add eval/scripts to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from table_utils import (
    load_all_models,
    get_result,
    fmt_pct,
    fmt_int,
    fmt_bold_best,
    latex_row,
    latex_toprule,
    latex_midrule,
    latex_bottomrule,
)

# Model order and the corresponding column headers
# (model_key, column_header)
# Edit this list according to the models you actually have
MODELS = [
    ("qwen_pl_moe_140k", "Qwen2.5-7B PL-MoE"),
    ("qwen_pl_moe_20k", "Qwen2.5-7B PL-MoE (20k)"),
    # Further models can be added, e.g.:
    # ("qwen3_4b_instruct_pl_moe_140k", "Qwen3-4B-Instruct PL-MoE"),
    # ("qwen25_7b_baseline", "Qwen2.5-7B Base"),
]

BENCHMARKS = [
    ("math500", "MATH-500"),
    ("aime24", "AIME-2024"),
    ("mmlu_stem", "MMLU-STEM"),
    ("gpqa_diamond", "GPQA-Diamond"),
]

MODE_LABELS = {
    "think": "Think",
    "no_think": "No-Think",
}


def generate_table(results_dir: str, output_path: str):
    all_data = load_all_models(results_dir)

    # Keep only the models that actually exist
    available_models = [(k, h) for k, h in MODELS if k in all_data]
    if not available_models:
        print(f"Warning: no matching models in results_dir. Available: {list(all_data.keys())}")
        return

    n_models = len(available_models)
    # Two columns per model: Acc, Len
    n_cols = 2 + n_models * 2  # Benchmark + Mode + (Acc + Len) * n_models

    lines = []
    # tabular environment
    col_spec = "ll" + "rr" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(latex_toprule())

    # Header row 1: model name (spanning 2 columns)
    header1_cells = ["", ""]
    for _, header in available_models:
        header1_cells.append(f"\\multicolumn{{2}}{{c}}{{{header}}}")
    # multicolumn spans 2 columns, so the number of & separators needs adjusting
    # Built manually
    header1 = " & ".join(["", ""] + [
        f"\\multicolumn{{2}}{{c}}{{{h}}}" for _, h in available_models
    ]) + " \\\\"
    lines.append(header1)

    # Header row 2: Acc / Len
    header2_cells = ["Benchmark", "Mode"]
    for _ in available_models:
        header2_cells.extend(["Acc", "Len"])
    lines.append(latex_row(*header2_cells))
    lines.append(latex_midrule())

    # Data rows
    for bench_key, bench_name in BENCHMARKS:
        for mode_idx, mode in enumerate(["think", "no_think"]):
            cells = []
            # Show the benchmark name only for the first mode
            if mode_idx == 0:
                cells.append(bench_name)
            else:
                cells.append("")
            cells.append(MODE_LABELS[mode])

            # Collect each model's acc and len to bold the best one
            acc_values = []
            len_values = []
            for model_key, _ in available_models:
                r = get_result(all_data[model_key], bench_key, mode)
                acc_values.append(r.get("accuracy") if r else None)
                len_values.append(r.get("avg_output_length") if r else None)

            # Formatting
            acc_fmt = [fmt_pct(v) for v in acc_values]
            len_fmt = [fmt_int(v) for v in len_values]

            # Bold the best value (higher accuracy is better; shorter length is better in no_think mode)
            if len(available_models) > 1:
                acc_fmt = fmt_bold_best(acc_values, acc_fmt, higher_is_better=True)
                if mode == "no_think":
                    len_fmt = fmt_bold_best(len_values, len_fmt, higher_is_better=False)

            for i in range(n_models):
                cells.extend([acc_fmt[i], len_fmt[i]])

            lines.append(latex_row(*cells))

        # Add a separator line between benchmarks (except after the last one)
        if bench_key != BENCHMARKS[-1][0]:
            lines.append(latex_midrule())

    lines.append(latex_bottomrule())
    lines.append("\\end{tabular}")

    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated: {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate the main results LaTeX table")
    parser.add_argument("--results-dir", type=str, default="eval/results_json")
    parser.add_argument("--output", type=str, default="eval/tables/tab_main_results.tex")
    args = parser.parse_args()
    generate_table(args.results_dir, args.output)


if __name__ == "__main__":
    main()
