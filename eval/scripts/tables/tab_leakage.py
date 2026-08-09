#!/usr/bin/env python3
"""
生成 Tab.2: Leakage 对比表（no_think 模式下 PL-MoE vs Baseline 的 reflective token 和 output length）。

Usage:
    python eval/scripts/tables/tab_leakage.py \
      --results-dir eval/results_json \
      --output eval/tables/tab_leakage.tex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from table_utils import (
    load_all_models,
    get_result,
    fmt_float,
    fmt_int,
    fmt_pct,
    fmt_reduction,
    latex_row,
    latex_toprule,
    latex_midrule,
    latex_bottomrule,
)

# (PL-MoE model key, Baseline model key, group label)
# 用户可根据实际可用模型修改
MODEL_PAIRS = [
    ("qwen_pl_moe_140k", "qwen25_7b_baseline", "Qwen2.5-7B"),
    # ("qwen3_4b_instruct_pl_moe_140k", "qwen3_4b_baseline", "Qwen3-4B-Instruct"),
]

BENCHMARKS = [
    ("math500", "MATH-500"),
    ("aime24", "AIME-2024"),
    ("mmlu_stem", "MMLU-STEM"),
    ("gpqa_diamond", "GPQA-Diamond"),
]


def generate_table(results_dir: str, output_path: str):
    all_data = load_all_models(results_dir)

    # 过滤出两端都可用的 model pair
    available_pairs = [
        (pl, bl, label) for pl, bl, label in MODEL_PAIRS
        if pl in all_data and bl in all_data
    ]

    # 如果没有 baseline，退化为仅展示 PL-MoE 的 no_think 数据
    solo_models = []
    if not available_pairs:
        # 没有 pair，展示所有可用模型的 no_think leakage 数据
        for pl, bl, label in MODEL_PAIRS:
            if pl in all_data:
                solo_models.append((pl, label))
        if not solo_models:
            print(f"警告: 无可用模型。可用: {list(all_data.keys())}")
            return

    lines = []

    if available_pairs:
        # 有 baseline 对比的完整表格
        for pair_idx, (pl_key, bl_key, group_label) in enumerate(available_pairs):
            pl_data = all_data[pl_key]
            bl_data = all_data[bl_key]

            if pair_idx == 0:
                lines.append("\\begin{tabular}{lrrrrrr}")
                lines.append(latex_toprule())
                header = latex_row(
                    "", "\\multicolumn{2}{c}{Baseline No-Think}",
                    "\\multicolumn{2}{c}{PL-MoE No-Think}",
                    "\\multicolumn{2}{c}{Reduction}",
                )
                lines.append(header)
                sub_header = latex_row(
                    "Benchmark", "Avg Refl.", "Avg Len",
                    "Avg Refl.", "Avg Len",
                    "Refl. $\\downarrow$", "Len $\\downarrow$",
                )
                lines.append(sub_header)
                lines.append(latex_midrule())

            for bench_key, bench_name in BENCHMARKS:
                bl_r = get_result(bl_data, bench_key, "no_think")
                pl_r = get_result(pl_data, bench_key, "no_think")

                bl_refl = bl_r.get("reflective_token_stats", {}).get("avg_count") if bl_r else None
                bl_len = bl_r.get("avg_output_length") if bl_r else None
                pl_refl = pl_r.get("reflective_token_stats", {}).get("avg_count") if pl_r else None
                pl_len = pl_r.get("avg_output_length") if pl_r else None

                lines.append(latex_row(
                    bench_name,
                    fmt_float(bl_refl), fmt_int(bl_len),
                    fmt_float(pl_refl), fmt_int(pl_len),
                    fmt_reduction(pl_refl, bl_refl),
                    fmt_reduction(pl_len, bl_len),
                ))

        lines.append(latex_bottomrule())
        lines.append("\\end{tabular}")

    else:
        # 仅展示 PL-MoE 的 no_think leakage 数据（无 baseline）
        lines.append("\\begin{tabular}{lrrr}")
        lines.append(latex_toprule())
        lines.append(latex_row("Benchmark", "Avg Refl.", "Frac w/ Refl.", "Avg Len"))
        lines.append(latex_midrule())

        for model_key, label in solo_models:
            data = all_data[model_key]
            for bench_key, bench_name in BENCHMARKS:
                r = get_result(data, bench_key, "no_think")
                refl = r.get("reflective_token_stats", {}).get("avg_count") if r else None
                frac = r.get("reflective_token_stats", {}).get("samples_with_reflective") if r else None
                avg_len = r.get("avg_output_length") if r else None

                frac_str = f"{fmt_pct(frac)}\\%" if frac is not None else "-"
                lines.append(latex_row(
                    bench_name,
                    fmt_float(refl),
                    frac_str,
                    fmt_int(avg_len),
                ))

        lines.append(latex_bottomrule())
        lines.append("\\end{tabular}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已生成: {output}")


def main():
    parser = argparse.ArgumentParser(description="生成 Leakage 对比表 LaTeX")
    parser.add_argument("--results-dir", type=str, default="eval/results_json")
    parser.add_argument("--output", type=str, default="eval/tables/tab_leakage.tex")
    args = parser.parse_args()
    generate_table(args.results_dir, args.output)


if __name__ == "__main__":
    main()
