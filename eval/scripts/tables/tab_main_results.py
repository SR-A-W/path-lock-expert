#!/usr/bin/env python3
"""
生成 Tab.1: 主结果表（所有模型 × 4 benchmarks × think/no_think 的 accuracy + avg output length）。

Usage:
    python eval/scripts/tables/tab_main_results.py \
      --results-dir eval/results_json \
      --output eval/tables/tab_main_results.tex
"""

import argparse
import sys
from pathlib import Path

# 添加 eval/scripts 到 path
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

# 模型顺序和对应的列标题
# (model_key, column_header)
# 用户可根据实际可用模型修改此列表
MODELS = [
    ("qwen_pl_moe_140k", "Qwen2.5-7B PL-MoE"),
    ("qwen_pl_moe_20k", "Qwen2.5-7B PL-MoE (20k)"),
    # 后续可添加:
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

    # 过滤出实际存在的模型
    available_models = [(k, h) for k, h in MODELS if k in all_data]
    if not available_models:
        print(f"警告: results_dir 中无匹配模型。可用: {list(all_data.keys())}")
        return

    n_models = len(available_models)
    # 每个模型 2 列: Acc, Len
    n_cols = 2 + n_models * 2  # Benchmark + Mode + (Acc + Len) * n_models

    lines = []
    # tabular 环境
    col_spec = "ll" + "rr" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(latex_toprule())

    # 表头第一行: 模型名（跨 2 列）
    header1_cells = ["", ""]
    for _, header in available_models:
        header1_cells.append(f"\\multicolumn{{2}}{{c}}{{{header}}}")
    # multicolumn 占 2 列，所以实际 & 数量需要调整
    # 手动构建
    header1 = " & ".join(["", ""] + [
        f"\\multicolumn{{2}}{{c}}{{{h}}}" for _, h in available_models
    ]) + " \\\\"
    lines.append(header1)

    # 表头第二行: Acc / Len
    header2_cells = ["Benchmark", "Mode"]
    for _ in available_models:
        header2_cells.extend(["Acc", "Len"])
    lines.append(latex_row(*header2_cells))
    lines.append(latex_midrule())

    # 数据行
    for bench_key, bench_name in BENCHMARKS:
        for mode_idx, mode in enumerate(["think", "no_think"]):
            cells = []
            # Benchmark 名只在第一个 mode 显示
            if mode_idx == 0:
                cells.append(bench_name)
            else:
                cells.append("")
            cells.append(MODE_LABELS[mode])

            # 收集各模型的 acc 和 len 用于加粗最优
            acc_values = []
            len_values = []
            for model_key, _ in available_models:
                r = get_result(all_data[model_key], bench_key, mode)
                acc_values.append(r.get("accuracy") if r else None)
                len_values.append(r.get("avg_output_length") if r else None)

            # 格式化
            acc_fmt = [fmt_pct(v) for v in acc_values]
            len_fmt = [fmt_int(v) for v in len_values]

            # 加粗最优（accuracy 越高越好，length 在 no_think 模式越短越好）
            if len(available_models) > 1:
                acc_fmt = fmt_bold_best(acc_values, acc_fmt, higher_is_better=True)
                if mode == "no_think":
                    len_fmt = fmt_bold_best(len_values, len_fmt, higher_is_better=False)

            for i in range(n_models):
                cells.extend([acc_fmt[i], len_fmt[i]])

            lines.append(latex_row(*cells))

        # benchmark 之间加分隔线（最后一个除外）
        if bench_key != BENCHMARKS[-1][0]:
            lines.append(latex_midrule())

    lines.append(latex_bottomrule())
    lines.append("\\end{tabular}")

    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已生成: {output}")


def main():
    parser = argparse.ArgumentParser(description="生成主结果表 LaTeX")
    parser.add_argument("--results-dir", type=str, default="eval/results_json")
    parser.add_argument("--output", type=str, default="eval/tables/tab_main_results.tex")
    args = parser.parse_args()
    generate_table(args.results_dir, args.output)


if __name__ == "__main__":
    main()
