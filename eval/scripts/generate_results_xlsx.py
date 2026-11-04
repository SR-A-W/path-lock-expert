#!/usr/bin/env python3
"""
Generate a formatted Excel summary of the experiment results.

Reads the CSV produced by summarize_results_csv.py
and writes an xlsx file with conditional formatting, colors, bold text, etc.

Usage:
    conda run -n evalscope python eval/scripts/generate_results_xlsx.py \
        --input /tmp/data_analyst_results.csv \
        --output eval/results/results_summary.xlsx
"""

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# ── Directory name -> display name mapping ─────────────────────────────────────────────────────

FORMAL_NAMES = {
    # Baselines
    "qwen3_4b_instruct_2507": "Qwen3-4B-Instruct (Baseline)",
    "qwen3_4b_instruct_140k_2phase": "Qwen3-4B-Instruct Baseline naive_mix_140k_2phase",
    "qwen3_4b_instruct_superior_65k_27k_2phase": "Qwen3-4B-Instruct Baseline superior_65k+27k_2phase",
    # Qwen3-4B-Instruct PL-MoE
    "qwen3_4b_instruct_pl_moe_20k": "Qwen3-4B-Instruct PL-MoE naive_mix_20k",
    "qwen3_4b_instruct_pl_moe_20k_a100": "Qwen3-4B-Instruct PL-MoE naive_mix_20k (A100)",
    "qwen3_4b_instruct_pl_moe_140k": "Qwen3-4B-Instruct PL-MoE naive_mix_140k",
    "qwen3_4b_instruct_pl_moe_superior_27k_27k": "Qwen3-4B-Instruct PL-MoE superior_reasoning_27k+27k",
    "qwen3_4b_instruct_pl_moe_superior_65k_27k": "Qwen3-4B-Instruct PL-MoE superior_reasoning_65k+27k",
    # Qwen3-4B-Base PL-MoE
    "qwen3_4b_base_pl_moe_20k": "Qwen3-4B-Base PL-MoE naive_mix_20k",
    "qwen3_4b_base_pl_moe_140k": "Qwen3-4B-Base PL-MoE naive_mix_140k",
    "qwen3_4b_base_pl_moe_superior_27k_27k": "Qwen3-4B-Base PL-MoE superior_reasoning_27k+27k",
    "qwen3_4b_base_pl_moe_superior_65k_27k": "Qwen3-4B-Base PL-MoE superior_reasoning_65k+27k",
    # Qwen3-4B-Base→Instruct Stage2 PL-MoE
    "qwen3_4b_base_instruct_stage2_pl_moe_superior_65k_27k": "Qwen3-4B-Base→Instruct Stage2 PL-MoE superior_reasoning_65k+27k",
    # Qwen3-4B (original) PL-MoE
    "qwen3_4b_pl_moe_20k": "Qwen3-4B PL-MoE naive_mix_20k",
    "qwen3_4b_pl_moe_superior_27k_27k": "Qwen3-4B PL-MoE superior_reasoning_27k+27k",
    "qwen3_4b_pl_moe_superior_27k_27k_v2": "Qwen3-4B PL-MoE superior_reasoning_27k+27k_v2",
    "qwen3_4b_pl_moe_superior_27k_27k_ckpt2400": "Qwen3-4B PL-MoE superior_reasoning_27k+27k (ckpt2400)",
    "qwen3_4b_pl_moe_superior_27k_27k_aime24_64k": "Qwen3-4B PL-MoE superior_reasoning_27k+27k (AIME24_64k)",
    # Qwen2.5-7B-Instruct PL-MoE
    "qwen_pl_moe_20k": "Qwen2.5-7B-Instruct PL-MoE naive_mix_20k",
    "qwen_pl_moe_140k": "Qwen2.5-7B-Instruct PL-MoE naive_mix_140k",
    # Qwen2.5-7B-Instruct PL-MoE
    "qwen2_pl_moe_superior_27k_27k": "Qwen2.5-7B-Instruct PL-MoE superior_reasoning_27k+27k",
    "qwen2_pl_moe_stage2_superior_27k_27k": "Qwen2.5-7B-Instruct PL-MoE Stage2_superior_reasoning_27k+27k",
    # LLaMA-3-8B PL-MoE
    "llama3_8b_pl_moe_20k": "LLaMA-3-8B-Instruct PL-MoE naive_mix_20k",
    "llama3_8b_pl_moe_superior_27k_27k": "LLaMA-3-8B-Instruct PL-MoE superior_reasoning_27k+27k",
    # Phi-4-Mini PL-MoE
    "phi4_mini_pl_moe_superior_27k_27k": "Phi-4-Mini PL-MoE superior_reasoning_27k+27k",
}

# Model grouping and ordering
MODEL_GROUP_ORDER = [
    # (group name, [list of directory names])
    ("Baseline", [
        "qwen3_4b_instruct_2507",
        "qwen3_4b_instruct_140k_2phase",
        "qwen3_4b_instruct_superior_65k_27k_2phase",
    ]),
    ("Qwen3-4B-Instruct PL-MoE (naive-mix)", [
        "qwen3_4b_instruct_pl_moe_20k",
        "qwen3_4b_instruct_pl_moe_20k_a100",
        "qwen3_4b_instruct_pl_moe_140k",
    ]),
    ("Qwen3-4B-Instruct PL-MoE (superior-reasoning)", [
        "qwen3_4b_instruct_pl_moe_superior_27k_27k",
        "qwen3_4b_instruct_pl_moe_superior_65k_27k",
    ]),
    ("Qwen3-4B-Base PL-MoE", [
        "qwen3_4b_base_pl_moe_20k",
        "qwen3_4b_base_pl_moe_140k",
        "qwen3_4b_base_pl_moe_superior_27k_27k",
        "qwen3_4b_base_pl_moe_superior_65k_27k",
        "qwen3_4b_base_instruct_stage2_pl_moe_superior_65k_27k",
    ]),
    ("Qwen3-4B PL-MoE", [
        "qwen3_4b_pl_moe_20k",
        "qwen3_4b_pl_moe_superior_27k_27k",
        "qwen3_4b_pl_moe_superior_27k_27k_v2",
        "qwen3_4b_pl_moe_superior_27k_27k_ckpt2400",
        "qwen3_4b_pl_moe_superior_27k_27k_aime24_64k",
    ]),
    ("Qwen2.5-7B-Instruct PL-MoE", [
        "qwen_pl_moe_20k",
        "qwen_pl_moe_140k",
        "qwen2_pl_moe_superior_27k_27k",
        "qwen2_pl_moe_stage2_superior_27k_27k",
    ]),
    ("LLaMA-3-8B-Instruct PL-MoE", [
        "llama3_8b_pl_moe_20k",
        "llama3_8b_pl_moe_superior_27k_27k",
    ]),
    ("Phi-4-Mini PL-MoE", [
        "phi4_mini_pl_moe_superior_27k_27k",
    ]),
]

BENCHMARKS = ["math500", "aime24", "mmlu_stem", "gpqa_diamond"]
BENCHMARK_DISPLAY = {
    "math500": "MATH500",
    "aime24": "AIME24",
    "mmlu_stem": "MMLU-STEM",
    "gpqa_diamond": "GPQA-Diamond",
}
METRICS = ["acc", "avg_len", "#reflective", "#wait"]
METRIC_DISPLAY = {
    "acc": "Accuracy",
    "avg_len": "Avg Length",
    "#reflective": "Avg #Reflective",
    "#wait": "Avg #Wait",
}

# Analysis section styles
ANALYSIS_TITLE_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
ANALYSIS_TITLE_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=12)
ANALYSIS_BODY_FONT = Font(name="Calibri", size=10)
ANALYSIS_BODY_FILL = PatternFill(start_color="F8F9FA", end_color="F8F9FA", fill_type="solid")
ANALYSIS_BULLET_FONT = Font(name="Calibri", bold=True, size=10, color="2F5496")

# ── Style definitions ──────────────────────────────────────────────────────────────────

# Colors
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
SUBHEADER_FILL = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
SUBHEADER_FONT = Font(name="Calibri", bold=True, size=10)

GROUP_HEADER_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
GROUP_HEADER_FONT = Font(name="Calibri", bold=True, size=10, color="375623")

BASELINE_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

THINK_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
NO_THINK_FILL = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

BEST_FONT = Font(name="Calibri", bold=True, color="006100", size=10)
BEST_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")

WARN_FONT = Font(name="Calibri", color="9C0006", size=10)
WARN_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

NORMAL_FONT = Font(name="Calibri", size=10)
THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)
GROUP_BOTTOM_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="medium", color="8DB4E2"),
)


def write_analysis_section(ws, start_row: int, title: str, lines: list[str], total_cols: int):
    """
    Write the analysis section at start_row of the sheet.
    title: analysis title
    lines: list of analysis entries, one per row
    total_cols: total number of columns for the merged cells
    """
    row = start_row + 1  # leave one blank row

    # Title row
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=total_cols)
    cell = ws.cell(row=row, column=1, value=title)
    cell.font = ANALYSIS_TITLE_FONT
    cell.fill = ANALYSIS_TITLE_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center")
    for c in range(2, total_cols + 1):
        ws.cell(row=row, column=c).fill = ANALYSIS_TITLE_FILL
    row += 1

    # Content rows
    for line in lines:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=total_cols)
        cell = ws.cell(row=row, column=1, value=line)
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        if line.startswith("▸"):
            cell.font = ANALYSIS_BULLET_FONT
        else:
            cell.font = ANALYSIS_BODY_FONT
        cell.fill = ANALYSIS_BODY_FILL
        for c in range(2, total_cols + 1):
            ws.cell(row=row, column=c).fill = ANALYSIS_BODY_FILL
        ws.row_dimensions[row].height = 20
        row += 1

    return row




def read_csv_data(csv_path: str) -> tuple[dict, dict]:
    """Read the CSV and return (data, source_dirs).
    data: {(dir_name, mode): {(benchmark, metric): value}}
    source_dirs: {(dir_name, mode): source_dir_path}
    """
    data = {}
    source_dirs = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            mode = row["mode"]
            key = (model, mode)
            metrics = {}
            for bench in BENCHMARKS:
                for metric in METRICS:
                    col = f"{bench}_{metric}"
                    val = row.get(col, "")
                    if val:
                        try:
                            metrics[(bench, metric)] = float(val)
                        except ValueError:
                            pass
            data[key] = metrics
            source_dirs[key] = row.get("source_dir", "")
    return data, source_dirs


def find_best_values(data: dict) -> dict:
    """
    Find the maximum accuracy for each (benchmark, mode).
    Compared only among PL-MoE models (baselines excluded).
    Returns {(bench, mode): best_acc}
    """
    best = {}
    for (model, mode), metrics in data.items():
        # Exclude baselines
        if "2507" in model or mode == "-":
            continue
        for bench in BENCHMARKS:
            acc = metrics.get((bench, "acc"))
            if acc is not None:
                key = (bench, mode)
                if key not in best or acc > best[key]:
                    best[key] = acc
    return best


def find_length_warnings(data: dict) -> set:
    """
    Flag cells whose avg_len is abnormally long in no_think mode.
    Condition: no_think avg_len > 3000 (possible reasoning leakage)
    """
    warnings = set()
    for (model, mode), metrics in data.items():
        if mode != "no_think":
            continue
        for bench in BENCHMARKS:
            avg_len = metrics.get((bench, "avg_len"))
            if avg_len is not None and avg_len > 3000:
                warnings.add((model, bench))
    return warnings


def create_accuracy_sheet(wb: Workbook, data: dict, best_values: dict, length_warnings: set,
                          source_dirs: dict):
    """Create the main Accuracy sheet."""
    ws = wb.active
    ws.title = "Accuracy"

    src_col = 3 + len(BENCHMARKS)  # column index of source_dir
    total_cols = src_col

    # ── Header ──
    row = 1
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=1)
    ws.cell(row=row, column=1, value="Model").font = HEADER_FONT
    ws.cell(row=row, column=1).fill = HEADER_FILL
    ws.cell(row=row, column=1).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=1).fill = HEADER_FILL

    ws.merge_cells(start_row=row, start_column=2, end_row=row + 1, end_column=2)
    ws.cell(row=row, column=2, value="Mode").font = HEADER_FONT
    ws.cell(row=row, column=2).fill = HEADER_FILL
    ws.cell(row=row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=2).fill = HEADER_FILL

    col = 3
    for bench in BENCHMARKS:
        ws.cell(row=row, column=col, value=BENCHMARK_DISPLAY[bench]).font = HEADER_FONT
        ws.cell(row=row, column=col).fill = HEADER_FILL
        ws.cell(row=row, column=col).alignment = Alignment(horizontal="center")
        # Sub-header
        ws.cell(row=row + 1, column=col, value="Accuracy").font = SUBHEADER_FONT
        ws.cell(row=row + 1, column=col).fill = SUBHEADER_FILL
        ws.cell(row=row + 1, column=col).alignment = Alignment(horizontal="center")
        col += 1

    # Source Dir header
    ws.merge_cells(start_row=row, start_column=src_col, end_row=row + 1, end_column=src_col)
    ws.cell(row=row, column=src_col, value="Source Dir").font = HEADER_FONT
    ws.cell(row=row, column=src_col).fill = HEADER_FILL
    ws.cell(row=row, column=src_col).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=src_col).fill = HEADER_FILL

    row = 3  # first data row

    for group_name, model_list in MODEL_GROUP_ORDER:
        # Check whether the group has any data
        group_has_data = False
        for model_dir in model_list:
            for mode in ["think", "no_think", "-"]:
                if (model_dir, mode) in data:
                    group_has_data = True
                    break
            if group_has_data:
                break
        if not group_has_data:
            continue

        # Group title row
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=total_cols)
        ws.cell(row=row, column=1, value=group_name).font = GROUP_HEADER_FONT
        ws.cell(row=row, column=1).fill = GROUP_HEADER_FILL
        for c in range(1, total_cols + 1):
            ws.cell(row=row, column=c).fill = GROUP_HEADER_FILL
        row += 1

        for model_dir in model_list:
            # Determine which modes this model has
            modes_present = []
            for mode in ["think", "no_think", "-"]:
                if (model_dir, mode) in data:
                    modes_present.append(mode)
            if not modes_present:
                continue

            formal_name = FORMAL_NAMES.get(model_dir, model_dir)
            src_written = False  # source_dir is written only on the first row of each model

            for mode in modes_present:
                metrics = data[(model_dir, mode)]
                mode_display = mode if mode != "-" else "default"

                is_baseline = (mode == "-")
                bg_fill = BASELINE_FILL if is_baseline else (
                    THINK_FILL if mode == "think" else NO_THINK_FILL
                )

                ws.cell(row=row, column=1, value=formal_name).font = NORMAL_FONT
                ws.cell(row=row, column=1).fill = bg_fill
                ws.cell(row=row, column=2, value=mode_display).font = NORMAL_FONT
                ws.cell(row=row, column=2).fill = bg_fill
                ws.cell(row=row, column=2).alignment = Alignment(horizontal="center")

                col = 3
                for bench in BENCHMARKS:
                    acc = metrics.get((bench, "acc"))
                    cell = ws.cell(row=row, column=col)
                    cell.fill = bg_fill
                    cell.alignment = Alignment(horizontal="center")
                    cell.number_format = "0.000"

                    if acc is not None:
                        cell.value = acc
                        cell.font = NORMAL_FONT

                        # Highlight the best one (best among PL-MoE models)
                        if not is_baseline and (bench, mode) in best_values:
                            if acc == best_values[(bench, mode)]:
                                cell.font = BEST_FONT
                                cell.fill = BEST_FILL
                    else:
                        cell.value = "—"
                        cell.font = Font(name="Calibri", size=10, color="AAAAAA")

                    col += 1

                # Source Dir (written on the model's first row only; the second row stays empty)
                src_cell = ws.cell(row=row, column=src_col)
                src_cell.fill = bg_fill
                if not src_written:
                    src_cell.value = source_dirs.get((model_dir, mode), "")
                    src_cell.font = Font(name="Calibri", size=8, color="808080")
                    src_cell.alignment = Alignment(horizontal="left", vertical="center")
                    src_written = True

                row += 1

        # Group separator: add a bottom border to the last row
        for c in range(1, total_cols + 1):
            ws.cell(row=row - 1, column=c).border = GROUP_BOTTOM_BORDER

    # ── Column widths ──
    ws.column_dimensions["A"].width = 50
    ws.column_dimensions["B"].width = 12
    for i, bench in enumerate(BENCHMARKS):
        ws.column_dimensions[get_column_letter(3 + i)].width = 16
    ws.column_dimensions[get_column_letter(src_col)].width = 55

    # Freeze panes
    ws.freeze_panes = "C3"

    return row


def _write_source_dir_header(ws, row, src_col):
    """Write the Source Dir header (two merged rows)."""
    ws.merge_cells(start_row=row, start_column=src_col, end_row=row + 1, end_column=src_col)
    ws.cell(row=row, column=src_col, value="Source Dir").font = HEADER_FONT
    ws.cell(row=row, column=src_col).fill = HEADER_FILL
    ws.cell(row=row, column=src_col).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=src_col).fill = HEADER_FILL


def _write_source_dir_cell(ws, row, src_col, source_dirs, model_dir, mode, bg_fill, src_written):
    """Write the Source Dir cell of a data row. Returns the updated src_written."""
    src_cell = ws.cell(row=row, column=src_col)
    src_cell.fill = bg_fill
    if not src_written:
        src_cell.value = source_dirs.get((model_dir, mode), "")
        src_cell.font = Font(name="Calibri", size=8, color="808080")
        src_cell.alignment = Alignment(horizontal="left", vertical="center")
        return True
    return src_written


def create_length_sheet(wb: Workbook, data: dict, length_warnings: set, source_dirs: dict):
    """Create the Output Length sheet."""
    ws = wb.create_sheet(title="Output Length")

    src_col = 3 + len(BENCHMARKS)
    total_cols = src_col

    # ── Header ──
    row = 1
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=1)
    ws.cell(row=row, column=1, value="Model").font = HEADER_FONT
    ws.cell(row=row, column=1).fill = HEADER_FILL
    ws.cell(row=row, column=1).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=1).fill = HEADER_FILL

    ws.merge_cells(start_row=row, start_column=2, end_row=row + 1, end_column=2)
    ws.cell(row=row, column=2, value="Mode").font = HEADER_FONT
    ws.cell(row=row, column=2).fill = HEADER_FILL
    ws.cell(row=row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=2).fill = HEADER_FILL

    col = 3
    for bench in BENCHMARKS:
        ws.cell(row=row, column=col, value=BENCHMARK_DISPLAY[bench]).font = HEADER_FONT
        ws.cell(row=row, column=col).fill = HEADER_FILL
        ws.cell(row=row, column=col).alignment = Alignment(horizontal="center")
        ws.cell(row=row + 1, column=col, value="Avg Length").font = SUBHEADER_FONT
        ws.cell(row=row + 1, column=col).fill = SUBHEADER_FILL
        ws.cell(row=row + 1, column=col).alignment = Alignment(horizontal="center")
        col += 1

    _write_source_dir_header(ws, row, src_col)

    row = 3

    for group_name, model_list in MODEL_GROUP_ORDER:
        group_has_data = any(
            (m, mode) in data
            for m in model_list for mode in ["think", "no_think", "-"]
        )
        if not group_has_data:
            continue

        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=total_cols)
        ws.cell(row=row, column=1, value=group_name).font = GROUP_HEADER_FONT
        ws.cell(row=row, column=1).fill = GROUP_HEADER_FILL
        for c in range(1, total_cols + 1):
            ws.cell(row=row, column=c).fill = GROUP_HEADER_FILL
        row += 1

        for model_dir in model_list:
            modes_present = [m for m in ["think", "no_think", "-"] if (model_dir, m) in data]
            if not modes_present:
                continue

            formal_name = FORMAL_NAMES.get(model_dir, model_dir)
            src_written = False

            for mode in modes_present:
                metrics = data[(model_dir, mode)]
                mode_display = mode if mode != "-" else "default"
                is_baseline = (mode == "-")
                bg_fill = BASELINE_FILL if is_baseline else (
                    THINK_FILL if mode == "think" else NO_THINK_FILL
                )

                ws.cell(row=row, column=1, value=formal_name).font = NORMAL_FONT
                ws.cell(row=row, column=1).fill = bg_fill
                ws.cell(row=row, column=2, value=mode_display).font = NORMAL_FONT
                ws.cell(row=row, column=2).fill = bg_fill
                ws.cell(row=row, column=2).alignment = Alignment(horizontal="center")

                col = 3
                for bench in BENCHMARKS:
                    avg_len = metrics.get((bench, "avg_len"))
                    cell = ws.cell(row=row, column=col)
                    cell.fill = bg_fill
                    cell.alignment = Alignment(horizontal="center")
                    cell.number_format = "#,##0"

                    if avg_len is not None:
                        cell.value = avg_len
                        if mode == "no_think" and (model_dir, bench) in length_warnings:
                            cell.font = WARN_FONT
                            cell.fill = WARN_FILL
                        else:
                            cell.font = NORMAL_FONT
                    else:
                        cell.value = "—"
                        cell.font = Font(name="Calibri", size=10, color="AAAAAA")

                    col += 1

                src_written = _write_source_dir_cell(
                    ws, row, src_col, source_dirs, model_dir, mode, bg_fill, src_written)

                row += 1

        for c in range(1, total_cols + 1):
            ws.cell(row=row - 1, column=c).border = GROUP_BOTTOM_BORDER

    ws.column_dimensions["A"].width = 50
    ws.column_dimensions["B"].width = 12
    for i in range(len(BENCHMARKS)):
        ws.column_dimensions[get_column_letter(3 + i)].width = 16
    ws.column_dimensions[get_column_letter(src_col)].width = 55
    ws.freeze_panes = "C3"


def create_reflective_sheet(wb: Workbook, data: dict, source_dirs: dict):
    """Create the Reflective Tokens sheet (only #reflective and #wait)."""
    ws = wb.create_sheet(title="Reflective Tokens")

    src_col = 3 + len(BENCHMARKS) * 2
    total_cols = src_col

    row = 1
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=1)
    ws.cell(row=row, column=1, value="Model").font = HEADER_FONT
    ws.cell(row=row, column=1).fill = HEADER_FILL
    ws.cell(row=row, column=1).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=1).fill = HEADER_FILL

    ws.merge_cells(start_row=row, start_column=2, end_row=row + 1, end_column=2)
    ws.cell(row=row, column=2, value="Mode").font = HEADER_FONT
    ws.cell(row=row, column=2).fill = HEADER_FILL
    ws.cell(row=row, column=2).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(row=row + 1, column=2).fill = HEADER_FILL

    col = 3
    for bench in BENCHMARKS:
        ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + 1)
        ws.cell(row=row, column=col, value=BENCHMARK_DISPLAY[bench]).font = HEADER_FONT
        ws.cell(row=row, column=col).fill = HEADER_FILL
        ws.cell(row=row, column=col).alignment = Alignment(horizontal="center")
        ws.cell(row=row, column=col + 1).fill = HEADER_FILL

        ws.cell(row=row + 1, column=col, value="#Reflective").font = SUBHEADER_FONT
        ws.cell(row=row + 1, column=col).fill = SUBHEADER_FILL
        ws.cell(row=row + 1, column=col).alignment = Alignment(horizontal="center")

        ws.cell(row=row + 1, column=col + 1, value="#Wait").font = SUBHEADER_FONT
        ws.cell(row=row + 1, column=col + 1).fill = SUBHEADER_FILL
        ws.cell(row=row + 1, column=col + 1).alignment = Alignment(horizontal="center")
        col += 2

    _write_source_dir_header(ws, row, src_col)

    row = 3

    for group_name, model_list in MODEL_GROUP_ORDER:
        group_has_data = any(
            (m, mode) in data
            for m in model_list for mode in ["think", "no_think", "-"]
        )
        if not group_has_data:
            continue

        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=total_cols)
        ws.cell(row=row, column=1, value=group_name).font = GROUP_HEADER_FONT
        ws.cell(row=row, column=1).fill = GROUP_HEADER_FILL
        for c in range(1, total_cols + 1):
            ws.cell(row=row, column=c).fill = GROUP_HEADER_FILL
        row += 1

        for model_dir in model_list:
            modes_present = [m for m in ["think", "no_think", "-"] if (model_dir, m) in data]
            if not modes_present:
                continue

            formal_name = FORMAL_NAMES.get(model_dir, model_dir)
            src_written = False

            for mode in modes_present:
                metrics = data[(model_dir, mode)]
                mode_display = mode if mode != "-" else "default"
                is_baseline = (mode == "-")
                bg_fill = BASELINE_FILL if is_baseline else (
                    THINK_FILL if mode == "think" else NO_THINK_FILL
                )

                ws.cell(row=row, column=1, value=formal_name).font = NORMAL_FONT
                ws.cell(row=row, column=1).fill = bg_fill
                ws.cell(row=row, column=2, value=mode_display).font = NORMAL_FONT
                ws.cell(row=row, column=2).fill = bg_fill
                ws.cell(row=row, column=2).alignment = Alignment(horizontal="center")

                col = 3
                for bench in BENCHMARKS:
                    for metric_key in ["#reflective", "#wait"]:
                        val = metrics.get((bench, metric_key))
                        cell = ws.cell(row=row, column=col)
                        cell.fill = bg_fill
                        cell.alignment = Alignment(horizontal="center")
                        cell.number_format = "0.0000"

                        if val is not None:
                            cell.value = val
                            if mode == "no_think" and val > 0.01:
                                cell.font = WARN_FONT
                                cell.fill = WARN_FILL
                            else:
                                cell.font = NORMAL_FONT
                        else:
                            cell.value = "—"
                            cell.font = Font(name="Calibri", size=10, color="AAAAAA")
                        col += 1

                src_written = _write_source_dir_cell(
                    ws, row, src_col, source_dirs, model_dir, mode, bg_fill, src_written)

                row += 1

        for c in range(1, total_cols + 1):
            ws.cell(row=row - 1, column=c).border = GROUP_BOTTOM_BORDER

    ws.column_dimensions["A"].width = 50
    ws.column_dimensions["B"].width = 12
    for i in range(len(BENCHMARKS) * 2):
        ws.column_dimensions[get_column_letter(3 + i)].width = 14
    ws.column_dimensions[get_column_letter(src_col)].width = 55
    ws.freeze_panes = "C3"


def create_legend_sheet(wb: Workbook):
    """Create the legend sheet."""
    ws = wb.create_sheet(title="Legend")

    ws.cell(row=1, column=1, value="Formatting legend").font = Font(name="Calibri", bold=True, size=14)

    items = [
        (3, BASELINE_FILL, NORMAL_FONT, "Yellow fill", "Baseline model (original model, not trained with PL-MoE)"),
        (4, THINK_FILL, NORMAL_FONT, "White fill", "think mode (uses expert 1, CoT reasoning)"),
        (5, NO_THINK_FILL, NORMAL_FONT, "Gray fill", "no_think mode (uses expert 0, direct answers)"),
        (6, BEST_FILL, BEST_FONT, "Green fill, bold", "Highest accuracy among PL-MoE models for this benchmark + mode"),
        (7, WARN_FILL, WARN_FONT, "Red fill", "Outlier warning (no_think output too long or reflective tokens present)"),
    ]
    for row_num, fill, font, label, desc in items:
        ws.cell(row=row_num, column=1, value=label).fill = fill
        ws.cell(row=row_num, column=1).font = font
        ws.cell(row=row_num, column=2, value=desc).font = NORMAL_FONT

    ws.cell(row=9, column=1, value="Sheet descriptions").font = Font(name="Calibri", bold=True, size=14)
    sheets_info = [
        (10, "Accuracy", "Accuracy of each model on the 4 benchmarks; green highlights the best within the same mode"),
        (11, "Output Length", "Average number of output tokens; red marks no_think outliers above 3000 tokens (possible reasoning leakage)"),
        (12, "Reflective Tokens", "Average number of reflective tokens (#reflective counts wait/hmm/okay/alternatively/let me think; #wait counts only 'wait, ')"),
    ]
    for row_num, sheet_name, desc in sheets_info:
        ws.cell(row=row_num, column=1, value=sheet_name).font = Font(name="Calibri", bold=True, size=10)
        ws.cell(row=row_num, column=2, value=desc).font = NORMAL_FONT

    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 80


def main():
    parser = argparse.ArgumentParser(description="Generate a formatted Excel summary of the experiment results")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output xlsx file path")
    args = parser.parse_args()

    # Read the data
    data, source_dirs = read_csv_data(args.input)
    print(f"Read {len(data)} rows")

    # Compute highlight information
    best_values = find_best_values(data)
    length_warnings = find_length_warnings(data)

    # Create the workbook
    wb = Workbook()
    create_accuracy_sheet(wb, data, best_values, length_warnings, source_dirs)
    create_length_sheet(wb, data, length_warnings, source_dirs)
    create_reflective_sheet(wb, data, source_dirs)
    create_legend_sheet(wb)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
