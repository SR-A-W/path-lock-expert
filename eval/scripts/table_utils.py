#!/usr/bin/env python3
"""
表格数据工具：标准化 JSON 的读取/查询接口与格式化函数。

纯函数式模块，无状态。供 eval/scripts/tables/ 下的各表格脚本调用。
"""

import json
from pathlib import Path
from typing import Any, Optional


# ── 数据加载 ──────────────────────────────────────────────────────────────────


def load_model_results(json_path: str | Path) -> dict:
    """加载单个标准化结果 JSON 文件。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_models(results_dir: str | Path = "eval/results_json") -> dict[str, dict]:
    """
    加载 results_dir 下所有 .json 文件。
    返回 {model_name: results_dict}，model_name 取自 JSON 中的 "model" 字段。
    """
    results_dir = Path(results_dir)
    models = {}
    for p in sorted(results_dir.glob("*.json")):
        data = load_model_results(p)
        name = data.get("model", p.stem)
        models[name] = data
    return models


def get_result(model_data: dict, benchmark: str, mode: str) -> dict | None:
    """查询某模型在特定 benchmark/mode 下的结果。"""
    return model_data.get("results", {}).get(benchmark, {}).get(mode)


def get_per_sample(model_data: dict, benchmark: str, mode: str) -> list[dict] | None:
    """查询某模型在特定 benchmark/mode 下的 per-sample 数据。"""
    return model_data.get("per_sample_data", {}).get(benchmark, {}).get(mode)


# ── 格式化 ────────────────────────────────────────────────────────────────────


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """格式化为百分比，如 0.832 → "83.2"。None → "-"。"""
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}"


def fmt_int(value: float | None) -> str:
    """格式化为整数，如 1234.5 → "1235"。None → "-"。"""
    if value is None:
        return "-"
    return str(round(value))


def fmt_float(value: float | None, decimals: int = 2) -> str:
    """格式化浮点数。None → "-"。"""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def fmt_bold_best(
    values: list[float | None],
    formatted: list[str],
    higher_is_better: bool = True,
) -> list[str]:
    """
    将最优值包裹在 \\textbf{} 中。
    values 和 formatted 应等长。None 值不参与比较。
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return list(formatted)
    if higher_is_better:
        best_idx = max(valid, key=lambda x: x[1])[0]
    else:
        best_idx = min(valid, key=lambda x: x[1])[0]
    result = list(formatted)
    result[best_idx] = f"\\textbf{{{result[best_idx]}}}"
    return result


def fmt_delta(value: float | None, baseline: float | None, decimals: int = 1) -> str:
    """格式化差值（带符号），如 "+3.2" 或 "-1.5"。"""
    if value is None or baseline is None:
        return "-"
    diff = (value - baseline) * 100  # 假设输入是 0-1 比例
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.{decimals}f}"


def fmt_reduction(new: float | None, old: float | None, decimals: int = 1) -> str:
    """格式化减少比例，如 old=100, new=20 → "-80.0%"。"""
    if new is None or old is None or old == 0:
        return "-"
    reduction = (new - old) / old * 100
    return f"{reduction:.{decimals}f}\\%"


# ── LaTeX 辅助 ─────────────────────────────────────────────────────────────────


def latex_row(*cells: str, end: bool = True) -> str:
    """用 ' & ' 连接 cells，末尾加 ' \\\\'。"""
    row = " & ".join(cells)
    if end:
        row += " \\\\"
    return row


def latex_hline() -> str:
    return "\\hline"


def latex_midrule() -> str:
    return "\\midrule"


def latex_toprule() -> str:
    return "\\toprule"


def latex_bottomrule() -> str:
    return "\\bottomrule"


# ── 比较工具 ──────────────────────────────────────────────────────────────────


def compute_leakage_reduction(
    pl_moe_data: dict,
    baseline_data: dict,
    benchmark: str,
) -> dict:
    """
    计算 no_think 模式下的 leakage 减少指标。
    返回 {reflective_reduction_pct, length_reduction_pct}。
    """
    pl_moe = get_result(pl_moe_data, benchmark, "no_think")
    baseline = get_result(baseline_data, benchmark, "no_think")

    result = {"reflective_reduction_pct": None, "length_reduction_pct": None}
    if not pl_moe or not baseline:
        return result

    # reflective token reduction
    pl_refl = pl_moe.get("reflective_token_stats", {}).get("avg_count")
    bl_refl = baseline.get("reflective_token_stats", {}).get("avg_count")
    if pl_refl is not None and bl_refl is not None and bl_refl > 0:
        result["reflective_reduction_pct"] = round((pl_refl - bl_refl) / bl_refl * 100, 2)

    # output length reduction
    pl_len = pl_moe.get("avg_output_length")
    bl_len = baseline.get("avg_output_length")
    if pl_len is not None and bl_len is not None and bl_len > 0:
        result["length_reduction_pct"] = round((pl_len - bl_len) / bl_len * 100, 2)

    return result


def compute_accuracy_delta(
    pl_moe_data: dict,
    baseline_data: dict,
    benchmark: str,
    mode: str,
) -> float | None:
    """计算 PL-MoE 与 baseline 的 accuracy 差。"""
    pl_moe = get_result(pl_moe_data, benchmark, mode)
    baseline = get_result(baseline_data, benchmark, mode)
    if not pl_moe or not baseline:
        return None
    pl_acc = pl_moe.get("accuracy")
    bl_acc = baseline.get("accuracy")
    if pl_acc is None or bl_acc is None:
        return None
    return pl_acc - bl_acc
