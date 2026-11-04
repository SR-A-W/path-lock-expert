#!/usr/bin/env python3
"""
Table data utilities: read/query interface for the standardized result JSON files, plus formatting helpers.

Stateless, purely functional module used by the table scripts under eval/scripts/tables/.
"""

import json
from pathlib import Path
from typing import Any, Optional


# ── Data loading ──────────────────────────────────────────────────────────────────


def load_model_results(json_path: str | Path) -> dict:
    """Load a single standardized result JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_models(results_dir: str | Path = "eval/results_json") -> dict[str, dict]:
    """
    Load all .json files under results_dir.
    Returns {model_name: results_dict}; model_name is taken from the "model" field of each JSON.
    """
    results_dir = Path(results_dir)
    models = {}
    for p in sorted(results_dir.glob("*.json")):
        data = load_model_results(p)
        name = data.get("model", p.stem)
        models[name] = data
    return models


def get_result(model_data: dict, benchmark: str, mode: str) -> dict | None:
    """Look up a model's results for a given benchmark/mode."""
    return model_data.get("results", {}).get(benchmark, {}).get(mode)


def get_per_sample(model_data: dict, benchmark: str, mode: str) -> list[dict] | None:
    """Look up a model's per-sample data for a given benchmark/mode."""
    return model_data.get("per_sample_data", {}).get(benchmark, {}).get(mode)


# ── Formatting ────────────────────────────────────────────────────────────────────


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Format as a percentage, e.g. 0.832 -> "83.2". None -> "-"."""
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}"


def fmt_int(value: float | None) -> str:
    """Format as an integer, e.g. 1234.5 -> "1235". None -> "-"."""
    if value is None:
        return "-"
    return str(round(value))


def fmt_float(value: float | None, decimals: int = 2) -> str:
    """Format a float. None -> "-"."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def fmt_bold_best(
    values: list[float | None],
    formatted: list[str],
    higher_is_better: bool = True,
) -> list[str]:
    """
    Wrap the best value in \\textbf{}.
    values and formatted must have equal length. None values are ignored in the comparison.
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
    """Format a signed difference, e.g. "+3.2" or "-1.5"."""
    if value is None or baseline is None:
        return "-"
    diff = (value - baseline) * 100  # inputs are assumed to be 0-1 fractions
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.{decimals}f}"


def fmt_reduction(new: float | None, old: float | None, decimals: int = 1) -> str:
    """Format a relative reduction, e.g. old=100, new=20 -> "-80.0%"."""
    if new is None or old is None or old == 0:
        return "-"
    reduction = (new - old) / old * 100
    return f"{reduction:.{decimals}f}\\%"


# ── LaTeX helpers ─────────────────────────────────────────────────────────────────


def latex_row(*cells: str, end: bool = True) -> str:
    """Join cells with ' & ' and append ' \\\\' at the end."""
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


# ── Comparison helpers ──────────────────────────────────────────────────────────────────


def compute_leakage_reduction(
    pl_moe_data: dict,
    baseline_data: dict,
    benchmark: str,
) -> dict:
    """
    Compute the leakage-reduction metrics in no_think mode.
    Returns {reflective_reduction_pct, length_reduction_pct}.
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
    """Compute the accuracy difference between PL-MoE and the baseline."""
    pl_moe = get_result(pl_moe_data, benchmark, mode)
    baseline = get_result(baseline_data, benchmark, mode)
    if not pl_moe or not baseline:
        return None
    pl_acc = pl_moe.get("accuracy")
    bl_acc = baseline.get("accuracy")
    if pl_acc is None or bl_acc is None:
        return None
    return pl_acc - bl_acc
