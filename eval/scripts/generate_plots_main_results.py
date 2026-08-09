"""
Table 3 -- Each (model x benchmark) saved as a standalone figure.
X-axis grouped by mode (Think / No-think), bars represent methods.
Color by mode: blue=Think, green=No-think.
Fill: Ours=solid, baselines=hatched. Instruct=gray hatched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os

# ── 颜色方案: 按 mode 分配颜色 ──
DEEP_BLUE  = "#1B3A5C"   # Think 色
DEEP_GREEN = "#1B5E20"   # No-think 色
GRAY       = "#888888"   # Instruct baseline

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16, "axes.labelsize": 17,
    "axes.titlesize": 19, "axes.titleweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# All data: (think, no_think) — None means N/A
DATA_ACC = {
    "Qwen2.5-7B": {
        "Instruct":    {"MATH500": (None,  59.94), "AIME24": (None,   6.67)},
        "Prev. Study": {"MATH500": (86.50, 63.56), "AIME24": (30.00,  3.33)},
        "Ours":        {"MATH500": (86.00, 83.20), "AIME24": (40.00, 26.00)},
    },
    "Qwen3-4B": {
        "Hybrid": {"MATH500": (92.02, 82.22), "AIME24": (61.67, 20.67)},
        "Ours":   {"MATH500": (94.80, 80.80), "AIME24": (60.00, 40.00)},
    },
}

DATA_LEN = {
    "Qwen2.5-7B": {
        "Instruct":    {"MATH500": (None,  703.11),  "AIME24": (None,    1729.22)},
        "Prev. Study": {"MATH500": (4488.40, 593.09), "AIME24": (12517.06, 1037.20)},
        "Ours":        {"MATH500": (3861.34, 614.46), "AIME24": (11794.06, 1328.60)},
    },
    "Qwen3-4B": {
        "Hybrid": {"MATH500": (4679.54, 1004.13), "AIME24": (11595.51, 4636.37)},
        "Ours":   {"MATH500": (6050.45, 676.78),  "AIME24": (31733.18, 4708.2)},
    },
}

DATA_REFL = {
    "Qwen2.5-7B": {
        "Instruct":    {"MATH500": (None,  0),    "AIME24": (None,  0)},
        "Prev. Study": {"MATH500": (33.04, 0.31), "AIME24": (76.44, 0.00)},
        "Ours":        {"MATH500": (24.64, 0.01), "AIME24": (62.76, 0.02)},
    },
    "Qwen3-4B": {
        "Hybrid": {"MATH500": (19.60, 0.12), "AIME24": (45.85, 2.54)},
        "Ours":   {"MATH500": (3.56,  0.13), "AIME24": (7.02,  0.39)},
    },
}

METRIC_INFO = {
    "acc":  {"data": DATA_ACC,  "ylabel": "Accuracy (%)",          "fmt": ".1f", "prefix": "exp_acc",  "higher_better": True},
    "len":  {"data": DATA_LEN,  "ylabel": "Avg. Output Length",    "fmt": ".0f", "prefix": "exp_len",  "higher_better": False},
    "refl": {"data": DATA_REFL, "ylabel": "#Reflective / Answer",  "fmt": ".2f", "prefix": "exp_refl", "higher_better": False},
}

OUTDIR = "plots/experiment"
os.makedirs(OUTDIR, exist_ok=True)

panels = [
    ("Qwen2.5-7B", "MATH500"),
    ("Qwen3-4B",   "MATH500"),
    ("Qwen2.5-7B", "AIME24"),
    ("Qwen3-4B",   "AIME24"),
]

import re

for metric_key, minfo in METRIC_INFO.items():
  DATA = minfo["data"]
  ylabel = minfo["ylabel"]
  vfmt = minfo["fmt"]
  prefix = minfo["prefix"]
  higher_better = minfo["higher_better"]

  for model, bm in panels:
    fig, ax = plt.subplots(figsize=(3.8, 4.0))
    methods = list(DATA[model].keys())

    def sort_key(m):
        if m == "Ours": return 0
        if m == "Instruct": return 99
        return 1
    sorted_methods = sorted(methods, key=sort_key)

    # 收集数据，None 也保留（留空位）
    think_methods, think_vals = [], []
    nothink_methods, nothink_vals = [], []
    for m in sorted_methods:
        tv, ntv = DATA[model][m][bm]
        # Think: 除 Instruct 外都占位
        if m != "Instruct":
            think_methods.append(m)
            think_vals.append(tv)  # 可能是 None
        # No-think: 所有方法都占位
        nothink_methods.append(m)
        nothink_vals.append(ntv)  # 可能是 None

    n_think = len(think_methods)
    n_nothink = len(nothink_methods)

    bw = 0.50
    gap_within = 0.12
    group_gap = 0.7
    think_xs = np.arange(n_think) * (bw + gap_within)
    think_center = think_xs.mean() if n_think else 0
    nothink_start = (think_xs[-1] + bw + group_gap) if n_think > 0 else 0
    nothink_xs = nothink_start + np.arange(n_nothink) * (bw + gap_within)
    nothink_center = nothink_xs.mean() if n_nothink else 0

    # ── Y 轴 (先设置, 再画 bar, 这样数值标签可以用 axes 坐标紧贴 bar) ──
    all_v = [v for v in think_vals + nothink_vals if v is not None]
    ymax_data = max(all_v) if all_v else 100
    ymin_auto = min(all_v) if all_v else 0
    ymin = ymin_auto * 0.7 if (ymin_auto > ymax_data * 0.3) else 0
    if metric_key == "refl":
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
        ymax = ymax_data * 4.0
        ax.set_ylim(-0.002, ymax)
    else:
        ymax = ymax_data * 1.35
        ax.set_ylim(ymin, ymax)

    # 数值标签: 用 blended transform 紧贴 bar 顶部 (axes y = bar顶 + 固定小偏移)
    from matplotlib.transforms import blended_transform_factory
    trans_val = blended_transform_factory(ax.transData, ax.transAxes)

    def _bar_label(x_pos, val, fmt):
        """在 bar 顶部紧贴放置数值标签"""
        disp = ax.transData.transform((0, val))
        _, val_axes_y = ax.transAxes.inverted().transform(disp)
        ax.text(x_pos, val_axes_y + 0.03, f"{val:{fmt}}", transform=trans_val,
                ha="center", va="bottom", fontsize=12, color="#222", fontweight="bold")

    # ── 画 Think 组 ──
    for i, (m, v) in enumerate(zip(think_methods, think_vals)):
        is_ours = (m == "Ours")
        if v is not None:
            ax.bar(think_xs[i], v, bw, color=DEEP_BLUE, edgecolor="white", lw=0.6,
                   zorder=3, hatch=None if is_ours else "///", alpha=1.0 if is_ours else 0.70)
            _bar_label(think_xs[i], v, vfmt)
        else:
            ax.bar(think_xs[i], 0, bw, color="#eeeeee", edgecolor="#cccccc", lw=0.5, zorder=2)
            ax.text(think_xs[i], 0.02, "N/A", transform=trans_val,
                    ha="center", va="bottom", fontsize=10, color="#999", fontstyle="italic")

    # ── 画 No-think 组 ──
    for i, (m, v) in enumerate(zip(nothink_methods, nothink_vals)):
        is_ours = (m == "Ours")
        col = GRAY if m == "Instruct" else DEEP_GREEN
        if v is not None:
            ax.bar(nothink_xs[i], v, bw, color=col, edgecolor="white", lw=0.6,
                   zorder=3, hatch=None if is_ours else "///", alpha=1.0 if is_ours else 0.70)
            _bar_label(nothink_xs[i], v, vfmt)
        else:
            ax.bar(nothink_xs[i], 0, bw, color="#eeeeee", edgecolor="#cccccc", lw=0.5, zorder=2)
            ax.text(nothink_xs[i], 0.02, "N/A", transform=trans_val,
                    ha="center", va="bottom", fontsize=10, color="#999", fontstyle="italic")

    # ── Delta (per-group positioning) ──
    def draw_deltas(group_methods, group_vals, group_xs, is_think_group=False):
        if "Ours" not in group_methods or len(group_methods) < 2:
            return
        ours_idx = group_methods.index("Ours")
        ours_val = group_vals[ours_idx]
        if ours_val is None:
            return
        baselines = [(m, v, group_xs[j])
                     for j, (m, v) in enumerate(zip(group_methods, group_vals))
                     if m != "Ours" and v is not None]
        if not baselines:
            return

        from matplotlib.transforms import blended_transform_factory
        trans_blend = blended_transform_factory(ax.transData, ax.transAxes)

        for rank, (_, bl_v, bl_x) in enumerate(baselines):
            delta = ours_val - bl_v
            if higher_better:
                sign = "+" if delta >= 0 else ""
                delta_text = f"$\\Delta${sign}{delta:{vfmt}}"
                d_color = "#2B8C2B" if delta >= 0 else "#CC3333"
            else:
                pct = (delta / bl_v * 100) if bl_v != 0 else 0
                sign = "+" if pct >= 0 else ""
                delta_text = f"$\\Delta${sign}{pct:.0f}%"
                if is_think_group:
                    d_color = "#444444"
                else:
                    d_color = "#2B8C2B" if delta <= 0 else "#CC3333"

            # 将 baseline bar 顶端转为 axes 坐标, 在其上方放 delta
            disp_bl = ax.transData.transform((0, bl_v))
            _, bl_axes_y = ax.transAxes.inverted().transform(disp_bl)
            # 每个 delta 独立定位在自己 baseline bar 上方, 留空间给数值标签
            label_y = min(bl_axes_y + 0.18 + rank * 0.10, 0.97)
            ax.text(bl_x, label_y, delta_text, transform=trans_blend,
                    ha="center", va="bottom", fontsize=11, color=d_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=d_color, alpha=0.9, lw=0.8))

    draw_deltas(think_methods, think_vals, think_xs, is_think_group=True)
    draw_deltas(nothink_methods, nothink_vals, nothink_xs, is_think_group=False)

    # ── X 轴 ──
    all_xs = list(think_xs) + list(nothink_xs)
    all_labels = think_methods + nothink_methods
    ax.set_xticks(all_xs)
    ax.set_xticklabels(all_labels, fontsize=13, rotation=35, ha="right", rotation_mode="anchor")
    trans = ax.get_xaxis_transform()
    if n_think > 0:
        ax.text(think_center, -0.22, "Think", transform=trans,
                ha="center", va="top", fontsize=15, fontweight="bold", color="#2B4C7E")
    if n_nothink > 0:
        ax.text(nothink_center, -0.22, "No-think", transform=trans,
                ha="center", va="top", fontsize=15, fontweight="bold", color="#2E7D32")

    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3, width=0.5)

    # ── Legend ──
    has_instruct = "Instruct" in nothink_methods
    has_bl_t = any(m != "Ours" for m in think_methods)
    has_bl_nt = any(m not in ("Ours", "Instruct") for m in nothink_methods)
    filtered = []
    if n_think > 0:    filtered.append(Patch(fc=DEEP_BLUE, ec="w", label="Ours (Think)"))
    if n_nothink > 0:  filtered.append(Patch(fc=DEEP_GREEN, ec="w", label="Ours (No-think)"))
    if has_bl_t:       filtered.append(Patch(fc=DEEP_BLUE, ec="w", label="Baseline (Think)", hatch="///", alpha=0.70))
    if has_bl_nt:      filtered.append(Patch(fc=DEEP_GREEN, ec="w", label="Baseline (No-think)", hatch="///", alpha=0.70))
    if has_instruct:   filtered.append(Patch(fc=GRAY, ec="w", label="Instruct (No-think)", hatch="///", alpha=0.70))
    # Legend 在所有 refl 图中显示, 放右上角
    if metric_key == "refl":
        ax.legend(handles=filtered, fontsize=10, loc="upper right",
                  handlelength=1.2, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.2, framealpha=0.95, edgecolor="#ccc")

    # ── Save ──
    tag = f"{model.replace('.', '').replace('-', '_').lower()}_{bm.lower()}"
    fig.savefig(f"{OUTDIR}/{prefix}_{tag}.pdf")
    fig.savefig(f"{OUTDIR}/{prefix}_{tag}.png")
    plt.close(fig)
    print(f"[ok] {metric_key} {model} {bm}")

print("\nAll figures saved.")
