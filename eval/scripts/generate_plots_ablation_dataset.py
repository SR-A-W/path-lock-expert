"""
Ablation Study: Dataset comparison (tab:ablation_dataset_by_base_weight).
Each (model x benchmark x metric) saved as a standalone figure.
X-axis grouped by mode (Think / No-think), bars represent datasets.
Color by mode: blue=Think, green=No-think.
Dataset distinction: Superior=solid, OpenR1=hatched ///.
2 models x 2 benchmarks x 3 metrics = 12 figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import os

# ── 颜色方案 ──
DEEP_BLUE  = "#1B3A5C"   # Think 色
DEEP_GREEN = "#1B5E20"   # No-think 色

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
    "hatch.linewidth": 1.5,
})

# ── 数据集样式: Superior=实心(主角), OpenR1=斜线(baseline) ──
DATASETS = ["Superior", "OpenR1"]
DATASET_STYLE = {
    "Superior": {"hatch": None,  "alpha": 1.00},
    "OpenR1":   {"hatch": "///", "alpha": 0.70},
}

# ── 数据: tab:ablation_dataset_by_base_weight ──
# 结构: {model: {dataset: {benchmark: (think, nothink)}}}

DATA_ACC = {
    "Qwen3-4B": {
        "OpenR1":   {"MATH500": (92.80, 51.60), "AIME24": (52.00, 16.00)},
        "Superior": {"MATH500": (94.80, 80.80), "AIME24": (60.00, 40.00)},
    },
    "Qwen2.5-7B": {
        "OpenR1":   {"MATH500": (86.00, 83.20), "AIME24": (36.00, 24.00)},
        "Superior": {"MATH500": (86.40, 74.80), "AIME24": (32.00, 22.00)},
    },
}

DATA_LEN = {
    "Qwen3-4B": {
        "OpenR1":   {"MATH500": (3409, 529),    "AIME24": (9775, 906)},
        "Superior": {"MATH500": (6050, 677),    "AIME24": (31733, 4708)},
    },
    "Qwen2.5-7B": {
        "OpenR1":   {"MATH500": (3861, 614),    "AIME24": (11794, 1329)},
        "Superior": {"MATH500": (2924, 525),    "AIME24": (13809, 4120)},
    },
}

DATA_REFL = {
    "Qwen3-4B": {
        "OpenR1":   {"MATH500": (0.02, 0.04),   "AIME24": (25.98, 0.00)},
        "Superior": {"MATH500": (3.56, 0.13),   "AIME24": (7.02, 0.39)},
    },
    "Qwen2.5-7B": {
        "OpenR1":   {"MATH500": (38.14, 0.04),  "AIME24": (62.76, 0.02)},
        "Superior": {"MATH500": (2.78, 0.00),   "AIME24": (8.16, 0.02)},
    },
}

METRIC_INFO = {
    "acc":  {"data": DATA_ACC,  "ylabel": "Accuracy (%)",          "fmt": ".1f", "prefix": "abl_dataset_acc"},
    "len":  {"data": DATA_LEN,  "ylabel": "Avg. Output Length",    "fmt": ".0f", "prefix": "abl_dataset_len"},
    "refl": {"data": DATA_REFL, "ylabel": "#Reflective / Answer",  "fmt": ".2f", "prefix": "abl_dataset_refl"},
}

MODELS = ["Qwen3-4B", "Qwen2.5-7B"]
BENCHMARKS = ["MATH500", "AIME24"]

OUTDIR = "plots/ablation"
os.makedirs(OUTDIR, exist_ok=True)

for metric_key, minfo in METRIC_INFO.items():
    DATA = minfo["data"]
    ylabel = minfo["ylabel"]
    vfmt = minfo["fmt"]
    prefix = minfo["prefix"]

    for model in MODELS:
        for bm in BENCHMARKS:
            fig, ax = plt.subplots(figsize=(3.0, 4.0))

            n_ds = len(DATASETS)
            bw = 0.50
            gap_within = 0.12
            group_gap = 0.6

            # Think 组 X 坐标
            think_xs = np.arange(n_ds) * (bw + gap_within)
            think_center = think_xs.mean()

            # No-think 组 X 坐标
            nothink_start = think_xs[-1] + bw + group_gap
            nothink_xs = nothink_start + np.arange(n_ds) * (bw + gap_within)
            nothink_center = nothink_xs.mean()

            # 收集数据
            think_vals = [DATA[model][ds][bm][0] for ds in DATASETS]
            nothink_vals = [DATA[model][ds][bm][1] for ds in DATASETS]
            all_v = [v for v in think_vals + nothink_vals if v is not None]
            ymax_data = max(all_v) if all_v else 100

            # ── Y 轴设置 ──
            if metric_key == "refl":
                ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
                ymax = ymax_data * 8.0
                ax.set_ylim(-0.002, ymax)
            else:
                ymin_auto = min(all_v) if all_v else 0
                ymin = ymin_auto * 0.7 if (ymin_auto > ymax_data * 0.3) else 0
                ymax = ymax_data * 1.35
                ax.set_ylim(ymin, ymax)

            trans_val = blended_transform_factory(ax.transData, ax.transAxes)

            def _bar_label(x_pos, val, fmt, is_last_in_group=False):
                if metric_key == "refl" and val == 0:
                    ax.text(x_pos, 0.06, "0", transform=trans_val,
                            ha="center", va="bottom", fontsize=12, color="#222", fontweight="bold")
                    return
                display_fmt = fmt
                fs = 12
                if metric_key == "refl" and val >= 10:
                    display_fmt = ".0f"
                    fs = 10
                disp = ax.transData.transform((0, val))
                _, val_axes_y = ax.transAxes.inverted().transform(disp)
                label_y = min(val_axes_y + 0.03, 0.93)
                # 每组最右 bar 的标签右对齐, 防止溢出图边界
                h_align = "right" if is_last_in_group else "center"
                ax.text(x_pos, label_y, f"{val:{display_fmt}}", transform=trans_val,
                        ha=h_align, va="bottom", fontsize=fs, color="#222", fontweight="bold",
                        clip_on=False)

            def _draw_bar(x_pos, val, base_color, ds_name, is_last_in_group=False):
                style = DATASET_STYLE[ds_name]
                if val is not None:
                    ax.bar(x_pos, val, bw, color=base_color, edgecolor="white", lw=0.6,
                           zorder=3, hatch=style["hatch"], alpha=style["alpha"])
                    _bar_label(x_pos, val, vfmt, is_last_in_group=is_last_in_group)
                else:
                    ax.bar(x_pos, 0, bw, color="#eeeeee", edgecolor="#cccccc", lw=0.5, zorder=2)
                    ax.text(x_pos, 0.02, "N/A", transform=trans_val,
                            ha="center", va="bottom", fontsize=10, color="#999", fontstyle="italic")

            # ── 画 Think 组 (蓝色) ──
            for i, ds in enumerate(DATASETS):
                _draw_bar(think_xs[i], think_vals[i], DEEP_BLUE, ds,
                          is_last_in_group=(i == n_ds - 1))

            # ── 画 No-think 组 (绿色) ──
            for i, ds in enumerate(DATASETS):
                _draw_bar(nothink_xs[i], nothink_vals[i], DEEP_GREEN, ds,
                          is_last_in_group=(i == n_ds - 1))

            # ── X 轴标签 ──
            all_xs = list(think_xs) + list(nothink_xs)
            all_labels = list(DATASETS) + list(DATASETS)
            ax.set_xticks(all_xs)
            ax.set_xticklabels(all_labels, fontsize=12, rotation=35, ha="right", rotation_mode="anchor")

            # 分组标签
            trans = ax.get_xaxis_transform()
            ax.text(think_center, -0.22, "Think", transform=trans,
                    ha="center", va="top", fontsize=15, fontweight="bold", color="#2B4C7E")
            ax.text(nothink_center, -0.22, "No-think", transform=trans,
                    ha="center", va="top", fontsize=15, fontweight="bold", color="#2E7D32")

            ax.set_ylabel(ylabel)
            ax.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=3, width=0.5)

            # ── Legend (只在 refl 图中显示) ──
            if metric_key == "refl":
                legend_handles = [
                    Patch(fc=DEEP_BLUE, ec="w", label="Superior (Think)"),
                    Patch(fc=DEEP_GREEN, ec="w", label="Superior (No-think)"),
                    Patch(fc=DEEP_BLUE, ec="w", hatch="///", alpha=0.70, label="OpenR1 (Think)"),
                    Patch(fc=DEEP_GREEN, ec="w", hatch="///", alpha=0.70, label="OpenR1 (No-think)"),
                ]
                ax.legend(handles=legend_handles, fontsize=9, loc="upper right",
                          handlelength=1.2, handletextpad=0.3, borderpad=0.3,
                          labelspacing=0.25, framealpha=0.95, edgecolor="#ccc")

            # ── Save ──
            model_tag = model.replace(".", "").replace("-", "_").lower()
            tag = f"{model_tag}_{bm.lower()}"
            fig.savefig(f"{OUTDIR}/{prefix}_{tag}.pdf")
            fig.savefig(f"{OUTDIR}/{prefix}_{tag}.png")
            plt.close(fig)
            print(f"[ok] {metric_key} {model} {bm}")

print("\nAll ablation dataset figures saved.")
