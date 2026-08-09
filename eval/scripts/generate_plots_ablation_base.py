"""
Ablation Study: Base model weight comparison (tab:ablation_base_superior_27k_27k).
Each (benchmark x metric) saved as a standalone figure.
X-axis grouped by mode (Think / No-think), bars represent base models.
Color by mode: blue=Think, green=No-think.
Model distinction: color saturation (HSL) + hatch pattern (方案C).
  - Qwen3-4B:        高饱和度 + 实心 (主角)
  - Qwen3-4B Base:   低饱和度 + 斜线 ///
  - Qwen2.5-7B Inst: 中饱和度 + 点阵 ..
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import colorsys
import os

# ── HSL 调色工具 ──
def hex_to_hsl(hex_color):
    """hex → HSL (H: 0-360, S: 0-1, L: 0-1)"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    # colorsys.rgb_to_hls returns (H, L, S)
    h_, l, s = colorsys.rgb_to_hls(r, g, b)
    return h_ * 360, s, l

def hsl_to_hex(h, s, l):
    """HSL (H: 0-360, S: 0-1, L: 0-1) → hex"""
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def adjust_saturation(hex_color, sat_ratio):
    """调整颜色的饱和度 (sat_ratio=1.0 不变, <1 降低饱和度), 保持色相和明度不变"""
    h, s, l = hex_to_hsl(hex_color)
    s_new = s * sat_ratio
    return hsl_to_hex(h, s_new, l)

# ── 基准颜色 ──
DEEP_BLUE  = "#1B3A5C"   # Think 色 (高饱和)
DEEP_GREEN = "#1B5E20"   # No-think 色 (高饱和)

# ── 每个模型的视觉样式: (饱和度比例, hatch, alpha) ──
# Qwen3-4B: 主角 — 高饱和, 实心
# Qwen3-4B Base: 低饱和, 斜线
# Qwen2.5-7B Inst.: 中饱和, 点阵
MODEL_STYLE = {
    "Qwen3-4B Base":    {"sat": 0.30, "hatch": "///",  "alpha": 0.85},
    "Qwen3-4B":         {"sat": 1.00, "hatch": None,   "alpha": 1.00},
    "Qwen2.5-7B Inst.": {"sat": 0.60, "hatch": "..",   "alpha": 0.85},
}

# 模型顺序 (X 轴 bar 顺序)
MODELS = ["Qwen3-4B", "Qwen3-4B Base", "Qwen2.5-7B Inst."]

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

# ── 数据: tab:ablation_base_superior_27k_27k (忽略 Qwen3-4B-Instruct) ──
DATA_ACC = {
    "Qwen3-4B Base":        {"MATH500": (90.00, 74.00), "AIME24": (20.00, 30.00)},
    "Qwen3-4B":             {"MATH500": (94.80, 80.80), "AIME24": (60.00, 40.00)},
    "Qwen2.5-7B Inst.":     {"MATH500": (86.40, 74.80), "AIME24": (32.00, 22.00)},
}

DATA_LEN = {
    "Qwen3-4B Base":        {"MATH500": (15769, 16224),   "AIME24": (16384, 16384)},
    "Qwen3-4B":             {"MATH500": (6050, 677),      "AIME24": (31733, 4708)},
    "Qwen2.5-7B Inst.":     {"MATH500": (2924, 525),      "AIME24": (13809, 4120)},
}

DATA_REFL = {
    "Qwen3-4B Base":        {"MATH500": (7.00, 0.00),     "AIME24": (5.00, 0.00)},
    "Qwen3-4B":             {"MATH500": (3.56, 0.13),     "AIME24": (7.02, 0.39)},
    "Qwen2.5-7B Inst.":     {"MATH500": (2.78, 0.00),     "AIME24": (8.16, 0.02)},
}

METRIC_INFO = {
    "acc":  {"data": DATA_ACC,  "ylabel": "Accuracy (%)",          "fmt": ".1f", "prefix": "abl_base_acc",  "higher_better": True},
    "len":  {"data": DATA_LEN,  "ylabel": "Avg. Output Length",    "fmt": ".0f", "prefix": "abl_base_len",  "higher_better": False},
    "refl": {"data": DATA_REFL, "ylabel": "#Reflective / Answer",  "fmt": ".2f", "prefix": "abl_base_refl", "higher_better": False},
}

BENCHMARKS = ["MATH500", "AIME24"]

OUTDIR = "plots/ablation"
os.makedirs(OUTDIR, exist_ok=True)

for metric_key, minfo in METRIC_INFO.items():
    DATA = minfo["data"]
    ylabel = minfo["ylabel"]
    vfmt = minfo["fmt"]
    prefix = minfo["prefix"]

    for bm in BENCHMARKS:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))

        n_models = len(MODELS)
        bw = 0.50
        gap_within = 0.15
        group_gap = 0.8

        # Think 组 X 坐标
        think_xs = np.arange(n_models) * (bw + gap_within)
        think_center = think_xs.mean()

        # No-think 组 X 坐标
        nothink_start = think_xs[-1] + bw + group_gap
        nothink_xs = nothink_start + np.arange(n_models) * (bw + gap_within)
        nothink_center = nothink_xs.mean()

        # 收集所有有效数值用于 Y 轴范围
        think_vals = [DATA[m][bm][0] for m in MODELS]
        nothink_vals = [DATA[m][bm][1] for m in MODELS]
        all_v = [v for v in think_vals + nothink_vals if v is not None]
        ymax_data = max(all_v) if all_v else 100

        # ── Y 轴设置 ──
        if metric_key == "refl":
            ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
            ymax = ymax_data * 4.0
            ax.set_ylim(-0.002, ymax)
        else:
            ymin_auto = min(all_v) if all_v else 0
            ymin = ymin_auto * 0.7 if (ymin_auto > ymax_data * 0.3) else 0
            ymax = ymax_data * 1.35
            ax.set_ylim(ymin, ymax)

        trans_val = blended_transform_factory(ax.transData, ax.transAxes)

        def _bar_label(x_pos, val, fmt):
            """在 bar 顶部紧贴放置数值标签"""
            if metric_key == "refl" and val == 0:
                ax.text(x_pos, 0.06, "0", transform=trans_val,
                        ha="center", va="bottom", fontsize=12, color="#222", fontweight="bold")
                return
            disp = ax.transData.transform((0, val))
            _, val_axes_y = ax.transAxes.inverted().transform(disp)
            ax.text(x_pos, val_axes_y + 0.03, f"{val:{fmt}}", transform=trans_val,
                    ha="center", va="bottom", fontsize=12, color="#222", fontweight="bold")

        def _draw_bar(x_pos, val, base_color, model_name):
            """画一个 bar, 根据模型应用对应的饱和度和纹理"""
            style = MODEL_STYLE[model_name]
            color = adjust_saturation(base_color, style["sat"])
            if val is not None:
                ax.bar(x_pos, val, bw, color=color, edgecolor="white", lw=0.6,
                       zorder=3, hatch=style["hatch"], alpha=style["alpha"])
                _bar_label(x_pos, val, vfmt)
            else:
                ax.bar(x_pos, 0, bw, color="#eeeeee", edgecolor="#cccccc", lw=0.5, zorder=2)
                ax.text(x_pos, 0.02, "N/A", transform=trans_val,
                        ha="center", va="bottom", fontsize=10, color="#999", fontstyle="italic")

        # ── 画 Think 组 (蓝色系) ──
        for i, m in enumerate(MODELS):
            _draw_bar(think_xs[i], think_vals[i], DEEP_BLUE, m)

        # ── 画 No-think 组 (绿色系) ──
        for i, m in enumerate(MODELS):
            _draw_bar(nothink_xs[i], nothink_vals[i], DEEP_GREEN, m)

        # ── X 轴标签 ──
        all_xs = list(think_xs) + list(nothink_xs)
        all_labels = list(MODELS) + list(MODELS)
        ax.set_xticks(all_xs)
        ax.set_xticklabels(all_labels, fontsize=11, rotation=35, ha="right", rotation_mode="anchor")

        # 分组标签
        trans = ax.get_xaxis_transform()
        ax.text(think_center, -0.28, "Think", transform=trans,
                ha="center", va="top", fontsize=15, fontweight="bold", color="#2B4C7E")
        ax.text(nothink_center, -0.28, "No-think", transform=trans,
                ha="center", va="top", fontsize=15, fontweight="bold", color="#2E7D32")

        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=3, width=0.5)

        # ── Legend (只在 refl 图中显示) ──
        if metric_key == "refl":
            # 用实际颜色构建 legend, 展示模型 + 模式的组合
            legend_handles = [
                Patch(fc=adjust_saturation(DEEP_BLUE, 0.30), ec="w", hatch="///",
                      label="Qwen3-4B Base"),
                Patch(fc=adjust_saturation(DEEP_BLUE, 1.00), ec="w",
                      label="Qwen3-4B"),
                Patch(fc=adjust_saturation(DEEP_BLUE, 0.60), ec="w", hatch="..",
                      label="Qwen2.5-7B Inst."),
            ]
            ax.legend(handles=legend_handles, fontsize=10, loc="upper right",
                      handlelength=1.2, handletextpad=0.3, borderpad=0.3,
                      labelspacing=0.25, framealpha=0.95, edgecolor="#ccc")

        # ── Save ──
        tag = bm.lower()
        fig.savefig(f"{OUTDIR}/{prefix}_{tag}.pdf")
        fig.savefig(f"{OUTDIR}/{prefix}_{tag}.png")
        plt.close(fig)
        print(f"[ok] {metric_key} {bm}")

print("\nAll ablation base figures saved.")
