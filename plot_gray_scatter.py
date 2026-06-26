"""Grayscale Analysis-1 scatter for a chosen metric (L1 or L2): per-frame pixel change
(x) vs embedding change (y), cuts excluded, static+fast overlaid, shared axes per figure.
Embedding axis uses each model's OWN scale (log y) because emb-L2/L1 magnitudes differ
wildly across models; pixel axis is shared. One figure per metric.
Usage: python plot_gray_scatter.py            (makes l1 and l2)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "out_gray")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
MODELS = [("clip", "CLIP-L/14"), ("dinov2", "DINOv2-L/14"),
          ("sam2", "SAM2.1-Hiera-B+"), ("sam3", "SAM3-ViT"),
          ("intern25", "InternVL2.5-8B"), ("intern3", "InternVL3-8B"),
          ("llava_ov", "LLaVA-OneVision-0.5B"),
          ("qwen25", "Qwen2.5-VL-7B"), ("qwen3", "Qwen3-VL-8B")]
VID_STYLE = {"static": ("#7ec8f0", "#08306b", "static"),
             "fast":   ("#ffc580", "#cc4400", "fast")}


def load_xy(sub, vid, metric):
    d = np.load(f"{ROOT}/{sub}/{vid}/frames.npz")
    x = np.asarray(d[f"pix_{metric}"], np.float64)
    y = np.asarray(d[f"emb_{metric}"], np.float64)
    cj = json.load(open(f"{ROOT}/{sub}/{vid}/scene_cuts.json"))
    cut = set(int(c - 1) for c in cj.get("cut_frames", []))
    keep = np.array([i for i in range(len(x)) if i not in cut])
    return x[keep], y[keep]


def pearson(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def make(metric):
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (sub, name) in zip(axes.ravel(), MODELS):
        data = {v: load_xy(sub, v, metric) for v in VID_STYLE}
        xmax = max(x.max() for x, _ in data.values()) * 1.05
        txt = []
        for vid, (pc, lc, vl) in VID_STYLE.items():
            x, y = data[vid]
            ax.scatter(x, y, s=6, alpha=0.30, color=pc, edgecolors="none")
            if x.std() > 1e-12:
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=lc, lw=2.4)
            txt.append((f"{vl}: r={pearson(x, y):.3f}", lc))
        ax.set_xlim(0, xmax)
        ax.set_title(name, fontsize=11, pad=3)
        for k, (s, c) in enumerate(txt):
            ax.text(0.04, 0.96 - k * 0.08, s, transform=ax.transAxes, va="top",
                    ha="left", fontsize=9, color=c, fontweight="bold")
        ax.set_xlabel(f"pixel {metric.upper()} change (grayscale)", fontsize=9)
        ax.set_ylabel(f"embedding {metric.upper()} change", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.legend(handles=[
        Line2D([0], [0], color="#08306b", lw=2.4, marker="o", markerfacecolor="#7ec8f0",
               markeredgecolor="none", markersize=8, label="static (locked camera)"),
        Line2D([0], [0], color="#cc4400", lw=2.4, marker="o", markerfacecolor="#ffc580",
               markeredgecolor="none", markersize=8, label="fast (downhill MTB POV)")],
        loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle(f"Analysis 1 (grayscale) — pixel {metric.upper()} vs embedding {metric.upper()} "
                 f"change, per-model y-scale (cuts excluded)", fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"{RES}/gray_analysis1_scatter_{metric}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("written", out)


for mt in ("l1", "l2"):
    make(mt)
