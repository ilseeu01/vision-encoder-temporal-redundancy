"""Grayscale Analysis-1 SHARED-AXIS overlay, the grayscale analogue of
analysis1_scatter_overlay.png. Raw embedding L1/L2 magnitudes differ ~6000x
across models, so we cannot share the y-axis on raw values. Instead we normalise
each model's per-frame embedding change by THAT model's mean embedding change
(pooled over both videos, cuts excluded) -> a unit-less "relative embedding
change" (1.0 = that model's typical change). The pixel axis is already shared
because grayscale pixel L1=L2 is identical for every model. static + fast are
overlaid with per-video OLS lines; all 9 panels share one global x/y range.
One figure per metric (l1, l2). Usage: python plot_gray_overlay_shared.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/home/hjlee/video_delta_analysis/out_gray"
RES = "/home/hjlee/video_delta_analysis/results"
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
    # load raw, then normalise embedding by per-model pooled mean (both videos)
    data = {}
    for sub, _ in MODELS:
        vids = {v: load_xy(sub, v, metric) for v in VID_STYLE}
        scale = np.concatenate([vids[v][1] for v in VID_STYLE]).mean()
        scale = scale if scale > 1e-12 else 1.0
        data[sub] = {v: (vids[v][0], vids[v][1] / scale) for v in VID_STYLE}
    # global shared ranges (robust 99.5pct cap so a few outliers don't flatten it)
    allx = np.concatenate([xy[0] for m in data.values() for xy in m.values()])
    ally = np.concatenate([xy[1] for m in data.values() for xy in m.values()])
    gxmax = np.percentile(allx, 99.5) * 1.05
    gymax = np.percentile(ally, 99.5) * 1.05

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (sub, name) in zip(axes.ravel(), MODELS):
        txt = []
        for vid, (pc, lc, vl) in VID_STYLE.items():
            x, y = data[sub][vid]
            ax.scatter(x, y, s=6, alpha=0.28, color=pc, edgecolors="none")
            if x.std() > 1e-12:
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=lc, lw=2.6)
            txt.append((f"{vl}: r = {pearson(x, y):.3f}", lc))
        ax.set_xlim(0, gxmax)
        ax.set_ylim(0, gymax)
        ax.set_title(name, fontsize=11, pad=4)
        for k, (s, c) in enumerate(txt):
            ax.text(0.04, 0.96 - k * 0.08, s, transform=ax.transAxes, va="top",
                    ha="left", fontsize=9, color=c, fontweight="bold")
        ax.set_xlabel(f"pixel {metric.upper()} change (grayscale)", fontsize=9)
        ax.set_ylabel(f"embedding {metric.upper()} change / model mean", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.legend(handles=[
        Line2D([0], [0], color="#08306b", lw=2.6, marker="o", markerfacecolor="#7ec8f0",
               markeredgecolor="none", markersize=8, label="static (locked camera)"),
        Line2D([0], [0], color="#cc4400", lw=2.6, marker="o", markerfacecolor="#ffc580",
               markeredgecolor="none", markersize=8, label="fast (downhill MTB POV)")],
        loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle(f"Analysis 1 (grayscale) — pixel {metric.upper()} vs normalised embedding "
                 f"{metric.upper()} change, SHARED axes (cuts excluded)", fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"{RES}/gray_overlay_shared_{metric}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("written", out, "| gxmax=%.4f gymax=%.3f" % (gxmax, gymax))


for mt in ("l1", "l2"):
    make(mt)
