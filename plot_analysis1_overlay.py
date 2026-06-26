"""Analysis-1 overlay: per model, static (blue) AND fast (orange) scatter of
pixel cosine-distance (x) vs embedding cosine-distance (y), cuts excluded, with
per-video OLS lines. All 9 panels share ONE global x/y range (max over both videos)
so embedding-change magnitude is comparable at a glance across models and videos."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/home/hjlee/video_delta_analysis/out"
RES = "/home/hjlee/video_delta_analysis/results"
os.makedirs(RES, exist_ok=True)

MODELS = [
    ("clip", "CLIP-L/14"), ("dinov2", "DINOv2-L/14"),
    ("sam2", "SAM2.1-Hiera-B+"), ("sam3", "SAM3-ViT"),
    ("intern25", "InternVL2.5-8B"), ("intern3", "InternVL3-8B"),
    ("llava_ov", "LLaVA-OneVision-0.5B"),
    ("qwen25", "Qwen2.5-VL-7B"), ("qwen3", "Qwen3-VL-8B"),
]
# (point color, line color, label) — light points + dark saturated line so the
# scatter cloud and its OLS fit are clearly distinguishable.
VID_STYLE = {
    "static": ("#7ec8f0", "#08306b", "static"),   # sky-blue points / navy line
    "fast":   ("#ffc580", "#cc4400", "fast"),      # light-orange points / dark-orange line
}


def load_xy(sub, vid):
    d = np.load(f"{ROOT}/{sub}/{vid}/frames.npz")
    x = np.asarray(d["pix_cos"], np.float64)
    y = np.asarray(d["emb_cos"], np.float64)
    cj = json.load(open(f"{ROOT}/{sub}/{vid}/scene_cuts.json"))
    cut = set(int(c - 1) for c in cj.get("cut_frames", []))
    keep = np.array([i for i in range(len(x)) if i not in cut])
    return x[keep], y[keep]


def pearson(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# preload everything + global axis maxima over BOTH videos
data = {sub: {v: load_xy(sub, v) for v in VID_STYLE} for sub, _ in MODELS}
gxmax = max(xy[0].max() for m in data.values() for xy in m.values()) * 1.05
gymax = max(xy[1].max() for m in data.values() for xy in m.values()) * 1.05

fig, axes = plt.subplots(3, 3, figsize=(13, 12))
for ax, (sub, name) in zip(axes.ravel(), MODELS):
    txt = []
    for vid, (pcolor, lcolor, vlab) in VID_STYLE.items():
        x, y = data[sub][vid]
        ax.scatter(x, y, s=6, alpha=0.28, color=pcolor, edgecolors="none")
        if x.std() > 1e-12:
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=lcolor, lw=2.6)
        txt.append((f"{vlab}: r = {pearson(x, y):.3f}", lcolor))
    ax.set_xlim(0, gxmax)
    ax.set_ylim(0, gymax)
    ax.set_title(name, fontsize=11, pad=4)
    for k, (s, c) in enumerate(txt):
        ax.text(0.04, 0.96 - k * 0.08, s, transform=ax.transAxes, va="top",
                ha="left", fontsize=9, color=c, fontweight="bold")
    ax.set_xlabel("pixel cosine-distance", fontsize=9)
    ax.set_ylabel("embedding cosine-distance", fontsize=9)
    ax.tick_params(labelsize=8)

fig.legend(handles=[
    Line2D([0], [0], color="#08306b", lw=2.6, marker="o", markerfacecolor="#7ec8f0",
           markeredgecolor="none", markersize=8, label="static (locked camera)"),
    Line2D([0], [0], color="#cc4400", lw=2.6, marker="o", markerfacecolor="#ffc580",
           markeredgecolor="none", markersize=8, label="fast (downhill MTB POV)")],
    loc="upper right", fontsize=10, framealpha=0.9)
fig.suptitle("Analysis 1 — pixel vs embedding change, static + fast overlaid "
             "(shared axes, cuts excluded)", fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = f"{RES}/analysis1_scatter_overlay.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print("written", out, "| gxmax=%.4f gymax=%.4f" % (gxmax, gymax))
