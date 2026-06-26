"""Analysis-1 scatter: per-frame pixel cosine-distance (x) vs embedding cosine-distance (y),
cuts excluded, with OLS fit line and r / R^2 / slope annotation. One 3x3 figure per video."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "out")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

# (out-dir, display name) in a stable, type-grouped order
MODELS = [
    ("clip", "CLIP-L/14"), ("dinov2", "DINOv2-L/14"),
    ("sam2", "SAM2.1-Hiera-B+"), ("sam3", "SAM3-ViT"),
    ("intern25", "InternVL2.5-8B"), ("intern3", "InternVL3-8B"),
    ("llava_ov", "LLaVA-OneVision-0.5B"),
    ("qwen25", "Qwen2.5-VL-7B"), ("qwen3", "Qwen3-VL-8B"),
]
VIDS = {"static": "static video (locked camera)", "fast": "fast video (downhill MTB POV)"}


def load_xy(sub, vid):
    d = np.load(f"{ROOT}/{sub}/{vid}/frames.npz")
    x = np.asarray(d["pix_cos"], np.float64)   # (N-1,)
    y = np.asarray(d["emb_cos"], np.float64)
    cj = json.load(open(f"{ROOT}/{sub}/{vid}/scene_cuts.json"))
    cut_pair = set(int(c - 1) for c in cj.get("cut_frames", []))
    keep = np.array([i for i in range(len(x)) if i not in cut_pair])
    return x[keep], y[keep]


def stats(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0, 0.0, 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    return r, r * r, slope


for vid, vlabel in VIDS.items():
    data = {sub: load_xy(sub, vid) for sub, _ in MODELS}
    # static: unify both axes to a shared range (max over all 9 panels) for at-a-glance
    # comparison of embedding-change magnitude; fast keeps per-panel auto-scaling.
    shared_xmax = shared_ymax = None
    if vid == "static":
        shared_xmax = max(x.max() for x, _ in data.values()) * 1.05
        shared_ymax = max(y.max() for _, y in data.values()) * 1.05

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (sub, name) in zip(axes.ravel(), MODELS):
        x, y = data[sub]
        r, r2, slope = stats(x, y)
        ax.scatter(x, y, s=6, alpha=0.35, color="#1f77b4", edgecolors="none")
        if x.std() > 1e-12:
            xs = np.linspace(x.min(), x.max(), 50)
            b = np.polyfit(x, y, 1)
            ax.plot(xs, np.polyval(b, xs), color="#d62728", lw=1.6)
        ax.set_title(name, fontsize=11, pad=4)
        ax.text(0.04, 0.96,
                f"r = {r:.3f}\n$R^2$ = {r2:.3f}\nslope = {slope:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
        if shared_ymax is not None:
            ax.set_xlim(0, shared_xmax)
            ax.set_ylim(0, shared_ymax)
        ax.set_xlabel("pixel cosine-distance", fontsize=9)
        ax.set_ylabel("embedding cosine-distance", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Analysis 1 — pixel vs embedding change ({vlabel}, cuts excluded)",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"{RES}/analysis1_scatter_{vid}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("written", out)
