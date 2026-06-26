"""Aggregate out_gray/<model>/<video>/meta.json into one JSON dump with everything
needed for the Notion update: timing, per-video L1/L2/cos means, analysis-1
correlation+slope (L1, L2, cos), analysis-2 CF (L1, L2, cos), and a mean embedding
norm proxy for scale-normalised sensitivity discussion."""
import json, glob, os

ROOT = os.environ.get("VDA_OUT_ROOT", "/home/hjlee/video_delta_analysis/out_gray")
ORDER = [("clip", "CLIP-L/14"), ("dinov2", "DINOv2-L/14"),
         ("sam2", "SAM2.1-Hiera-B+"), ("sam3", "SAM3-ViT"),
         ("intern25", "InternVL2.5-8B"), ("intern3", "InternVL3-8B"),
         ("llava_ov", "LLaVA-OneVision-0.5B"),
         ("qwen25", "Qwen2.5-VL-7B"), ("qwen3", "Qwen3-VL-8B")]
TYPE = {"clip": "VFM", "dinov2": "VFM", "sam2": "VFM", "sam3": "VFM",
        "intern25": "LMM", "intern3": "LMM", "llava_ov": "LMM",
        "qwen25": "LMM", "qwen3": "LMM"}


def g(sub, vid):
    p = f"{ROOT}/{sub}/{vid}/meta.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


rows = []
for sub, name in ORDER:
    rec = {"sub": sub, "name": name, "type": TYPE[sub]}
    for vid in ("fast", "static"):
        m = g(sub, vid)
        if not m:
            rec[vid] = None
            continue
        s = m["delta_summary"]
        a = s["averages_excluding_cuts"]
        c = s["analysis1_correlation"]
        mc = s["motion_concentration"]["p90"]
        rec[vid] = {
            "gray": m.get("grayscale", False),
            "n_frames": m.get("n_frames"),
            "num_cuts": s.get("num_cuts"),
            "timing_ms": m["timing"]["per_frame_ms_mean"],
            "timing_std": m["timing"]["per_frame_ms_std"],
            "pix": {k: a[k]["pixel"] for k in ("l1", "l2", "cos")},
            "emb": {k: a[k]["embedding"] for k in ("l1", "l2", "cos")},
            "corr": {k: c.get(f"pix{k.upper()}_vs_emb{k.upper()}") for k in ("l1", "l2", "cos")},
            "cf": {k: mc["R_over_area"][k] for k in ("l1", "l2", "cos")},
            "R": {k: mc["R"][k] for k in ("l1", "l2", "cos")},
            "area": mc["area_fraction"],
        }
    rows.append(rec)

print(json.dumps(rows, indent=1, ensure_ascii=False))
with open(f"{os.path.dirname(ROOT)}/results/gray_summary.json", "w") as f:
    json.dump(rows, f, indent=1, ensure_ascii=False)
