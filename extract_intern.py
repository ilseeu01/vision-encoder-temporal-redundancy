"""
InternVL3-8B: extract per-frame InternViT PRE-PROJECTOR patch embeddings
(vision_model(...).last_hidden_state[:, 1:, :], i.e. before pixel-shuffle + MLP)
for every original frame, then compute pixel & embedding deltas + analysis.
"""
import os, sys, gc
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import common

VIDEO = "/home/hjlee/cd/sam2/assets/caterpillar_9_24.mp4"
OUT   = "/home/hjlee/video_delta_analysis/out_intern"
NPZ   = "/home/hjlee/video_delta_analysis/cache/intern_frames.npz"
MODEL = "OpenGVLab/InternVL3-8B"
RES   = common.FIXED_RES          # 448
DEV   = "cuda:0"
THUMB = 160
MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_frames():
    vr = VideoReader(VIDEO, ctx=cpu(0))
    frames = []
    for i in range(len(vr)):
        arr = vr[i].asnumpy()                       # (H,W,3) RGB uint8
        img = Image.fromarray(arr).resize((RES, RES), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def main():
    from transformers import AutoModel

    print("loading InternVL3-8B ...")
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True).eval().to(DEV)
    vision = model.vision_model

    frames = load_frames()
    N = len(frames)
    print(f"{N} frames @ {RES}x{RES}")

    embs = []
    gh = gw = None
    for i, fr in enumerate(frames):
        x = (fr.astype(np.float32) / 255.0 - MEAN) / STD          # (RES,RES,3)
        pv = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV, torch.bfloat16)
        with torch.no_grad():
            out = vision(pixel_values=pv).last_hidden_state        # (1, 1+P, D)
        tok = out[0, 1:, :].float().cpu().numpy()                  # (P, D) drop CLS
        if gh is None:
            P = tok.shape[0]; gh = gw = int(round(P ** 0.5))
            print(f"patch grid = {gh}x{gw}, dim={tok.shape[-1]}")
        embs.append(tok.astype(np.float16))
        if (i + 1) % 50 == 0:
            print(f"  encoded {i+1}/{N}")
    del model, vision; gc.collect(); torch.cuda.empty_cache()

    fr01 = [f.astype(np.float32) / 255.0 for f in frames]
    scal = {f"{mod}_{m}": [] for mod in ("pix", "emb") for m in common.METRICS}
    mp   = {f"{mod}map_{m}": [] for mod in ("pix", "emb") for m in common.METRICS}
    for i in range(1, N):
        pm = common.pixel_delta_maps(fr01[i - 1], fr01[i], gh, gw)
        em = common.embed_delta_maps(embs[i - 1], embs[i], gh, gw)
        for m in common.METRICS:
            scal[f"pix_{m}"].append(float(pm[m].mean())); mp[f"pixmap_{m}"].append(pm[m])
            scal[f"emb_{m}"].append(float(em[m].mean())); mp[f"embmap_{m}"].append(em[m])

    thumbs = np.stack([np.asarray(Image.fromarray(f).resize((THUMB, THUMB)), dtype=np.uint8)
                       for f in frames])
    save = {"n_frames": N, "gh": gh, "gw": gw, "thumbs": thumbs}
    for k, v in scal.items(): save[k] = np.asarray(v, dtype=np.float32)
    for k, v in mp.items():   save[k] = np.asarray(v, dtype=np.float32)
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, **save)
    print("saved", NPZ)

    common.run_analysis(NPZ, OUT, "InternVL3-8B")


if __name__ == "__main__":
    main()
