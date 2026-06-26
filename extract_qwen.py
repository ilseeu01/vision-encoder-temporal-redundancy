"""
Qwen3-VL-8B: extract per-frame vision-encoder PRE-PROJECTOR patch embeddings
(captured as the input to model.visual.merger) for every original frame, then
compute pixel & embedding deltas and run the shared analysis.
"""
import os, sys, gc
import numpy as np
import torch
import av
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import common

VIDEO = os.environ.get("VDA_VIDEO", os.path.expanduser("~/cd/sam2/assets/caterpillar_9_24.mp4"))
OUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_qwen")
NPZ   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "qwen_frames.npz")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
RES   = common.FIXED_RES          # 448
DEV   = "cuda:0"
THUMB = 160


def load_frames():
    c = av.open(VIDEO)
    frames = []
    for fr in c.decode(video=0):
        img = fr.to_image().convert("RGB").resize((RES, RES), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.uint8))
    c.close()
    return frames                 # list of (RES,RES,3) uint8


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print("loading Qwen3-VL-8B ...")
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": DEV}, attn_implementation="sdpa")
    model.eval()
    visual = model.model.visual

    # capture merger INPUT = pre-projector patch tokens
    grab = {}
    def pre_hook(mod, args):
        grab["x"] = args[0].detach().float().cpu()
    h = visual.merger.register_forward_pre_hook(pre_hook)

    frames = load_frames()
    N = len(frames)
    print(f"{N} frames @ {RES}x{RES}")

    img_proc = proc.image_processor
    embs = []          # (gh*gw, D) per frame
    gh = gw = None
    for i, fr in enumerate(frames):
        pil = Image.fromarray(fr)
        enc = img_proc(images=[pil], return_tensors="pt")
        pv = enc["pixel_values"].to(DEV, torch.bfloat16)
        grid = enc["image_grid_thw"].to(DEV)            # [[1,h,w]]
        with torch.no_grad():
            visual(pv, grid_thw=grid)
        x = grab["x"]                                   # (h*w, D)
        t, hh, ww = [int(v) for v in grid[0]]
        if gh is None:
            gh, gw = hh, ww
            print(f"patch grid = {gh}x{gw}, dim={x.shape[-1]}")
        embs.append(x.numpy().astype(np.float16))
        if (i + 1) % 50 == 0:
            print(f"  encoded {i+1}/{N}")
    h.remove()
    del model, visual; gc.collect(); torch.cuda.empty_cache()

    # --- deltas
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

    common.run_analysis(NPZ, OUT, "Qwen3-VL-8B")


if __name__ == "__main__":
    main()
