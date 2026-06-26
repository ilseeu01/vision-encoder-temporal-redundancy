"""CLIP ViT-L/14 vision tower patch tokens + timing (env: qwen3vl). Native 224, no CLS."""
import os, sys, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/hjlee/video_delta_analysis"); import common

DEV = "cuda:0"
MODEL = "openai/clip-vit-large-patch14"; NAME = "CLIP-L/14"
VIDS = json.load(open("/home/hjlee/video_delta_analysis/videos.json"))
OUT = "/home/hjlee/video_delta_analysis/out/clip"
NRES = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def main():
    from transformers import CLIPVisionModel
    model = CLIPVisionModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval().to(DEV)

    def pv(fr):
        img = Image.fromarray(fr).resize((NRES, NRES), Image.BICUBIC)
        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV, torch.bfloat16)

    def fwd(fr):
        with torch.no_grad():
            out = model(pixel_values=pv(fr))
        return out.last_hidden_state[0, 1:, :]       # drop CLS

    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="VFM", encoder="CLIP ViT-L/14 (image-text, no CLS)",
                dtype="bf16", native_res=f"{NRES}",
                total_params=common.count_params(model),
                vision_encoder_params=common.count_params(model))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
