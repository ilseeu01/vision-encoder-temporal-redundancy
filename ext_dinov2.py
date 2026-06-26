"""DINOv2-large patch tokens + timing (env: qwen3vl). 448 input, interpolated pos-enc, no CLS."""
import os, sys, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/hjlee/video_delta_analysis"); import common

DEV = "cuda:0"
MODEL = "facebook/dinov2-large"; NAME = "DINOv2-L/14"
VIDS = json.load(open("/home/hjlee/video_delta_analysis/videos.json"))
OUT = "/home/hjlee/video_delta_analysis/out/dinov2"
RES = common.FIXED_RES
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval().to(DEV)

    def pv(fr):
        x = (fr.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV, torch.bfloat16)

    def fwd(fr):
        with torch.no_grad():
            out = model(pixel_values=pv(fr), interpolate_pos_encoding=True)
        return out.last_hidden_state[0, 1:, :]       # drop CLS

    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="VFM", encoder="DINOv2 ViT-L/14 (SSL, no CLS)",
                dtype="bf16", native_res="518 default; here 448 w/ interp pos-enc",
                total_params=common.count_params(model),
                vision_encoder_params=common.count_params(model))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
