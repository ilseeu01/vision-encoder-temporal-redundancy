"""InternVL3-8B / InternVL2.5-8B InternViT pre-projector patch tokens + timing (env: internvl).
Usage: python ext_intern.py {3|25}   (drops CLS; vision_model.last_hidden_state[:,1:,:])."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/home/hjlee/video_delta_analysis"); import common

DEV = "cuda:0"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "3"
CFG = {"3":  ("OpenGVLab/InternVL3-8B",  "InternVL3-8B",  "intern3"),
       "25": ("OpenGVLab/InternVL2_5-8B", "InternVL2.5-8B", "intern25")}[VARIANT]
MODEL, NAME, SUB = CFG
VIDS = json.load(open("/home/hjlee/video_delta_analysis/videos.json"))
OUT = f"/home/hjlee/video_delta_analysis/out/{SUB}"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RES = common.FIXED_RES


def main():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, low_cpu_mem_usage=True).eval().to(DEV)
    vision = model.vision_model

    def pv(fr):
        x = (fr.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV, torch.bfloat16)

    def fwd(fr):
        with torch.no_grad():
            out = vision(pixel_values=pv(fr)).last_hidden_state
        return out[0, 1:, :]

    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="LMM", encoder="InternViT-300M (pre pixel-shuffle, CLS dropped)",
                dtype="bf16", native_res=f"{RES} (single tile)",
                total_params=common.count_params(model),
                vision_encoder_params=common.count_params(vision))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
