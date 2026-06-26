"""LLaVA-OneVision-Qwen2-0.5b-ov SigLIP vision tower patch tokens + timing (env: qwen3vl).
The checkpoint is in the ORIGINAL LLaVA format (flat config), so the HF
LlavaOnevisionForConditionalGeneration class fails to load. We therefore rebuild the
SigLIP-so400m/14-384 vision tower and load its (fine-tuned) weights directly from the
checkpoint (prefix model.vision_tower.vision_tower.vision_model.*). Single tile, no CLS."""
import os, sys, json, glob
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
sys.path.insert(0, "/home/hjlee/video_delta_analysis"); import common

DEV = "cuda:0"
MODEL = "lmms-lab/llava-onevision-qwen2-0.5b-ov"; NAME = "LLaVA-OneVision-0.5B"
VIDS = json.load(open("/home/hjlee/video_delta_analysis/videos.json"))
OUT = "/home/hjlee/video_delta_analysis/out/llava_ov"
NRES = 384
SNAP = sorted(glob.glob(f"/home/hjlee/.cache/huggingface/hub/models--{MODEL.replace('/','--')}/snapshots/*/"))[-1]
PREFIX = "model.vision_tower.vision_tower."   # -> leaves "vision_model.*"


def main():
    from transformers import AutoConfig, SiglipVisionModel
    vcfg = AutoConfig.from_pretrained("google/siglip-so400m-patch14-384").vision_config
    model = SiglipVisionModel(vcfg)

    # load fine-tuned vision-tower weights from the LLaVA-OV checkpoint; tally total params
    sd, total = {}, 0
    for st in glob.glob(SNAP + "*.safetensors"):
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k); total += t.numel()
                if k.startswith(PREFIX):
                    sd[k[len(PREFIX):]] = t                # "vision_model.*"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{NAME}] loaded vision tower: {len(sd)} tensors, "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(DEV, torch.bfloat16).eval()

    def pv(fr):
        img = Image.fromarray(fr).resize((NRES, NRES), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5   # SigLIP norm
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV, torch.bfloat16)

    def fwd(fr):
        # LLaVA-OV taps mm_vision_select_layer=-2 (second-to-last); the last SigLIP layer
        # and pooling head are unused (absent from ckpt). select_feature="patch" (no CLS).
        with torch.no_grad():
            out = model(pixel_values=pv(fr), output_hidden_states=True)
        return out.hidden_states[-2][0]

    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="LMM", encoder="SigLIP-so400m/14-384 (LLaVA-OV vision tower, fine-tuned)",
                dtype="bf16", native_res=f"{NRES} (single tile, anyres bypassed)",
                total_params=int(total),
                vision_encoder_params=common.count_params(model))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
