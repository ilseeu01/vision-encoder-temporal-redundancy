"""Qwen3-VL-8B vision-encoder delta + timing (env: qwen3vl). Pre-merger ViT patch tokens."""
import os, sys, json
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import common

DEV = "cuda:0"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"; NAME = "Qwen3-VL-8B"
VIDS = json.load(open(os.path.join(HERE, "videos.json")))
OUT = os.path.join(HERE, "out", "qwen3")


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    ip = proc.image_processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": DEV},
        attn_implementation="sdpa").eval()
    visual = common.get_submodule(model, ["model.visual", "visual"])

    grab = {}
    visual.merger.register_forward_pre_hook(lambda m, a: grab.__setitem__("x", a[0]))

    def run(fr):
        enc = ip(images=[Image.fromarray(fr)], return_tensors="pt")
        pv = enc["pixel_values"].to(DEV, torch.bfloat16)
        grid = enc["image_grid_thw"].to(DEV)
        with torch.no_grad():
            visual(pv, grid_thw=grid)
        return grab["x"]

    fwd = lambda fr: run(fr)
    tok = lambda fr: run(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="LMM", encoder="Qwen3-VL ViT (pre-merger patch tokens)",
                dtype="bf16", native_res="dynamic (Qwen smart-resize)",
                total_params=common.count_params(model),
                vision_encoder_params=common.count_params(visual))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
