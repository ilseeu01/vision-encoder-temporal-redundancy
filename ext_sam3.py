"""SAM3 image-model ViT trunk feature map delta + timing (env: sam3).
trunk outputs [...,(1,1024,72,72)] at 1008 input -> 5184 tokens, dim 1024. Norm 0.5/0.5."""
import os, sys, json
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import common
sys.path.insert(0, os.environ.get("SAM3_REPO", os.path.expanduser("~/cd/sam3")))

DEV = "cuda"
NAME = "SAM3-ViT"
VIDS = json.load(open(os.path.join(HERE, "videos.json")))
OUT = os.path.join(HERE, "out", "sam3")
NRES = 1008


def main():
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(device=DEV, eval_mode=True)
    trunk = common.get_submodule(model, ["backbone.vision_backbone.trunk",
                                         "backbone.visual.trunk"])

    def pv(fr):
        img = Image.fromarray(fr).resize((NRES, NRES), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV).float()

    def fwd(fr):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = trunk(pv(fr))
        feat = out[-1] if isinstance(out, (list, tuple)) else out  # (1,1024,72,72)
        c = feat.shape[1]
        return feat[0].reshape(c, -1).T              # (5184,1024)

    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="VFM", encoder="SAM3 ViT-L/14 trunk (img 1008)",
                dtype="bf16-autocast", native_res=f"{NRES}",
                total_params=common.count_params(model),
                vision_encoder_params=common.count_params(trunk))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
