"""SAM2.1 Hiera-base+ image-encoder feature map delta + timing (env: sam2).
vision_features (1,256,64,64) -> 4096 tokens, dim 256."""
import os, sys, json
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import common
sys.path.insert(0, os.environ.get("SAM2_REPO", os.path.expanduser("~/cd/sam2")))

DEV = "cuda:0"
NAME = "SAM2.1-Hiera-B+"
CKPT = os.environ.get("SAM2_CKPT", os.path.join(os.environ.get("SAM2_REPO", os.path.expanduser("~/cd/sam2")), "checkpoints", "sam2.1_hiera_base_plus.pt"))
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
VIDS = json.load(open(os.path.join(HERE, "videos.json")))
OUT = os.path.join(HERE, "out", "sam2")
NRES = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    from sam2.build_sam import build_sam2
    sam = build_sam2(CFG, CKPT, device=DEV, mode="eval")
    enc = sam.image_encoder

    def pv(fr):
        img = Image.fromarray(fr).resize((NRES, NRES), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV).float()

    def fwd(fr):
        with torch.no_grad():
            out = enc(pv(fr))
        feat = out["vision_features"]                # (1,256,64,64)
        c = feat.shape[1]
        return feat[0].reshape(c, -1).T              # (4096,256)

    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)
    meta = dict(model_type="VFM", encoder="SAM2.1 Hiera-B+ image encoder (vision_features)",
                dtype="fp32", native_res=f"{NRES}",
                total_params=common.count_params(sam),
                vision_encoder_params=common.count_params(enc))
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"{OUT}/{tag}", NAME, fwd, tok, dict(meta))


if __name__ == "__main__":
    main()
