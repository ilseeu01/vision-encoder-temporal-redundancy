#!/bin/bash
# Download all HuggingFace-hosted model weights used in the analysis.
# SAM2.1 / SAM3 are NOT here: they load from source repos + local checkpoints
# (set SAM2_REPO / SAM2_CKPT / SAM3_REPO env vars, see README "Setup").
# `hf` ships with huggingface_hub; override the binary via the HF env var.
HF="${HF:-hf}"
LOG="$(cd "$(dirname "$0")" && pwd)/dl_models.log"
: > "$LOG"
MODELS=(
  "openai/clip-vit-large-patch14"            # CLIP-L/14        (env qwen3vl)
  "facebook/dinov2-large"                    # DINOv2-L/14      (env qwen3vl)
  "Qwen/Qwen2.5-VL-7B-Instruct"              # Qwen2.5-VL-7B    (env qwen3vl)
  "Qwen/Qwen3-VL-8B-Instruct"                # Qwen3-VL-8B      (env qwen3vl)
  "OpenGVLab/InternVL2_5-8B"                 # InternVL2.5-8B   (env internvl)
  "OpenGVLab/InternVL3-8B"                   # InternVL3-8B     (env internvl)
  "lmms-lab/llava-onevision-qwen2-0.5b-ov"   # LLaVA-OneVision-0.5B (env qwen3vl)
)
for M in "${MODELS[@]}"; do
  echo "=== $(date +%H:%M:%S) downloading $M ===" >> "$LOG"
  "$HF" download "$M" >> "$LOG" 2>&1 && echo "OK $M" >> "$LOG" || echo "FAIL $M" >> "$LOG"
done
echo "=== ALL DONE $(date +%H:%M:%S) ===" >> "$LOG"
echo "SAM2.1 / SAM3: clone their repos and place checkpoints, then set SAM2_REPO/SAM3_REPO." >> "$LOG"
