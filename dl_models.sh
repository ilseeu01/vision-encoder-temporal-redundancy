#!/bin/bash
HF=/home/hjlee/miniconda3/envs/qwen3vl/bin/hf
LOG=/home/hjlee/video_delta_analysis/dl_models.log
: > "$LOG"
for M in "Qwen/Qwen2.5-VL-7B-Instruct" "OpenGVLab/InternVL2_5-8B" "lmms-lab/llava-onevision-qwen2-0.5b-ov" "facebook/dinov2-large"; do
  echo "=== $(date +%H:%M:%S) downloading $M ===" >> "$LOG"
  $HF download "$M" >> "$LOG" 2>&1 && echo "OK $M" >> "$LOG" || echo "FAIL $M" >> "$LOG"
done
echo "=== ALL DONE $(date +%H:%M:%S) ===" >> "$LOG"
