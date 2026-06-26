#!/bin/bash
# Full GRAYSCALE extraction for all 9 models, 4 GPUs (one sequential queue per GPU).
cd /home/hjlee/video_delta_analysis
export VDA_GRAY=1
PQ=/home/hjlee/miniconda3/envs/qwen3vl/bin/python
PI=/home/hjlee/miniconda3/envs/internvl/bin/python
P2=/home/hjlee/miniconda3/envs/sam2/bin/python
P3=/home/hjlee/miniconda3/envs/sam3/bin/python
LOG=/home/hjlee/video_delta_analysis/run_all_gray.log
: > "$LOG"
mkdir -p logs_gray
run() {
  local gpu="$1" tag="$2" cmd="$3"
  echo "[$(date +%H:%M:%S)] GPU$gpu START $tag" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu $cmd > "logs_gray/$tag.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] GPU$gpu OK    $tag" >> "$LOG" \
    || echo "[$(date +%H:%M:%S)] GPU$gpu FAIL  $tag" >> "$LOG"
}
( run 0 sam3 "$P3 ext_sam3.py"; run 0 clip "$PQ ext_clip.py" ) &
( run 1 sam2 "$P2 ext_sam2.py"; run 1 dinov2 "$PQ ext_dinov2.py" ) &
( run 2 qwen3 "$PQ ext_qwen3.py"; run 2 qwen25 "$PQ ext_qwen25.py" ) &
( run 3 intern3 "$PI ext_intern.py 3"; run 3 intern25 "$PI ext_intern.py 25"; run 3 llava "$PQ ext_llava_ov.py" ) &
wait
echo "[$(date +%H:%M:%S)] ALL GRAYSCALE EXTRACTION DONE" >> "$LOG"
