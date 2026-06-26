#!/bin/bash
YT=/home/hjlee/miniconda3/envs/qwen3vl/bin/yt-dlp
FF=/home/hjlee/miniconda3/envs/qwen3vl/bin
D=/home/hjlee/video_delta_analysis/videos/cand
LOG=/home/hjlee/video_delta_analysis/dl_videos.log
mkdir -p "$D"; : > "$LOG"
FMT='bv*[height<=480]+ba/b[height<=480]/bv*+ba/b'
declare -A Q=(
  [fast1]="ytsearch1:motocross onboard pov helmet cam"
  [fast2]="ytsearch1:formula 1 onboard hotlap"
  [fast3]="ytsearch1:downhill mountain bike pov fast"
  [static1]="ytsearch1:locked off tripod shot empty room interview b roll"
  [static2]="ytsearch1:static security camera street sample footage"
  [static3]="ytsearch1:still calm landscape long take no camera movement"
)
for k in fast1 fast2 fast3 static1 static2 static3; do
  echo "=== $k : ${Q[$k]} ===" >> "$LOG"
  $YT --ffmpeg-location "$FF" -f "$FMT" --download-sections "*0-45" \
      --recode-video mp4 -o "$D/$k.%(ext)s" "${Q[$k]}" >> "$LOG" 2>&1 \
      && echo "OK $k" >> "$LOG" || echo "FAIL $k" >> "$LOG"
done
echo "=== VIDEOS DONE ===" >> "$LOG"
ls -la "$D" >> "$LOG"
