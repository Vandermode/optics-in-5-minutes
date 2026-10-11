#!/usr/bin/env bash
# The one shareable copy of a finished episode:  common/release_encode.sh <episode_dir> [out_dir=<repo>/dist]
#   <slug>_1080p.mp4   clean 1080p60 picture + an English subtitle track viewers can toggle (mov_text), narration audio
# A slow-preset x264 re-encode of the render's soft-subtitle film (the render itself uses superfast CRF 17 for speed):
# about half the size, visually transparent (E01: 52 -> 22 MB, 47-51 dB PSNR). Audio is copied unchanged.
# Env: THREADS (x264 threads, default 8).
set -euo pipefail
EP=$(cd "$1" && pwd)
source "$(dirname "$0")/env.sh"
OUTD=${2:-$O5_ROOT/dist}
mkdir -p "$OUTD"
SLUG=$("$O5_PY" -c "import json;print(json.load(open('$EP/episode.json'))['slug'])")
FF=${O5_FFMPEG:-ffmpeg}
SOFT="$EP/${SLUG}_1080p60_softsubs.mp4"
[ -f "$SOFT" ] || { echo "no $SOFT: run make render first" >&2; exit 1; }
nice -n 5 "$FF" -y -hide_banner -loglevel error -i "$SOFT" -map 0:v -map 0:a -map 0:s \
    -c:v libx264 -preset slow -crf 20 -tune animation -pix_fmt yuv420p -threads "${THREADS:-8}" \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
    -c:a copy -c:s mov_text -metadata:s:s:0 language=eng -movflags +faststart "$OUTD/${SLUG}_1080p.mp4"
ls -la "$OUTD/${SLUG}_1080p.mp4"
