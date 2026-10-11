#!/usr/bin/env bash
# Compact copies of a finished episode for sharing:  common/release_encode.sh <episode_dir> [out_dir=<repo>/dist]
#   <slug>_1080p.mp4          burned-in subtitles, x264 preset slow CRF 20 (the render uses superfast CRF 17 for speed)
#   <slug>_1080p_softsubs.mp4 clean picture + English subtitle track, same encode
#   <slug>_720p.mp4           burned-in subtitles, 1280x720 CRF 23: a small review copy
# Audio is copied unchanged. Run after render.sh. Env: THREADS (x264 threads, default 8).
set -euo pipefail
EP=$(cd "$1" && pwd)
source "$(dirname "$0")/env.sh"
OUTD=${2:-$O5_ROOT/dist}
mkdir -p "$OUTD"
SLUG=$("$O5_PY" -c "import json;print(json.load(open('$EP/episode.json'))['slug'])")
FF=${O5_FFMPEG:-ffmpeg}
T=${THREADS:-8}
X=(-c:v libx264 -tune animation -pix_fmt yuv420p -threads "$T" -colorspace bt709 -color_primaries bt709 -color_trc bt709
   -color_range tv -movflags +faststart)
SRC="$EP/${SLUG}_1080p60.mp4"
SOFT="$EP/${SLUG}_1080p60_softsubs.mp4"
[ -f "$SRC" ] || { echo "no $SRC: run render.sh first" >&2; exit 1; }
nice -n 5 "$FF" -y -hide_banner -loglevel error -i "$SRC" -map 0:v -map 0:a "${X[@]}" -preset slow -crf 20 \
    -c:a copy "$OUTD/${SLUG}_1080p.mp4"
nice -n 5 "$FF" -y -hide_banner -loglevel error -i "$SOFT" -map 0:v -map 0:a -map 0:s "${X[@]}" -preset slow -crf 20 \
    -c:a copy -c:s mov_text -metadata:s:s:0 language=eng "$OUTD/${SLUG}_1080p_softsubs.mp4"
nice -n 5 "$FF" -y -hide_banner -loglevel error -i "$SRC" -map 0:v -map 0:a -vf scale=1280:720:flags=lanczos \
    "${X[@]}" -preset slow -crf 23 -c:a copy "$OUTD/${SLUG}_720p.mp4"
ls -la "$OUTD/${SLUG}"_*
