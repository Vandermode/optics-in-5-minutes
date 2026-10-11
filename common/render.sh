#!/usr/bin/env bash
# Full pipeline for one episode:  common/render.sh <episode_dir> [--skip-audio]
#   1. narration: Kokoro-82M (kokoro-onnx, 4 CPU threads) -> audio/, timing.json      (skip with --skip-audio)
#   2. check: faster-whisper transcribes it back; subtitles timed to the spoken words
#   3. picture + encode: fast_render.py (windowed parallel Manim -> FFmpeg pieces -> stream-copy stitch)
# Env: WORKERS (default 6; keep <= 8 on a shared server); paths in common/env.sh / config.local.sh.
set -euo pipefail
EP=$(cd "$1" && pwd); shift || true
source "$(dirname "$0")/env.sh"
if [[ "${1:-}" != "--skip-audio" ]]; then
  "$O5_COMMON/audio.sh" "$EP"
fi
WORK=$(mktemp -d)
TMPDIR=$WORK nice -n 5 "$O5_PY" "$O5_COMMON/fast_render.py" "$EP" "${WORKERS:-6}"
rmdir "$WORK" 2>/dev/null || true
