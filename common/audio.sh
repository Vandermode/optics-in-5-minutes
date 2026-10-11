#!/usr/bin/env bash
# Narration only:  common/audio.sh <episode_dir>   (TTS -> Whisper round-trip check -> subtitles)
set -euo pipefail
EP=$(cd "$1" && pwd)
source "$(dirname "$0")/env.sh"
OMP_NUM_THREADS=4 nice -n 5 "$O5_PY" "$O5_COMMON/tts.py" "$EP"
OMP_NUM_THREADS=4 nice -n 5 "$O5_PY" "$O5_COMMON/check_audio.py" "$EP"
"$O5_PY" "$O5_COMMON/make_subs.py" "$EP"
