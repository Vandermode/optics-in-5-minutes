#!/usr/bin/env bash
# Narration:  common/audio.sh <episode_dir>   (Kokoro TTS -> Whisper round-trip check -> subtitles)
# Fast path: TTS re-synthesises only segments whose `say` changed (audio/cache.json). If the resulting timing.json is
# byte-identical to before (e.g. a fresh clone rebuilding the WAVs, which git does not track), the Whisper check and
# subtitles are already up to date and are skipped. FORCE_CHECK=1 always runs them.
set -euo pipefail
EP=$(cd "$1" && pwd)
source "$(dirname "$0")/env.sh"
T="$EP/audio/timing.json"
before=$( [ -f "$T" ] && cksum < "$T" || echo none )
OMP_NUM_THREADS=4 nice -n 5 "$O5_PY" "$O5_COMMON/tts.py" "$EP"
after=$(cksum < "$T")
if [[ "$before" == "$after" && -f "$EP/audio/words.json" && -f "$EP/subtitles.ass" && -f "$EP/subtitles.srt" \
      && "${FORCE_CHECK:-0}" != "1" ]]; then
  echo "timing unchanged: Whisper check and subtitles are up to date (FORCE_CHECK=1 to rerun)"
else
  OMP_NUM_THREADS=4 nice -n 5 "$O5_PY" "$O5_COMMON/check_audio.py" "$EP"
  "$O5_PY" "$O5_COMMON/make_subs.py" "$EP"
fi
