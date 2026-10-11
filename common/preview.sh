#!/usr/bin/env bash
# Low-res preview of chosen chapters (default: all), at most $PAR Manim processes at a time (default 2).
#   common/preview.sh <episode_dir> [Ch0Hook Ch3Foo ...]   -> $OUT/<Chapter>/####.png  (960x540, 15 fps)
# Also warms the episode's LaTeX cache (<episode_dir>/.cache/Tex) from the preview media dirs.
set -u
EP=$(cd "$1" && pwd); shift
source "$(dirname "$0")/env.sh"
SLUG=$("$O5_PY" -c "import json;print(json.load(open('$EP/episode.json'))['slug'])")
OUT=${OUT:-${TMPDIR:-/tmp}/o5pv_$SLUG}
RES=${RES:-960,540}
FPS=${FPS:-15}
PAR=${PAR:-2}
CH=("$@")
if [ ${#CH[@]} -eq 0 ]; then
  mapfile -t CH < <("$O5_PY" -c "import json;print('\n'.join(json.load(open('$EP/episode.json'))['chapters']))")
fi
mkdir -p "$OUT/logs" "$EP/.cache/Tex"
run() {
  local c=$1
  rm -rf "$OUT/media_$c"
  (cd "$EP" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 5 "$O5_PY" -m manim render \
      -r "$RES" --fps "$FPS" --format png-sequence --disable_caching \
      --media_dir "$OUT/media_$c" scenes.py "$c" > "$OUT/logs/$c.log" 2>&1)
  local d
  d=$(find "$OUT/media_$c/images" -type d -name "$c" 2>/dev/null | head -1)
  rm -rf "$OUT/$c"; [ -n "$d" ] && mv "$d" "$OUT/$c"
  cp -n "$OUT/media_$c"/Tex/* "$EP/.cache/Tex/" 2>/dev/null
  echo "$c: $(ls "$OUT/$c" 2>/dev/null | wc -l) frames; $(grep -E 'LATE|chapter [0-9]|Error|Traceback' "$OUT/logs/$c.log" | tr '\n' ' ' | cut -c1-400)"
}
n=0
for c in "${CH[@]}"; do
  run "$c" &
  n=$((n + 1))
  if [ $n -ge "$PAR" ]; then wait -n; n=$((n - 1)); fi
done
wait
echo "previews in $OUT"
