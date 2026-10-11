#!/usr/bin/env bash
# Create (or update) the `optics5min` environment and fetch the speech models.
#   ./setup_env.sh                 # uses micromamba, mamba or conda (first found; override with CONDA_TOOL=...)
#   ./setup_env.sh --no-models     # skip the ~800 MB model download (run `make models` later)
# Needs: a C compiler (gcc, or `xcode-select --install` on macOS) to build ManimPango; a LaTeX distribution with
# dvisvgm for Manim's maths; FFmpeg with libass for burned-in subtitles.
#   Ubuntu: sudo apt install build-essential texlive texlive-latex-extra dvisvgm ffmpeg
#   macOS:  xcode-select --install; brew install ffmpeg; MacTeX
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME=${ENV_NAME:-optics5min}
TOOL=${CONDA_TOOL:-}
if [ -z "$TOOL" ]; then
  for t in micromamba mamba conda; do command -v "$t" >/dev/null && { TOOL=$t; break; }; done
fi
[ -n "$TOOL" ] || { echo "need micromamba, mamba or conda on PATH" >&2; exit 1; }
echo "== using $TOOL, environment $ENV_NAME"

# 1. conda-forge packages + pip wheels (everything except manim/ManimPango)
if "$TOOL" run -n "$ENV_NAME" python -c "pass" 2>/dev/null; then
  if [ "$TOOL" = micromamba ]; then "$TOOL" update -y -n "$ENV_NAME" -f environment.yml
  else "$TOOL" env update -n "$ENV_NAME" -f environment.yml; fi
else
  if [ "$TOOL" = micromamba ]; then "$TOOL" create -y -n "$ENV_NAME" -f environment.yml
  else "$TOOL" env create -n "$ENV_NAME" -f environment.yml; fi
fi
P=$("$TOOL" run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")
PY=$P/bin/python
echo "== environment at $P"

# 2. ManimPango 0.7 (required by manim 0.22): no binary wheel or conda-forge build exists, so compile it against
#    this environment's pango, with an rpath so it never picks up a system pango at run time
PKG_CONFIG_PATH=$P/lib/pkgconfig CFLAGS="-I$P/include" LDFLAGS="-Wl,-rpath,$P/lib -L$P/lib" \
  "$PY" -m pip install --no-binary manimpango "manimpango==0.7.0"
"$PY" -m pip install "manim==0.22.0"

# 3. speech models (Kokoro-82M, Whisper small.en) into .models/
if [[ "${1:-}" != "--no-models" ]]; then
  "$PY" common/fetch_models.py
fi

# 4. report
O5_PY=$PY "$PY" common/check_env.py || true
cat <<EOF

Next:  $TOOL activate $ENV_NAME      (or put  O5_PY=$PY  in config.local.sh)
       make check-env && make episode E=E01
EOF
