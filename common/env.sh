# Sourced by every shell entry point (render.sh, audio.sh, preview.sh, new_episode.sh, release_encode.sh) and the
# Makefile. Machine-specific settings go in <repo>/config.local.sh (gitignored); see config.local.sh.example and
# common/o5env.py for the variables.
O5_COMMON=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
O5_ROOT=$(dirname "$O5_COMMON")
if [ -f "$O5_ROOT/config.local.sh" ]; then
  # shellcheck disable=SC1091
  source "$O5_ROOT/config.local.sh"
fi
: "${O5_PY:=python}"                     # python of the optics5min environment (conda activate optics5min)
export O5_PY
for _v in O5_MODELS O5_KOKORO_DIR O5_HF_HOME O5_FFMPEG O5_FFPROBE O5_FONTS_DIR O5_DATA_URL; do
  [ -n "${!_v:-}" ] && export "${_v?}"
done
unset _v
