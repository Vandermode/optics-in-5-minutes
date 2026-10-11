"""Machine-specific paths for the pipeline: environment variables with portable defaults.

Set any of these in the environment, or in <repo>/config.local.sh (gitignored; sourced by common/env.sh, which all
shell entry points and the Makefile go through):

    O5_PY          python of the optics5min environment            (default: `python` on PATH)
    O5_MODELS      model root                                      (default: <repo>/.models)
    O5_KOKORO_DIR  kokoro-v1.0.onnx + voices-v1.0.bin              (default: $O5_MODELS/kokoro)
    O5_HF_HOME     faster-whisper model cache                      (default: $O5_MODELS/hf)
    O5_FFMPEG      ffmpeg built with libass (the `ass` filter)     (default: `ffmpeg` on PATH)
    O5_FFPROBE     ffprobe                                         (default: next to O5_FFMPEG, else on PATH)
    O5_FONTS_DIR   fonts for Manim text and burned-in subtitles    (default: <repo>/common/fonts, vendored)
"""
import os
import shutil

COMMON = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(COMMON)


def _env(name, default):
    v = os.environ.get(name, "").strip()
    return os.path.expanduser(v) if v else default


MODELS = _env("O5_MODELS", os.path.join(ROOT, ".models"))
KOKORO_DIR = _env("O5_KOKORO_DIR", os.path.join(MODELS, "kokoro"))
HF_HOME = _env("O5_HF_HOME", os.path.join(MODELS, "hf"))
FFMPEG = _env("O5_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
_probe_sibling = os.path.join(os.path.dirname(FFMPEG), "ffprobe")
FFPROBE = _env("O5_FFPROBE", _probe_sibling if os.path.isfile(_probe_sibling) else (shutil.which("ffprobe") or "ffprobe"))
FONTS_DIR = _env("O5_FONTS_DIR", os.path.join(COMMON, "fonts"))
