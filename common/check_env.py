"""Check that this machine can build episodes:  python common/check_env.py   (or: make check-env)

Prints one line per requirement (OK / MISSING / WARN) with a hint, and exits non-zero if anything required is missing.
"""
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys

import o5env

PINS = {"manim": "0.22.0", "manimpango": None, "cairo": None, "numpy": None, "scipy": None, "PIL": None,
        "kokoro_onnx": "0.6.1", "onnxruntime": "1.31.0", "faster_whisper": "1.2.1", "soundfile": None, "jax": None}
bad = 0


def line(ok, what, hint="", warn=False):
    global bad
    tag = "OK     " if ok else ("WARN   " if warn else "MISSING")
    bad += (not ok and not warn)
    print(f"{tag} {what}" + (f"\n          -> {hint}" if hint and not ok else ""))


print(f"python {sys.version.split()[0]} at {sys.executable}")
DIST = {"cairo": "pycairo", "PIL": "pillow", "kokoro_onnx": "kokoro-onnx", "faster_whisper": "faster-whisper"}
for mod, pin in PINS.items():
    try:
        importlib.import_module(mod)
        try:
            v = importlib.metadata.version(DIST.get(mod, mod))
        except importlib.metadata.PackageNotFoundError:
            v = "?"
        line(pin is None or v == pin, f"{mod} {v}" + (f" (pinned {pin})" if pin else ""),
             "conda env create -f environment.yml  (or update it)", warn=pin is not None and v != pin)
    except Exception as e:  # noqa: BLE001
        line(mod == "jax", f"{mod}: {e.__class__.__name__}", "conda env create -f environment.yml", warn=mod == "jax")

for tool, hint in (("latex", "install a TeX distribution (Debian/Ubuntu: texlive texlive-latex-extra; macOS: MacTeX)"),
                   ("dvisvgm", "comes with TeX Live (Debian/Ubuntu: dvisvgm)")):
    p = shutil.which(tool)
    line(bool(p), f"{tool}: {p}", hint)

ff = o5env.FFMPEG
try:
    filt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    has_ass = any(l.split()[1:2] == ["ass"] for l in filt.splitlines() if len(l.split()) > 1)
    line(has_ass, f"ffmpeg with libass: {ff}", "set O5_FFMPEG in config.local.sh to an ffmpeg built with libass "
                                              "(e.g. /usr/bin/ffmpeg on Ubuntu)")
except OSError:
    line(False, f"ffmpeg: {ff} not runnable", "conda env create -f environment.yml, or set O5_FFMPEG")
line(bool(shutil.which(o5env.FFPROBE) or os.path.isfile(o5env.FFPROBE)), f"ffprobe: {o5env.FFPROBE}", "set O5_FFPROBE")

fonts = sorted(f for f in os.listdir(o5env.FONTS_DIR)) if os.path.isdir(o5env.FONTS_DIR) else []
line(all(f in fonts for f in ("Lato-Regular.ttf", "Lato-Bold.ttf", "DejaVuSansMono.ttf")),
     f"fonts in {o5env.FONTS_DIR}", "restore common/fonts from git")

for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
    p = os.path.join(o5env.KOKORO_DIR, name)
    line(os.path.isfile(p), f"Kokoro model {p}", "python common/fetch_models.py   (or: make models)")
hf = os.path.join(o5env.HF_HOME, "hub", "models--Systran--faster-whisper-small.en")
line(os.path.isdir(hf), f"Whisper small.en cache {hf}", "python common/fetch_models.py   (or: make models)")

try:
    import jax
    devs = [d.platform for d in jax.devices()]
    print(f"INFO    jax devices: {devs}" + ("" if "gpu" in devs else "  (CPU only: physics.py works, some are slower; "
                                                                      "GPU: pip install -U 'jax[cuda12]==0.8.0')"))
except Exception:  # noqa: BLE001
    pass
print("\nall required checks passed" if not bad else f"\n{bad} required item(s) missing")
sys.exit(1 if bad else 0)
