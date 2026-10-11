"""Download the speech models into $O5_MODELS (default <repo>/.models), with checksums.

    python common/fetch_models.py                 # download Kokoro-82M (~340 MB) + Whisper small.en (~480 MB)
    python common/fetch_models.py --from DIR      # copy kokoro-v1.0.onnx / voices-v1.0.bin from DIR instead

Kokoro: kokoro-onnx GitHub release "model-files-v1.0" (Apache-2.0 weights). Whisper: faster-whisper "small.en",
fetched into the Hugging Face cache at $O5_HF_HOME on first use; this script just triggers that download.
"""
import hashlib
import os
import shutil
import sys
import urllib.request

import o5env

KOKORO = {
    "kokoro-v1.0.onnx": ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                         "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    "voices-v1.0.bin": ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    src = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    os.makedirs(o5env.KOKORO_DIR, exist_ok=True)
    for name, (url, digest) in KOKORO.items():
        dst = os.path.join(o5env.KOKORO_DIR, name)
        if os.path.exists(dst) and sha256(dst) == digest:
            print(f"ok       {dst}")
            continue
        tmp = dst + ".part"
        if src:
            shutil.copyfile(os.path.join(os.path.expanduser(src), name), tmp)
        else:
            print(f"download {url}")
            urllib.request.urlretrieve(url, tmp)
        got = sha256(tmp)
        if got != digest:
            os.remove(tmp)
            sys.exit(f"checksum mismatch for {name}: {got}")
        os.replace(tmp, dst)
        print(f"ok       {dst}")
    os.environ.setdefault("HF_HOME", o5env.HF_HOME)
    from faster_whisper import WhisperModel
    WhisperModel("small.en", device="cpu", compute_type="int8")
    print(f"ok       whisper small.en in {os.environ['HF_HOME']}")


if __name__ == "__main__":
    main()
