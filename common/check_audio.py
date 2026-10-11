"""Round-trip check of the narration with faster-whisper (CPU, 4 threads) + word timestamps.

    python check_audio.py <episode_dir>       (Whisper cache: $O5_HF_HOME, see o5env.py)
        -> prints script vs heard per segment (flags mismatching words), writes <episode_dir>/audio/words.json
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

import o5env

os.environ.setdefault("HF_HOME", o5env.HF_HOME)
from faster_whisper import WhisperModel  # noqa: E402

EP = os.path.abspath(sys.argv[1])
d = os.path.join(EP, "audio")
timing = json.load(open(os.path.join(d, "timing.json")))
narr = {s["id"]: s for s in json.load(open(os.path.join(EP, "narration.json")))["segments"]}
model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
x, sr = sf.read(os.path.join(d, "narration.wav"), dtype="float32")
t16 = np.arange(int(len(x) * 16000 / sr)) / 16000          # resample to 16 kHz (speech band only)
x16 = np.interp(t16, np.arange(len(x)) / sr, x).astype(np.float32)
segs, _ = model.transcribe(x16, word_timestamps=True, beam_size=5, language="en")
words = [dict(w=w.word.strip(), s=round(w.start, 3), e=round(w.end, 3)) for s in segs for w in s.words]
json.dump(words, open(os.path.join(d, "words.json"), "w"), indent=0)


def norm(s):
    return re.findall(r"[a-z0-9]+", s.lower())


bad = 0
for seg in timing["segments"]:
    heard = " ".join(w["w"] for w in words if seg["start"] - 0.15 <= w["s"] < seg["end"] + 0.1)
    say = narr[seg["id"]]["say"]
    a, b = norm(say), norm(heard)
    missing = [w for w in a if w not in b]
    flag = "" if not missing else f"   <-- check: {' '.join(missing[:8])}"
    bad += bool(missing)
    print(f"[{seg['id']}]{flag}\n   say  : {say}\n   heard: {heard}")
print(f"\n{bad} of {len(timing['segments'])} segments have words not heard verbatim (often harmless: numbers, spelling)")
