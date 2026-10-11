"""Narration audio with Kokoro-82M (open weights, Apache-2.0) via kokoro-onnx, 4 CPU threads.

    python tts.py <episode_dir> [model_dir]          (model_dir default: $O5_KOKORO_DIR, see o5env.py)

Reads <episode_dir>/narration.json; writes <episode_dir>/audio/<id>.wav (trimmed), audio/narration.wav
(24 kHz mono, peak -1 dBFS) and audio/timing.json with the start/end of every segment on the film timeline.
Segments whose text+say+voice settings are unchanged are reused from audio/cache.json (fast re-runs).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

import o5env

EP = os.path.abspath(sys.argv[1])
model_dir = sys.argv[2] if len(sys.argv) > 2 else o5env.KOKORO_DIR
out = os.path.join(EP, "audio")
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(EP, "narration.json")))
cache_path = os.path.join(out, "cache.json")
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
_tts = None


def engine():
    global _tts
    if _tts is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(os.path.join(model_dir, "kokoro-v1.0.onnx"), so, providers=["CPUExecutionProvider"])
        _tts = Kokoro.from_session(sess, os.path.join(model_dir, "voices-v1.0.bin"))
    return _tts


def trim(x, sr, thr=0.008, pad=0.04):
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * sr))
    b = min(len(x), idx[-1] + int(pad * sr))
    return x[a:b]


t = cfg["lead_in_s"]
pieces, timing = [], []
sr = 24000
segs = cfg["segments"]
ids = [s["id"] for s in segs]
assert len(ids) == len(set(ids)), "duplicate segment ids"
new_cache = {}
for si, seg in enumerate(segs):
    key = hashlib.sha1(json.dumps([seg["say"], cfg["voice"], cfg["speed"]]).encode()).hexdigest()
    wav = os.path.join(out, seg["id"] + ".wav")
    if cache.get(seg["id"]) == key and os.path.exists(wav):
        audio, sr = sf.read(wav, dtype="float32")
    else:
        audio, sr = engine().create(seg["say"], voice=cfg["voice"], speed=cfg["speed"], lang="en-us")
        audio = trim(np.asarray(audio, np.float32), sr)
        sf.write(wav, audio, sr)
    new_cache[seg["id"]] = key
    d = len(audio) / sr
    timing.append(dict(id=seg["id"], ch=seg.get("ch", 0), text=seg["text"], start=round(t, 3), end=round(t + d, 3)))
    pieces.append((t, audio))
    t += d + cfg["gap_s"] + seg.get("pause_after", 0.0)
    if si + 1 < len(segs) and segs[si + 1].get("ch", 0) != seg.get("ch", 0):
        t += cfg.get("chapter_pause_s", 0.0)              # breathing room between chapters
total = t - cfg["gap_s"] - segs[-1].get("pause_after", 0.0) + cfg["tail_s"]
track = np.zeros(int(np.ceil(total * sr)), np.float32)
for t0, a in pieces:
    i = int(round(t0 * sr))
    track[i:i + len(a)] += a
peak = np.abs(track).max()
track *= 0.89 / peak                                   # peak -1 dBFS
sf.write(os.path.join(out, "narration.wav"), track, sr)
json.dump(dict(total=round(total, 3), segments=timing), open(os.path.join(out, "timing.json"), "w"), indent=1)
json.dump(new_cache, open(cache_path, "w"), indent=0)
for f in os.listdir(out):                              # drop WAVs of deleted segments
    if f.endswith(".wav") and f != "narration.wav" and f[:-4] not in new_cache:
        os.remove(os.path.join(out, f))
words = sum(len(s["text"].split()) for s in segs)
for s in timing:
    print(f"{s['start']:7.2f}-{s['end']:7.2f}  ch{s['ch']} {s['id']}")
print(f"total {total:.2f} s ({total / 60:.2f} min), {len(segs)} segments, {words} words")
