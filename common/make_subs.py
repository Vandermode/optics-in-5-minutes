"""Subtitles from the narration script, timed with the Whisper word timestamps.

    python make_subs.py <episode_dir>  ->  <episode_dir>/subtitles.srt, subtitles.ass

The subtitle text is narration.json's "text" (write it display-ready: plain Unicode, e.g. "kx", "λ", "e^(ikz)").
Ported from angular_spectrum_method_5min/make_subs.py.
"""
import json
import os
import re
import sys

EP = os.path.abspath(sys.argv[1])
d = os.path.join(EP, "audio")
timing = json.load(open(os.path.join(d, "timing.json")))
words = json.load(open(os.path.join(d, "words.json")))
MAXC = 72


def clean(t):
    t = t.replace("kₓ²", "kx²").replace("k_z²", "kz²").replace("kₓ", "kx").replace("k_x", "kx")
    return t.replace("k_z", "kz")


def split(t):
    if len(t) <= MAXC:
        return [t]
    best, bi = None, None
    for m in re.finditer(r"[.:;,?] ", t):
        i = m.end()
        if min(i, len(t) - i) < 0.3 * len(t):        # keep the two halves balanced
            continue
        score = abs(i - len(t) / 2)
        if best is None or score < best:
            best, bi = score, i
    if bi is None:
        sp = [m.start() for m in re.finditer(" ", t)]
        bi = min(sp, key=lambda i: abs(i - len(t) / 2)) + 1
    return split(t[:bi].strip()) + split(t[bi:].strip())


cues = []
for seg in timing["segments"]:
    ws = [w for w in words if seg["start"] - 0.3 <= w["s"] < seg["end"] + 0.1]
    t0, t1 = seg["start"], seg["end"]
    text = clean(seg["text"])
    parts = split(text)
    # map character fractions of the display text onto the spoken-word timeline
    cum, acc = [(0.0, t0)], 0
    tot = sum(len(w["w"]) for w in ws) or 1
    for w in ws:
        acc += len(w["w"])
        cum.append((acc / tot, w["e"]))
    cum[-1] = (1.0, t1)

    def at(f):
        for (fa, ta), (fb, tb) in zip(cum[:-1], cum[1:]):
            if f <= fb:
                return ta + (tb - ta) * (f - fa) / max(fb - fa, 1e-9)
        return t1
    n, c0 = len(text), 0
    for k, p in enumerate(parts):
        c1 = c0 + len(p) + (1 if k < len(parts) - 1 else 0)
        a = t0 if k == 0 else at(c0 / n)
        b = t1 if k == len(parts) - 1 else at(c1 / n)
        cues.append([a, b, p])
        c0 = c1
for i in range(len(cues)):                       # hold each cue until shortly before the next one
    nxt = cues[i + 1][0] if i + 1 < len(cues) else cues[i][1] + 1.0
    cues[i][1] = min(max(cues[i][1] + 0.25, cues[i][0] + 1.0), nxt - 0.04)


def ts(t, sep=","):
    h, r = divmod(t, 3600)
    m, s_ = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s_:06.3f}".replace(".", sep)


with open(os.path.join(EP, "subtitles.srt"), "w") as f:
    for i, (a, b, p) in enumerate(cues, 1):
        f.write(f"{i}\n{ts(a)} --> {ts(b)}\n{p}\n\n")
with open(os.path.join(EP, "subtitles.ass"), "w") as f:
    f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 2\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Lato,42,&H00F2F2F2,&H00F2F2F2,&H00000000,&H96000000,0,0,0,0,100,100,0.3,0,1,2.2,1.2,2,80,80,32,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    for a, b, p in cues:
        f.write(f"Dialogue: 0,{ts(a, '.')[1:-1]},{ts(b, '.')[1:-1]},Default,,0,0,0,,{{\\fad(120,120)}}{p}\n")
print(f"{len(cues)} cues -> subtitles.srt / subtitles.ass")
