"""Verification checklist for a finished episode (FAST_ANIMATION_PIPELINE.md section 9, items 2, 3, 5, 6).

    python verify.py <episode_dir> [--no-scan]

1. frame count == round(total*60), duration, soft-subtitle packets == cues
2. seam/pop scan (frame-to-frame jumps > 3x local median + 1.5) -- hits should be intended transitions only
3. integrated loudness (target about -16 LUFS)
4. stills: 3 per chapter (30/60/90 % of its window) -> <episode_dir>/.verify/sheet_<n>.png (3 chapters per sheet)
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

EP = os.path.abspath(sys.argv[1])
SCAN = "--no-scan" not in sys.argv
import o5env  # noqa: E402

FF = o5env.FFMPEG
FFPROBE = o5env.FFPROBE
meta = json.load(open(os.path.join(EP, "episode.json")))
tim = json.load(open(os.path.join(EP, "audio", "timing.json")))
segs, total = tim["segments"], tim["total"]
subs = os.path.join(EP, f"{meta['slug']}_1080p60.mp4")
soft = os.path.join(EP, f"{meta['slug']}_1080p60_softsubs.mp4")
vdir = os.path.join(EP, ".verify")
os.makedirs(vdir, exist_ok=True)
ok = True

# 1. counts --------------------------------------------------------------------------
r = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,nb_read_frames",
                    "-show_entries", "format=duration", "-of", "json", subs], capture_output=True, text=True)
info = json.loads(r.stdout)
nv = [int(s["nb_read_frames"]) for s in info["streams"] if s["codec_type"] == "video"][0]
dur = float(info["format"]["duration"])
exp = int(round(total * 60))
print(f"[counts] video frames {nv} (expected {exp}) {'OK' if nv == exp else 'MISMATCH'}; duration {dur:.3f}s "
      f"(timing total {total:.3f}s)")
ok &= nv == exp
r = subprocess.run([FFPROBE, "-v", "error", "-count_packets", "-show_entries", "stream=codec_type,nb_read_packets",
                    "-of", "json", soft], capture_output=True, text=True)
sp = [int(s["nb_read_packets"]) for s in json.loads(r.stdout)["streams"] if s["codec_type"] == "subtitle"]
ncue = open(os.path.join(EP, "subtitles.srt")).read().count(" --> ")
print(f"[counts] soft-sub packets {sp} for {ncue} cues {'OK' if sp and ncue <= sp[0] <= 2 * ncue + 1 else 'CHECK'}"
      " (mov_text adds empty packets for gaps)")

# 2. seam / pop scan -------------------------------------------------------------------
chs = sorted(set(s["ch"] for s in segs))
b = [0.0]
for c in chs[1:]:
    b.append(0.5 * (max(s["end"] for s in segs if s["ch"] == c - 1) + min(s["start"] for s in segs if s["ch"] == c)))
b.append(total)
win = {c: (b[i], b[i + 1]) for i, c in enumerate(chs)}
if SCAN:
    p = subprocess.Popen([FF, "-v", "error", "-i", subs, "-vf", "scale=320:180", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE)
    fr = 320 * 180
    prev, d = None, []
    while True:
        buf = p.stdout.read(fr)
        if len(buf) < fr:
            break
        a = np.frombuffer(buf, np.uint8).astype(np.float32)
        if prev is not None:
            d.append(np.abs(a - prev).mean())
        prev = a
    p.wait()
    d = np.array(d)
    t = (np.arange(len(d)) + 1) / 60
    w = 30
    pad = np.pad(d, w, mode="edge")
    run = np.array([np.median(pad[i:i + 2 * w + 1]) for i in range(len(d))])
    hits = np.where(d > 3 * run + 1.5)[0]
    groups = []
    for i in hits:
        if groups and i - groups[-1][-1] <= 3:
            groups[-1].append(i)
        else:
            groups.append([i])
    print(f"[scan] {len(d) + 1} frames; {len(groups)} jump(s): " +
          ", ".join(f"{t[g[0]]:.2f}s" + (f"-{t[g[-1]]:.2f}s" if len(g) > 1 else "") for g in groups))
    prof = os.path.join(EP, ".cache", "piece_speed.json")
    if os.path.exists(prof):
        starts = sorted(r_["fa"] / 60 for r_ in json.load(open(prof)))
        near = [f"{t[g[0]]:.2f}" for g in groups if any(abs(t[g[0]] - s) < 0.05 for s in starts)]
        print(f"[scan] jumps within 50 ms of a piece start (inspect these!): {near if near else 'none'}")

# 3. loudness ----------------------------------------------------------------------------
r = subprocess.run([FF, "-hide_banner", "-nostats", "-i", subs, "-map", "0:a", "-af", "ebur128", "-f", "null", "-"],
                   capture_output=True, text=True)
m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
print(f"[loudness] integrated {m[-1] if m else '?'} LUFS (target about -16)")

# 4. stills ----------------------------------------------------------------------------
names = meta["chapters"]
tiles = []
for c in chs:
    a, bb = win[c]
    for f in (0.3, 0.6, 0.9):
        tt = a + f * (bb - a)
        png = os.path.join(vdir, f"t{tt:07.2f}.png")
        subprocess.run([FF, "-v", "error", "-y", "-ss", f"{tt:.3f}", "-i", subs, "-frames:v", "1", "-vf", "scale=640:360", png],
                       check=True)
        tiles.append((c, tt, png))
for k in range(0, len(chs), 3):
    sel = [x for x in tiles if x[0] in chs[k:k + 3]]
    rows = len(set(x[0] for x in sel))
    sh = Image.new("RGB", (3 * 640 + 8, rows * 360 + (rows - 1) * 4), (255, 0, 0))
    for j, (c, tt, png) in enumerate(sel):
        x, y = (j % 3) * 644, (j // 3) * 364
        sh.paste(Image.open(png).convert("RGB"), (x, y))
        ImageDraw.Draw(sh).text((x + 6, y + 4), f"{tt:.1f}s {names[c]}", fill=(255, 80, 80))
    out = os.path.join(vdir, f"sheet_{k // 3}.png")
    sh.save(out)
    print(f"[stills] {out}")
print("ALL COUNTS OK" if ok else "COUNT PROBLEM")
