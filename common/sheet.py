"""Contact sheet of preview frames at global times (seconds of film time).

    python sheet.py <episode_dir> <preview_out_dir> out.png t1 t2 ...    [env FPS=15 to match preview.sh]

Each tile is labelled with its time and chapter; the red line marks SUB_Y (subtitles live below it).
"""
import json
import os
import sys

from PIL import Image, ImageDraw

EP, d, out = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
times = [float(t) for t in sys.argv[4:]]
fps = int(os.environ.get("FPS", 15))
meta = json.load(open(os.path.join(EP, "episode.json")))
tim = json.load(open(os.path.join(EP, "audio", "timing.json")))
segs, total = tim["segments"], tim["total"]
chs = sorted(set(s["ch"] for s in segs))
b = [0.0]
for c in chs[1:]:
    b.append(0.5 * (max(s["end"] for s in segs if s["ch"] == c - 1) + min(s["start"] for s in segs if s["ch"] == c)))
b.append(total)
win = {c: (b[i], b[i + 1]) for i, c in enumerate(chs)}
ims = []
for t in times:
    ch = max(c for c, (a, _) in win.items() if a <= t)
    f0 = int(round(win[ch][0] * fps))
    i = int(round(t * fps)) - f0
    name = meta["chapters"][ch]
    p = f"{d}/{name}/{i:04d}.png"
    if not os.path.exists(p):
        files = sorted(os.listdir(f"{d}/{name}")) if os.path.isdir(f"{d}/{name}") else []
        p = f"{d}/{name}/{files[-1]}" if files else None
    im = Image.open(p).convert("RGB") if p else Image.new("RGB", (960, 540), (60, 0, 0))
    ims.append((t, name, im))
w, h = ims[0][2].size
cols = 2
rows = (len(ims) + cols - 1) // cols
sh = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (255, 0, 0))
ysub = int(h * (0.5 + 3.25 / 8.0))                       # SUB_Y = -3.25 of 8 scene units
for k, (t, name, im) in enumerate(ims):
    x, y = (k % cols) * (w + 4), (k // cols) * (h + 4)
    sh.paste(im, (x, y))
    dr = ImageDraw.Draw(sh)
    dr.text((x + 6, y + 4), f"{t:.1f}s {name}", fill=(255, 80, 80))
    dr.line([(x, y + ysub), (x + w, y + ysub)], fill=(120, 0, 0), width=1)
sh.save(out)
print(out)
