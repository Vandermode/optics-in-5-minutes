"""Fast end-to-end render of one Optics-in-5-Minutes episode.

    python fast_render.py <episode_dir> [workers]        (run with the Manim venv python)

<episode_dir>/episode.json: {"slug": "...", "chapters": ["Ch0Hook", ...]} (chapter order = chapter index).
1. split each chapter's frame range into pieces of roughly equal *cost* (profile: <episode_dir>/.cache/piece_speed.json)
2. render pieces on a pool of workers (fastrender.py: windowed Manim -> FFmpeg, clean + subtitled pieces)
3. meanwhile: loudness-normalise + AAC-encode the narration once
4. stitch pieces by stream copy (no re-encode) and mux audio (+ a soft subtitle track for the clean version)
Outputs: <episode_dir>/<slug>_1080p60.mp4 (burned-in subtitles) and <slug>_1080p60_softsubs.mp4.
Ported from angular_spectrum_method_5min/fast_render.py.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

COMMON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMMON)
EP = os.path.abspath(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
FPS = 60
import o5env  # noqa: E402

FF = o5env.FFMPEG
PY = sys.executable
PIECE_COST = 22.0                      # target seconds of rendering per piece
STARTUP = 2.6                          # measured per-worker start + fast-forward cost (s)
META = json.load(open(os.path.join(EP, "episode.json")))
SCENES = META["chapters"]
OUT_SUBS = os.path.join(EP, f"{META['slug']}_1080p60.mp4")
OUT_SOFT = os.path.join(EP, f"{META['slug']}_1080p60_softsubs.mp4")
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")   # 1 BLAS thread/worker


def windows():
    t = json.load(open(os.path.join(EP, "audio", "timing.json")))
    segs, total = t["segments"], t["total"]
    chs = sorted(set(s["ch"] for s in segs))
    b = [0.0]
    for c in chs[1:]:
        last = max(s["end"] for s in segs if s["ch"] == c - 1)
        first = min(s["start"] for s in segs if s["ch"] == c)
        b.append(0.5 * (last + first))
    b.append(total)
    return {c: (b[i], b[i + 1]) for i, c in enumerate(chs)}


def plan():
    """Equal-cost pieces from the measured per-frame cost profile (.cache/piece_speed.json)."""
    prof_path = os.path.join(EP, ".cache", "piece_speed.json")
    prof = json.load(open(prof_path)) if os.path.exists(prof_path) else []
    pieces = []
    for c, (a, b) in sorted(windows().items()):
        fa, fb = int(round(a * FPS)), int(round(b * FPS))
        cost = np.full(fb - fa, 1 / 25.0)                      # default: 25 frames/s per worker
        for r in prof:
            if r["scene"] == SCENES[c]:
                lo, hi = max(r["fa"], fa), min(r["fb"], fb)
                if hi > lo:
                    cost[lo - fa:hi - fa] = 1 / r["fps"]
        cum = np.concatenate([[0.0], np.cumsum(cost)])
        n = max(1, int(round(cum[-1] / PIECE_COST)))
        edges = [fa] + [fa + int(np.searchsorted(cum, cum[-1] * i / n)) for i in range(1, n)] + [fb]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi > lo:
                pieces.append(dict(scene=SCENES[c], fa=lo, fb=hi, cost=float(cum[hi - fa] - cum[lo - fa]) + STARTUP))
    return pieces


def run_piece(p, work):
    prefix = os.path.join(work, f"p{p['fa']:05d}")
    t = time.time()
    r = subprocess.run(["nice", "-n", "5", PY, os.path.join(COMMON, "fastrender.py"), EP, p["scene"], str(p["fa"]),
                        str(p["fb"]), prefix], capture_output=True, text=True, cwd=EP, env=ENV1)
    line = [l for l in r.stdout.splitlines() if l.startswith("PIECE")]
    if r.returncode != 0 or not line or "OK" not in line[0]:
        raise RuntimeError(f"piece {p} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return prefix, time.time() - t, line[0]


def main():
    t0 = time.time()
    os.makedirs(os.path.join(EP, ".cache"), exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"o5fast_{META['slug']}_")
    pieces = plan()
    print(f"{len(pieces)} pieces on {WORKERS} workers; work dir {work}", flush=True)
    audio = os.path.join(work, "audio.m4a")
    aproc = subprocess.Popen([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", os.path.join(EP, "audio", "narration.wav"),
                              "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "192k", audio])
    done, measured = {}, []
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(run_piece, p, work): p for p in sorted(pieces, key=lambda p: -p["cost"])}
        for f in cf.as_completed(futs):
            prefix, dt, line = f.result()
            p = futs[f]
            done[p["fa"]] = prefix
            render_s = max(dt - STARTUP, 0.5)                   # self-calibrate the cost profile for next time
            measured.append(dict(scene=p["scene"], fa=p["fa"], fb=p["fb"], fps=round((p["fb"] - p["fa"]) / render_s, 2)))
            print(f"  [{time.time() - t0:6.1f}s] {line}", flush=True)
    json.dump(measured, open(os.path.join(EP, ".cache", "piece_speed.json"), "w"), indent=0)
    t_frames = time.time() - t0
    aproc.wait()
    lists = {}
    for kind in ("clean", "subs"):
        lists[kind] = os.path.join(work, f"{kind}.txt")
        with open(lists[kind], "w") as fh:
            for fa in sorted(done):
                fh.write(f"file '{done[fa]}_{kind}.ts'\n")
    common = [FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0"]
    subprocess.run(common + ["-i", lists["subs"], "-i", audio, "-map", "0:v", "-map", "1:a", "-c", "copy",
                             "-movflags", "+faststart", OUT_SUBS], check=True)
    subprocess.run(common + ["-i", lists["clean"], "-i", audio, "-i", os.path.join(EP, "subtitles.srt"),
                             "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
                             "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", OUT_SOFT], check=True)
    if os.environ.get("KEEP_PIECES", "0") != "1":
        shutil.rmtree(work, ignore_errors=True)
    total = time.time() - t0
    print(f"frames rendered+encoded in {t_frames:.1f}s; total {total:.1f}s ({total / 60:.2f} min)", flush=True)
    print(f"wrote {OUT_SUBS}\n      {OUT_SOFT}", flush=True)


if __name__ == "__main__":
    main()
