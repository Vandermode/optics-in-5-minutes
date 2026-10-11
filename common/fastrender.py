"""Windowed, PNG-free renderer for one Optics-in-5-Minutes episode (one worker).

    python fastrender.py <episode_dir> <SceneClass> <frame_a> <frame_b> <piece_prefix> [--null | --dump]

Renders the global frames [frame_a, frame_b) of one chapter scene (from <episode_dir>/scenes.py) with Manim CE
(Cairo) and streams the raw frames straight into FFmpeg, which writes two MPEG-TS pieces in one pass:
    <prefix>_clean.ts   picture only
    <prefix>_subs.ts    subtitles (<episode_dir>/subtitles.ass) burned in with libass, timed with the global clock
Animations that end before frame_a are skipped (Manim jumps them to their end state without drawing);
the clock advances by whole frames exactly as in a full render, so the pieces join seamlessly.
--null discards the frames (timing tests); --dump keeps raw frames -> <prefix>_raw.npy (exactness tests).
Ported unchanged (apart from paths) from angular_spectrum_method_5min/fastrender.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

EP = os.path.abspath(sys.argv[1])
SCENE, FA, FB, PREFIX = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
NULL = "--null" in sys.argv
DUMP = "--dump" in sys.argv
W, H, FPS = 1920, 1080, 60
sys.path.insert(0, EP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(EP)

from manim import config  # noqa: E402

config.pixel_width, config.pixel_height, config.frame_rate = W, H, FPS
config.format = "none"
config.disable_caching = True
config.progress_bar = "none"
config.verbosity = "ERROR"
# private copy of the pre-warmed LaTeX cache: Manim writes/deletes temporary SVGs next to cached ones,
# which races when several workers share one directory
import shutil  # noqa: E402
import tempfile  # noqa: E402

_media = tempfile.mkdtemp(prefix="o5w_")
_tex = os.path.join(EP, ".cache", "Tex")
if os.path.isdir(_tex):
    shutil.copytree(_tex, os.path.join(_media, "Tex"))
config.media_dir = _media

from PIL import Image  # noqa: E402

import o5env  # noqa: E402
from manim.camera.camera import Camera  # noqa: E402
from manim.renderer.cairo_renderer import CairoRenderer  # noqa: E402

import scenes  # noqa: E402


class EndWindow(Exception):
    pass


# --------------------------------------------------------------- frame sink ----
dumped = []
if NULL or DUMP:
    sink = None
else:
    ass = os.path.join(EP, "subtitles.ass")
    off = FA / FPS
    conv = "scale=out_color_matrix=bt709:out_range=tv:flags=fast_bilinear,format=yuv420p"
    x264 = ["-c:v", "libx264", "-preset", "superfast", "-crf", "17", "-tune", "animation", "-threads", "2",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    cmd = [o5env.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-filter_complex",
           f"[0:v]{conv},split=2[va][b];"
           f"[b]setpts=PTS+{off}/TB,ass={ass}:fontsdir={o5env.FONTS_DIR},setpts=PTS-{off}/TB[vb]",
           "-map", "[va]", *x264, "-f", "mpegts", PREFIX + "_clean.ts",
           "-map", "[vb]", *x264, "-f", "mpegts", PREFIX + "_subs.ts"]
    sink = subprocess.Popen(cmd, stdin=subprocess.PIPE)
written = 0


# --------------------------------------------------------- renderer patches ----
def play(self, scene, *args, **kwargs):
    self.skip_animations = self._original_skipping_status
    self.update_skipping_status()
    scene.compile_animation_data(*args, **kwargs)
    n = len(np.arange(0, scene.duration, 1 / FPS))           # frames this play produces in a full render
    g = scene.f0 + int(round(self.time * FPS))               # global index of its first frame
    if g >= FB:
        raise EndWindow
    if g + n <= FA:
        self.skip_animations = True                           # entirely before the window: jump to its end
    self.file_writer.add_partial_movie_file(None)
    self.animations_hashes.append(None)
    self.file_writer.begin_animation(not self.skip_animations, animation_index=self.num_plays)
    scene.begin_animations()
    if self.skip_animations:
        self.static_image = None
        scene.play_internal()
        self.time += n / FPS                                  # same whole-frame clock as a rendered play
    else:
        self.save_static_frame_data(scene, scene.static_mobjects)
        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene, mobjects=scene.moving_mobjects)
            self.freeze_current_frame(scene.duration)
        else:
            scene.play_internal()
    self.file_writer.end_animation(not self.skip_animations)
    self.num_plays += 1


def render(self, scene, t, moving_mobjects=None):
    if self.skip_animations:
        return
    g = scene.f0 + int(round(self.time * FPS))
    if g < FA:                                                # straddling play: advance without drawing
        self.time += 1 / FPS
        return
    self.update_frame(scene, moving_mobjects)
    self.add_frame(self.camera.pixel_array)                   # streamed immediately: no copy needed


def add_frame(self, frame, num_frames=1):
    global written
    if self.skip_animations:
        return
    for _ in range(num_frames):
        g = self._scene.f0 + int(round(self.time * FPS))
        if g >= FB:
            raise EndWindow
        if g >= FA:
            if sink is not None:
                sink.stdin.write(frame.tobytes())
            elif DUMP:
                dumped.append(frame.copy())
            written += 1
        self.time += 1 / FPS


def display_image_mobject(self, image_mobject, pixel_array):
    """Axis-aligned images only: resize to the target box and alpha-blend just that box."""
    corners = self.points_to_subpixel_coords(image_mobject, image_mobject.points)
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    src = image_mobject.get_pixel_array()
    if src.shape[1] != w or src.shape[0] != h:
        src = np.asarray(Image.fromarray(src, mode="RGBA").resize((w, h), resample=image_mobject.resampling_algorithm))
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x1, self.pixel_width), min(y1, self.pixel_height)
    if X1 <= X0 or Y1 <= Y0:
        return
    s = src[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
    d = pixel_array[Y0:Y1, X0:X1].astype(np.float32)
    a = s[..., 3:4] / 255.0
    out = np.empty_like(d)
    out[..., :3] = s[..., :3] * a + d[..., :3] * (1 - a)
    out[..., 3:4] = 255.0 * (a + d[..., 3:4] / 255.0 * (1 - a))
    pixel_array[Y0:Y1, X0:X1] = (out + 0.5).astype(np.uint8)


CairoRenderer.play = play
CairoRenderer.render = render
CairoRenderer.add_frame = add_frame
Camera.display_image_mobject = display_image_mobject

# ---------------------------------------------------------------------- run ----
cls = getattr(scenes, SCENE)
t_start = time.time()
scene = cls()
scene.renderer._scene = scene
try:
    scene.render()
except EndWindow:
    pass
if sink is not None:
    sink.stdin.close()
    sink.wait()
if DUMP:
    np.save(PREFIX + "_raw.npy", np.stack(dumped))
dt = time.time() - t_start
shutil.rmtree(_media, ignore_errors=True)
status = "OK" if written == FB - FA else f"MISMATCH (expected {FB - FA})"
print(f"PIECE {SCENE} [{FA},{FB}) frames={written} {status} time={dt:.1f}s fps={written / max(dt, 1e-9):.1f}", flush=True)
