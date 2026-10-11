"""Shared style, timing and drawing helpers for the "Optics in 5 Minutes" episodes.

Ported from the pilot (angular_spectrum_method_5min/asm5_common.py) with the episode folder as a parameter.
Usage at the top of an episode's scenes.py:

    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "common"))
    import o5common
    o5common.load_timing(HERE)
    from o5common import *

Scene rules (FAST_ANIMATION_PIPELINE.md, section 5) apply unchanged: state is a pure function of self.now and the
plays before it, every chapter syncs with self.at(...) and ends with self.finish(), per-frame redraws use
fast_arrow / fast_dashed and never build Tex/MathTex/Arrow/DashedLine.

Colour code (series-wide): field U BLUE, k_x YELLOW, k_z TEAL, transfer function / pupil GREEN,
evanescent / loss RED, imaginary part PURPLE, reference wave / source points GOLD, matter GREY_B,
grating vector ORANGE, adjoint field PINK.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from manim import *

import o5env  # noqa: E402

config.background_color = BLACK

# vendored fonts (Lato for cards/kickers, DejaVu Sans Mono for code cards): register them with Pango so every
# machine renders the same typography whether or not the fonts are installed system-wide
if os.path.isdir(o5env.FONTS_DIR):
    import manimpango
    for _f in sorted(os.listdir(o5env.FONTS_DIR)):
        if _f.lower().endswith((".ttf", ".otf")):
            manimpango.register_font(os.path.join(o5env.FONTS_DIR, _f))
TAU_ = 2 * np.pi

C_U, C_KX, C_KZ, C_H, C_EV, C_IM = BLUE, YELLOW, TEAL, GREEN, RED, PURPLE_B
C_REF, C_MAT, C_K, C_ADJ = GOLD, GREY_B, ORANGE, PINK
SUB_Y = -3.25                 # keep content above this line (subtitles below)
MONO = "DejaVu Sans Mono"

# ------------------------------------------------------------------ timing ------
TIM = None
SEGS = []
SEG = {}
TOTAL = 0.0
WIN = {}
EP_DIR = None


def chapter_windows(segs, total):
    chs = sorted(set(s["ch"] for s in segs))
    b = [0.0]
    for c in chs[1:]:
        last = max(s["end"] for s in segs if s["ch"] == c - 1)
        first = min(s["start"] for s in segs if s["ch"] == c)
        b.append(0.5 * (last + first))
    b.append(total)
    return {c: (b[i], b[i + 1]) for i, c in enumerate(chs)}


def load_timing(ep_dir):
    """Bind the episode's audio/timing.json (call before `from o5common import *`)."""
    global TIM, SEGS, SEG, TOTAL, WIN, EP_DIR
    EP_DIR = os.path.abspath(ep_dir)
    TIM = json.load(open(os.path.join(EP_DIR, "audio", "timing.json")))
    SEGS = TIM["segments"]
    SEG = {s["id"]: s for s in SEGS}
    TOTAL = TIM["total"]
    WIN = chapter_windows(SEGS, TOTAL)
    return WIN


def data_path(*parts):
    return os.path.join(EP_DIR, "data", *parts)


# ------------------------------------------------------------------ colour maps --
def smoothstep(e):
    e = np.clip(e, 0.0, 1.0)
    return e * e * (3 - 2 * e)


def lut(stops, n=1024):
    v = np.linspace(0, 1, n)
    return np.stack([np.interp(v, [s[0] for s in stops], [s[1][c] for s in stops]) for c in range(3)], -1).astype(np.uint8)


HEAT = lut([(0.0, (0, 0, 0)), (0.25, (12, 50, 70)), (0.5, (28, 117, 138)), (0.75, (88, 196, 221)),
            (0.92, (255, 255, 120)), (1.0, (255, 255, 255))])
DIVERGE = lut([(0.0, (88, 196, 221)), (0.35, (20, 70, 90)), (0.5, (0, 0, 0)), (0.65, (70, 60, 20)),
               (1.0, (255, 230, 120))])
PHASE_HUE = lut([(0.0, (255, 80, 80)), (0.17, (255, 200, 60)), (0.33, (120, 230, 90)), (0.5, (60, 210, 230)),
                 (0.67, (90, 120, 255)), (0.83, (220, 90, 230)), (1.0, (255, 80, 80))], 2048)
GREYS = lut([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])


def _rgba(rgb, alpha=255):
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out


def heat_rgba(A, vmax, gamma=0.8, flip=True):
    """Non-negative map (|U| or intensity) -> RGBA with the series' HEAT map; flip puts row 0 at the bottom."""
    v = np.clip(np.asarray(A, float) / vmax, 0, 1) ** gamma
    rgb = HEAT[(v * 1023).astype(int)]
    if flip:
        rgb = rgb[::-1]
    return _rgba(rgb)


def grey_rgba(A, vmax, gamma=1.0, flip=True):
    v = np.clip(np.asarray(A, float) / vmax, 0, 1) ** gamma
    rgb = GREYS[(v * 1023).astype(int)]
    if flip:
        rgb = rgb[::-1]
    return _rgba(rgb)


def diverge_rgba(F, scale, flip=True):
    """Signed real field -> RGBA (cyan negative, black zero, gold positive)."""
    v = np.clip(np.asarray(F, float) / scale * 0.5 + 0.5, 0, 1)
    rgb = DIVERGE[(v * 1023).astype(int)]
    if flip:
        rgb = rgb[::-1]
    return _rgba(rgb)


def phase_rgba(U, vmax=None, gamma=0.7, flip=True, floor=0.0):
    """Complex field -> RGBA: hue = phase (PHASE_HUE), brightness = |U|^gamma (floor keeps dark phase visible)."""
    U = np.asarray(U)
    a = np.abs(U)
    vmax = vmax if vmax is not None else max(a.max(), 1e-12)
    b = floor + (1 - floor) * np.clip(a / vmax, 0, 1) ** gamma
    hue = (np.angle(U) % TAU_) / TAU_
    rgb = (PHASE_HUE[(hue * 2047).astype(int)].astype(np.float32) * b[..., None]).astype(np.uint8)
    if flip:
        rgb = rgb[::-1]
    return _rgba(rgb)


def colour_wheel_rgba(n=200):
    """A small phase colour wheel (for legends)."""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)
    R = np.hypot(X, Y)
    hue = (np.arctan2(Y, X) % TAU_) / TAU_
    rgb = PHASE_HUE[(hue * 2047).astype(int)]
    out = _rgba(rgb)
    out[..., 3] = (255 * np.clip((1.0 - R) / 0.02, 0, 1) * np.clip((R - 0.45) / 0.02, 0, 1)).astype(np.uint8)
    return out


# ------------------------------------------------------------------ physics ------
def asm_rows(u0, dx, zs, lam=1.0):
    """Angular-spectrum propagation of a 1D field u0 (spacing dx) to every z in zs -> complex (nz, N)."""
    k = TAU_ / lam
    kx = TAU_ * np.fft.fftfreq(len(u0), d=dx)
    kz = np.sqrt((k ** 2 - kx ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    F0 = np.fft.fft(u0)
    return np.array([np.fft.ifft(F0 * np.exp(1j * kz * z)) for z in zs])


def asm2d(u0, dx, z, lam=1.0):
    """Exact scalar angular-spectrum propagation of a 2D field (square grid spacing dx) by z."""
    k = TAU_ / lam
    ny, nx = u0.shape
    kx = TAU_ * np.fft.fftfreq(nx, d=dx)
    ky = TAU_ * np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    kz = np.sqrt((k ** 2 - KX ** 2 - KY ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    return np.fft.ifft2(np.fft.fft2(u0) * np.exp(1j * kz * z))


# ============================================================ scene base ========
class Chapter(Scene):
    """One chapter = one Scene rendering exactly its window of the narration timeline."""
    CH = 0
    TAG = None

    def setup(self):
        self.always_update_mobjects = True
        fps = config.frame_rate
        a, b = WIN[self.CH]
        self.f0, self.f1 = int(round(a * fps)), int(round(b * fps))
        self.t0, self.t_end = self.f0 / fps, self.f1 / fps
        self.log = []

    @property
    def now(self):
        return self.t0 + self.renderer.time

    @staticmethod
    def S(k):
        return SEG[k]["start"]

    @staticmethod
    def E(k):
        return SEG[k]["end"]

    def at(self, t):
        """Wait until global time t, counted in whole frames: the clock's float noise differs between rendered
        and skipped plays, and must not change the wait's length when t lies on the frame grid
        (FAST_ANIMATION_PIPELINE.md section 10, "One-frame timing step at a seam")."""
        x = (t - self.now) * config.frame_rate                # frames still to go
        if x > 0.5 + 1e-6:
            k = int(np.ceil(x - 1e-6))
            self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
        elif x < -0.5 - 1e-6:
            self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")

    def show_tag(self, run_time=0.4):
        if self.TAG:
            self.tag = Tex(self.TAG, font_size=30, color=GREY_B).to_corner(UL, buff=0.32)
            self.play(FadeIn(self.tag, shift=0.1 * RIGHT), run_time=run_time)

    def finish(self, fade=0.45):
        """Fade everything out and end on exactly the chapter's last frame."""
        fps = config.frame_rate
        target = self.f1 - self.f0
        nf = max(1, int(round(fade * fps)))
        gap = target - nf - int(round(self.renderer.time * fps))
        if gap > 0:
            self.wait((gap - 0.5) / fps)
        for m in self.mobjects:
            for sm in m.get_family():
                sm.clear_updaters()
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=(nf - 0.5) / fps)
        rem = target - int(round(self.renderer.time * fps))
        if rem > 0:
            self.wait((rem - 0.5) / fps)
        n = int(round(self.renderer.time * fps))
        self.log.append(f"chapter {self.CH}: {n} frames (target {target}), ends at {self.now:.2f}s")
        print("--- timeline ---\n" + "\n".join(self.log), file=sys.stderr)

    # ---------------------------------------------------------------- helpers --
    def live_image(self, fn, width, center, height=None, nearest=False):
        """ImageMobject redrawn every frame by fn(t) (RGBA uint8); opacity via the returned tracker."""
        img0 = fn(self.now)
        m = ImageMobject(np.array(img0, copy=True))
        m.stretch_to_fit_width(width)
        m.stretch_to_fit_height(height if height else width * img0.shape[0] / img0.shape[1])
        m.move_to(center)
        native = abs(width * config.pixel_height / config.frame_height - img0.shape[1]) < 1.5
        if nearest or native:
            m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        fade = ValueTracker(0.0)

        def upd(mob):
            img = np.array(fn(self.now), copy=True)
            img[..., 3] = (img[..., 3].astype(np.float32) * fade.get_value()).astype(np.uint8)
            mob.pixel_array = img
        m.add_updater(upd)
        upd(m)
        return m, fade

    def static_image(self, rgba, width, center, height=None, nearest=False):
        """Static RGBA image, faded through the returned tracker (same contract as live_image)."""
        arr = np.ascontiguousarray(rgba)
        return self.live_image(lambda t: arr, width, center, height, nearest)

    def sweep_image(self, rgba, width, center, height=None, axis="up", soft=0.012):
        """Image revealed by a wipe: returns (mobject, fade tracker, sweep tracker 0..1).
        axis: 'up' (bottom to top), 'right' (left to right), 'down', 'left'."""
        base = np.ascontiguousarray(rgba)
        h, w = base.shape[:2]
        if axis in ("up", "down"):
            pos = np.arange(h)[::-1] / max(h - 1, 1) if axis == "up" else np.arange(h) / max(h - 1, 1)
        else:
            pos = np.arange(w) / max(w - 1, 1) if axis == "right" else np.arange(w)[::-1] / max(w - 1, 1)
        sw = ValueTracker(0.0)

        def fn(t):
            img = base.copy()
            m = np.clip((sw.get_value() - pos) / soft, 0, 1)
            if axis in ("up", "down"):
                img[..., 3] = (img[..., 3] * m[:, None]).astype(np.uint8)
            else:
                img[..., 3] = (img[..., 3] * m[None, :]).astype(np.uint8)
            return img
        m, fd = self.live_image(fn, width, center, height)
        return m, fd, sw


# ======================================================== series cards ==========
def series_title(ep, title, subtitle=None, y=0.9, size=64):
    """Episode title card: small kicker 'OPTICS IN 5 MINUTES · n' above the title (and an optional subtitle)."""
    kick = Text(f"OPTICS IN 5 MINUTES  ·  {ep}", font="Lato", font_size=26, color=GREY_B, weight=BOLD)
    ttl = Tex(title, font_size=size)
    parts = [kick, ttl]
    if subtitle:
        parts.append(Tex(subtitle, font_size=34, color=GREY_A))
    g = VGroup(*parts).arrange(DOWN, buff=0.28).move_to([0, y, 0])
    back = BackgroundRectangle(g, fill_opacity=0.65, buff=0.3).set_fill(BLACK)
    return VGroup(back, g)


def next_card(text, y=-1.9):
    """'Next time' teaser card."""
    a = Text("NEXT", font="Lato", font_size=24, color=YELLOW, weight=BOLD)
    b = Tex(text, font_size=36)
    g = VGroup(a, b).arrange(RIGHT, buff=0.35).move_to([0, y, 0])
    box = SurroundingRectangle(g, buff=0.25, corner_radius=0.12, color=GREY_B, stroke_width=1.5)
    return VGroup(box, g)


def code_line(l, font_size=24, color=GREY_A):
    """One monospace line that keeps its indentation: Text drops leading whitespace from its glyph box, so the
    indent is filled with invisible placeholder glyphs (left-aligned rows then line up like real code)."""
    n = len(l) - len(l.lstrip(" "))
    if not l.strip():
        return Text(" ", font=MONO, font_size=font_size, color=color)
    if n == 0:
        return Text(l, font=MONO, font_size=font_size, color=color)
    t = Text("x" * n + l[n:], font=MONO, font_size=font_size, color=color)
    for g in t[:n]:
        g.set_opacity(0)
    return t


def code_card(lines, font_size=24, color=GREY_A, title=None, width=None):
    """Monospace code card (build once; never per frame). Leading spaces are kept as indentation."""
    rows = VGroup(*[code_line(l, font_size, color) for l in lines])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    g = VGroup(rows)
    if title:
        t = Text(title, font="Lato", font_size=font_size - 2, color=YELLOW, weight=BOLD)
        t.next_to(rows, UP, aligned_edge=LEFT, buff=0.25)
        g.add(t)
    box = SurroundingRectangle(g, buff=0.3, corner_radius=0.12, color=GREY_B, stroke_width=1.5).set_fill(GREY_E, 0.6)
    if width and box.width < width:
        box.stretch_to_fit_width(width).align_to(g, LEFT).shift(0.3 * LEFT)
    return VGroup(box, g)


def caption(text, size=28, color=GREY_A, **kw):
    return Tex(text, font_size=size, color=color, **kw)


def boxed(mob, color=YELLOW, buff=0.15, width=2.5):
    return VGroup(SurroundingRectangle(mob, buff=buff, color=color, stroke_width=width, corner_radius=0.06), mob)


# ======================================================== diagram builders ======
def clip_to_box(p, q, cx, cy, hx, hy):
    t0, t1 = 0.0, 1.0
    d = q - p
    for i, (c, h) in enumerate(((cx, hx), (cy, hy))):
        for sgn, lim in ((-1, c - h), (1, c + h)):
            num, den = (lim - p[i]) * sgn, d[i] * sgn
            if abs(den) < 1e-12:
                if num < 0:
                    return None
                continue
            r = num / den
            if den > 0:
                t1 = min(t1, r)
            else:
                t0 = max(t0, r)
    return (p + t0 * d, p + t1 * d) if t0 < t1 else None


def fast_arrow(start, end, color=WHITE, width=4.0, tip=0.18, opacity=1.0):
    """Cheap arrow for per-frame redraws: a line plus a filled triangle (Manim's Arrow is ~10x costlier)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    v = end - start
    L = np.linalg.norm(v)
    if L < 1e-6:
        return VGroup()
    u = v / L
    tl = min(tip, 0.45 * L)
    nrm = np.array([-u[1], u[0], 0.0])
    base = end - tl * u
    shaft = Line(start, base + 0.02 * u).set_stroke(color, width, opacity=opacity)
    head = Polygon(end, base + 0.5 * tl * nrm, base - 0.5 * tl * nrm).set_stroke(width=0).set_fill(color, opacity)
    return VGroup(shaft, head)


def fast_dashed(p, q, dash=0.06, color=GREY_B, width=1.5, opacity=1.0):
    """Cheap dashed line: one VMobject with a sub-path per dash."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = np.linalg.norm(q - p)
    vm = VMobject().set_stroke(color, width, opacity=opacity)
    if L < 1e-6:
        return vm
    u = (q - p) / L
    n = max(1, int(L / (2 * dash)))
    step = L / n
    for i in range(n):
        a = p + u * (i * step)
        vm.start_new_path(a)
        vm.add_line_to(a + u * (0.5 * step))
    return vm


def fade_vgroup(g, o):
    """Multiply stroke and fill opacities of every sub-mobject by o (keeps unfilled paths unfilled)."""
    if o >= 0.999:
        return g
    for sm in g.family_members_with_points():
        sm.set_stroke(opacity=sm.get_stroke_opacity() * o)
        sm.set_fill(opacity=sm.get_fill_opacity() * o)
    return g


def graph_line(fn, x0, x1, y0, xscale, yscale, n=500, color=C_U, width=3):
    xx = np.linspace(x0, x1, n)
    return VMobject().set_points_as_corners([[x * xscale, y0 + yscale * v, 0] for x, v in zip(xx, fn(xx))]).set_stroke(color, width)


def poly_line(xs, ys, color=C_U, width=3, opacity=1.0):
    """Polyline through scene points (xs, ys) -- cheap enough for per-frame redraws with a few hundred points."""
    pts = np.zeros((len(xs), 3))
    pts[:, 0], pts[:, 1] = xs, ys
    return VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity=opacity)


def plot_xy(x, y, box_center, box_w, box_h, xlim, ylim, color=C_U, width=3, opacity=1.0):
    """Data curve mapped into a box (data limits xlim, ylim) -> scene polyline."""
    cx, cy = box_center[0], box_center[1]
    X = cx - box_w / 2 + (np.asarray(x, float) - xlim[0]) / (xlim[1] - xlim[0]) * box_w
    Y = cy - box_h / 2 + (np.asarray(y, float) - ylim[0]) / (ylim[1] - ylim[0]) * box_h
    return poly_line(X, Y, color, width, opacity)


def stems(xs, ys, y0, color=C_KX, width=3, dot=0.035):
    g = VGroup()
    for x, y in zip(xs, ys):
        g.add(Line([x, y0, 0], [x, y, 0]).set_stroke(color, width))
        if dot:
            g.add(Dot([x, y, 0], radius=dot, color=color))
    return g


def phasor(center, z, scale=1.0, color=WHITE, width=4, tip=0.15, opacity=1.0):
    """Arrow for the complex number z drawn from center (scene units per unit magnitude = scale)."""
    c = np.asarray(center, float)
    return fast_arrow(c, c + scale * np.array([np.real(z), np.imag(z), 0.0]), color, width, tip, opacity)


def panel_frame(P0, H, xl="x", zl="z"):
    sq = Square(side_length=2 * H).move_to(P0).set_stroke(GREY_B, 2)
    px = MathTex(xl, font_size=30, color=GREY_B).next_to(sq, DOWN, buff=0.1).shift((H - 0.2) * RIGHT)
    pz = MathTex(zl, font_size=30, color=GREY_B).next_to(sq, LEFT, buff=0.1).shift((H - 0.2) * UP)
    return VGroup(sq, px, pz)


def wave_panel(scene, P0, H, kx_t, lam_d=0.78, speed=0.75, reveal=None, op=None, color=C_U, width=3):
    """Crests of the plane wave with k_x/k = kx_t inside a 2H x 2H panel (x right, z up).
    For k_x/k > 1 the wave is evanescent: vertical crests whose opacity decays as e^{-kappa z}."""
    def waves():
        g = VGroup()
        kx = kx_t.get_value()
        o = op.get_value() if op is not None else 1.0
        if o <= 0.002:
            return g
        ph = (speed * scene.now / lam_d) % 1.0
        top = P0[1] + H if reveal is None else P0[1] - H + 2 * H * np.clip(reveal.get_value(), 0, 1)
        hy = 0.5 * (top - (P0[1] - H))
        cy = P0[1] - H + hy
        if hy <= 1e-3:
            return g
        if kx <= 1.0:
            th = np.arcsin(np.clip(kx, -1, 1))
            d = np.array([np.sin(th), np.cos(th), 0])
            p = np.array([np.cos(th), -np.sin(th), 0])
            for n in range(-9, 10):
                c = P0 + (n + ph) * lam_d * d
                seg = clip_to_box(c - 4 * H * p, c + 4 * H * p, P0[0], cy, H, hy)
                if seg:
                    g.add(Line(*seg).set_stroke(color, width, opacity=o))
        else:
            lam_x = lam_d / kx
            kap = TAU_ / lam_d * np.sqrt(kx ** 2 - 1)
            nseg = 36
            for n in range(-14, 15):
                x = P0[0] + (n + ph) * lam_x
                if abs(x - P0[0]) > H:
                    continue
                for j in range(nseg):
                    z0, z1 = 2 * H * j / nseg, 2 * H * (j + 1) / nseg
                    a = np.exp(-kap * 0.5 * (z0 + z1))
                    if a < 0.01:
                        break
                    g.add(Line([x, P0[1] - H + z0, 0], [x, P0[1] - H + z1, 0]).set_stroke(C_EV, 5, opacity=o * a ** 0.6))
        return g
    return always_redraw(waves)


def k_circle(kc, R, kx_t, labels=True):
    """Static circle + axes, and a live wave-vector (tip on the circle, or on the k_x axis if evanescent)."""
    circ = Circle(radius=R).move_to(kc).set_stroke(GREY_A, 2.5)
    kxax = Arrow(kc + (R + 0.45) * LEFT, kc + (R + 0.75) * RIGHT, buff=0, stroke_width=2, color=GREY_B,
                 max_tip_length_to_length_ratio=0.03)
    kzax = Arrow(kc + 0.3 * DOWN, kc + (R + 0.45) * UP, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04)
    static = VGroup(circ, kxax, kzax)
    if labels:
        static.add(MathTex("k_x", font_size=32, color=C_KX).next_to(kxax, RIGHT, buff=0.06),
                   MathTex("k_z", font_size=32, color=C_KZ).next_to(kzax, UP, buff=0.06),
                   MathTex("k", font_size=32).move_to(kc + (R + 0.25) * np.array([-0.72, 0.69, 0])))
    ev_lab = MathTex("k_z", "=", "i\\kappa", font_size=32).set_color(C_EV)

    def vec():
        kx = kx_t.get_value()
        if kx <= 1.0:
            tip = kc + R * np.array([kx, np.sqrt(max(0.0, 1 - kx ** 2)), 0])
            return VGroup(fast_arrow(kc, tip, WHITE, 5, 0.22),
                          Line(kc, kc + R * kx * RIGHT).set_stroke(C_KX, 6),
                          fast_dashed(kc + R * kx * RIGHT, tip, 0.06, C_KZ, 4),
                          Dot(tip, radius=0.05))
        tip = kc + R * kx * RIGHT
        return VGroup(Line(kc, tip).set_stroke(C_KX, 6), Dot(tip, radius=0.07, color=C_EV),
                      ev_lab.copy().move_to(tip + np.array([0.55, 0.62, 0])))
    return static, always_redraw(vec)


def axis_pair(center, w, h, xlab=None, ylab=None, color=GREY_B, size=28):
    """Simple L-shaped axes (origin at lower-left of a w x h box centred at center) with optional labels."""
    c = np.asarray(center, float)
    o = c + np.array([-w / 2, -h / 2, 0])
    ax = VGroup(Arrow(o, o + (w + 0.25) * RIGHT, buff=0, stroke_width=2, color=color, max_tip_length_to_length_ratio=0.04),
                Arrow(o, o + (h + 0.25) * UP, buff=0, stroke_width=2, color=color, max_tip_length_to_length_ratio=0.05))
    if xlab:
        ax.add(MathTex(xlab, font_size=size, color=color).next_to(ax[0], RIGHT, buff=0.08))
    if ylab:
        ax.add(MathTex(ylab, font_size=size, color=color).next_to(ax[1], UP, buff=0.08))
    return ax
