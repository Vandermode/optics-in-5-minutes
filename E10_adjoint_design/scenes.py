"""Designing Light with Gradients: the adjoint method -- chapter scenes (Optics in 5 Minutes, E10, season finale).

Render: ./render.sh --skip-audio ; preview: ./preview.sh Ch3Backward
Every physical field is loaded from data/*.npz (written by physics.py); nothing heavy is computed here.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403


def _at_whole_frames(self, t):
    """Chapter.at with whole-frame waits (FAST_ANIMATION_PIPELINE.md section 10: no one-frame step at seams)."""
    x = (t - self.now) * config.frame_rate                # frames still to go
    if x > 0.5 + 1e-6:
        k = int(np.ceil(x - 1e-6))
        self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
    elif x < -0.5 - 1e-6:
        self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


Chapter.at = _at_whole_frames

# ------------------------------------------------------------------------------------------------ data
_CACHE = {}


def D(name):
    if name not in _CACHE:
        _CACHE[name] = dict(np.load(data_path(name + ".npz")))
    return _CACHE[name]


_WORDS = None


def word_time(word, t0, t1, default):
    """Start time of the first Whisper word (audio/words.json) starting with `word` (str or tuple) inside [t0, t1];
    `default` if the transcript does not contain it there."""
    global _WORDS
    if _WORDS is None:
        p = os.path.join(HERE, "audio", "words.json")
        _WORDS = json.load(open(p)) if os.path.exists(p) else []
    cands = (word,) if isinstance(word, str) else tuple(word)
    for w in _WORDS:
        txt = "".join(ch for ch in w["w"].lower() if ch.isalpha())
        if t0 - 0.2 <= w["s"] <= t1 and any(txt.startswith(c) for c in cands):
            return w["s"]
    return default


def dur(scene, t_end, minimum=0.2):
    """Run time that ends a play on the frame grid at t_end (whole frames, like Chapter.at)."""
    k = max(int(round(minimum * 60)), int(np.ceil((t_end - scene.now) * 60 - 1e-6)))
    return (k - 0.5) / 60


def axes_lr(center, w, h, xlab, ylab, size=26, color=GREY_B):
    """L-shaped axes (origin lower-left of a w x h box); x label centred below, y label above the y axis."""
    c = np.asarray(center, float)
    o = c + np.array([-w / 2, -h / 2, 0])
    ax = VGroup(Arrow(o, o + (w + 0.2) * RIGHT, buff=0, stroke_width=2, color=color, max_tip_length_to_length_ratio=0.04),
                Arrow(o, o + (h + 0.2) * UP, buff=0, stroke_width=2, color=color, max_tip_length_to_length_ratio=0.05))
    ax.add(Tex(xlab, font_size=size, color=color).next_to(o + w / 2 * RIGHT, DOWN, buff=0.12))
    ax.add(Tex(ylab, font_size=size, color=color).next_to(o + (h + 0.2) * UP, UP, buff=0.08))
    return ax


# ------------------------------------------------------------------------------------------------ colours
OMEGA = TAU_ / 1.25                       # one optical period per 1.25 s of film time (e^{-i omega t})
RGB_U = np.array([88, 196, 221], np.float32)      # forward field: BLUE
RGB_U2 = np.array([20, 52, 104], np.float32)
RGB_A = np.array([232, 88, 200], np.float32)      # adjoint field: PINK
RGB_A2 = np.array([92, 24, 84], np.float32)
RGB_S = np.array([240, 172, 95], np.float32)      # source / wavelet: GOLD
RGB_S2 = np.array([104, 64, 18], np.float32)
RGB_SI = np.array([66, 70, 84], np.float32)       # silicon (matter, grey)
HEAT_F = HEAT.astype(np.float32)


def to_img(a):
    """(x, y) array -> image orientation (rows = y from the top, columns = x)."""
    return np.ascontiguousarray(np.asarray(a).T[::-1])


def pool2(a):
    n, m = a.shape
    return a.reshape(n // 2, 2, m // 2, 2).mean((1, 3))


class Cplx:
    """Complex field (image orientation) with float32 real/imag parts for fast per-frame snapshots."""

    def __init__(self, c, vmax=None):
        c = to_img(c)
        self.re, self.im = c.real.astype(np.float32), c.imag.astype(np.float32)
        self.vmax = float(vmax if vmax is not None else np.abs(c).max())

    def snap(self, t):
        """Re{c e^{-i omega t}} / vmax."""
        ph = OMEGA * t
        return (self.re * np.float32(np.cos(ph)) + self.im * np.float32(np.sin(ph))) / np.float32(self.vmax)


def wave_rgb(v, cp, cn, gamma=0.6):
    pos = np.clip(v, 0, 1) ** gamma
    neg = np.clip(-v, 0, 1) ** gamma
    return pos[..., None] * cp + neg[..., None] * cn


def heat_rgb(a, gamma=0.8):
    v = np.clip(a, 0, 1) ** gamma
    return HEAT_F[(v * 1023).astype(np.int32)]


def rgba_of(rgb, alpha=255):
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = alpha
    return out


def nearest_up(a, f):
    return np.kron(a, np.ones((f, f), np.float32))


# ------------------------------------------------------------------------------------------------ helpers
class Panel:
    """Live image whose content is render(t) and whose position/size are driven by trackers (so it can move while
    its content keeps animating -- an .animate on the image itself would suspend its updater)."""

    def __init__(self, scene, render, cx, cy, w, aspect=0.75, nearest=False):
        self.scene, self.render, self.aspect = scene, render, aspect
        self.cx, self.cy, self.w = ValueTracker(cx), ValueTracker(cy), ValueTracker(w)
        self.fade = ValueTracker(0.0)
        self.m = ImageMobject(np.array(render(scene.now), copy=True))
        if nearest:
            self.m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        def upd(mob):
            im = np.array(self.render(self.scene.now), copy=True)
            f = self.fade.get_value()
            if f < 0.999:
                im[..., 3] = (im[..., 3].astype(np.float32) * f).astype(np.uint8)
            mob.pixel_array = im
            self.place()
        self.m.add_updater(upd)
        upd(self.m)

    def place(self):
        w = self.w.get_value()
        self.m.stretch_to_fit_width(w)
        self.m.stretch_to_fit_height(w * self.aspect)
        self.m.move_to([self.cx.get_value(), self.cy.get_value(), 0])

    def pt(self, u, v):
        """Panel coordinates (u right, v up, both 0..1) -> scene point (current geometry)."""
        w = self.w.get_value()
        h = w * self.aspect
        return np.array([self.cx.get_value() + (u - 0.5) * w, self.cy.get_value() + (v - 0.5) * h, 0.0])

    def geom(self, cx, cy, w):
        return [self.cx.animate.set_value(cx), self.cy.animate.set_value(cy), self.w.animate.set_value(w)]


class Counter:
    """Integer display built from pre-made digit glyphs (no Tex is built per frame)."""

    def __init__(self, size=34, color=WHITE):
        self.g = {c: MathTex(c, font_size=size, color=color) for c in "0123456789"}
        self.adv = max(m.width for m in self.g.values()) * 1.08
        self.gap = self.adv * 0.35

    def build(self, n, left):
        s = f"{int(n):d}"
        g = VGroup()
        x = left[0]
        for k, ch in enumerate(s):
            if k > 0 and (len(s) - k) % 3 == 0:
                x += self.gap
            m = self.g[ch].copy()
            m.move_to([x + self.adv / 2, left[1], 0])
            g.add(m)
            x += self.adv
        return g


def box_outline(p, ci0, ci1, cj0, cj1, color=GREY_B, width=1.5, dash=0.06, opacity=1.0):
    """Dashed outline of a cell rectangle (crop cells: 160 x 120) on a Panel -- cheap enough per frame."""
    a, b = p.pt(ci0 / 160, cj0 / 120), p.pt(ci1 / 160, cj1 / 120)
    c1, c2 = np.array([a[0], b[1], 0]), np.array([b[0], a[1], 0])
    return VGroup(*[fast_dashed(q, r, dash, color, width, opacity) for q, r in ((a, c2), (c2, b), (b, c1), (c1, a))])


def code_card_ind(lines, font_size=24, color=GREY_A, title=None):
    """Like o5common.code_card, but keeps leading indentation (Text drops leading spaces from its bounds)."""
    cw = Text("0" * 20, font=MONO, font_size=font_size).width / 20
    rows = VGroup(*[Text(l.lstrip(" \u00a0") or " ", font=MONO, font_size=font_size, color=color) for l in lines])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    for r, l in zip(rows, lines):
        r.shift((len(l) - len(l.lstrip(" \u00a0"))) * cw * RIGHT)
    g = VGroup(rows)
    if title:
        t = Text(title, font="Lato", font_size=font_size - 2, color=YELLOW, weight=BOLD)
        t.next_to(rows, UP, aligned_edge=LEFT, buff=0.25)
        g.add(t)
    box = SurroundingRectangle(g, buff=0.3, corner_radius=0.12, color=GREY_B, stroke_width=1.5).set_fill(GREY_E, 0.6)
    return VGroup(box, g)


def label_box(tex, size=28, color=WHITE, pad=0.12):
    t = Tex(tex, font_size=size, color=color)
    return VGroup(BackgroundRectangle(t, fill_opacity=0.75, buff=pad).set_fill(BLACK), t)


# geometry of the splitter crop (cells): design box x 36..86, y 20..100; outputs at y 30 and 90; input y 60
BOX = (36, 86, 20, 100)


def splitter_layers():
    """Device masks at 2x crop resolution (320 x 240, x/y arrays) -> image-orientation float maps."""
    d = D("splitter")
    full = pool2(d["ymask_hi"].astype(np.float32))                  # Y-branch fill fraction (anti-aliased)
    guides = full.copy()
    guides[72:172, :] = 0.0
    arms = full - guides
    binm = guides.copy()
    binm[72:172, 40:200] = nearest_up(d["bin_opt"].astype(np.float32), 2)
    return to_img(guides), to_img(arms), to_img(binm)


# ================================================================================ 0  hook ==========
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S, E = self.S, self.E
        d = D("splitter")
        G, ARMS, BIN = splitter_layers()
        XY = Cplx(d["x_y"], 0.9 * float(d["vref"]))
        XR = Cplx(d["x_ref"], 0.9 * float(d["vref"]))                    # incident mode alone (no device yet)
        XO = Cplx(d["x_opt"], 0.9 * float(d["vref"]))
        LO = Cplx(d["lam_opt"], 0.75 * float(np.abs(d["lam_opt"]).max()))
        ucol = (np.arange(320) + 0.5) / 320
        front, yarm = ValueTracker(0.0), ValueTracker(0.0)
        wf, wa = ValueTracker(1.0), ValueTracker(0.0)

        def render_y(t):
            m = np.clip((front.get_value() - ucol) / 0.025, 0, 1).astype(np.float32)[None, :, None]
            y_ = yarm.get_value()
            fld = XY.snap(t) if y_ > 0.997 else (1 - y_) * XR.snap(t) + y_ * XY.snap(t)
            rgb = (G + y_ * ARMS)[..., None] * RGB_SI + m * wave_rgb(fld, RGB_U, RGB_U2)
            return rgba_of(rgb)

        def render_o(t):
            rgb = BIN[..., None] * RGB_SI
            if wf.get_value() > 0.003:
                rgb = rgb + wf.get_value() * wave_rgb(XO.snap(t), RGB_U, RGB_U2)
            if wa.get_value() > 0.003:
                rgb = rgb + wa.get_value() * wave_rgb(LO.snap(t), RGB_A, RGB_A2)
            return rgba_of(rgb)

        pY = Panel(self, render_y, -0.6, -0.1, 7.6)
        box_op = ValueTracker(0.0)
        boxY = always_redraw(lambda: box_outline(pY, *BOX, opacity=box_op.get_value()))
        cap = Tex("computed: 2D wave simulation, silicon in glass, $\\lambda = 1.55\\,\\mu$m", font_size=24,
                  color=GREY_B).move_to([0, 3.42, 0])
        qmark = MathTex("?", font_size=64, color=GREY_B).move_to(pY.pt(61 / 160, 0.5))
        wg_lab = Tex("silicon waveguide", font_size=26, color=GREY_A).move_to(pY.pt(17 / 160, 0.5) + 0.5 * DOWN)
        self.add(pY.m, boxY)
        self.play(pY.fade.animate.set_value(1.0), box_op.animate.set_value(1.0), FadeIn(cap), run_time=0.8)
        self.at(S("h1") + 0.1)
        self.play(front.animate.set_value(36 / 160), FadeIn(wg_lab), FadeIn(qmark), run_time=2.4, rate_func=linear)
        goal = VGroup(*[Tex("50\\%", font_size=34, color=GREY_A).move_to(pY.pt(1.0, v) + 0.62 * RIGHT) for v in (0.75, 0.25)])
        goal_t = Tex("goal", font_size=28, color=GREY_B).next_to(goal[0], UP, buff=0.2)
        self.at(S("h1") + 3.3)
        self.play(FadeIn(goal), FadeIn(goal_t), run_time=0.6)

        # ---- h2: the human design
        human = label_box("a human design", 30, GREY_A).move_to(pY.pt(0.2, 0.9))
        self.at(S("h2"))
        self.play(yarm.animate.set_value(1.0), FadeOut(qmark), FadeIn(human), run_time=0.8)
        self.at(S("h2") + 1.4)
        self.play(front.animate.set_value(1.0), run_time=2.6, rate_func=linear)
        T_y = float(d["T_y"])
        got = VGroup(*[Tex(f"{50 * T_y:.0f}\\%", font_size=34).move_to(g_) for g_ in goal])
        tot = VGroup(MathTex(f"{100 * T_y:.0f}\\%", font_size=56), Tex("delivered", font_size=28, color=GREY_A)).arrange(DOWN, buff=0.12)
        tot.move_to([5.35, 0.0, 0])
        self.at(S("h2") + 4.3)
        self.play(ReplacementTransform(goal, got), FadeOut(goal_t), FadeIn(tot, shift=0.1 * LEFT), run_time=0.7)

        # ---- h3: the computer's design, side by side
        pO = Panel(self, render_o, 3.45, -0.15, 6.3)
        self.at(S("h3"))
        self.play(FadeOut(VGroup(got, tot, wg_lab, human)), run_time=0.4)
        self.play(*pY.geom(-3.45, -0.15, 6.3), run_time=1.0)
        boxO = always_redraw(lambda: box_outline(pO, *BOX, opacity=box_op.get_value()))
        self.add(pO.m, boxO)
        t_l = Tex("human design", font_size=30, color=GREY_A).move_to([-3.45, 2.55, 0])
        t_r = Tex("computer design", font_size=30, color=GREY_A).move_to([3.45, 2.55, 0])
        self.play(pO.fade.animate.set_value(1.0), FadeIn(t_l), FadeIn(t_r), run_time=0.9)
        n_l = Tex(f"{100 * T_y:.0f}\\% delivered", font_size=34).move_to([-3.45, -2.82, 0])
        n_r = Tex(f"{100 * float(d['T_opt']):.1f}\\% delivered", font_size=34, color=GREEN_B).move_to([3.45, -2.82, 0])
        self.play(FadeIn(n_l), run_time=0.5)
        self.at(S("h3") + 4.9)
        self.play(FadeIn(n_r, scale=1.1), run_time=0.6)
        cap2 = Tex("both are all silicon or glass, in the same $1.5\\times2.4\\,\\mu$m box", font_size=24,
                   color=GREY_B).move_to([0, 3.42, 0])
        self.play(FadeOut(cap), FadeIn(cap2), run_time=0.5)

        # ---- h4: the question
        hl = SurroundingRectangle(VGroup(Dot(pO.pt(BOX[0] / 160, BOX[2] / 120)), Dot(pO.pt(BOX[1] / 160, BOX[3] / 120))),
                                  buff=0.0, color=YELLOW, stroke_width=3)
        self.at(S("h4") + 0.2)
        self.play(Create(hl), run_time=0.7)

        # ---- h5: run light backward -> the adjoint field; title card
        self.at(S("h5"))
        self.play(FadeOut(hl), wf.animate.set_value(0.3), wa.animate.set_value(1.0), run_time=1.0)
        back = Tex("light sent backward", font_size=28, color=C_ADJ).move_to([3.45, 2.55, 0])
        self.play(FadeOut(t_r), FadeIn(back), run_time=0.4)
        title = series_title(10, "Designing Light with Gradients", "the adjoint method", y=0.25)
        self.at(word_time("adjoint", S("h5"), E("h5"), S("h5") + 2.2) - 0.15)
        self.play(FadeIn(title), pY.fade.animate.set_value(0.35), pO.fade.animate.set_value(0.35),
                  box_op.animate.set_value(0.3), run_time=0.8)
        self.finish()


# ================================================================================ 1  landscape =====
class Ch1Landscape(Chapter):
    CH = 1
    TAG = "1\\quad Climbing the landscape"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = D("splitter")
        K_IT = 6
        G, _, _ = splitter_layers()
        rp = d["rp_snaps"][K_IT].astype(np.float32)                    # (50, 80) design values p in [0, 1]
        dev = np.array(G)
        devxy = np.zeros((320, 240), np.float32)
        devxy[72:172, 40:200] = nearest_up(rp, 2)
        DEV = np.maximum(dev, to_img(devxy))
        ABSX = to_img(np.repeat(np.repeat(d["absx_snaps"][K_IT].astype(np.float32), 2, 0), 2, 1))
        vref = float(d["vref"])
        heat_op = ValueTracker(0.0)
        dim_t = ValueTracker(0.0)               # device dimming under the field (separate, so re-simulating never pops)
        sweep = ValueTracker(1.0)
        ucol = (np.arange(320) + 0.5) / 320
        visit = ValueTracker(0.0)
        vis_idx = np.zeros((320, 240), np.float32) - 1
        order = np.arange(50 * 80).reshape(80, 50).T                     # raster order: rows of the box, top first
        vis_idx[72:172, 40:200] = nearest_up(order[:, ::-1].astype(np.float32), 2)
        VIS = to_img(vis_idx)
        grey_rgb = DEV[..., None] * np.array([235, 235, 240], np.float32)
        grey_rgb = np.where((G > 0)[..., None], G[..., None] * np.array([110, 112, 124], np.float32), grey_rgb)

        def render(t):
            rgb = grey_rgb.copy()
            h = heat_op.get_value()
            rgb = rgb * (1 - 0.55 * dim_t.get_value())
            if h > 0.003:
                m = np.clip((sweep.get_value() - ucol) / 0.03, 0, 1).astype(np.float32)[None, :, None]
                rgb = rgb + h * m * heat_rgb((ABSX / vref) ** 2 / 0.9, 0.5)
            n = visit.get_value()
            if n > 0.5:
                sel = (VIS >= 0) & (VIS < n)
                rgb[sel] = 0.5 * rgb[sel] + 0.5 * np.array([255, 220, 60], np.float32)
            return rgba_of(rgb)
        P = Panel(self, render, -3.35, -0.25, 6.2)
        boxl = always_redraw(lambda: box_outline(P, *BOX, color=GREY_B))
        self.at(S("l1"))
        self.add(P.m, boxl)
        self.play(P.fade.animate.set_value(1.0), run_time=0.8)
        # zoom inset: 5 x 5 pixels with their values
        i0, j0 = 22, 44
        cells = VGroup()
        side = 0.66
        inset_c = np.array([3.3, 0.55, 0])
        for a in range(5):
            for b in range(5):
                v = float(rp[i0 + a, j0 + 4 - b])
                sq = Square(side).set_stroke(GREY_C, 1.5).set_fill(rgb_to_color(np.array([v, v, v]) * 0.92), 1.0)
                sq.move_to(inset_c + np.array([(a - 2) * side, (2 - b) * side, 0]))
                num = MathTex(f"{v:.2f}", font_size=22, color=BLACK if v > 0.55 else WHITE).move_to(sq)
                cells.add(VGroup(sq, num))
        src_a = P.pt((BOX[0] + i0 / 1) / 160, (BOX[2] + j0) / 120)
        src_b = P.pt((BOX[0] + i0 + 5) / 160, (BOX[2] + j0 + 5) / 120)
        src = Rectangle(width=src_b[0] - src_a[0], height=src_b[1] - src_a[1]).move_to((src_a + src_b) / 2).set_stroke(YELLOW, 2.5)
        links = VGroup(Line(src.get_corner(UR), cells.get_corner(UL)), Line(src.get_corner(DR), cells.get_corner(DL))).set_stroke(YELLOW, 1.5, 0.7)
        bar = ImageMobject(rgba_of(np.repeat(np.linspace(0, 235, 256, dtype=np.float32)[None, :, None], 12, 0) * np.ones(3, np.float32)))
        bar.stretch_to_fit_width(2.4).stretch_to_fit_height(0.24).move_to([3.3, -1.75, 0])
        bar_f = SurroundingRectangle(bar, buff=0.02, color=GREY_B, stroke_width=1)
        b0 = VGroup(MathTex("p=0", font_size=26, color=GREY_A).next_to(bar, DOWN, buff=0.14).align_to(bar, LEFT),
                    Tex("glass", font_size=26, color=GREY_A).next_to(bar, LEFT, buff=0.2))
        b1 = VGroup(MathTex("p=1", font_size=26, color=GREY_A).next_to(bar, DOWN, buff=0.14).align_to(bar, RIGHT),
                    Tex("silicon", font_size=26, color=GREY_A).next_to(bar, RIGHT, buff=0.2))
        self.at(S("l1") + 2.4)
        self.play(Create(src), Create(links), LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.02), run_time=1.2)
        self.play(FadeIn(bar), FadeIn(bar_f), FadeIn(b0), FadeIn(b1), run_time=0.6)

        # ---- l2: the merit
        Fv = float(d["F_hist"][K_IT])
        fq = MathTex("F(p)", font_size=60).move_to([3.3, 1.6, 0])
        fdesc = Tex("fraction of the light\\\\that reaches the outputs", font_size=30, color=GREY_A).next_to(fq, DOWN, buff=0.3)
        fval = MathTex(f"F = {Fv:.2f}", font_size=40, color=YELLOW).next_to(fdesc, DOWN, buff=0.45)
        fval_l = Tex("for this design (computed)", font_size=24, color=GREY_B).next_to(fval, DOWN, buff=0.15)
        outs = VGroup(*[fast_arrow(P.pt(1.0, v) + 0.12 * RIGHT, P.pt(1.0, v) + 0.62 * RIGHT, YELLOW, 4, 0.16) for v in (0.75, 0.25)])
        self.at(S("l2"))
        self.play(FadeOut(VGroup(cells, src, links, bar_f, b0, b1)), FadeOut(bar), run_time=0.5)
        self.play(heat_op.animate.set_value(1.0), dim_t.animate.set_value(1.0), FadeIn(fq), FadeIn(outs), run_time=1.0)
        self.play(FadeIn(fdesc), run_time=0.6)
        self.at(S("l2") + 3.6)
        self.play(FadeIn(fval), FadeIn(fval_l), run_time=0.6)

        # ---- l3: the landscape (schematic)
        L = D("landscape")
        Fl = L["F"].astype(np.float64)
        lo, hi = Fl.min(), Fl.max()
        land_c, land_s = np.array([3.4, -0.08, 0]), 4.25
        land = ImageMobject(heat_rgba((Fl - lo) / (hi - lo), 1.0, gamma=1.1))
        land.stretch_to_fit_width(land_s).stretch_to_fit_height(land_s).move_to(land_c)
        import contourpy
        gx = np.linspace(-1, 1, Fl.shape[1])
        cg = contourpy.contour_generator(gx, gx, Fl)
        conts = VGroup()
        for lev in np.linspace(lo + 0.08 * (hi - lo), hi - 0.04 * (hi - lo), 9):
            for seg in cg.lines(lev):
                if len(seg) < 4:
                    continue
                pts = np.zeros((len(seg), 3))
                pts[:, 0] = land_c[0] + seg[:, 0] * land_s / 2
                pts[:, 1] = land_c[1] + seg[:, 1] * land_s / 2
                conts.add(VMobject().set_points_as_corners(pts).set_stroke(WHITE, 1.0, 0.35))
        land_f = SurroundingRectangle(land, buff=0.0, color=GREY_B, stroke_width=1.5)
        ax_l = VGroup(MathTex("p_1", font_size=30, color=GREY_A).next_to(land_f, DOWN, buff=0.12),
                      MathTex("p_2", font_size=30, color=GREY_A).next_to(land_f, LEFT, buff=0.12))
        land_t = Tex("$F$ over all designs", font_size=30).next_to(land_f, UP, buff=0.15)
        land_cap = Tex("(schematic: 2 of 4000 directions)", font_size=22, color=GREY_B).move_to([3.4, -3.0, 0])
        path = L["path"]
        pts = np.zeros((len(path), 3))
        pts[:, 0] = land_c[0] + path[:, 0] * land_s / 2
        pts[:, 1] = land_c[1] + path[:, 1] * land_s / 2
        prog = ValueTracker(0.0)

        def walker():
            k = int(np.clip(prog.get_value(), 0, 1) * (len(pts) - 1))
            g = VGroup()
            if k > 0:
                g.add(VMobject().set_points_as_corners(pts[:k + 1]).set_stroke(WHITE, 3))
            p0 = pts[k]
            dvec = pts[min(k + 6, len(pts) - 1)] - pts[max(k - 6, 0)]
            if np.linalg.norm(dvec) > 1e-6 and k < len(pts) - 3:
                g.add(fast_arrow(p0, p0 + 0.75 * dvec / np.linalg.norm(dvec), YELLOW, 5, 0.2))
            g.add(Dot(p0, radius=0.08, color=WHITE))
            return g
        walk = always_redraw(walker)
        grad_l = MathTex("\\nabla F", font_size=34, color=YELLOW)
        grad_l.add_updater(lambda m: m.move_to(pts[int(np.clip(prog.get_value(), 0, 1) * (len(pts) - 1))] + np.array([0.55, -0.3, 0])))
        self.at(S("l3"))
        self.play(FadeOut(VGroup(fq, fdesc, fval, fval_l)), run_time=0.4)
        self.play(FadeIn(land), FadeIn(land_f), FadeIn(conts), FadeIn(ax_l), FadeIn(land_t), FadeIn(land_cap), run_time=1.0)
        self.add(walk)
        self.play(FadeIn(grad_l), run_time=0.4)
        self.at(S("l3") + 4.2)
        self.play(prog.animate.set_value(1.0), run_time=2.6, rate_func=rate_functions.ease_in_out_sine)
        grad_l.clear_updaters()

        # ---- l4: finite differences, one pixel at a time
        self.at(S("l4"))
        self.play(FadeOut(VGroup(land_f, conts, ax_l, land_t, land_cap, grad_l, walk)), FadeOut(land), run_time=0.5)
        self.remove(walk)
        cnt = Counter(40, WHITE)
        nsim = ValueTracker(0)
        sim_t = Tex("simulations:", font_size=34, color=GREY_A).move_to([2.0, 1.4, 0])
        cnt_m = always_redraw(lambda: cnt.build(nsim.get_value(), sim_t.get_right() + np.array([0.25, 0, 0])))
        fd = MathTex("\\frac{\\Delta F}{\\Delta p_1}", font_size=44).move_to([3.4, -0.3, 0])
        fd_t = Tex("nudge pixel 1, re-simulate", font_size=28, color=GREY_A).next_to(fd, DOWN, buff=0.25)
        pix = [(10, 58), (11, 58), (12, 58)]

        def pix_sq(i, j):
            a = P.pt((BOX[0] + i) / 160, (BOX[2] + j) / 120)
            b = P.pt((BOX[0] + i + 1) / 160, (BOX[2] + j + 1) / 120)
            return Square(side_length=max(b[0] - a[0], 0.07) * 1.6).move_to((a + b) / 2).set_stroke(YELLOW, 3)
        self.add(cnt_m)
        self.play(FadeIn(sim_t), run_time=0.4)
        sq = pix_sq(*pix[0])
        self.play(Create(sq), FadeIn(fd), FadeIn(fd_t), run_time=0.6)
        for k in range(3):
            if k > 0:
                nsq = pix_sq(*pix[k])
                fdk = MathTex(f"\\frac{{\\Delta F}}{{\\Delta p_{k + 1}}}", font_size=44).move_to(fd)
                ftk = Tex(f"nudge pixel {k + 1}, re-simulate", font_size=28, color=GREY_A).move_to(fd_t)
                self.play(Transform(sq, nsq), Transform(fd, fdk), Transform(fd_t, ftk), run_time=0.3)
            self.play(heat_op.animate.set_value(0.0), run_time=0.25)
            sweep.set_value(0.0)
            heat_op.set_value(1.0)
            self.play(sweep.animate.set_value(1.0), nsim.animate.set_value(k + 1), run_time=0.9 if k == 0 else 0.55,
                      rate_func=linear)

        # ---- l5: one simulation per pixel: 4000 here, millions for a metalens
        self.at(S("l5"))
        self.play(FadeOut(VGroup(sq, fd, fd_t)), run_time=0.3)
        self.play(visit.animate.set_value(4000), nsim.animate.set_value(4000), run_time=2.6, rate_func=rate_functions.ease_in_quad)
        meta = Tex("a metalens: millions of pixels", font_size=32, color=GREY_A).move_to([3.0, 0.2, 0])
        self.at(word_time("millions", S("l5"), E("l5"), S("l5") + 3.6) - 0.1)
        self.play(FadeIn(meta, shift=0.1 * UP), run_time=0.5)

        # ---- l6: a million pixels x a minute = two years per step
        cost = MathTex("10^6", "\\times", "1\\,\\text{min}", "\\approx", "2\\ \\text{years}", font_size=50).move_to([3.0, -1.0, 0])
        cost[4].set_color(C_EV)
        per = Tex("for one step uphill", font_size=30, color=C_EV).next_to(cost, DOWN, buff=0.25)
        self.at(S("l6") + 0.3)
        self.play(Write(cost), run_time=1.4)
        self.play(FadeIn(per), nsim.animate.set_value(1000000), run_time=1.2, rate_func=rate_functions.ease_in_quad)
        self.finish()


# ================================================================================ 2  adjoint trick ==
class Ch2Adjoint(Chapter):
    CH = 2
    TAG = "2\\quad The adjoint trick"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = D("splitter")
        # ---- a1: A x = b as blocks (a 6x6 grid -> 36 unknowns; computed operator pattern)
        A6 = d["A6"]
        n = A6.shape[0]
        img = np.zeros((n, n, 3), np.float32) + 14
        off = (np.abs(A6) > 1e-9) & ~np.eye(n, dtype=bool)
        img[off] = [120, 122, 132]
        dg = np.diag(A6)
        dv = (dg - dg.min()) / (dg.max() - dg.min())
        for k in range(n):
            img[k, k] = np.array([200, 200, 210]) * (0.55 + 0.45 * dv[k])
        Aimg = ImageMobject(rgba_of(np.kron(img, np.ones((8, 8, 1), np.float32))))
        Aimg.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        Aimg.stretch_to_fit_width(3.0).stretch_to_fit_height(3.0).move_to([-2.2, -0.45, 0])
        Af = SurroundingRectangle(Aimg, buff=0.03, color=GREY_B, stroke_width=1.5)
        xv = np.abs(np.asarray(d["x0"])[::9, 120][:n])
        xcol = VGroup(*[Rectangle(width=0.42, height=3.0 / n).set_stroke(width=0).set_fill(
            interpolate_color(BLACK, C_U, float(0.25 + 0.75 * v / xv.max())), 1) for v in xv]).arrange(DOWN, buff=0)
        xcol.move_to([0.4, -0.45, 0])
        bvals = np.zeros(n)
        bvals[14] = 1.0
        bcol = VGroup(*[Rectangle(width=0.42, height=3.0 / n).set_stroke(width=0).set_fill(C_REF if v > 0 else GREY_E, 1)
                        for v in bvals]).arrange(DOWN, buff=0).move_to([2.0, -0.45, 0])
        frames = VGroup(SurroundingRectangle(xcol, buff=0.03, color=GREY_B, stroke_width=1.5),
                        SurroundingRectangle(bcol, buff=0.03, color=GREY_B, stroke_width=1.5))
        eqs = MathTex("=", font_size=60).move_to([1.2, -0.45, 0])
        top = MathTex("A", "\\,x", "=", "b", font_size=72).move_to([0, 2.35, 0])
        top[1].set_color(C_U)
        top[3].set_color(C_REF)
        top2 = MathTex("A(p)", "\\,x", "=", "b", font_size=72).move_to([0, 2.35, 0])
        top2[1].set_color(C_U)
        top2[3].set_color(C_REF)
        cap = Tex("(computed: $A$ for a tiny $6\\times6$ grid; the splitter's $A$ has 43{,}200 rows)",
                  font_size=22, color=GREY_B).move_to([0, -2.95, 0])
        # a grid of 36 unknowns (like E09's grid): field values in blue, one source point in gold
        gc, gsp = np.array([-0.4, -0.45, 0]), 0.56
        gpos = [gc + np.array([(k // 6 - 2.5) * gsp, (2.5 - k % 6) * gsp, 0]) for k in range(n)]
        dots = VGroup(*[Dot(gpos[k], radius=0.11, color=interpolate_color(BLACK, C_U, float(0.25 + 0.75 * xv[k] / xv.max())))
                        for k in range(n)])
        glines = VGroup(*[Line(gc + np.array([(i - 2.5) * gsp, -2.8 * gsp, 0]), gc + np.array([(i - 2.5) * gsp, 2.8 * gsp, 0]))
                          for i in range(6)],
                        *[Line(gc + np.array([-2.8 * gsp, (j - 2.5) * gsp, 0]), gc + np.array([2.8 * gsp, (j - 2.5) * gsp, 0]))
                          for j in range(6)]).set_stroke(GREY_D, 1.5)
        src_ring = Circle(radius=0.2).move_to(gpos[14]).set_stroke(C_REF, 3)
        g_t = Tex("a grid: field values, one source", font_size=28, color=GREY_A).next_to(glines, DOWN, buff=0.3)
        self.at(S("a1") + 0.3)
        self.play(Create(glines), LaggedStart(*[GrowFromCenter(d_) for d_ in dots], lag_ratio=0.02), run_time=1.3)
        self.play(Create(src_ring), FadeIn(g_t), run_time=0.6)
        self.at(word_time("linear", S("a1") + 3.0, E("a1"), S("a1") + 4.6) - 0.35)
        self.play(*[ReplacementTransform(dots[k], xcol[k]) for k in range(n)], ReplacementTransform(src_ring, bcol[14]),
                  FadeIn(VGroup(*[bcol[k] for k in range(n) if k != 14])), FadeOut(glines), FadeOut(g_t),
                  FadeIn(Aimg), Create(Af), Create(frames), FadeIn(eqs), FadeIn(cap), run_time=1.1)
        self.at(word_time("times", S("a1") + 4.0, E("a1"), S("a1") + 6.0) - 0.35)
        self.play(Write(top), run_time=1.2)

        # ---- a2: what each symbol is
        lb = Tex("source", font_size=30, color=C_REF).next_to(frames[1], RIGHT, buff=0.3)
        lx = Tex("field everywhere", font_size=30, color=C_U).next_to(frames[0], UP, buff=0.2)
        la = Tex("material on\\\\the diagonal", font_size=30).next_to(Af, LEFT, buff=0.3)
        diag = Line(Af.get_corner(UL) + np.array([0.06, -0.06, 0]), Af.get_corner(DR) + np.array([-0.06, 0.06, 0])).set_stroke(YELLOW, 3, 0.7)
        self.at(S("a2"))
        self.play(FadeIn(lb), Indicate(bcol, color=C_REF, scale_factor=1.05), run_time=0.7)
        self.at(word_time("x", S("a2") + 0.6, E("a2"), S("a2") + 1.2) - 0.1)
        self.play(FadeIn(lx), run_time=0.6)
        self.at(word_time("matrix", S("a2") + 1.5, E("a2"), S("a2") + 3.0) - 0.1)
        self.play(FadeIn(la), Create(diag), run_time=0.8)
        self.at(word_time("depends", S("a2") + 2.5, E("a2"), S("a2") + 5.4) - 0.3)
        self.play(TransformMatchingTex(top, top2), run_time=0.8)

        # ---- a3: differentiate
        l1 = MathTex("\\frac{\\partial A}{\\partial p}\\,x", "+", "A\\,\\frac{\\partial x}{\\partial p}", "=", "0", font_size=50)
        l2 = MathTex("\\frac{\\partial x}{\\partial p}", "=", "-\\,A^{-1}", "\\,\\frac{\\partial A}{\\partial p}\\,x", font_size=50)
        l1.move_to([-0.6, 0.6, 0])
        l2.move_to([0, -1.0, 0])
        hint = Tex("($b$ does not\\\\depend on $p$)", font_size=26, color=GREY_B).next_to(l1, RIGHT, buff=0.5)
        self.at(S("a3"))
        self.play(FadeOut(Group(Aimg, Af, xcol, bcol, frames, eqs, cap, lb, lx, la, diag)), top2.animate.scale(0.7).move_to([0, 2.72, 0]),
                  run_time=0.6)
        self.play(Write(l1), FadeIn(hint), run_time=1.3)
        self.at(S("a3") + 2.7)
        self.play(Write(l2), run_time=1.2)

        # ---- a4: the chain of three factors (blocks below, connected to their factors)
        ch = MathTex("\\frac{dF}{dp}", "=", "-", "\\frac{\\partial F}{\\partial x}", "\\;A^{-1}\\;", "\\frac{\\partial A}{\\partial p}\\,x",
                     font_size=56).move_to([0.3, 1.45, 0])
        ch[3].set_color(YELLOW)
        yb = -1.05
        rowb = Rectangle(width=2.5, height=0.24).set_stroke(YELLOW, 2).set_fill(YELLOW, 0.35).move_to([-4.6, yb + 1.13, 0])
        sqb = Rectangle(width=2.5, height=2.5).set_stroke(GREY_A, 2).set_fill(GREY_D, 0.5).move_to([-1.6, yb, 0])
        sq_t = MathTex("A^{-1}", font_size=44).move_to(sqb)
        widb = VGroup(Rectangle(width=4.2, height=2.5).set_stroke(GREY_A, 2))
        ncols = 28
        for k in range(ncols):
            widb.add(Rectangle(width=4.2 / ncols * 0.72, height=2.36).set_stroke(width=0).set_fill(C_U, 0.45)
                     .move_to([(k + 0.5) / ncols * 4.2 - 2.1, 0, 0]))
        widb.move_to([2.4, yb, 0])
        links = VGroup(Line(ch[3].get_bottom() + 0.08 * DOWN, rowb.get_top() + 0.06 * UP),
                       Line(ch[4].get_bottom() + 0.08 * DOWN, sqb.get_top() + 0.06 * UP),
                       Line(ch[5].get_bottom() + 0.08 * DOWN, widb.get_top() + 0.06 * UP)).set_stroke(GREY_C, 1.5)
        r_l = Tex("a row", font_size=28, color=YELLOW).next_to(rowb, DOWN, buff=0.25)
        w_l = Tex("one column per pixel", font_size=28, color=C_U).next_to(widb, DOWN, buff=0.2)
        self.at(S("a4"))
        self.play(FadeOut(VGroup(l1, hint)), l2.animate.scale(0.7).move_to([3.3, 2.72, 0]), top2.animate.move_to([-1.6, 2.72, 0]),
                  run_time=0.6)
        self.play(Write(ch), run_time=1.3)
        self.at(word_time("row", S("a4") + 2.0, E("a4"), S("a4") + 3.0) - 0.1)
        self.play(FadeIn(rowb), FadeIn(r_l), Create(links[0]), run_time=0.5)
        self.play(FadeIn(sqb), FadeIn(sq_t), Create(links[1]), run_time=0.5)
        self.at(word_time("column", S("a4") + 3.0, E("a4"), S("a4") + 4.6) - 0.1)
        self.play(FadeIn(widb), FadeIn(w_l), Create(links[2]), run_time=0.6)

        # ---- a5: right to left = one solve per column
        rl = Arrow([5.0, -2.85, 0], [-1.2, -2.85, 0], buff=0, stroke_width=4, color=C_EV, max_tip_length_to_length_ratio=0.04)
        rl_t = Tex("right to left", font_size=28, color=C_EV).next_to(rl, DOWN, buff=0.06).shift(1.6 * RIGHT)
        col = ValueTracker(0.0)
        hl = always_redraw(lambda: Rectangle(width=4.2 / ncols, height=2.5).set_stroke(C_EV, 3).move_to(
            [2.4 - 2.1 + (min(int(col.get_value()), ncols - 1) + 0.5) / ncols * 4.2, yb, 0]))
        cnt = Counter(40, C_EV)
        nsol = ValueTracker(0)
        sol_t = Tex("solves:", font_size=32, color=C_EV).move_to([-5.9, -1.3, 0])
        cnt_m = always_redraw(lambda: cnt.build(nsol.get_value(), sol_t.get_right() + np.array([0.2, 0, 0])))
        self.at(S("a5"))
        self.play(FadeOut(VGroup(r_l, w_l)), GrowArrow(rl), FadeIn(rl_t), FadeIn(sol_t), run_time=0.6)
        self.add(hl, cnt_m)
        self.play(col.animate.set_value(ncols - 0.01), nsol.animate.set_value(4000), run_time=3.6, rate_func=rate_functions.ease_in_quad)
        self.remove(hl)
        self.play(widb[1:].animate.set_fill(opacity=0.15), run_time=0.3)

        # ---- a6: left to right = one extra solve
        lr = Arrow([-5.8, -2.85, 0], [0.4, -2.85, 0], buff=0, stroke_width=4, color=GREEN, max_tip_length_to_length_ratio=0.04)
        lr_t = Tex("left to right", font_size=28, color=GREEN).next_to(lr, DOWN, buff=0.06).shift(1.6 * LEFT)
        lam_row = Rectangle(width=5.6, height=0.3).set_stroke(C_ADJ, 2).set_fill(C_ADJ, 0.5).move_to([-3.1, yb + 1.13, 0])
        lam_l = MathTex("\\lambda^{T}", font_size=44, color=C_ADJ).next_to(lam_row, DOWN, buff=0.2)
        note = Tex("$\\lambda$: the adjoint field (not a wavelength)", font_size=24, color=C_ADJ).next_to(lam_l, DOWN, buff=0.15)
        adj = MathTex("A^{T}\\lambda", "=", "\\left(\\frac{\\partial F}{\\partial x}\\right)^{T}", font_size=44).move_to([-3.1, -1.55, 0])
        adj[0].set_color(C_ADJ)
        one = Tex("1 extra solve", font_size=32, color=GREEN).next_to(adj, DOWN, buff=0.2)
        merge_links = Line(ch[4].get_bottom() + 0.08 * DOWN, lam_row.get_top() + np.array([1.5, 0.06, 0])).set_stroke(GREY_C, 1.5)
        self.at(S("a6"))
        self.play(FadeOut(VGroup(rl, rl_t, cnt_m, sol_t)), widb[1:].animate.set_fill(opacity=0.45), GrowArrow(lr), FadeIn(lr_t), run_time=0.6)
        self.remove(cnt_m)
        self.play(ReplacementTransform(VGroup(rowb, sqb, sq_t), lam_row), ReplacementTransform(VGroup(links[0], links[1]), merge_links),
                  ch[3].animate.set_color(C_ADJ), ch[4].animate.set_color(C_ADJ), run_time=1.2)
        self.play(FadeIn(lam_l), FadeIn(note), run_time=0.5)
        self.at(S("a6") + 4.0)
        self.play(FadeOut(note), Write(adj), run_time=1.2)
        self.play(FadeIn(one), run_time=0.5)

        # ---- a7: cheap products -> the whole gradient
        grow = ValueTracker(0.0)
        gvals = d["g0"].astype(np.float64).ravel()[::143][:ncols]
        gsc = np.abs(gvals).max()

        def grad_row():
            g = VGroup()
            m = int(np.clip(grow.get_value(), 0, 1) * ncols + 1e-6)
            for k in range(m):
                v = gvals[k] / gsc
                c = interpolate_color(BLACK, GOLD if v > 0 else BLUE_B, min(1.0, 0.25 + abs(v)))
                g.add(Rectangle(width=4.2 / ncols, height=0.3).set_stroke(width=0).set_fill(c, 1).move_to(
                    [2.4 - 2.1 + (k + 0.5) / ncols * 4.2, -2.85, 0]))
            return g
        grw = always_redraw(grad_row)
        gl = MathTex("\\frac{dF}{dp}", font_size=34, color=GOLD).next_to([0.3, -2.85, 0], LEFT, buff=0.1)
        fin = MathTex("\\frac{dF}{dp}", "=", "-\\,\\lambda^{T}", "\\frac{\\partial A}{\\partial p}", "\\,x", font_size=56).move_to([0.3, 1.45, 0])
        fin[2].set_color(C_ADJ)
        fin[4].set_color(C_U)
        fbox = SurroundingRectangle(fin, buff=0.2, color=YELLOW, stroke_width=2.5)
        foot = Tex("complex fields: twice the\\\\real part of this (Wirtinger)",
                   font_size=24, color=GREY_B).move_to([4.75, 1.45, 0])
        two = Tex("2 simulations,\\\\any number of pixels", font_size=30, color=GREEN).move_to([-3.6, -2.0, 0])
        self.at(S("a7"))
        self.play(FadeOut(VGroup(lr, lr_t, adj, one)), run_time=0.4)
        self.add(grw)
        self.play(grow.animate.set_value(1.0), FadeIn(gl), run_time=1.6, rate_func=linear)
        self.play(FadeOut(merge_links), FadeOut(links[2]), ReplacementTransform(ch, fin), run_time=1.0)
        self.play(Create(fbox), FadeIn(foot), FadeIn(two), run_time=0.7)
        self.finish()


# ================================================================================ 3  backward ======
class Ch3Backward(Chapter):
    CH = 3
    TAG = "3\\quad Light run backward"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = D("splitter")
        G, _, _ = splitter_layers()
        vref = float(d["vref"])
        X0 = Cplx(d["x0"], 0.9 * vref)
        L0 = Cplx(d["lam0"], 0.8 * float(np.abs(d["lam0"]).max()))
        NU = Cplx(d["nudge"], 1.2 * float(np.percentile(np.abs(d["nudge"]), 99)))
        XO = Cplx(d["x_opt"], 0.9 * vref)
        LO = Cplx(d["lam_opt"], 0.75 * float(np.abs(d["lam_opt"]).max()))
        # devices
        box0 = np.zeros((320, 240), np.float32)
        box0[72:172, 40:200] = 0.5
        DEV0 = np.maximum(G, to_img(box0))
        _, _, BIN = splitter_layers()
        rps = d["rp_snaps"].astype(np.float32)
        absx = d["absx_snaps"].astype(np.float32)
        nit = len(rps)
        ucol = (np.arange(320) + 0.5) / 320
        wf, wa, wn = ValueTracker(1.0), ValueTracker(0.0), ValueTracker(0.0)
        afront = ValueTracker(1.0)          # adjoint reveal: from the outputs (u = 1) backward to u = 0
        nrad = ValueTracker(0.0)            # wavelet reveal radius (panel u units)
        mode = ValueTracker(0.0)            # 0: iteration-0 waves; 1: morph; 2: final waves
        itv = ValueTracker(0.0)
        pi_, pj_ = (int(v) for v in d["nudge_pix"])
        pu, pv = (BOX[0] + pi_ + 0.5) / 160, (BOX[2] + pj_ + 0.5) / 120
        VV, UU = np.meshgrid((np.arange(240)[::-1] + 0.5) / 240, ucol, indexing="ij")
        RAD = np.hypot(UU - pu, (VV - pv) * 0.75)

        def render(t):
            md = mode.get_value()
            if 0.002 < md < 0.998:
                return ((1 - md) * render_at(t, 0.0).astype(np.float32) + md * render_at(t, 1.0).astype(np.float32)).astype(np.uint8)
            return render_at(t, md)

        def render_at(t, md):
            if md < 0.5:
                rgb = DEV0[..., None] * RGB_SI
                if wf.get_value() > 0.003:
                    rgb = rgb + wf.get_value() * wave_rgb(X0.snap(t), RGB_U, RGB_U2, 0.8)
                if wa.get_value() > 0.003:
                    m = np.clip((ucol - afront.get_value()) / 0.03, 0, 1).astype(np.float32)[None, :, None]
                    rgb = rgb + wa.get_value() * m * wave_rgb(L0.snap(t), RGB_A, RGB_A2, 0.8)
                if wn.get_value() > 0.003:
                    m = np.clip((nrad.get_value() - RAD) / 0.05, 0, 1).astype(np.float32)[..., None]
                    rgb = rgb + wn.get_value() * m * wave_rgb(NU.snap(t), RGB_S, RGB_S2, 0.8)
                return rgba_of(rgb)
            if md < 1.5:
                v = float(np.clip(itv.get_value(), 0, nit - 1))
                k = min(int(v), nit - 2)
                f = np.float32(v - k)
                rp = (1 - f) * rps[k] + f * rps[k + 1]
                ax = (1 - f) * absx[k] + f * absx[k + 1]
                dev = np.zeros((320, 240), np.float32)
                dev[72:172, 40:200] = nearest_up(rp, 2)
                DEV = np.maximum(G, to_img(dev))
                AX = to_img(np.repeat(np.repeat(ax, 2, 0), 2, 1))
                rgb = DEV[..., None] * np.array([92, 96, 112], np.float32) + heat_rgb((AX / vref) ** 2 / 0.9, 0.5) * 0.95
                return rgba_of(rgb)
            rgb = BIN[..., None] * RGB_SI
            rgb = rgb + wf.get_value() * wave_rgb(XO.snap(t), RGB_U, RGB_U2) + wa.get_value() * wave_rgb(LO.snap(t), RGB_A, RGB_A2)
            return rgba_of(rgb)
        P = Panel(self, render, -2.0, -0.2, 6.8)
        bop = ValueTracker(1.0)
        boxl = always_redraw(lambda: box_outline(P, *BOX, color=GREY_B, opacity=bop.get_value()))
        src_m = VGroup(fast_arrow(P.pt(0.0, 0.5) + np.array([-0.55, -0.55, 0]), P.pt(0.0, 0.5) + np.array([-0.05, -0.08, 0]), C_REF, 4, 0.16),
                       Tex("source", font_size=26, color=C_REF).move_to(P.pt(0.0, 0.5) + np.array([-0.55, -0.8, 0])))
        det_x = 156 / 160
        det = VGroup(fast_dashed(P.pt(det_x, 0.04), P.pt(det_x, 0.96), 0.07, WHITE, 2.5),
                     Tex("detector", font_size=26).next_to(P.pt(det_x, 0.96), UP, buff=0.12))
        self.at(S("b1"))
        self.add(P.m, boxl)
        self.play(P.fade.animate.set_value(1.0), FadeIn(src_m), FadeIn(det), run_time=0.8)
        # reciprocity cartoon (right column)
        rc = np.array([4.35, 0.6, 0])
        sdot = Dot(rc + np.array([-1.15, 0.0, 0]), radius=0.1, color=C_REF)
        ddot = Dot(rc + np.array([1.15, 0.0, 0]), radius=0.1, color=WHITE)
        blobs = VGroup(*[Circle(radius=r).move_to(rc + np.array(o)).set_stroke(GREY_B, 1.5).set_fill(GREY_D, 0.6)
                         for r, o in ((0.24, (-0.3, 0.05, 0)), (0.2, (0.3, -0.12, 0)), (0.15, (0.05, 0.3, 0)))])
        fw = CurvedArrow(sdot.get_center() + 0.12 * UP, ddot.get_center() + 0.12 * UP, angle=-1.2, color=C_U, stroke_width=3)
        bw = CurvedArrow(ddot.get_center() + 0.12 * DOWN, sdot.get_center() + 0.12 * DOWN, angle=-1.2, color=C_ADJ, stroke_width=3)
        rlab = VGroup(Tex("source", font_size=24, color=C_REF).next_to(sdot, LEFT, buff=0.15),
                      Tex("detector", font_size=24).next_to(ddot, RIGHT, buff=0.15))
        rtitle = Tex("reciprocity", font_size=36).move_to(rc + np.array([0, 1.75, 0]))
        rsub = Tex("swap them: same signal", font_size=28, color=GREY_A).move_to(rc + np.array([0, -1.45, 0]))
        self.at(word_time("reciprocity", S("b1"), E("b1"), S("b1") + 2.8) - 0.1)
        self.play(FadeIn(rtitle), FadeIn(sdot), FadeIn(ddot), FadeIn(blobs), FadeIn(rlab), run_time=0.6)
        self.play(Create(fw), run_time=0.7)
        self.at(word_time("swap", S("b1"), E("b1"), S("b1") + 4.6))
        self.play(Create(bw), FadeIn(rsub), run_time=0.8)

        # ---- b2: A^T = A -> lambda is light sent backward from the outputs
        at_ = MathTex("A^{T}", "=", "A", font_size=60).move_to([4.35, 1.5, 0])
        same = MathTex("A\\,", "\\lambda", "=", "\\frac{\\partial F}{\\partial x}", font_size=48).move_to([4.35, 0.15, 0])
        same[1].set_color(C_ADJ)
        same_t = Tex("same simulation,\\\\new source at the detector", font_size=28, color=GREY_A).next_to(same, DOWN, buff=0.3)
        adj_l = Tex("adjoint field $\\lambda$", font_size=30, color=C_ADJ).move_to([4.35, -2.4, 0])
        self.at(S("b2"))
        self.play(FadeOut(VGroup(rtitle, sdot, ddot, blobs, rlab, fw, bw, rsub)), run_time=0.4)
        self.play(Write(at_), run_time=0.9)
        self.at(word_time("comes", S("b2"), E("b2"), S("b2") + 3.6) - 0.2)
        self.play(FadeIn(same), FadeIn(same_t), run_time=0.8)
        self.at(word_time("outputs", S("b2"), E("b2"), S("b2") + 6.0) - 0.4)
        afront.set_value(1.0)
        self.play(wa.animate.set_value(1.0), wf.animate.set_value(0.25), run_time=0.4)
        self.play(afront.animate.set_value(0.0), FadeIn(adj_l), run_time=2.2, rate_func=linear)

        # ---- b3: nudge one pixel -> a wavelet
        psq = Square(side_length=0.22).move_to(P.pt(pu, pv)).set_stroke(YELLOW, 3)
        w_t = Tex("a nudged pixel\\\\scatters a wavelet,", font_size=30).move_to([4.35, 1.3, 0])
        w_t2 = Tex("strength $\\propto$ forward field $x_j$", font_size=30, color=C_U).next_to(w_t, DOWN, buff=0.25)
        self.at(S("b3"))
        self.play(FadeOut(VGroup(at_, same, same_t, adj_l)), wa.animate.set_value(0.0), wf.animate.set_value(0.45), Create(psq), run_time=0.6)
        wn.set_value(1.0)
        self.play(nrad.animate.set_value(0.32), FadeIn(w_t), run_time=2.4, rate_func=linear)
        self.play(FadeIn(w_t2), nrad.animate.set_value(0.45), run_time=0.8, rate_func=linear)

        # ---- b4: the backward field at that pixel: does the wavelet arrive in step?
        xj, lj = complex(d["nudge_x"]), complex(d["nudge_lam"])
        pc = np.array([4.35, -0.9, 0])
        R = 0.95
        prod = -xj * lj
        dial = VGroup(Circle(radius=R).move_to(pc).set_stroke(GREY_C, 1.5),
                      Line(pc + (R + 0.25) * LEFT, pc + (R + 0.25) * RIGHT).set_stroke(GREY_C, 1),
                      phasor(pc, R * xj / abs(xj), 1.0, C_U, 5, 0.17),
                      phasor(pc, R * lj / abs(lj), 1.0, C_ADJ, 5, 0.17))
        dl = VGroup(MathTex("x_j", font_size=30, color=C_U).move_to(pc + 1.25 * R * np.array([np.cos(np.angle(xj)), np.sin(np.angle(xj)), 0])),
                    MathTex("\\lambda_j", font_size=30, color=C_ADJ).move_to(pc + 1.25 * R * np.array([np.cos(np.angle(lj)), np.sin(np.angle(lj)), 0])))
        pr_ar = phasor(pc, R * prod / abs(prod), 1.0, WHITE, 5, 0.17)
        shadow = Line(pc, pc + R * np.real(prod / abs(prod)) * RIGHT).set_stroke(GOLD if np.real(prod) > 0 else BLUE_B, 8)
        pl = MathTex("-x_j\\lambda_j", font_size=30).move_to(pc + 1.3 * R * np.array([np.cos(np.angle(prod)), np.sin(np.angle(prod)), 0]))
        instep = Tex("in step: adding silicon helps" if np.real(prod) > 0 else "out of step: removing silicon helps",
                     font_size=26, color=GOLD if np.real(prod) > 0 else BLUE_B).move_to(pc + np.array([0, -1.45, 0]))
        self.at(S("b4"))
        self.play(FadeOut(VGroup(w_t, w_t2)), wa.animate.set_value(0.85), wn.animate.set_value(0.4), wf.animate.set_value(0.3),
                  nrad.animate.set_value(1.1), run_time=1.2, rate_func=linear)
        bt = Tex("forward and backward\\\\fields at that pixel", font_size=28, color=GREY_A).move_to([4.35, 1.45, 0])
        self.play(FadeIn(dial), FadeIn(dl), FadeIn(bt), run_time=0.8)
        self.at(S("b4") + 3.2)
        self.play(GrowFromPoint(pr_ar, pc), FadeIn(pl), run_time=0.7)
        self.play(Create(shadow), FadeIn(instep), run_time=0.7)

        # ---- b5: the sensitivity map, every pixel at once
        g = d["dFde0"].astype(np.float32)
        sc = float(np.percentile(np.abs(g), 97))
        smap = diverge_rgba(g.T, sc)                                      # rows = y (flipped up); gold: add, cyan: remove
        a_, b_ = P.pt(BOX[0] / 160, BOX[2] / 120), P.pt(BOX[1] / 160, BOX[3] / 120)
        sm, sm_f, sm_s = self.sweep_image(smap, b_[0] - a_[0], (a_ + b_) / 2, b_[1] - a_[1], axis="left")
        eqs = MathTex("\\frac{\\partial F}{\\partial \\varepsilon_j}", "=", "-2k_0^2\\,\\mathrm{Re}\\{", "\\lambda_j", "\\,x_j", "\\}",
                      font_size=44).move_to([4.35, 1.2, 0])
        eqs[3].set_color(C_ADJ)
        eqs[4].set_color(C_U)
        leg = VGroup(VGroup(Square(0.3).set_fill(GOLD, 1).set_stroke(width=0), Tex("add silicon", font_size=28)).arrange(RIGHT, buff=0.2),
                     VGroup(Square(0.3).set_fill(rgb_to_color(np.array([88, 196, 221]) / 255), 1).set_stroke(width=0),
                            Tex("remove silicon", font_size=28)).arrange(RIGHT, buff=0.2)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        leg.move_to([4.35, -0.6, 0])
        smt = Tex("the sensitivity map", font_size=32, color=YELLOW).move_to([4.35, -2.2, 0])
        self.at(S("b5"))
        self.play(FadeOut(VGroup(dial, dl, pr_ar, shadow, pl, instep, bt, psq)), wn.animate.set_value(0.0), run_time=0.6)
        self.play(Write(eqs), run_time=1.2)
        self.add(sm)
        sm_f.set_value(1.0)
        self.at(word_time("sensitivity", S("b5"), E("b5"), S("b5") + 4.2) - 0.5)
        self.play(sm_s.animate.set_value(1.0), wa.animate.set_value(0.35), wf.animate.set_value(0.35), FadeIn(leg), FadeIn(smt),
                  run_time=1.6, rate_func=linear)

        # ---- b6/b7: update, re-simulate, repeat -> the device takes shape (iterations 0..159)
        ax_c, ax_w, ax_h = np.array([4.5, 0.05, 0]), 3.4, 2.6
        axes = axes_lr(ax_c, ax_w, ax_h, "iteration", "merit $F$", size=26)
        tk = VGroup(MathTex("1", font_size=24, color=GREY_B).next_to(ax_c + np.array([-ax_w / 2, ax_h / 2, 0]), LEFT, buff=0.1),
                    MathTex("0", font_size=24, color=GREY_B).next_to(ax_c + np.array([-ax_w / 2, -ax_h / 2, 0]), LEFT, buff=0.1),
                    MathTex(str(nit), font_size=24, color=GREY_B).next_to(ax_c + np.array([ax_w / 2, -ax_h / 2, 0]), DOWN, buff=0.1))
        Fh = d["F_hist"].astype(np.float64)
        xs_ = np.arange(nit)

        def curve():
            k = int(np.clip(itv.get_value(), 0, nit - 1))
            g_ = VGroup()
            if k >= 1:
                g_.add(plot_xy(xs_[:k + 1], Fh[:k + 1], ax_c, ax_w, ax_h, (0, nit), (0, 1), YELLOW, 3))
            p_ = ax_c + np.array([-ax_w / 2 + k / nit * ax_w, -ax_h / 2 + Fh[k] * ax_h, 0])
            g_.add(Dot(p_, radius=0.06, color=YELLOW))
            return g_
        cur = always_redraw(curve)
        cnt = Counter(30, WHITE)
        it_t = Tex("iteration", font_size=28, color=GREY_A).move_to([3.3, 2.15, 0])
        it_m = always_redraw(lambda: cnt.build(itv.get_value() + 1, it_t.get_right() + np.array([0.2, 0, 0])))
        self.at(S("b6"))
        self.play(FadeOut(VGroup(eqs, leg, smt)), sm_f.animate.set_value(0.0), run_time=0.5)
        self.remove(sm)
        self.play(mode.animate.set_value(1.0), FadeIn(axes), FadeIn(tk), FadeIn(it_t), run_time=0.5)
        self.add(cur, it_m)
        t_end = S("b7") + 1.6
        self.play(itv.animate.set_value(nit - 1), run_time=dur(self, t_end), rate_func=lambda a: a ** 1.6)
        fin_t = Tex(f"{100 * float(d['T_opt']):.1f}\\% (all silicon or glass)", font_size=28, color=GREEN_B).move_to([4.4, -2.0, 0])
        self.play(FadeIn(fin_t), run_time=0.4)
        self.finish()


# ================================================================================ 4  autodiff ======
class Ch4Autodiff(Chapter):
    CH = 4
    TAG = "4\\quad Differentiable optics"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        a = D("asm")
        phs, ins = a["phase"].astype(np.float32), a["inten"].astype(np.float32)
        its = a["snap_its"]
        nsn = len(its)
        # ---- d1: no derivation needed: a phase mask and free-space propagation (E01)
        PINK_LUT = lut([(0.0, (0, 0, 0)), (0.55, (209, 71, 189)), (1.0, (255, 225, 245))])

        def pink_rgba(v):
            return rgba_of(PINK_LUT[(np.clip(v, 0, 1) ** 0.6 * 1023).astype(int)].astype(np.float32))
        hand = MathTex("\\frac{dF}{dp}", "=", "-\\,\\lambda^{T}", "\\frac{\\partial A}{\\partial p}", "\\,x", font_size=48).move_to([0, 0.6, 0])
        hand[2].set_color(C_ADJ)
        hand[4].set_color(C_U)
        hand_t = Tex("derived by hand, for one kind of simulation", font_size=28, color=GREY_A).next_to(hand, DOWN, buff=0.3)
        self.at(S("d1") + 0.2)
        self.play(FadeIn(hand), FadeIn(hand_t), run_time=0.7)
        yp, sz = -1.6, 2.0
        pm0 = phase_rgba(np.exp(1j * phs[0]), 1.0, flip=False)
        mask = ImageMobject(pm0).stretch_to_fit_width(sz).stretch_to_fit_height(sz).move_to([-4.6, yp, 0])
        mask_f = SurroundingRectangle(mask, buff=0.02, color=GREY_B, stroke_width=1.5)
        mask_t = Tex("phase mask $\\varphi$", font_size=28).next_to(mask_f, UP, buff=0.15)
        scr0 = ImageMobject(heat_rgba(ins[0], 1.0, 0.7, flip=False)).stretch_to_fit_width(sz).stretch_to_fit_height(sz).move_to([4.4, yp, 0])
        scr_f = SurroundingRectangle(scr0, buff=0.02, color=GREY_B, stroke_width=1.5)
        scr_t = Tex("screen at distance $z$", font_size=28).next_to(scr_f, UP, buff=0.15)
        beam = VGroup(*[Arrow([-3.35, yp + y, 0], [3.15, yp + y, 0], buff=0, stroke_width=3, color=C_U, max_tip_length_to_length_ratio=0.03)
                        for y in (-0.6, 0.0, 0.6)])
        asm_t = Tex("free-space propagation (E01)", font_size=28, color=GREY_A).move_to([-0.1, yp + 0.95, 0])
        self.at(word_time("take", S("d1") + 2.0, E("d1"), S("d1") + 2.8) - 0.1)
        self.play(FadeOut(VGroup(hand, hand_t)), FadeIn(mask), Create(mask_f), FadeIn(mask_t), run_time=0.7)
        self.at(word_time(("then", "episode"), S("d1") + 3.0, E("d1"), S("d1") + 4.2) - 0.1)
        self.play(LaggedStart(*[GrowArrow(b_) for b_ in beam], lag_ratio=0.2), FadeIn(asm_t), FadeIn(scr0), Create(scr_f), FadeIn(scr_t),
                  run_time=1.2)

        # ---- d2: the forward chain
        names = ["U_0", "\\times e^{i\\varphi}", "\\mathrm{FFT}", "\\times H", "\\mathrm{FFT}^{-1}", "|\\cdot|^2", "L"]
        xsn = np.linspace(-5.7, 5.7, 7)
        yf, yb = 2.35, 1.0

        def node(tex, x, y, color=WHITE, stroke=GREY_B):
            t = MathTex(tex, font_size=34, color=color)
            r = RoundedRectangle(corner_radius=0.12, width=max(1.35, t.width + 0.35), height=0.66).set_stroke(stroke, 2)
            return VGroup(r, t).move_to([x, y, 0])
        fwd = VGroup(*[node(n_, x, yf) for n_, x in zip(names, xsn)])
        fwd[3][1].set_color(C_H)
        fars = VGroup(*[Arrow(fwd[i].get_right(), fwd[i + 1].get_left(), buff=0.06, stroke_width=3, color=C_U,
                              max_tip_length_to_length_ratio=0.25) for i in range(6)])
        f_l = Tex("forward", font_size=26, color=C_U).next_to(fwd, UP, buff=0.14).align_to(fwd, LEFT)
        word_ids = ["mask", "f", "transfer", "inverse", "square"]
        self.at(S("d2"))
        self.play(FadeIn(fwd[0]), FadeIn(f_l), run_time=0.4)
        tprev = S("d2") + 0.9
        for i in range(1, 7):
            tw = word_time(word_ids[i - 1], tprev, E("d2"), tprev + 0.8) if i <= 5 else tprev + 0.8
            self.at(max(tw - 0.15, self.now))
            self.play(GrowArrow(fars[i - 1]), FadeIn(fwd[i]), run_time=0.45)
            tprev = max(tw, self.now)
        pulse = ValueTracker(0.0)

        def dotf():
            u = pulse.get_value()
            if u <= 0.001 or u >= 0.999:
                return VGroup()
            return Dot([-5.7 + 11.4 * u, yf - 0.48, 0], radius=0.08, color=C_U)
        pf = always_redraw(dotf)
        self.add(pf)
        self.play(pulse.animate.set_value(1.0), run_time=1.0, rate_func=linear)
        self.remove(pf)

        # ---- d3: backward pass
        bnames = ["\\frac{\\partial L}{\\partial \\varphi}", "\\times e^{-i\\varphi}", "\\mathrm{FFT}^{-1}", "\\times H^{*}",
                  "\\mathrm{FFT}", "\\times 2U", "1"]
        bwd = VGroup(*[node(n_, x, yb, C_ADJ if i in (2, 3, 4) else GREY_A, C_ADJ if i in (2, 3, 4) else GREY_D)
                       for i, (n_, x) in enumerate(zip(bnames, xsn))])
        bars = VGroup(*[Arrow(bwd[i + 1].get_left(), bwd[i].get_right(), buff=0.06, stroke_width=3, color=C_ADJ,
                              max_tip_length_to_length_ratio=0.25) for i in range(6)])
        b_l = Tex("backward (automatic differentiation)", font_size=26, color=C_ADJ).next_to(bwd, DOWN, buff=0.12).align_to(bwd, LEFT)
        bp = ValueTracker(0.0)

        def dotb():
            u = bp.get_value()
            if u <= 0.001 or u >= 0.999:
                return VGroup()
            return Dot([5.7 - 11.4 * u, yb + 0.48, 0], radius=0.08, color=C_ADJ)
        pb = always_redraw(dotb)
        self.at(word_time("runs", S("d3"), E("d3"), S("d3") + 3.1) - 0.1)
        self.add(pb)
        steps = [FadeIn(bwd[6])] + [AnimationGroup(GrowArrow(bars[5 - i]), FadeIn(bwd[5 - i])) for i in range(6)]
        self.play(FadeIn(b_l), bp.animate.set_value(1.0), LaggedStart(*steps, lag_ratio=0.5), run_time=2.0, rate_func=linear)
        self.remove(pb)
        swap1 = Line(fwd[2].get_bottom(), bwd[2].get_top()).set_stroke(C_ADJ, 2, 0.6)
        swap2 = Line(fwd[4].get_bottom(), bwd[4].get_top()).set_stroke(C_ADJ, 2, 0.6)
        swap3 = Line(fwd[3].get_bottom(), bwd[3].get_top()).set_stroke(C_ADJ, 2, 0.6)
        self.at(word_time("ffts", S("d3") + 3.5, E("d3"), S("d3") + 5.6) - 0.1)
        self.play(Create(swap1), Create(swap2), Indicate(bwd[2], color=C_ADJ), Indicate(bwd[4], color=C_ADJ), run_time=1.0)
        self.at(word_time("conjugate", S("d3") + 5.0, E("d3") + 0.3, S("d3") + 7.6) - 0.6)
        self.play(Create(swap3), Indicate(bwd[3], color=C_ADJ, scale_factor=1.3), run_time=1.0)

        # ---- d4: H* = propagate by -z: the error, computed, travels back to the mask
        hz = MathTex("H", "=", "e^{\\,i k_z z}", "\;\;\\Rightarrow\;\;", "H^{*}", "=", "e^{\\,i k_z (-z)}", font_size=40).move_to([-0.1, yp + 0.95, 0])
        hz[0].set_color(C_H)
        hz[2].set_color(C_H)
        hz[4].set_color(C_ADJ)
        hz[6].set_color(C_ADJ)
        self.at(S("d4"))
        self.play(FadeOut(asm_t), Write(hz), run_time=1.3)
        err_m = ImageMobject(pink_rgba(a["err0"].astype(np.float32) / 0.6)).stretch_to_fit_width(sz).stretch_to_fit_height(sz).move_to(scr0)
        back_m = ImageMobject(pink_rgba(a["back0"].astype(np.float32) / 0.8)).stretch_to_fit_width(sz).stretch_to_fit_height(sz).move_to(mask)
        rbeam = VGroup(*[Arrow([3.15, yp + y, 0], [-3.35, yp + y, 0], buff=0, stroke_width=3, color=C_ADJ, max_tip_length_to_length_ratio=0.03)
                         for y in (-0.6, 0.0, 0.6)])
        err_t = Tex("error on the screen", font_size=28, color=C_ADJ).move_to(scr_t)
        back_t = Tex("sent back by $-z$", font_size=28, color=C_ADJ).move_to(mask_t)
        ev = Tex("(evanescent waves just decay: $e^{-\\kappa z}$ is not undone)", font_size=24, color=GREY_B).move_to([0, -2.95, 0])
        chk = Tex(f"computed: equals autodiff to ${a['relerr'] * 1e7:.1f}\\times 10^{{-7}}$", font_size=24, color=GREY_B).move_to([-0.1, yp - 0.95, 0])
        self.at(word_time("backward", S("d4") + 3.0, E("d4"), S("d4") + 4.6) - 0.3)
        self.play(ReplacementTransform(beam, rbeam), FadeIn(err_m), FadeTransform(scr_t, err_t), run_time=0.8)
        self.play(FadeIn(back_m), FadeTransform(mask_t, back_t), FadeIn(chk), run_time=0.8)
        self.play(FadeIn(ev), run_time=0.5)

        # ---- d5: gradient descent shapes the mask (computed, 300 Adam steps)
        itf = ValueTracker(0.0)

        def interp(arr):
            v = float(np.clip(itf.get_value(), 0, nsn - 1))
            k = min(int(v), nsn - 2)
            f = v - k
            return arr[k], arr[k + 1], f

        MASKS = [phase_rgba(np.exp(1j * phs[k]), 1.0, flip=False) for k in range(nsn)]
        IMGS = [heat_rgba(ins[k], 1.0, 0.7, flip=False).astype(np.float32) for k in range(nsn)]

        def mask_fn(t):
            _, _, f = interp(phs)
            k = min(int(np.clip(itf.get_value(), 0, nsn - 1)), nsn - 2)
            return MASKS[k] if f < 0.5 else MASKS[k + 1]

        def img_fn(t):
            _, _, f = interp(ins)
            k = min(int(np.clip(itf.get_value(), 0, nsn - 1)), nsn - 2)
            return ((1 - f) * IMGS[k] + f * IMGS[k + 1]).astype(np.uint8)
        mk, mk_f = self.live_image(mask_fn, 3.0, np.array([-4.3, -0.55, 0]), 3.0)
        im, im_f = self.live_image(img_fn, 3.0, np.array([-0.55, -0.55, 0]), 3.0)
        mkF = SurroundingRectangle(mk, buff=0.02, color=GREY_B, stroke_width=1.5)
        imF = SurroundingRectangle(im, buff=0.02, color=GREY_B, stroke_width=1.5)
        mk_t = Tex("phase mask $\\varphi$", font_size=28).next_to(mkF, UP, buff=0.15)
        im_t = Tex("projected image", font_size=28).next_to(imF, UP, buff=0.15)
        tg = ImageMobject(heat_rgba(a["target"].astype(np.float32), 1.0, 0.7, flip=False)).stretch_to_fit_width(1.0).stretch_to_fit_height(1.0)
        tg.move_to([1.65, 0.45, 0])
        tgF = SurroundingRectangle(tg, buff=0.02, color=GREY_B, stroke_width=1)
        tg_t = Tex("target", font_size=24, color=GREY_A).next_to(tgF, DOWN, buff=0.1)
        hist = a["hist"].astype(np.float64)
        lc, lw, lh = np.array([4.6, -0.75, 0]), 3.0, 2.5
        lax = axes_lr(lc, lw, lh, "gradient step", "error (log)", size=26)
        lgh = np.log10(hist)
        ylim = (np.floor(lgh.min()), np.ceil(lgh.max()))
        ytk = VGroup(MathTex(f"10^{{{int(ylim[1])}}}", font_size=22, color=GREY_B).next_to(lc + np.array([-lw / 2, lh / 2, 0]), LEFT, buff=0.08),
                     MathTex(f"10^{{{int(ylim[0])}}}", font_size=22, color=GREY_B).next_to(lc + np.array([-lw / 2, -lh / 2, 0]), LEFT, buff=0.08))

        def loss_curve():
            v = float(np.clip(itf.get_value(), 0, nsn - 1))
            k = min(int(v), nsn - 2)
            stepf = its[k] + (v - k) * (its[k + 1] - its[k])
            m = int(stepf)
            g_ = VGroup()
            if m >= 1:
                g_.add(plot_xy(np.arange(m + 1), lgh[:m + 1], lc, lw, lh, (0, 300), ylim, C_ADJ, 3))
            g_.add(Dot(lc + np.array([-lw / 2 + m / 300 * lw, -lh / 2 + (lgh[m] - ylim[0]) / (ylim[1] - ylim[0]) * lh, 0]), radius=0.06, color=C_ADJ))
            return g_
        lcur = always_redraw(loss_curve)
        demo_cap = Tex("computed: $256\\times256$ phase mask, exact ASM, 300 gradient steps",
                       font_size=24, color=GREY_B).move_to([0.6, 2.75, 0])
        self.at(S("d5"))
        self.play(FadeOut(Group(mask, mask_f, scr0, scr_f, err_m, back_m, fwd, fars, f_l, bwd, bars, b_l, swap1, swap2, swap3, hz, ev,
                                chk, rbeam, err_t, back_t)), run_time=0.5)          # keep the original z-order (no pop)
        self.add(mk, im)
        self.play(mk_f.animate.set_value(1.0), im_f.animate.set_value(1.0), Create(mkF), Create(imF), FadeIn(mk_t), FadeIn(im_t),
                  FadeIn(tg), FadeIn(tgF), FadeIn(tg_t), FadeIn(lax), FadeIn(ytk), FadeIn(demo_cap), run_time=0.8)
        self.add(lcur)
        self.at(word_time("watch", S("d5"), E("d5"), S("d5") + 3.4) - 0.2)
        self.play(itf.animate.set_value(nsn - 1), run_time=dur(self, E("d5") + 0.6), rate_func=lambda u: u ** 1.15)

        # ---- d6: Gerchberg-Saxton (E05) was a heuristic loop; gradients take any goal
        gs = Tex("E05's Gerchberg--Saxton: a clever loop", font_size=30, color=GREY_A).move_to([-2.6, -2.75, 0])
        gd = Tex("gradients: any goal you can write down", font_size=30, color=YELLOW).move_to([3.4, -2.75, 0])
        self.at(S("d6") + 0.3)
        self.play(FadeIn(gs), run_time=0.6)
        self.at(word_time("gradients", S("d6") + 2.0, E("d6"), S("d6") + 3.9) - 0.1)
        self.play(FadeIn(gd), run_time=0.6)
        self.finish()


# ================================================================================ 5  end-to-end ====
class Ch5EndToEnd(Chapter):
    CH = 5
    TAG = "5\\quad End-to-end design"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        e = D("edof")
        scene_img = e["scene"].astype(np.float32)
        raw = e["learned_raw"][1].astype(np.float32)
        rest = e["learned_rest"][1].astype(np.float32)

        def grey_img(a, w, c):
            m = ImageMobject(grey_rgba(np.clip(a, 0, 1), 1.0, flip=False)).stretch_to_fit_width(w).stretch_to_fit_height(w).move_to(c)
            return Group(m, SurroundingRectangle(m, buff=0.02, color=GREY_B, stroke_width=1.2))
        yp = 0.6
        xsn = [-5.55, -3.3, -1.05, 1.2, 3.45, 5.6]
        n_scene = grey_img(scene_img, 1.5, [xsn[0], yp, 0])
        pup = e["pupil_snaps"][-1]
        pup_rgba = phase_rgba(np.exp(1j * np.nan_to_num(pup)), 1.0, flip=False)
        pup_rgba[..., 3] = np.where(np.isnan(pup), 0, 255).astype(np.uint8)
        n_opt = Group(ImageMobject(pup_rgba).stretch_to_fit_width(1.4).stretch_to_fit_height(1.4).move_to([xsn[1], yp, 0]),
                      Circle(radius=0.72).move_to([xsn[1], yp, 0]).set_stroke(GREY_B, 1.5))
        n_raw = grey_img(raw / np.percentile(raw, 99.5), 1.5, [xsn[2], yp, 0])
        n_w = VGroup(RoundedRectangle(corner_radius=0.12, width=1.5, height=1.0).set_stroke(GREY_B, 2), MathTex("W", font_size=44)).move_to([xsn[3], yp, 0])
        n_rest = grey_img(rest, 1.5, [xsn[4], yp, 0])
        n_loss = VGroup(MathTex("\\| \\hat{s} - s \\|^2", font_size=40)).move_to([xsn[5], yp, 0])
        nodes = [n_scene, n_opt, n_raw, n_w, n_rest, n_loss]
        labels = ["scene $s$", "optics (PSF)", "sensor $+$ noise", "Wiener filter (E03)", "image $\\hat{s}$", "error vs truth"]
        labs = VGroup(*[Tex(t, font_size=24, color=GREY_A).move_to([x, yp - 1.12, 0]) for t, x in zip(labels, xsn)])
        fars = VGroup(*[Arrow([xsn[i] + 0.8, yp, 0], [xsn[i + 1] - 0.8, yp, 0], buff=0, stroke_width=4, color=C_U,
                              max_tip_length_to_length_ratio=0.3) for i in range(5)])
        self.at(S("e1"))
        self.play(FadeIn(n_scene), FadeIn(labs[0]), FadeIn(n_opt), FadeIn(labs[1]), run_time=0.8)
        self.at(S("e2"))
        self.play(GrowArrow(fars[0]), run_time=0.4)
        self.play(GrowArrow(fars[1]), FadeIn(n_raw), FadeIn(labs[2]), run_time=0.6)
        self.at(word_time("then", S("e2") + 2.0, E("e2"), S("e2") + 3.3) - 0.1)
        self.play(GrowArrow(fars[2]), FadeIn(n_w), FadeIn(labs[3]), run_time=0.6)
        self.at(word_time("sharpens", S("e2") + 3.0, E("e2"), S("e2") + 6.2) - 0.2)
        self.play(GrowArrow(fars[3]), FadeIn(n_rest), FadeIn(labs[4]), run_time=0.6)

        # ---- e3: compare with the truth, send the gradient back into the optics
        bars = VGroup(*[CurvedArrow([xsn[i + 1] - 0.2, yp - 1.42, 0], [xsn[i] + 0.2, yp - 1.42, 0], angle=-0.9, color=C_ADJ, stroke_width=4)
                        for i in range(5)])
        self.at(S("e3"))
        self.play(GrowArrow(fars[4]), FadeIn(n_loss), FadeIn(labs[5]), run_time=0.6)
        self.at(word_time("send", S("e3"), E("e3"), S("e3") + 2.4) - 0.1)
        self.play(LaggedStart(*[Create(bars[4 - i]) for i in range(5)], lag_ratio=0.55), run_time=3.2)
        glow = Circle(radius=0.8).move_to([xsn[1], yp, 0]).set_stroke(C_ADJ, 5)
        self.play(Create(glow), run_time=0.5)

        # ---- e4: an ordinary lens is sharp only in focus
        show = e["show"]
        psis = e["psis"]
        psf_lens = e["psf_lens"].astype(np.float32)
        snaps = e["psf_snaps"].astype(np.float32)
        nsn = len(snaps)
        cc = slice(14, 50)
        px = np.linspace(-4.15, -0.35, 5)
        yr1, yr2 = 0.05, -2.05
        ips = 0.85

        def psf_img(p):
            q = p[cc, cc]
            return heat_rgba(q / q.max(), 1.0, 0.55, flip=False)
        lens_psfs = Group(*[ImageMobject(psf_img(psf_lens[k])).stretch_to_fit_width(ips).stretch_to_fit_height(ips).move_to([px[k], yr1, 0])
                            for k in range(5)])
        lens_fr = VGroup(*[SurroundingRectangle(m, buff=0.01, color=GREY_C, stroke_width=1) for m in lens_psfs])
        defoc = VGroup(*[MathTex(f"{psis[s_] / TAU_:+.2g}\\lambda".replace("+0\\lambda", "0").replace("+", "+"), font_size=24, color=GREY_B)
                         .move_to([px[k], yr1 + 0.68, 0]) for k, s_ in enumerate(show)])
        defoc_t = Tex("point-spread function at defocus", font_size=26, color=GREY_A).move_to([px.mean(), yr1 + 1.5, 0])
        r1 = Tex("ordinary\\\\lens", font_size=28).move_to([-5.75, yr1, 0])
        r2 = Tex("learned\\\\optics", font_size=28, color=C_ADJ).move_to([-5.75, yr2, 0])
        imx = [1.35, 3.25, 5.15]
        iw = 1.75
        lens_im = Group(*[grey_img(e["lens_rest"][q].astype(np.float32), iw, [imx[q], yr1, 0]) for q in range(3)])
        lrn_im = Group(*[grey_img(e["learned_rest"][q].astype(np.float32), iw, [imx[q], yr2, 0]) for q in range(3)])
        heads = VGroup(*[Tex(t, font_size=24, color=GREY_B).move_to([imx[q], yr1 + 1.08, 0]) for q, t in enumerate(("near", "in focus", "far"))])
        im_t = Tex("image after the filter", font_size=26, color=GREY_A).move_to([imx[1], yr1 + 1.5, 0])
        self.at(S("e4"))
        self.play(FadeOut(Group(*nodes[1:], labs, fars, bars, glow, n_scene)), run_time=0.6)
        self.play(FadeIn(r1), FadeIn(defoc_t), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(Group(lens_psfs[k], lens_fr[k], defoc[k])) for k in (2, 1, 3, 0, 4)], lag_ratio=0.45), run_time=2.0)
        self.at(word_time("elsewhere", S("e4") + 2.0, E("e4"), S("e4") + 4.6) - 0.2)
        self.play(Indicate(lens_psfs[0], scale_factor=1.12, color=WHITE), Indicate(lens_psfs[4], scale_factor=1.12, color=WHITE), run_time=0.8)
        self.at(word_time("blurs", S("e4") + 4.0, E("e4") + 0.3, S("e4") + 7.8) - 0.6)
        self.play(FadeIn(lens_im), FadeIn(heads), FadeIn(im_t), run_time=0.8)

        # ---- e5: the learned optics: a PSF that barely changes with depth (training snapshots)
        itp = ValueTracker(0.0)

        def lrn_fn(k):
            def fn(t):
                v = float(np.clip(itp.get_value(), 0, nsn - 1))
                j = min(int(v), nsn - 2)
                f = v - j
                return psf_img((1 - f) * snaps[j, k] + f * snaps[j + 1, k])
            return fn
        lrn = []
        for k in range(5):
            m, fd = self.live_image(lrn_fn(k), ips, np.array([px[k], yr2, 0]), ips)
            lrn.append((m, fd))
        lrn_fr = VGroup(*[SurroundingRectangle(m, buff=0.01, color=GREY_C, stroke_width=1) for m, _ in lrn])
        tr_t = Tex("training", font_size=24, color=C_ADJ).move_to([-2.25, yr2 - 0.68, 0])
        self.at(S("e5"))
        for m, _ in lrn:
            self.add(m)
        self.play(FadeIn(r2), FadeIn(lrn_fr), FadeIn(tr_t), *[fd.animate.set_value(1.0) for _, fd in lrn], run_time=0.6)
        self.play(itp.animate.set_value(nsn - 1), run_time=3.2, rate_func=lambda u: u ** 0.7)
        self.at(word_time("undo", S("e5") + 3.0, E("e5"), S("e5") + 5.6) - 0.3)
        self.play(FadeIn(lrn_im), run_time=0.9)
        self.finish()


# ================================================================================ 6  in practice ===
class Ch6Practice(Chapter):
    CH = 6
    TAG = "6\\quad In practice"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = D("splitter")
        card = code_card_ind(["for step in range(300):",
                              "    F, g = merit_and_gradient(p)   # 2 simulations",
                              "    p = clip(p + lr * g, 0, 1)     # step uphill"], font_size=26, title="inverse design").move_to([0, 0.6, 0])
        rows = card[1][0]
        self.at(S("c1") + 0.1)
        self.play(FadeIn(card, shift=0.1 * DOWN), run_time=0.7)
        hl = SurroundingRectangle(rows[1], buff=0.08, color=YELLOW, stroke_width=2.5)
        hl2 = SurroundingRectangle(rows[2], buff=0.08, color=YELLOW, stroke_width=2.5)
        self.at(word_time("simulate", S("c1"), E("c1"), S("c1") + 2.0) - 0.1)
        self.play(Create(hl), run_time=0.5)
        self.at(word_time("step", S("c1") + 2.5, E("c1") + 0.2, S("c1") + 3.6) - 0.1)
        self.play(Transform(hl, hl2), run_time=0.5)
        self.play(FadeOut(hl), run_time=0.4)

        def dimg(arr, w, c, nearest=True):
            a_ = np.asarray(arr, np.float32)
            rgb = (to_img(a_)[..., None] * np.array([235, 235, 240], np.float32))
            m = ImageMobject(rgba_of(rgb))
            if nearest:
                m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            m.stretch_to_fit_width(w).stretch_to_fit_height(w * 1.6).move_to(c)
            return Group(m, SurroundingRectangle(m, buff=0.02, color=GREY_B, stroke_width=1.2))
        rps = d["rp_snaps"].astype(np.float32)
        Fh = d["F_hist"]
        xs3 = [-3.6, 0.0, 3.6]
        yd = -1.15
        w3 = 1.35
        stages = [(rps[25], f"grey: ${100 * Fh[25]:.1f}\\%$"), (rps[90], f"sharper: ${100 * Fh[90]:.1f}\\%$"),
                  (d["bin_opt"], f"binary: ${100 * float(d['T_opt']):.1f}\\%$")]
        st_g = Group(*[dimg(a_, w3, [x, yd, 0]) for (a_, _), x in zip(stages, xs3)])
        st_l = VGroup(*[Tex(t, font_size=28).next_to(st_g[k], DOWN, buff=0.12) for k, (_, t) in enumerate(stages)])
        st_l[2].set_color(GREEN_B)
        st_a = VGroup(*[Arrow([xs3[k] + 0.85, yd, 0], [xs3[k + 1] - 0.85, yd, 0], buff=0, stroke_width=3, color=GREY_B,
                              max_tip_length_to_length_ratio=0.2) for k in range(2)])
        fil = Circle(radius=4.0 / 50 * w3).set_stroke(YELLOW, 2.5).move_to(st_g[0][0].get_corner(UL) + np.array([0.35, -0.35, 0]))
        fil_t = Tex("min. feature", font_size=26, color=YELLOW).next_to(st_g[0], LEFT, buff=0.15).shift(0.8 * UP)
        self.at(S("c2"))
        self.play(card.animate.scale(0.8).move_to([0, 2.45, 0]), run_time=0.5)
        self.play(FadeIn(st_g[0]), FadeIn(st_l[0]), run_time=0.6)
        self.at(word_time("all", S("c2") + 1.0, E("c2"), S("c2") + 2.6) - 0.1)
        self.play(GrowArrow(st_a[0]), FadeIn(st_g[1]), FadeIn(st_l[1]), run_time=0.7)
        self.at(word_time("glass", S("c2") + 2.0, E("c2"), S("c2") + 3.6) - 0.1)
        self.play(GrowArrow(st_a[1]), FadeIn(st_g[2]), FadeIn(st_l[2]), run_time=0.7)
        self.at(word_time("feature", S("c2") + 3.0, E("c2"), S("c2") + 5.0) - 0.1)
        self.play(Create(fil), FadeIn(fil_t), run_time=0.6)
        self.play(Indicate(st_l[2], color=GREEN_B), run_time=0.8)

        # ---- c3: different starting guesses -> different designs
        alts = [d["bin_opt"], d["alt_bins"][0], d["alt_bins"][1]]
        starts = d["starts"].astype(np.float32)
        Ts = [float(d["T_opt"]), float(d["T_alts"][0]), float(d["T_alts"][1])]
        px3 = [-4.4, 0.0, 4.4]
        sw = 0.85
        st0 = Group(*[dimg(starts[k], sw, [x - 1.15, yd, 0], nearest=False) for k, x in enumerate(px3)])
        st0_l = VGroup(*[Tex("start", font_size=24, color=GREY_A).next_to(st0[k], UP, buff=0.12) for k in range(3)])
        al_g = Group(*[dimg(a_, w3, [x + 0.75, yd, 0]) for a_, x in zip(alts, px3)])
        al_a = VGroup(*[Arrow(st0[k].get_right() + 0.08 * RIGHT, al_g[k].get_left() + 0.08 * LEFT, buff=0, stroke_width=3, color=YELLOW,
                              max_tip_length_to_length_ratio=0.3) for k in range(3)])
        al_l = VGroup(*[Tex(f"${100 * T:.1f}\\%$", font_size=28).next_to(al_g[k], DOWN, buff=0.12) for k, T in enumerate(Ts)])
        self.at(S("c3"))
        self.play(FadeOut(Group(st_g, st_l, st_a, fil, fil_t)), run_time=0.5)
        self.at(word_time("starting", S("c3") + 1.0, E("c3"), S("c3") + 2.6) - 0.35)
        self.play(LaggedStart(*[FadeIn(Group(st0[k], st0_l[k])) for k in range(3)], lag_ratio=0.25), run_time=0.6)
        self.at(word_time("climb", S("c3") + 2.0, E("c3"), S("c3") + 3.4) - 0.1)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(al_a[k]), FadeIn(al_g[k], shift=0.1 * RIGHT)) for k in range(3)], lag_ratio=0.3),
                  run_time=1.4)
        self.at(word_time("good", S("c3") + 3.5, E("c3"), S("c3") + 5.2) - 0.3)
        self.play(LaggedStart(*[FadeIn(al_l[k]) for k in range(3)], lag_ratio=0.2), run_time=0.8)

        # ---- c4: the model gap: optimised under TEA, checked with a full-wave simulation
        gp = D("gap")
        xs = gp["xs"][::2]
        It, Ifd = gp["I_tea"].astype(np.float32), gp["I_fd"].astype(np.float32)
        vm = float(np.percentile(It, 99.7))
        mw, mh = 5.6, 2.8
        yc = -0.95
        mt, mt_f, mt_s = self.sweep_image(heat_rgba(It.T[::-1], vm, 0.55, flip=False), mw, np.array([-3.25, yc, 0]), mh, axis="up")
        mf, mf_f, mf_s = self.sweep_image(heat_rgba(Ifd.T[::-1], vm, 0.55, flip=False), mw, np.array([3.25, yc, 0]), mh, axis="up")
        frames = VGroup(*[Rectangle(width=mw, height=mh).move_to([cx, yc, 0]).set_stroke(GREY_B, 1.2) for cx in (-3.25, 3.25)])
        h = gp["h"][::2]

        def relief(cx):
            xx = cx - mw / 2 + (xs - xs[0]) / (xs[-1] - xs[0]) * mw
            yy = yc - mh / 2 + h / 12.0 * mh
            pts = [[xx[0], yc - mh / 2, 0]] + [[a_, b_, 0] for a_, b_ in zip(xx, yy)] + [[xx[-1], yc - mh / 2, 0]]
            return Polygon(*pts).set_stroke(GREY_A, 1.2).set_fill(GREY_B, 0.55)
        rel = VGroup(relief(-3.25), relief(3.25))
        XT, ZT = float(gp["XT"]), float(gp["ZT"])
        tgt = VGroup(*[Circle(radius=0.16).move_to([cx - mw / 2 + (XT - xs[0]) / (xs[-1] - xs[0]) * mw, yc - mh / 2 + ZT / 12 * mh, 0])
                       .set_stroke(WHITE, 2) for cx in (-3.25, 3.25)])
        tl = Tex("thin-element model (E06)", font_size=28).next_to(frames[0], UP, buff=0.15)
        tr = Tex("full Maxwell (2D FDFD)", font_size=28).next_to(frames[1], UP, buff=0.15)
        nl = Tex(f"${100 * float(gp['eta_tea']):.0f}\\%$ of the light in the focus", font_size=28).next_to(frames[0], DOWN, buff=0.15)
        nr = Tex(f"${100 * float(gp['eta_fdfd']):.0f}\\%$ of the light in the focus", font_size=28, color=C_EV).next_to(frames[1], DOWN, buff=0.15)
        g_t = Tex("one glass relief, optimised with the thin-element model", font_size=28, color=GREY_A).move_to([0, 2.55, 0])
        self.at(S("c4"))
        self.play(FadeOut(Group(st0, st0_l, al_g, al_a, al_l, card)), run_time=0.5)
        self.add(mt)
        mt_f.set_value(1.0)
        self.play(FadeIn(frames[0]), FadeIn(rel[0]), FadeIn(tl), FadeIn(g_t), run_time=0.6)
        self.play(mt_s.animate.set_value(1.0), run_time=1.6, rate_func=linear)
        self.play(FadeIn(tgt[0]), FadeIn(nl), run_time=0.5)
        self.at(word_time("loses", S("c4") + 3.0, E("c4"), S("c4") + 5.6) - 0.9)
        self.add(mf)
        mf_f.set_value(1.0)
        self.play(FadeIn(frames[1]), FadeIn(rel[1]), FadeIn(tr), run_time=0.5)
        self.play(mf_s.animate.set_value(1.0), run_time=1.4, rate_func=linear)
        self.play(FadeIn(tgt[1]), FadeIn(nr), run_time=0.5)

        # ---- c5: models that are fast and accurate
        rungs = ["thin element", "BPM", "RCWA", "FDTD / FDFD"]
        times = ["ms", "s", "min", "hours"]
        rx = np.linspace(-4.5, 4.5, 4)
        yl = -0.6
        line = Line([-5.6, yl, 0], [5.6, yl, 0]).set_stroke(GREY_B, 3)
        dots = VGroup(*[Dot([x, yl, 0], radius=0.11, color=WHITE) for x in rx])
        rl = VGroup(*[Tex(r_, font_size=30).move_to([x, yl + 0.55, 0]) for r_, x in zip(rungs, rx)])
        tl_ = VGroup(*[Tex(t_, font_size=26, color=GREY_B).move_to([x, yl - 0.5, 0]) for t_, x in zip(times, rx)])
        fa = VGroup(Arrow([-1.0, yl - 1.35, 0], [-5.4, yl - 1.35, 0], buff=0, stroke_width=4, color=GREEN, max_tip_length_to_length_ratio=0.06),
                    Tex("fast", font_size=30, color=GREEN).move_to([-3.2, yl - 1.7, 0]),
                    Arrow([1.0, yl - 1.35, 0], [5.4, yl - 1.35, 0], buff=0, stroke_width=4, color=C_KX, max_tip_length_to_length_ratio=0.06),
                    Tex("accurate", font_size=30, color=C_KX).move_to([3.2, yl - 1.7, 0]))
        need = Tex("inverse design needs both", font_size=34, color=YELLOW).move_to([0, 1.5, 0])
        self.at(S("c5"))
        self.play(FadeOut(Group(frames, rel, tgt, tl, tr, nl, nr, g_t)), mt_f.animate.set_value(0.0), mf_f.animate.set_value(0.0), run_time=0.5)
        self.remove(mt, mf)
        self.play(Create(line), FadeIn(dots), FadeIn(rl), FadeIn(tl_), run_time=0.9)
        self.play(FadeIn(fa), run_time=0.6)
        self.play(FadeIn(need), run_time=0.6)
        self.finish()


# ================================================================================ 7  recap =========
def icon(k, c, s=0.42):
    """Small line icons for the ten ideas (k = 1..10) centred at c (radius ~ s)."""
    c = np.asarray(c, float)
    size, s = s, 0.42
    g = VGroup()
    if k == 1:          # plane wave + k-circle
        g.add(Circle(radius=0.3 * s / 0.42).move_to(c).set_stroke(GREY_A, 2))
        g.add(fast_arrow(c, c + 0.3 * s / 0.42 * np.array([0.6, 0.8, 0]), C_KX, 3, 0.1))
        g.add(*[Line(c + np.array([-0.38, -0.25 + 0.17 * i, 0]) * s / 0.42, c + np.array([-0.12, 0.05 + 0.17 * i, 0]) * s / 0.42).set_stroke(C_U, 2)
                for i in range(2)])
    elif k == 2:        # lens + rays to a spot
        g.add(Ellipse(width=0.16 * s / 0.42, height=0.75 * s / 0.42).move_to(c + np.array([-0.12, 0, 0]) * s / 0.42).set_stroke(GREY_A, 2))
        for y in (-0.28, 0.0, 0.28):
            g.add(Line(c + np.array([-0.4, y, 0]) * s / 0.42, c + np.array([-0.12, y, 0]) * s / 0.42).set_stroke(C_U, 2))
            g.add(Line(c + np.array([-0.12, y, 0]) * s / 0.42, c + np.array([0.36, 0, 0]) * s / 0.42).set_stroke(C_U, 2))
    elif k == 3:        # Airy disc
        g.add(Dot(c, radius=0.07, color=YELLOW))
        g.add(*[Circle(radius=r * s / 0.42).move_to(c).set_stroke(YELLOW, 2, 0.8 - 0.3 * i) for i, r in enumerate((0.18, 0.3))])
    elif k == 4:        # fringes washing out
        xx = np.linspace(-0.38, 0.38, 60)
        g.add(VMobject().set_points_as_corners([c + np.array([x, 0.12 + 0.12 * np.cos(25 * x), 0]) * 1 for x in xx]).set_stroke(C_U, 2))
        g.add(VMobject().set_points_as_corners([c + np.array([x, -0.18 + 0.03 * np.cos(25 * x), 0]) for x in xx]).set_stroke(GREY_A, 2))
    elif k == 5:        # phase: arrow on a dial
        g.add(Circle(radius=0.3).move_to(c).set_stroke(GREY_A, 2), fast_arrow(c, c + 0.3 * np.array([np.cos(2.2), np.sin(2.2), 0]), C_IM, 3, 0.1))
    elif k == 6:        # kinoform sawtooth
        pts = []
        for i in range(4):
            x0 = -0.38 + 0.19 * i
            pts += [c + np.array([x0, -0.2, 0]), c + np.array([x0 + 0.19, 0.18, 0])]
        g.add(VMobject().set_points_as_corners(pts + [c + np.array([0.38, -0.2, 0])]).set_stroke(GREY_A, 2))
    elif k == 7:        # band gap: folded bands
        xx = np.linspace(-0.35, 0.35, 40)
        g.add(VMobject().set_points_as_corners([c + np.array([x, 0.06 + 0.5 * x * x + 0.05, 0]) for x in xx]).set_stroke(C_K, 2))
        g.add(VMobject().set_points_as_corners([c + np.array([x, -0.06 - 0.5 * x * x - 0.05 + 0.0, 0]) for x in xx]).set_stroke(C_K, 2))
    elif k == 8:        # waveguide + mode
        g.add(Line(c + np.array([-0.38, 0.12, 0]), c + np.array([0.38, 0.12, 0])).set_stroke(GREY_A, 2),
              Line(c + np.array([-0.38, -0.12, 0]), c + np.array([0.38, -0.12, 0])).set_stroke(GREY_A, 2))
        yy = np.linspace(-0.3, 0.3, 40)
        g.add(VMobject().set_points_as_corners([c + np.array([0.12 * np.exp(-(y / 0.1) ** 2) - 0.05, y, 0]) for y in yy]).set_stroke(C_U, 2))
    elif k == 9:        # Yee grid
        for i in range(3):
            for j in range(3):
                g.add(Dot(c + np.array([-0.24 + 0.24 * i, -0.24 + 0.24 * j, 0]), radius=0.03, color=GREY_A))
        g.add(fast_arrow(c + np.array([-0.12, -0.12, 0]), c + np.array([0.1, -0.12, 0]), C_U, 2.5, 0.08),
              fast_arrow(c + np.array([0.12, 0.0, 0]), c + np.array([0.12, 0.2, 0]), C_KZ, 2.5, 0.08))
    else:               # forward and backward
        g.add(fast_arrow(c + np.array([-0.36, 0.1, 0]), c + np.array([0.3, 0.1, 0]), C_U, 3, 0.12),
              fast_arrow(c + np.array([0.36, -0.1, 0]), c + np.array([-0.3, -0.1, 0]), C_ADJ, 3, 0.12))
    return g.scale(size / 0.42, about_point=c)


class Ch7Recap(Chapter):
    CH = 7
    TAG = "7\\quad Season recap"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = D("splitter")
        _, _, BIN = splitter_layers()
        XO = Cplx(d["x_opt"], 0.9 * float(d["vref"]))
        LO = Cplx(d["lam_opt"], 0.75 * float(np.abs(d["lam_opt"]).max()))

        def render(t):
            rgb = BIN[..., None] * RGB_SI + 0.85 * wave_rgb(XO.snap(t), RGB_U, RGB_U2) + 0.85 * wave_rgb(LO.snap(t), RGB_A, RGB_A2)
            return rgba_of(rgb)
        P = Panel(self, render, 0.0, 0.55, 5.6)
        one = Tex("one forward $+$ one backward simulation", font_size=34).move_to([0, -2.05, 0])
        two = Tex("$=$ the gradient for every pixel", font_size=34, color=YELLOW).move_to([0, -2.65, 0])
        self.at(S("r1"))
        self.add(P.m)
        self.play(P.fade.animate.set_value(1.0), FadeIn(one), run_time=0.8)
        self.at(word_time("gradient", S("r1"), E("r1"), S("r1") + 2.8) - 0.3)
        self.play(FadeIn(two), run_time=0.6)

        # ---- r2: the series map, lit as each idea is named
        names = ["plane waves", "lens = Fourier", "diffraction limit", "coherence", "phase", "flat optics",
                 "periodic\\\\structures", "trapped light", "Maxwell\\\\on a grid", "design by\\\\gradients"]
        keys = [("plain", "plane"), ("lenses", "lens"), ("diffraction",), ("coherence",), ("phase",), ("flat",), ("periodic",),
                ("trapped",), ("maxwell",), ("design",)]
        cols = [C_U, C_U, YELLOW, C_U, C_IM, GREY_A, C_K, C_U, C_KZ, C_ADJ]
        pos = [np.array([-6.0 + 12.0 / 9 * i, 0.72 if i % 2 == 0 else -0.72, 0]) for i in range(10)]
        rings = [Circle(radius=0.52).move_to(p).set_stroke(GREY_D, 2).set_fill(BLACK, 1) for p in pos]
        icons = [icon(i + 1, p, 0.36) for i, p in enumerate(pos)]
        nlab = []
        for i, (n_, p) in enumerate(zip(names, pos)):
            lab = VGroup(Tex(f"E{i + 1:02d}", font_size=22, color=GREY_C), Tex(n_, font_size=24, color=GREY_B)).arrange(DOWN, buff=0.06)
            lab.next_to(rings[i], UP if i % 2 == 0 else DOWN, buff=0.14)
            nlab.append(lab)
        thread = VMobject().set_points_smoothly(pos).set_stroke(GOLD, 3, 0.8)
        thread.set_z_index(-1)
        for ic in icons:
            fade_vgroup(ic, 0.25)
        self.at(S("r2"))
        self.play(FadeOut(VGroup(one, two)), P.fade.animate.set_value(0.0), run_time=0.6)
        self.remove(P.m)
        self.play(*[FadeIn(r) for r in rings], *[FadeIn(ic) for ic in icons], *[FadeIn(l_) for l_ in nlab], run_time=0.7)
        tprev = S("r2") + 1.0
        for i in range(10):
            tw = word_time(keys[i], tprev - 0.3, E("r2") + 0.3, tprev + 0.9)
            self.at(max(tw - 0.1, self.now))
            lit = icon(i + 1, pos[i], 0.36)
            self.play(rings[i].animate.set_stroke(cols[i], 3.5), Transform(icons[i], lit), nlab[i][1].animate.set_color(WHITE),
                      run_time=0.35)
            tprev = max(tw, self.now)

        # ---- r3: ten ideas, one thread
        self.at(S("r3"))
        self.play(Create(thread), run_time=1.3)
        self.play(rings[9].animate.set_stroke(C_ADJ, 6), run_time=0.4)

        # ---- r4: the closing line, over the dimmed map
        close = VGroup(Tex("Light is linear: in the right basis, the physics is simple;", font_size=44),
                       Tex("run it backward, and the design is too.", font_size=44, color=YELLOW)).arrange(DOWN, buff=0.35)
        close.move_to([0, 0.0, 0])
        self.at(S("r4") - 0.3)
        self.play(*[FadeOut(ic) for ic in icons], *[r.animate.set_stroke(opacity=0.18) for r in rings],
                  *[l_.animate.set_fill(opacity=0.18) for l_ in nlab], thread.animate.set_stroke(opacity=0.18), FadeIn(close[0]),
                  run_time=0.9)
        self.at(word_time("run", S("r4") + 2.5, E("r4"), S("r4") + 3.6) - 0.15)
        self.play(FadeIn(close[1]), run_time=0.8)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Landscape", "Ch2Adjoint", "Ch3Backward", "Ch4Autodiff", "Ch5EndToEnd", "Ch6Practice", "Ch7Recap"]
