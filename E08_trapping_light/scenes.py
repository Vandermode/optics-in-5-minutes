"""Trapping Light: Waveguides, Modes & Resonators -- chapter scenes (Optics in 5 Minutes, E08).
Render: ./render.sh ; preview: ./preview.sh Ch0Hook
All physical fields come from physics.py (data/*.npz); scenes only load them and draw.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403

C_RAY = WHITE
C_OK = GREEN
CORE_FILL, CLAD_FILL = "#4a4a4a", "#1e1e1e"


class EpChapter(Chapter):
    """Chapter with the whole-frame at() of the pilot (asm5_common): o5common's at() still waits a float duration,
    which can differ by one frame between workers when the target lies on the frame grid."""

    def at(self, t):
        x = (t - self.now) * config.frame_rate
        if x > 0.5 + 1e-6:
            k = int(np.ceil(x - 1e-6))
            self.wait((k - 0.5) / config.frame_rate)
        elif x < -0.5 - 1e-6:
            self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


def tex(s, size=30, color=WHITE, **kw):
    return Tex(s, font_size=size, color=color, **kw)


def mtex(s, size=36, color=WHITE, **kw):
    return MathTex(s, font_size=size, color=color, **kw)


# ------------------------------------------------------------------ ring field (FDFD data) ----
class RingField:
    """Live picture of the computed add-drop ring: per colour channel, brightness = |E|^gamma times a travelling crest
    pattern cos^2(arg E - omega t) (i.e. Re{E e^{-i omega t}} squared, softened), additively mixed."""
    COLS = {"R": (255, 70, 55), "G": (80, 255, 110), "B": (70, 130, 255)}

    def __init__(self, vmax=1.45, gamma=0.72, crop=None):
        r = np.load(data_path("ring.npz"))
        self.r = r
        sl = (slice(None), slice(None)) if crop is None else crop
        self.arr = r["arrival"][sl][::-1].astype(np.float32)
        core = r["core"][sl][::-1]
        self.bg = (core * 34.0).astype(np.float32)
        self.ch = {}
        for k in "RGB":
            E = r["E" + k][sl][::-1]
            env = (np.clip(np.abs(E) / vmax, 0, 1) ** gamma).astype(np.float32)
            P = np.angle(E)
            self.ch[k] = (env, np.cos(2 * P).astype(np.float32), np.sin(2 * P).astype(np.float32))
        self.shape = core.shape
        self.cols = {k: np.array(v, np.float32) for k, v in self.COLS.items()}

    def rgba(self, t, ops, fronts=None, bg_op=1.0, omega=TAU * 0.55, soft=1.0):
        rgb = np.zeros(self.shape + (3,), np.float32)
        if bg_op > 0:
            rgb += (self.bg * bg_op)[..., None]
        a = 2 * omega * t
        ca, sa = np.cos(a), np.sin(a)
        for k in "RGB":
            o = ops.get(k, 0.0)
            if o <= 0.002:
                continue
            env, c2, s2 = self.ch[k]
            b = env * (0.28 + 0.36 * (1.0 + c2 * ca + s2 * sa)) * o
            if fronts is not None and fronts.get(k, 1e9) < 60:
                b = b * np.clip((fronts[k] - self.arr) / soft, 0, 1)
            rgb += b[..., None] * self.cols[k]
        out = np.empty(self.shape + (4,), np.uint8)
        out[..., :3] = np.clip(rgb, 0, 255)
        out[..., 3] = 255
        return out

    def geometry(self, disp_h, center):
        """scene <- micron mapping for the image placed with height disp_h at center."""
        r = self.r
        xs, ys, h = r["x"], r["y"], float(r["h"])
        hum = ys[-1] - ys[0] + h
        wum = xs[-1] - xs[0] + h
        sc = disp_h / hum
        cx, cy = 0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1])
        c = np.asarray(center, float)
        return sc, wum * sc, (lambda xu, yu: c + np.array([(xu - cx) * sc, (yu - cy) * sc, 0.0]))


# ------------------------------------------------------------------ slab helpers (Ch1) --------
SLAB = np.load(data_path("slab.npz"))
N1, N2, DSL = float(SLAB["n1"]), float(SLAB["n2"]), float(SLAB["d"])
THETAS = SLAB["thetas"]
THMAX = float(SLAB["thmax"])


def slab_kg(th):
    kx = TAU * N1 * np.sin(th)
    g = TAU * np.sqrt(max((N1 * np.cos(th)) ** 2 - N2 ** 2, 1e-12))
    return kx, g


def round_trip_phase(th):
    kx, g = slab_kg(th)
    return 2 * kx * DSL - 4 * np.arctan(g / kx)


def shoot(th, xs):
    """Field across the slab (x in wavelengths, core |x| < d/2) built from the bottom: decaying tail below, matched
    in the core, and whatever the top tail has to be (it blows up unless th is a mode angle). Returns (E, B) with
    E normalised to max 1 in the core and B the coefficient of the growing exponential on top."""
    kx, g = slab_kg(th)
    a = -DSL / 2
    E = np.empty_like(xs)
    lo, hi = xs < a, xs > DSL / 2
    co = ~(lo | hi)
    E[lo] = np.exp(g * (xs[lo] - a))
    u = xs[co] - a
    E[co] = np.cos(kx * u) + (g / kx) * np.sin(kx * u)
    Et = np.cos(kx * DSL) + (g / kx) * np.sin(kx * DSL)
    dEt = -kx * np.sin(kx * DSL) + g * np.cos(kx * DSL)
    A, B = 0.5 * (Et - dEt / g), 0.5 * (Et + dEt / g)
    u = xs[hi] - DSL / 2
    E[hi] = A * np.exp(-g * u) + B * np.exp(np.minimum(g * u, 60))
    nrm = max(np.sqrt(1 + (g / kx) ** 2), 1e-9)          # max of the core cosine
    return E / nrm, B / nrm


def mode_profile(m, xs):
    E, _ = shoot(THETAS[m], xs)
    return E / np.abs(E).max()


# ============================================================ 0  hook ==========================
class Ch0Hook(EpChapter):
    CH = 0

    def construct(self):
        S = self.S
        rf = RingField()
        disp_h = 6.2
        C = np.array([-0.55, 0.2, 0])
        sc, disp_w, P = rf.geometry(disp_h, C)
        r = rf.r
        R, yb = float(r["R"]), float(r["yb"])
        ops = {k: ValueTracker(0.0) for k in "RGB"}
        fr = {k: ValueTracker(-1.0) for k in "RGB"}
        bgop = ValueTracker(0.0)
        dim = ValueTracker(1.0)

        def fn(t):
            dv = dim.get_value()
            return rf.rgba(t, {k: ops[k].get_value() * dv for k in "RGB"}, {k: fr[k].get_value() for k in "RGB"},
                           bgop.get_value() * dv)
        img, fade = self.live_image(fn, disp_w, C, disp_h)
        fade.set_value(1.0)
        self.add(img)
        x_l, x_r = C[0] - disp_w / 2, C[0] + disp_w / 2
        y_bot, y_top = P(0, yb)[1], P(0, -yb)[1]
        lab_in = VGroup(tex("in", 30), Arrow(ORIGIN, 0.55 * RIGHT, buff=0, stroke_width=4, color=GREY_A,
                                            max_tip_length_to_length_ratio=0.35)).arrange(RIGHT, buff=0.12)
        lab_in.next_to([x_l, y_bot, 0], LEFT, buff=0.1)
        lab_thr = VGroup(Arrow(ORIGIN, 0.55 * RIGHT, buff=0, stroke_width=4, color=GREY_A,
                               max_tip_length_to_length_ratio=0.35), tex("through", 30)).arrange(RIGHT, buff=0.12)
        lab_thr.next_to([x_r, y_bot, 0], RIGHT, buff=0.1)
        lab_drop = VGroup(tex("drop", 30), Arrow(0.55 * RIGHT, ORIGIN, buff=0, stroke_width=4, color=GREY_A,
                                                max_tip_length_to_length_ratio=0.35)).arrange(RIGHT, buff=0.12)
        lab_drop.next_to([x_l, y_top, 0], LEFT, buff=0.1)
        cap = tex("computed:\\\\ 2D wave equation", 24, GREY_B).move_to([5.25, -2.45, 0])
        bar = Line(P(1.0, -3.6), P(3.0, -3.6)).set_stroke(WHITE, 3)
        bar.move_to([5.25, -1.55, 0])
        bar_l = tex("$2\\,\\mu$m", 24, GREY_B).next_to(bar, UP, buff=0.08)
        ring_hi = Circle(radius=R * sc).move_to(P(0, 0)).set_stroke(YELLOW, 4)

        self.play(bgop.animate.set_value(1.0), run_time=0.8)
        # h1: light routed through a wire
        self.at(S("h1") + 0.3)
        self.play(ops["R"].animate.set_value(1.0), fr["R"].animate.set_value(9.0), run_time=3.0, rate_func=linear)
        self.play(fr["R"].animate.set_value(30.0), FadeIn(cap), FadeIn(bar), FadeIn(bar_l), run_time=2.0, rate_func=linear)
        # h2: this ring sorts colours; two colours pass straight by
        self.at(S("h2") + 0.1)
        self.play(Create(ring_hi), run_time=0.8)
        self.play(ring_hi.animate.set_stroke(opacity=0.0), run_time=0.7)
        self.remove(ring_hi)
        self.at(S("h2") + 2.5)
        self.play(ops["B"].animate.set_value(1.0), fr["B"].animate.set_value(9.0), FadeIn(lab_in), FadeIn(lab_thr),
                  run_time=2.0, rate_func=linear)
        self.play(fr["B"].animate.set_value(30.0), run_time=0.8, rate_func=linear)
        # h3: one colour drops into the ring and leaves through the other wire
        self.at(S("h3") + 0.05)
        self.play(ops["G"].animate.set_value(1.0), fr["G"].animate.set_value(12.5), run_time=2.0, rate_func=linear)
        self.play(fr["G"].animate.set_value(17.5), FadeIn(lab_drop), run_time=1.3, rate_func=linear)
        self.play(fr["G"].animate.set_value(30.0), run_time=1.2, rate_func=linear)
        # h4: the question, then the title
        card = series_title(8, "Trapping Light", "waveguides, modes and resonators", y=0.35)
        self.at(S("h4") + 0.9)
        self.play(dim.animate.set_value(0.33), FadeOut(VGroup(lab_in, lab_thr, lab_drop, cap, bar, bar_l)), run_time=0.7)
        self.play(FadeIn(card, shift=0.1 * DOWN), run_time=1.0)
        self.finish()


# ============================================================ 1  modes =========================
class Ch1Modes(EpChapter):
    CH = 1
    TAG = "1\\quad Bouncing waves make modes"

    def construct(self):
        S = self.S
        self.show_tag()
        U1 = 1.3 / DSL                          # scene units per wavelength
        ZL, ZR = -6.6, 2.5
        yc, hc = 1.4, DSL * U1                  # core centre and height
        clad = 1.0
        yb_, yt_ = yc - hc / 2, yc + hc / 2
        core = Rectangle(width=ZR - ZL, height=hc).move_to([(ZL + ZR) / 2, yc, 0]).set_stroke(width=0).set_fill(CORE_FILL, 1)
        cl_t = Rectangle(width=ZR - ZL, height=clad).next_to(core, UP, buff=0).set_stroke(width=0).set_fill(CLAD_FILL, 1)
        cl_b = Rectangle(width=ZR - ZL, height=clad).next_to(core, DOWN, buff=0).set_stroke(width=0).set_fill(CLAD_FILL, 1)
        walls = VGroup(Line([ZL, yb_, 0], [ZR, yb_, 0]), Line([ZL, yt_, 0], [ZR, yt_, 0])).set_stroke(GREY_B, 2)
        lab_core = tex("core, $n_1$", 30).next_to([ZR, yc, 0], RIGHT, buff=0.3)
        lab_clad = VGroup(tex("cladding,", 28, GREY_A), tex("$n_2 < n_1$", 28, GREY_A)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        lab_clad.next_to([ZR, yt_ + clad / 2, 0], RIGHT, buff=0.3)
        lab_clad2 = tex("cladding", 28, GREY_A).next_to([ZR, yb_ - clad / 2, 0], RIGHT, buff=0.3)
        self.at(S("m1"))
        self.play(FadeIn(cl_t), FadeIn(cl_b), FadeIn(core), Create(walls), run_time=1.0)
        self.at(S("m1") + 1.5)
        self.play(FadeIn(lab_core), run_time=0.5)
        self.at(S("m1") + 4.6)
        self.play(FadeIn(lab_clad), FadeIn(lab_clad2), run_time=0.5)

        # ---------------- m2: the zigzag ray (angle tracker th, reveal tracker rv)
        th = ValueTracker(float(THETAS[1]))
        rv = ValueTracker(0.0)
        ray_op = ValueTracker(1.0)

        def zig_points(theta, z0=ZL + 0.05):
            pts = [np.array([z0, yb_, 0.0])]
            run = hc / np.tan(theta)
            up = True
            while pts[-1][0] < ZR:
                p = pts[-1]
                q = np.array([p[0] + run, yt_ if up else yb_, 0.0])
                if q[0] > ZR:
                    f = (ZR - p[0]) / run
                    q = p + f * (q - p)
                pts.append(q)
                up = not up
            return np.array(pts)

        def ray():
            o = ray_op.get_value()
            if o <= 0.003:
                return VGroup()
            pts = zig_points(th.get_value())
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            Ltot = seg.sum()
            Lr = rv.get_value() * Ltot
            if Lr < 1e-3:
                return VGroup()
            cum = np.concatenate([[0], np.cumsum(seg)])
            k = int(np.searchsorted(cum, Lr) - 1)
            k = min(max(k, 0), len(seg) - 1)
            tip = pts[k] + (pts[k + 1] - pts[k]) * (Lr - cum[k]) / seg[k]
            pp = np.vstack([pts[:k + 1], tip])
            g = VGroup(VMobject().set_points_as_corners(pp).set_stroke(C_RAY, 3.5, opacity=o))
            if rv.get_value() < 0.999:
                d = pp[-1] - pp[-2]
                g.add(fast_arrow(pp[-1] - 0.3 * d / max(np.linalg.norm(d), 1e-6), pp[-1], C_RAY, 3.5, 0.2, o))
            return g
        rayg = always_redraw(ray)
        tir = tex("total internal reflection", 26, GREY_A)
        self.at(S("m2") + 0.1)
        self.add(rayg)
        self.play(rv.animate.set_value(1.0), run_time=4.4, rate_func=linear)
        p1 = zig_points(float(THETAS[1]))[1]
        tir.move_to([p1[0], yt_ + 0.45, 0])
        tir_ar = Arrow(tir.get_bottom() + 0.02 * DOWN, p1 + 0.05 * UP, buff=0.04, stroke_width=3, color=GREY_A,
                       max_tip_length_to_length_ratio=0.3)
        self.play(FadeIn(tir), GrowArrow(tir_ar), run_time=0.6)

        # ---------------- m3: two crossing plane waves (crests of the up- and down-going waves)
        kx1, g1 = slab_kg(float(THETAS[1]))
        beta1 = TAU * N1 * np.cos(float(THETAS[1]))
        kxs, bs = kx1 / U1, beta1 / U1                    # rad per scene unit
        cph = kx1 * DSL / 2 - np.arctan(g1 / kx1)
        om = 3.6
        cr_op = ValueTracker(0.0)

        def crests():
            o = cr_op.get_value()
            g = VGroup()
            if o <= 0.003:
                return g
            t = self.now
            n_lo = int(np.floor((-om * t - 6.0) / TAU)) - 1          # crests n with ZL <= z <= ZR
            n_hi = int(np.ceil((bs * (ZR - ZL) - om * t + 6.0) / TAU)) + 1
            for sgn, col in ((1, BLUE_C), (-1, BLUE_A)):
                for n in range(n_lo, n_hi + 1):
                    # sgn=+1: kx x + c + beta z - om t = 2 pi n  (up);  sgn=-1: -kx x - c + beta z - om t = 2 pi n (down)
                    zz = []
                    for xx in (-hc / 2, hc / 2):
                        zz.append(ZL + (TAU * n + om * t - sgn * (kxs * xx + cph)) / bs)
                    pa, pb = np.array([zz[0], yb_, 0]), np.array([zz[1], yt_, 0])
                    seg = clip_to_box(pa, pb, (ZL + ZR) / 2, yc, (ZR - ZL) / 2, hc / 2 + 1e-6)
                    if seg:
                        g.add(Line(*seg).set_stroke(col, 2.5, opacity=0.85 * o))
            return g
        crg = always_redraw(crests)
        kp = np.array([-5.7, -1.3, 0])
        L_ = 2.0
        ku = np.array([np.cos(THETAS[1]), np.sin(THETAS[1]), 0])
        kd = np.array([np.cos(THETAS[1]), -np.sin(THETAS[1]), 0])
        kb, kk = L_ * ku[0], L_ * ku[1]
        karrs = VGroup(Arrow(kp, kp + L_ * ku, buff=0, stroke_width=5, color=WHITE, max_tip_length_to_length_ratio=0.12),
                       Arrow(kp, kp + L_ * kd, buff=0, stroke_width=5, color=WHITE, max_tip_length_to_length_ratio=0.12))
        kxl = DoubleArrow(kp + kb * RIGHT + kk * DOWN, kp + kb * RIGHT + kk * UP, buff=0, stroke_width=4, color=C_KX,
                          max_tip_length_to_length_ratio=0.1).shift(0.18 * RIGHT)
        bl = Arrow(kp + (kk + 0.4) * DOWN, kp + (kk + 0.4) * DOWN + kb * RIGHT, buff=0, stroke_width=4, color=C_KZ,
                   max_tip_length_to_length_ratio=0.1)
        comp = VGroup(kxl, bl)
        kl = VGroup(tex("$\\pm k_x$: across", 30, C_KX).next_to(kxl, RIGHT, buff=0.15),
                    tex("$\\beta$: along", 30, C_KZ).next_to(bl, RIGHT, buff=0.15))
        kwl = VGroup(tex("up", 26, GREY_A).next_to(kp + 0.5 * L_ * ku, UL, buff=0.06),
                     tex("down", 26, GREY_A).next_to(kp + 0.5 * L_ * kd, DL, buff=0.06))
        self.at(S("m3") + 1.2)
        self.add(crg)
        self.play(cr_op.animate.set_value(1.0), ray_op.animate.set_value(0.35), FadeOut(tir), FadeOut(tir_ar), run_time=1.0)
        self.at(S("m3") + 3.4)
        self.play(GrowArrow(karrs[0]), FadeIn(kwl[0]), run_time=0.5)
        self.play(GrowArrow(karrs[1]), FadeIn(kwl[1]), run_time=0.5)
        self.play(FadeIn(kxl), GrowArrow(bl), FadeIn(kl), run_time=0.8)

        # ---------------- m4: standing across, travelling along (live computed mode field)
        nzp, nxp = 640, 230
        zs_img = np.linspace(ZL, ZR, nzp)
        xs_img = np.linspace(yt_ + clad, yb_ - clad, nxp)          # top row first
        prof_img = mode_profile(1, (xs_img - yc) / U1).astype(np.float32)

        def field2(t):                       # = sum of the two crest families above (same phase reference)
            ph = bs * (zs_img - ZL) - om * t
            v = prof_img[:, None] * np.cos(ph)[None, :]
            return diverge_rgba(v.astype(np.float32), 1.05, flip=False)
        fimg, ffade = self.live_image(field2, ZR - ZL, [(ZL + ZR) / 2, yc, 0], hc + 2 * clad)
        # profile panel (shares the vertical axis)
        PX, PA = 4.55, 1.15
        xs_p = np.linspace(-(DSL / 2 + clad / U1), DSL / 2 + clad / U1, 260)
        p_axis = Line([PX, yb_ - clad, 0], [PX, yt_ + clad, 0]).set_stroke(GREY_C, 1.5)
        p_walls = VGroup(fast_dashed([ZR + 0.15, yb_, 0], [PX + 1.6, yb_, 0], 0.07, GREY_B, 1.5),
                         fast_dashed([ZR + 0.15, yt_, 0], [PX + 1.6, yt_, 0], 0.07, GREY_B, 1.5))
        p_op = ValueTracker(0.0)
        fits_t = ValueTracker(0.0)

        def profile():
            o = p_op.get_value()
            if o <= 0.003:
                return VGroup()
            E, B = shoot(th.get_value(), xs_p)
            ok = abs(B) < 1e-3
            col = interpolate_color(C_U, C_OK, fits_t.get_value()) if ok else GREY_A
            big = np.where(np.abs(E) > 1.3)[0]
            if len(big) == 0:
                return poly_line(PX + PA * E, yc + xs_p * U1, col, 4, o)
            k = big[0]
            g = VGroup(poly_line(PX + PA * E[:k], yc + xs_p[:k] * U1, col, 4, o))
            p2 = np.array([PX + PA * np.sign(E[k]) * 1.3, yc + xs_p[k] * U1, 0])
            p1 = np.array([PX + PA * E[k - 2], yc + xs_p[k - 2] * U1, 0])
            g.add(fast_arrow(p1, p2 + 0.15 * (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-6), C_EV, 4, 0.2, o))
            return g
        prof = always_redraw(profile)
        lab_sw = tex("standing wave\\\\across", 26, GREY_A).move_to([PX + 0.25, yt_ + clad + 0.45, 0])
        lab_tr = VGroup(tex("travels along", 26, GREY_A), Arrow(ORIGIN, 0.7 * RIGHT, buff=0, stroke_width=4, color=GREY_A,
                                                                 max_tip_length_to_length_ratio=0.3)).arrange(RIGHT, buff=0.15)
        lab_tr.move_to([ZR - 1.6, yt_ + clad + 0.3, 0])
        self.at(S("m4"))
        self.add(fimg)
        self.bring_to_front(walls)
        self.add(prof)
        self.play(ffade.animate.set_value(1.0), cr_op.animate.set_value(0.0), FadeOut(karrs), FadeOut(comp), FadeOut(kwl),
                  FadeOut(kl), FadeOut(lab_core), FadeOut(lab_clad), FadeOut(lab_clad2), p_op.animate.set_value(1.0),
                  Create(p_axis), FadeIn(p_walls), run_time=1.0)
        self.remove(crg)
        self.play(FadeIn(lab_sw), run_time=0.5)
        self.at(S("m4") + 2.2)
        self.play(FadeIn(lab_tr), run_time=0.5)

        # ---------------- m5: the transverse round trip (dial)
        DC, DR = np.array([0.4, -1.75, 0]), 0.78
        dial_ring = Circle(radius=DR).move_to(DC).set_stroke(GREY_B, 2)
        tick = Line(DC + (DR - 0.12) * RIGHT, DC + (DR + 0.12) * RIGHT).set_stroke(C_OK, 5)
        dial_lab = tex("round-trip phase", 28, GREY_A).next_to(dial_ring, UP, buff=0.18)
        s_rt = ValueTracker(0.0)               # 0..1 along the transverse round trip (up, reflect, down, reflect)
        mode_dial = ValueTracker(0.0)          # 0: round-trip animation; 1: full round-trip phase of the current angle
        z_rt = ZR - 1.1

        def rt_phase(s):
            kx, g = slab_kg(th.get_value())
            phr = -2 * np.arctan(g / kx)
            # segments: up (0..0.42), reflection (0.42..0.5), down (0.5..0.92), reflection (0.92..1)
            if s <= 0.42:
                return kx * DSL * s / 0.42
            if s <= 0.5:
                return kx * DSL + phr * (s - 0.42) / 0.08
            if s <= 0.92:
                return kx * DSL + phr + kx * DSL * (s - 0.5) / 0.42
            return 2 * kx * DSL + phr + phr * (s - 0.92) / 0.08

        def dial():
            md = mode_dial.get_value()
            ang = rt_phase(s_rt.get_value()) if md < 0.5 else round_trip_phase(th.get_value())
            m_near = np.round(ang / TAU)
            ok = abs(ang - TAU * m_near) < 0.02 and (md > 0.5 or s_rt.get_value() > 0.999)
            col = interpolate_color(YELLOW, C_OK, fits_t.get_value()) if ok else YELLOW
            return fast_arrow(DC, DC + DR * np.array([np.cos(ang), np.sin(ang), 0]), col, 6, 0.2)
        dl = always_redraw(dial)

        def rt_path():
            s = s_rt.get_value()
            if s <= 1e-4 or mode_dial.get_value() > 0.5:
                return VGroup()
            g = VGroup()
            if s <= 0.46:
                yy = yb_ + hc * min(s / 0.42, 1)
                g.add(Line([z_rt - 0.1, yb_, 0], [z_rt - 0.1, yy, 0]).set_stroke(YELLOW, 5), Dot([z_rt - 0.1, yy, 0], 0.08, color=YELLOW))
            else:
                g.add(Line([z_rt - 0.1, yb_, 0], [z_rt - 0.1, yt_, 0]).set_stroke(YELLOW, 5, opacity=0.5))
                yy = yt_ - hc * min(max((s - 0.5) / 0.42, 0), 1)
                g.add(Line([z_rt + 0.1, yt_, 0], [z_rt + 0.1, yy, 0]).set_stroke(YELLOW, 5), Dot([z_rt + 0.1, yy, 0], 0.08, color=YELLOW))
            return g
        rtp = always_redraw(rt_path)
        ok_lab = tex("back in step", 28, C_OK).next_to(dial_ring, RIGHT, buff=0.3)
        self.at(S("m5") + 0.1)
        self.play(ffade.animate.set_value(0.0), FadeOut(lab_sw), FadeOut(lab_tr), ray_op.animate.set_value(0.55), run_time=0.8)
        fimg.clear_updaters()
        self.remove(fimg)
        self.at(S("m5") + 1.6)
        self.add(dl, rtp)
        self.play(Create(dial_ring), FadeIn(tick), FadeIn(dial_lab), run_time=0.6)
        self.play(s_rt.animate.set_value(1.0), run_time=3.4, rate_func=linear)
        self.play(fits_t.animate.set_value(1.0), FadeIn(ok_lab), run_time=0.5)

        # ---------------- m6: 2 k_x d + 2 phi_r = 2 pi m ; the angle slider
        eq = MathTex("2", "k_x", "d", "+", "2\\varphi_r", "=", "2\\pi m", font_size=44)
        eq[1].set_color(C_KX)
        eq[6].set_color(C_OK)
        eq.move_to([4.15, -1.35, 0])
        eq_n = tex("$\\varphi_r$: phase shift at each reflection", 24, GREY_B).next_to(eq, DOWN, buff=0.22)
        SL0, SL1, SLY = -6.0, -2.0, -1.75
        th0 = np.radians(4.0)
        def th2x(t_):
            return SL0 + (SL1 - SL0) * (t_ - th0) / (THMAX - th0)
        sl_line = Line([SL0, SLY, 0], [SL1, SLY, 0]).set_stroke(GREY_B, 3)
        sl_lab = tex("ray angle $\\theta$", 28, GREY_A).next_to(sl_line, UP, buff=0.32).align_to(sl_line, LEFT)
        sl_end = tex("critical", 22, GREY_B).next_to([SL1, SLY, 0], RIGHT, buff=0.15)
        knob = always_redraw(lambda: Dot([th2x(np.clip(th.get_value(), th0, THMAX)), SLY, 0], 0.11, color=YELLOW))
        self.at(S("m6") + 0.1)
        self.play(FadeOut(ok_lab), FadeOut(rtp), run_time=0.3)
        self.remove(rtp)
        self.play(Write(eq), run_time=1.2)
        self.play(FadeIn(eq_n), run_time=0.4)
        mode_dial.set_value(1.0)
        self.at(S("m6") + 2.3)
        self.add(knob)
        self.play(FadeIn(sl_line), FadeIn(sl_lab), FadeIn(sl_end), FadeIn(knob), th.animate.set_value(th0), run_time=0.9)
        ticks = VGroup()
        mlabs = VGroup()
        self.at(S("m6") + 3.3)
        for m in range(3):
            self.play(th.animate.set_value(float(THETAS[m])), run_time=0.95, rate_func=smooth)
            tk = Line([th2x(THETAS[m]), SLY - 0.17, 0], [th2x(THETAS[m]), SLY + 0.17, 0]).set_stroke(C_OK, 5)
            ml = mtex(f"m={m}", 26, C_OK).next_to(tk, DOWN, buff=0.08)
            ticks.add(tk)
            mlabs.add(ml)
            self.play(FadeIn(tk, scale=1.6), FadeIn(ml), run_time=0.3)
        self.play(th.animate.set_value(THMAX - 0.002), run_time=0.6)

        # ---------------- m7: the three modes, live, with evanescent tails
        self.at(S("m7") + 0.45)
        for mob in (rayg, knob, dl, prof):
            mob.clear_updaters()
        self.play(FadeOut(VGroup(core, cl_t, cl_b, walls, rayg, p_axis, p_walls, prof, dial_ring, tick, dial_lab, eq, eq_n,
                                 sl_line, sl_lab, sl_end, knob, dl, ticks, mlabs)), run_time=0.5)
        strips = []
        hc3, cl3 = 0.62, 0.42
        U3 = hc3 / DSL
        ycs = [2.35, 0.55, -1.25]
        Z3L, Z3R = -6.5, 2.2
        nz3, nx3 = 560, 120
        zs3 = np.linspace(Z3L, Z3R, nz3)
        PX3, PA3 = 4.3, 0.85
        stat = VGroup()
        imgs = []
        for m, y0 in enumerate(ycs):
            xs3 = np.linspace(DSL / 2 + cl3 / U3, -(DSL / 2 + cl3 / U3), nx3)
            pr = mode_profile(m, xs3).astype(np.float32)
            bm = TAU * N1 * np.cos(THETAS[m]) / U1                 # same scene wavelength scale as before (U1)

            def fm(t, pr=pr, bm=bm):
                v = pr[:, None] * np.cos(bm * (zs3 - Z3L) - om * t)[None, :]
                return diverge_rgba(v.astype(np.float32), 1.05, flip=False)
            im, fd = self.live_image(fm, Z3R - Z3L, [(Z3L + Z3R) / 2, y0, 0], hc3 + 2 * cl3)
            imgs.append((im, fd))
            stat.add(Line([Z3L, y0 - hc3 / 2, 0], [Z3R, y0 - hc3 / 2, 0]).set_stroke(GREY_B, 1.5),
                     Line([Z3L, y0 + hc3 / 2, 0], [Z3R, y0 + hc3 / 2, 0]).set_stroke(GREY_B, 1.5))
            xp = np.linspace(-(DSL / 2 + cl3 / U3), DSL / 2 + cl3 / U3, 240)
            E = mode_profile(m, xp)
            inside = np.abs(xp) <= DSL / 2
            stat.add(Line([PX3, y0 - hc3 / 2 - cl3, 0], [PX3, y0 + hc3 / 2 + cl3, 0]).set_stroke(GREY_C, 1.5))
            stat.add(poly_line(PX3 + PA3 * E[inside], y0 + xp[inside] * U3, C_U, 4))
            for sel in (xp < -DSL / 2, xp > DSL / 2):
                ii = np.where(sel)[0]
                ii = np.r_[ii, ii[-1] + 1] if sel[0] else np.r_[ii[0] - 1, ii]
                stat.add(poly_line(PX3 + PA3 * E[ii], y0 + xp[ii] * U3, C_EV, 4))
            stat.add(mtex(f"m={m}", 30).move_to([6.1, y0, 0]))
        ev_lab = tex("evanescent tails", 28, C_EV).move_to([4.5, -2.75, 0])
        y2 = ycs[2]
        ev_ar = VGroup(Arrow(ev_lab.get_top() + 0.25 * LEFT, [PX3 + 0.05, y2 - hc3 / 2 - 0.22, 0], buff=0.05, stroke_width=3,
                             color=C_EV, max_tip_length_to_length_ratio=0.25))
        for im, _ in imgs:
            self.add(im)
        self.play(*[fd.animate.set_value(1.0) for _, fd in imgs], FadeIn(stat), run_time=0.9)
        self.at(S("m7") + 4.0)
        self.play(FadeIn(ev_lab), GrowArrow(ev_ar[0]), run_time=0.6)
        self.finish()


# ============================================================ 2  eigenvectors ===================
class Ch2Eigen(EpChapter):
    CH = 2
    TAG = "2\\quad Modes are eigenvectors"

    def construct(self):
        S = self.S
        self.show_tag()
        sv = np.load(data_path("solver.npz"))
        n1, n2, d = float(sv["n1"]), float(sv["n2"]), float(sv["d"])
        # ---------------- level diagram (right): index profile n(x), modes drawn at their effective index
        XL, XR, Xc = 1.2, 5.6, 3.4
        sx = (XR - XL) / 4.0                       # scene units per wavelength (x from -2 to 2)
        NL, NH, YL, YH = 1.40, 2.06, -2.75, 1.55
        def ny(n):
            return YL + (n - NL) / (NH - NL) * (YH - YL)
        def nx_(x):
            return Xc + x * sx
        ax_x = Arrow([XL - 0.1, YL, 0], [XR + 0.35, YL, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03)
        ax_n = Arrow([XL - 0.1, YL, 0], [XL - 0.1, YH + 0.55, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04)
        lab_x = mtex("x", 30, GREY_B).next_to(ax_x, RIGHT, buff=0.08)
        lab_n = mtex("n", 32, GREY_B).next_to(ax_n, UP, buff=0.08)
        prof_pts = [[nx_(-2.0), ny(n2), 0], [nx_(-d / 2), ny(n2), 0], [nx_(-d / 2), ny(n1), 0], [nx_(d / 2), ny(n1), 0],
                    [nx_(d / 2), ny(n2), 0], [nx_(2.0), ny(n2), 0]]
        nprof = VMobject().set_points_as_corners(prof_pts).set_stroke(GREY_B, 4)
        nfill = Polygon(*prof_pts[1:5]).set_stroke(width=0).set_fill(GREY_D, 0.45)
        t_n1 = mtex("n_1", 30, GREY_A).next_to([XL - 0.1, ny(n1), 0], LEFT, buff=0.12)
        t_n2 = mtex("n_2", 30, GREY_A).next_to([XL - 0.1, ny(n2), 0], LEFT, buff=0.12)
        tk = VGroup(Line([XL - 0.18, ny(n1), 0], [XL - 0.02, ny(n1), 0]), Line([XL - 0.18, ny(n2), 0], [XL - 0.02, ny(n2), 0])).set_stroke(GREY_B, 2)
        lab_core = tex("core", 26, GREY_A).move_to([nx_(0), ny(n1) + 0.25, 0])
        self.at(S("e1"))
        self.play(GrowArrow(ax_x), GrowArrow(ax_n), FadeIn(lab_x), FadeIn(lab_n), run_time=0.7)
        self.play(Create(nprof), FadeIn(nfill), FadeIn(t_n1), FadeIn(t_n2), FadeIn(tk), FadeIn(lab_core), run_time=1.2)

        # ---------------- e2: mode ansatz -> eigenvalue problem
        L1 = MathTex("E(x,z)", "=", "E(x)", "\\,e^{i\\beta z}", font_size=44).move_to([-3.45, 2.7, 0])
        L1[0].set_color(C_U)
        L1[2].set_color(C_U)
        L1[3].set_color(C_KZ)
        wave = MathTex("\\frac{\\partial^2 E}{\\partial x^2}", "+", "\\frac{\\partial^2 E}{\\partial z^2}", "+",
                       "k_0^2\\, n(x)^2 E", "=", "0", font_size=38).move_to([-3.45, 1.5, 0])
        wave_l = tex("wave equation", 24, GREY_B).next_to(wave, UP, buff=0.12)
        L3 = MathTex("\\Big[\\frac{d^2}{dx^2}+k_0^2\\, n(x)^2\\Big]", "E", "=", "\\beta^2", "E", font_size=44).move_to([-3.45, 1.5, 0])
        L3[1].set_color(C_U)
        L3[4].set_color(C_U)
        L3[3].set_color(C_KZ)
        bH = Brace(L3[0], DOWN, buff=0.08, color=GREY_B)
        bH_l = tex("a matrix, $H$", 26, GREY_A).next_to(bH, DOWN, buff=0.06)
        bE = Brace(L3[3], DOWN, buff=0.08, color=C_KZ)
        bE_l = tex("eigenvalue", 26, C_KZ).next_to(bE, DOWN, buff=0.06)
        self.at(S("e2") + 0.3)
        self.play(Write(L1), run_time=1.4)
        self.at(S("e2") + 4.3)
        self.play(FadeIn(wave), FadeIn(wave_l), run_time=0.7)
        self.at(S("e2") + 6.3)
        self.play(FadeOut(wave_l), FadeTransform(wave, L3), run_time=1.0)
        self.play(GrowFromCenter(bH), FadeIn(bH_l), GrowFromCenter(bE), FadeIn(bE_l), run_time=0.6)

        # ---------------- e3: sample on a grid -> tridiagonal matrix + diagonal
        xc = sv["xc"]
        grid = VGroup(*[Dot([nx_(x), YL, 0], radius=0.03, color=YELLOW) for x in xc])
        D2, Dn = sv["D2"], sv["Dn"]
        Hm = D2 + Dn
        def mimg(M, scale):
            v = np.clip(M / scale * 0.5 + 0.5, 0, 1)
            rgb = DIVERGE[(v * 1023).astype(int)]
            out = np.empty(M.shape + (4,), np.uint8)
            out[..., :3] = rgb
            out[..., 3] = 255
            out[np.abs(M) < 1e-9] = (14, 14, 14, 255)
            return np.kron(out, np.ones((6, 6, 1), np.uint8))
        msz = 1.5
        my = -1.2
        mxs = [-5.6, -3.45, -1.3]
        mats = []
        for M, sc_, xm in ((D2, np.abs(D2).max(), mxs[0]), (Dn, np.abs(Dn).max(), mxs[1]), (Hm, np.abs(Hm).max(), mxs[2])):
            im, fd = self.static_image(mimg(M, sc_), msz, [xm, my, 0], msz, nearest=True)
            mats.append((im, fd, SurroundingRectangle(im, buff=0.03, color=GREY_B, stroke_width=1.5)))
        plus = mtex("+", 44).move_to([(mxs[0] + mxs[1]) / 2, my, 0])
        eqs = mtex("=", 44).move_to([(mxs[1] + mxs[2]) / 2, my, 0])
        ml = [MathTex("\\tfrac{1}{\\Delta x^2}", "[\\,1\;\\, {-2}\;\\, 1\\,]", font_size=28).next_to(mats[0][0], DOWN, buff=0.16),
              MathTex("k_0^2\\, n(x)^2", font_size=28).next_to(mats[1][0], DOWN, buff=0.16),
              MathTex("H", font_size=32).next_to(mats[2][0], DOWN, buff=0.16)]
        ml_t = [tex("second derivative", 24, GREY_B).next_to(ml[0], DOWN, buff=0.08),
                tex("index profile", 24, GREY_B).next_to(ml[1], DOWN, buff=0.08)]
        self.at(S("e3") + 0.2)
        self.play(LaggedStart(*[FadeIn(g, scale=0.5) for g in grid], lag_ratio=0.03), run_time=1.3)
        self.at(S("e3") + 2.4)
        self.add(mats[0][0])
        self.play(mats[0][1].animate.set_value(1.0), Create(mats[0][2]), FadeIn(ml[0]), FadeIn(ml_t[0]), run_time=0.8)
        self.at(S("e3") + 5.2)
        self.add(mats[1][0])
        self.play(mats[1][1].animate.set_value(1.0), Create(mats[1][2]), FadeIn(ml[1]), FadeIn(ml_t[1]), FadeIn(plus), run_time=0.8)
        self.add(mats[2][0])
        self.play(mats[2][1].animate.set_value(1.0), Create(mats[2][2]), FadeIn(ml[2]), FadeIn(eqs), run_time=0.8)

        # ---------------- e4: eigenvectors = mode shapes, eigenvalues -> beta
        x_f = sv["x"]
        prof = sv["prof"]
        neff = sv["neff"]
        sel = np.abs(x_f) <= 2.0
        amp = 0.42
        modes = VGroup()
        levels = VGroup()
        for m in range(len(neff)):
            yy = ny(neff[m])
            levels.add(fast_dashed([XL - 0.1, yy, 0], [XR, yy, 0], 0.07, C_KZ, 1.5, 0.7))
            modes.add(poly_line(nx_(x_f[sel]), yy + amp * prof[sel, m], C_U, 4))
        eig_ar = Arrow(mats[2][2].get_right() + 0.08 * RIGHT, [XL - 0.25, my + 0.75, 0], buff=0, stroke_width=4,
                       color=GREY_A, max_tip_length_to_length_ratio=0.25)
        eig_l = MathTex("\\mathrm{eig}", font_size=30, color=GREY_A).next_to(eig_ar.get_center(), UP, buff=0.12)
        eig_t = tex("eigenvectors = mode shapes", 28, GREY_A).move_to([Xc, YH + 0.75, 0])
        blabs = VGroup(*[mtex(f"\\beta_{m}", 30, C_KZ).next_to([XR, ny(neff[m]), 0], RIGHT, buff=0.12) for m in range(len(neff))])
        self.at(S("e4"))
        self.play(GrowArrow(eig_ar), FadeIn(eig_l), FadeOut(grid), FadeIn(eig_t), run_time=0.7)
        self.play(LaggedStart(*[Create(mm) for mm in modes], lag_ratio=0.35), run_time=1.8)
        self.at(S("e4") + 3.6)
        self.play(FadeIn(levels), FadeIn(blabs), run_time=0.8)

        # ---------------- e5: effective index between n2 and n1
        neff_l = MathTex("n_{\\rm eff}", "=", "\\beta / k_0", font_size=34).next_to(ax_n, UP, buff=0.1)
        neff_l[0].set_color(C_KZ)
        band = Rectangle(width=0.22, height=ny(n1) - ny(n2)).move_to([XL - 0.1, (ny(n1) + ny(n2)) / 2, 0])
        band.set_stroke(width=0).set_fill(C_KZ, 0.35)
        between = tex("guided: $n_2 < n_{\\rm eff} < n_1$", 30, C_KZ).next_to(neff_l, RIGHT, buff=0.7)
        self.at(S("e5") + 0.2)
        self.play(FadeOut(lab_n), FadeOut(eig_t), run_time=0.35)
        self.play(FadeIn(neff_l), run_time=0.5)
        self.at(S("e5") + 2.4)
        self.play(FadeIn(band), FadeIn(between), Indicate(t_n1, color=C_KZ), Indicate(t_n2, color=C_KZ), run_time=1.0)

        # ---------------- e6: any input = sum of modes, each with its own phase
        SUM = MathTex("E(x,z)", "=", "\\sum_m", "c_m", "\\,E_m(x)", "\\,e^{i\\beta_m z}", font_size=44).move_to([-3.45, 1.85, 0])
        SUM[0].set_color(C_U)
        SUM[4].set_color(C_U)
        SUM[5].set_color(C_KZ)
        rates = [3.0 * (neff[m] - n2) / (neff[0] - n2) for m in range(len(neff))]
        dial_op = ValueTracker(0.0)
        t_d0 = S("e6") + 1.2

        def dials():
            o = dial_op.get_value()
            g = VGroup()
            if o <= 0.003:
                return g
            for m in range(len(neff)):
                c = np.array([6.35, ny(neff[m]), 0])
                a = rates[m] * (self.now - t_d0)
                g.add(Circle(radius=0.27).move_to(c).set_stroke(GREY_B, 1.5, opacity=o),
                      fast_arrow(c, c + 0.27 * np.array([np.cos(a), np.sin(a), 0]), C_KZ, 3.5, 0.11, o))
            return g
        dl = always_redraw(dials)
        row1 = VGroup(tex("free space (episode 1):", 30, GREY_A), tex("plane waves,", 30), mtex("e^{ik_z z}", 34, C_KZ))
        row2 = VGroup(tex("waveguide:", 30, GREY_A), tex("modes,", 30), mtex("e^{i\\beta_m z}", 34, C_KZ))
        for rw in (row1, row2):
            rw.arrange(RIGHT, buff=0.2)
        row1.move_to([-3.45, 0.2, 0])
        row2.next_to(row1, DOWN, buff=0.35).align_to(row1, RIGHT)
        self.at(S("e6"))
        old = VGroup(L1, L3, bH, bH_l, bE, bE_l, plus, eqs, *ml, *ml_t, *[m_[2] for m_ in mats], eig_ar, eig_l)
        self.play(FadeOut(old), *[m_[1].animate.set_value(0.0) for m_ in mats], run_time=0.6)
        for m_ in mats:
            m_[0].clear_updaters()
            self.remove(m_[0])
        self.play(Write(SUM), run_time=1.3)
        self.add(dl)
        self.play(dial_op.animate.set_value(1.0), FadeOut(blabs), run_time=0.6)
        self.at(S("e6") + 4.5)
        self.play(FadeIn(row1, shift=0.1 * UP), run_time=0.7)
        self.play(FadeIn(row2, shift=0.1 * UP), run_time=0.7)

        # ---------------- e7: multimode interference and self-imaging (computed by exactly that sum)
        mm = np.load(data_path("mmi.npz"))
        I = mm["I"]
        xm, zm = mm["x"], mm["z"]
        Wg, Lpi = float(mm["W"]), float(mm["Lpi"])
        PW, PH = 12.4, 2.9
        PC = np.array([0.15, 0.35, 0])
        img = heat_rgba(I, I.max() * 0.85, gamma=0.62)
        mim, mfd, msw = self.sweep_image(img, PW, PC, PH, axis="right")
        sxm = PH / (xm[-1] - xm[0])
        gl = VGroup(Line([PC[0] - PW / 2, PC[1] + Wg / 2 * sxm, 0], [PC[0] + PW / 2, PC[1] + Wg / 2 * sxm, 0]),
                    Line([PC[0] - PW / 2, PC[1] - Wg / 2 * sxm, 0], [PC[0] + PW / 2, PC[1] - Wg / 2 * sxm, 0])).set_stroke(GREY_B, 2)
        def zx(z):
            return PC[0] - PW / 2 + z / zm[-1] * PW
        x0 = float(mm["x0"])
        inp = VGroup(Arrow([PC[0] - PW / 2 - 0.55, PC[1] + x0 * sxm, 0], [PC[0] - PW / 2 - 0.02, PC[1] + x0 * sxm, 0], buff=0,
                           stroke_width=5, color=YELLOW, max_tip_length_to_length_ratio=0.35))
        inp_l = tex("spot", 26, YELLOW).next_to(inp, DOWN, buff=0.1)
        c6 = Circle(radius=0.36).move_to([zx(float(mm["z6"])), PC[1] + x0 * sxm, 0]).set_stroke(YELLOW, 3)
        c6_l = tex("the spot again", 28, YELLOW).move_to([zx(float(mm["z6"])) - 1.3, PC[1] - PH / 2 - 0.75, 0])
        c6_ar = Arrow(c6_l.get_top(), c6.get_bottom(), buff=0.06, stroke_width=3, color=YELLOW, max_tip_length_to_length_ratio=0.2)
        c3 = DashedVMobject(Circle(radius=0.36), num_dashes=16).move_to([zx(float(mm["z3"])), PC[1] - x0 * sxm, 0]).set_stroke(GREY_A, 2.5)
        c3_l = tex("mirror image", 26, GREY_A).next_to(c3, UP, buff=0.55)
        zax = Arrow([PC[0] - PW / 2, PC[1] - PH / 2 - 0.25, 0], [PC[0] + PW / 2, PC[1] - PH / 2 - 0.25, 0], buff=0, stroke_width=2,
                    color=GREY_B, max_tip_length_to_length_ratio=0.015)
        zl = mtex("z", 30, GREY_B).next_to(zax, RIGHT, buff=0.08)
        cap = tex(f"computed: {int(mm['nmodes'])} guided modes, each advanced by $e^{{i\\beta_m z}}$ and summed", 24, GREY_B)
        cap.move_to([PC[0], PC[1] - PH / 2 - 1.35, 0])
        self.at(S("e7"))
        dl.clear_updaters()
        self.add(mim)
        mfd.set_value(1.0)
        self.play(FadeOut(VGroup(SUM, row1, row2, ax_x, ax_n, lab_x, neff_l, nprof, nfill, t_n1, t_n2, tk, lab_core, modes,
                                 levels, band, dl, between)), FadeIn(gl), GrowArrow(inp[0]), FadeIn(inp_l), FadeIn(zax),
                  FadeIn(zl), run_time=0.7)
        self.remove(dl)
        self.play(msw.animate.set_value(1.05), FadeIn(cap), run_time=5.0, rate_func=linear)
        self.play(Create(c3), FadeIn(c3_l), run_time=0.5)
        self.play(Create(c6), FadeIn(c6_l), GrowArrow(c6_ar), run_time=0.6)
        self.finish()


# ============================================================ 3  coupler =======================
class Ch3Coupler(EpChapter):
    CH = 3
    TAG = "3\\quad Two guides: light that sloshes"

    def construct(self):
        S = self.S
        self.show_tag()
        cp = np.load(data_path("coupler.npz"))
        xg, zg, I = cp["x"], cp["z"], cp["I"]
        Lc, kap = float(cp["Lc"]), float(cp["kappa"])
        c1, c2, w = float(cp["c1"]), float(cp["c2"]), float(cp["w"])
        MW, MH = 11.6, 2.55
        MC = np.array([0.7, 1.75, 0])
        mx0 = MC[0] - MW / 2
        sxm = MH / (xg[-1] - xg[0])
        def zx(z):
            return mx0 + z / zg[-1] * MW
        def xy(x):
            return MC[1] + x * sxm
        guides = VGroup(*[Rectangle(width=MW, height=w * sxm).move_to([MC[0], xy(c), 0]).set_stroke(width=0).set_fill(CORE_FILL, 1)
                          for c in (c1, c2)])
        frame = Rectangle(width=MW, height=MH).move_to(MC).set_stroke(GREY_C, 1.5)
        gl = VGroup(tex("guide 2", 26, GREY_A).next_to([mx0, xy(c2), 0], LEFT, buff=0.15),
                    tex("guide 1", 26, GREY_A).next_to([mx0, xy(c1), 0], LEFT, buff=0.15))
        # single-guide modes with overlapping tails (from the supermodes: (S + A)/2 and (S - A)/2)
        Sx, Ax = cp["S"], cp["A"]
        m1 = (Sx + Ax) / 2
        m2 = (Sx - Ax) / 2
        pX = mx0 + 1.0
        tails = VGroup(Line([pX, MC[1] - MH / 2, 0], [pX, MC[1] + MH / 2, 0]).set_stroke(GREY_B, 1.5),
                       poly_line(pX + 1.3 * m1 / m1.max(), xy(xg), C_U, 4),
                       poly_line(pX + 1.3 * m2 / m2.max(), xy(xg), BLUE_A, 4))
        gap = Rectangle(width=2.3, height=(c2 - c1 - w) * sxm).move_to([pX + 0.9, MC[1], 0]).set_stroke(width=0).set_fill(C_EV, 0.25)
        gap_l = tex("tails overlap", 26, C_EV).next_to(gap, RIGHT, buff=0.2)
        DROP = 1.1                                      # the map sits lower (centred) until the supermodes arrive
        for mob in (guides, frame, gl, tails, gap, gap_l):
            mob.shift(DROP * DOWN)
        self.at(S("c1"))
        self.play(FadeIn(guides), Create(frame), FadeIn(gl), run_time=0.9)
        self.at(S("c1") + 2.2)
        self.play(Create(tails), run_time=1.0)
        self.play(FadeIn(gap), FadeIn(gap_l), run_time=0.5)
        inp = Arrow([mx0 - 0.0, xy(c1), 0] + 0.75 * LEFT, [mx0 + 0.05, xy(c1), 0], buff=0, stroke_width=6, color=YELLOW,
                    max_tip_length_to_length_ratio=0.35).shift(DROP * DOWN)
        self.at(S("c1") + 4.9)
        self.play(FadeOut(gl[1]), GrowArrow(inp), run_time=0.6)

        # ---------------- c2: the computed BPM map, revealed along z
        img = heat_rgba(I, I.max() * 0.9, gamma=0.65)
        mim, mfd, msw = self.sweep_image(img, MW, MC, MH, axis="right")
        cap = tex("computed: split-step beam propagation", 24, GREY_B).next_to(frame, DOWN, buff=0.12).align_to(frame, RIGHT)
        zax = mtex("z", 30, GREY_B).next_to(frame, RIGHT, buff=0.12)
        mim.shift(DROP * DOWN)
        self.at(S("c2"))
        self.add(mim)
        mfd.set_value(1.0)
        self.bring_to_front(frame)
        self.play(FadeOut(tails), FadeOut(gap), FadeOut(gap_l), FadeIn(cap), FadeIn(zax), run_time=0.4)
        self.play(msw.animate.set_value(1.05), run_time=4.0, rate_func=linear)

        # ---------------- c3: supermodes (eigenvectors of the pair)
        cy, cs = -1.4, 0.3                              # column centre and scale (units per wavelength)
        def col_axes(x0):
            g = VGroup(Line([x0, cy - 3.0 * cs, 0], [x0, cy + 3.0 * cs, 0]).set_stroke(GREY_C, 1.5))
            for c in (c1, c2):
                g.add(Rectangle(width=1.45, height=w * cs).move_to([x0 + 0.45, cy + c * cs, 0]).set_stroke(width=0).set_fill(CORE_FILL, 0.8))
            return g
        CX = [-5.4, -2.6, 0.4]
        A_ = 0.85
        axS, axA, axT = col_axes(CX[0]), col_axes(CX[1]), col_axes(CX[2])
        curS = poly_line(CX[0] + A_ * Sx, cy + xg * cs, C_U, 4)
        cA = ValueTracker(1.0)                          # Re(e^{-2 i kappa z}) applied to A (relative phase)
        curA = always_redraw(lambda: poly_line(CX[1] + A_ * cA.get_value() * Ax, cy + xg * cs, C_U, 4))
        labS = VGroup(tex("symmetric", 26, GREY_A), mtex("\\beta+\\kappa", 32, C_KZ)).arrange(DOWN, buff=0.08)
        labA = VGroup(tex("antisymmetric", 26, GREY_A), mtex("\\beta-\\kappa", 32, C_KZ)).arrange(DOWN, buff=0.08)
        labS.next_to([CX[0] + 0.45, cy - 3.0 * cs, 0], DOWN, buff=0.15)
        labA.next_to([CX[1] + 0.45, cy - 3.0 * cs, 0], DOWN, buff=0.15)
        self.at(S("c3") + 0.3)
        self.play(Group(guides, mim, frame, gl[0], inp, cap, zax).animate.shift(DROP * UP), run_time=1.0)   # keep the map above the guides
        self.play(FadeIn(axS), FadeIn(axA), run_time=0.6)
        self.at(S("c3") + 3.6)
        self.play(Create(curS), FadeIn(labS[0]), run_time=0.8)
        self.add(curA)
        self.play(Create(curA), FadeIn(labA[0]), run_time=0.8)
        self.at(S("c3") + 6.0)
        self.play(FadeIn(labS[1]), FadeIn(labA[1]), run_time=0.6)

        # ---------------- c4: the sum, A falls behind, light crosses over after L = pi / 2 kappa
        zt = ValueTracker(0.0)                          # propagation distance (wavelengths)
        def sum_curve():
            ph = -2 * kap * zt.get_value()
            Ii = np.abs(Sx + Ax * np.exp(1j * ph)) ** 2 / 4.0
            return poly_line(CX[2] + 1.15 * Ii, cy + xg * cs, YELLOW, 4)
        curT = always_redraw(sum_curve)
        plus = mtex("+", 40).move_to([(CX[0] + 0.45 + 0.725 + CX[1] - A_) / 2, cy, 0])
        eqs = mtex("\\rightarrow", 40).move_to([(CX[1] + 0.45 + 0.725 + CX[2] - 0.1) / 2, cy, 0])
        labT = VGroup(tex("their sum", 26, GREY_A), tex("(intensity)", 22, GREY_B)).arrange(DOWN, buff=0.08)
        labT.next_to([CX[2] + 0.45, cy - 3.0 * cs, 0], DOWN, buff=0.15)
        DCc, DRr = np.array([CX[1] + 1.35, cy + 3.0 * cs + 0.42, 0]), 0.27
        def adial():
            a = -2 * kap * zt.get_value()
            return VGroup(Circle(radius=DRr).move_to(DCc).set_stroke(GREY_B, 1.5),
                          fast_arrow(DCc, DCc + DRr * np.array([np.cos(a), np.sin(a), 0]), C_KZ, 3.5, 0.11))
        adl = always_redraw(adial)
        adl_l = tex("falls behind", 24, GREY_B).next_to(DCc, RIGHT, buff=0.45)
        def marker():
            z = zt.get_value()
            return Line([zx(z), MC[1] - MH / 2, 0], [zx(z), MC[1] + MH / 2, 0]).set_stroke(WHITE, 3)
        mk = always_redraw(marker)
        Lmark = VGroup(Line([zx(Lc), MC[1] - MH / 2 - 0.18, 0], [zx(Lc), MC[1] - MH / 2, 0]).set_stroke(YELLOW, 3),
                       MathTex("L", "=", "\\pi/2\\kappa", font_size=32, color=YELLOW))
        Lmark[1].next_to(Lmark[0], DOWN, buff=0.05)
        self.at(S("c4"))
        self.add(curT)
        self.play(FadeIn(axT), FadeIn(plus), FadeIn(eqs), FadeIn(labT), Create(curT), run_time=0.9)
        self.at(S("c4") + 2.6)
        self.add(mk, adl)
        self.play(FadeIn(mk), FadeIn(adl), FadeIn(adl_l), run_time=0.4)
        self.play(zt.animate.set_value(Lc), cA.animate.set_value(-1.0), run_time=4.6, rate_func=linear)
        self.play(FadeIn(Lmark), run_time=0.5)

        # ---------------- c5: in the basis of the two guides: a 2x2 rotation
        rot = MathTex("\\begin{pmatrix} a_1 \\\\ a_2 \\end{pmatrix}", "=",
                      "\\begin{pmatrix} \\cos\\kappa z & i\\sin\\kappa z \\\\ i\\sin\\kappa z & \\cos\\kappa z \\end{pmatrix}",
                      "\\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix}", font_size=38).move_to([-3.3, -1.25, 0])
        rot_l = tex("amplitudes in guide 1 and guide 2", 24, GREY_B).next_to(rot, DOWN, buff=0.25)
        RC, RR = np.array([3.4, -1.95, 0]), 1.0
        rax = VGroup(Arrow(RC + 1.35 * LEFT, RC + 1.4 * RIGHT, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04),
                     Arrow(RC + 1.2 * DOWN, RC + 1.3 * UP, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04),
                     Circle(radius=RR).move_to(RC).set_stroke(GREY_C, 1.5))
        rlab = VGroup(tex("guide 1", 24, GREY_A).next_to(rax[0], RIGHT, buff=0.08),
                      tex("guide 2", 24, GREY_A).next_to(rax[1], UP, buff=0.06))
        def rvec():
            a = kap * zt.get_value()
            tip = RC + RR * np.array([np.cos(a), np.sin(a), 0])
            return VGroup(Line(RC, RC + RR * np.cos(a) * RIGHT).set_stroke(C_U, 6),
                          Line(RC + RR * np.cos(a) * RIGHT, tip).set_stroke(YELLOW, 6),
                          fast_arrow(RC, tip, WHITE, 5, 0.18))
        rv = always_redraw(rvec)
        kz_l = always_redraw(lambda: MathTex("\\kappa z", font_size=30, color=GREY_A).move_to(
            RC + 0.55 * np.array([np.cos(kap * zt.get_value() / 2), np.sin(kap * zt.get_value() / 2), 0])))
        self.at(S("c5"))
        for mob in (curA, curT, adl):
            mob.clear_updaters()
        self.play(FadeOut(VGroup(axS, axA, axT, curS, curA, curT, labS, labA, labT, plus, eqs, adl, adl_l)), run_time=0.5)
        self.remove(curA, curT, adl)
        self.play(Write(rot), run_time=1.6)
        self.play(FadeIn(rot_l), FadeIn(rax), FadeIn(rlab), run_time=0.6)
        self.add(rv)
        self.play(FadeIn(rv), run_time=0.4)
        self.at(S("c5") + 4.2)
        self.play(zt.animate.set_value(0.0), run_time=1.3)
        self.add(kz_l)
        self.play(zt.animate.set_value(2 * Lc), run_time=3.0, rate_func=linear)

        # ---------------- c6: half a crossing = 50:50 splitter
        half = VGroup(Line([zx(Lc / 2), MC[1] - MH / 2 - 0.18, 0], [zx(Lc / 2), MC[1] - MH / 2, 0]).set_stroke(YELLOW, 3),
                      tex("$L/2$: 50\\,:\\,50", 30, YELLOW))
        half[1].next_to(half[0], DOWN, buff=0.06)
        uses = tex("interferometers $\\cdot$ switches $\\cdot$ optical processors", 28, GREY_A).move_to([-3.3, -2.85, 0])
        self.at(S("c6"))
        self.play(zt.animate.set_value(Lc / 2), FadeOut(Lmark), run_time=1.6)
        self.play(FadeIn(half), run_time=0.5)
        self.at(S("c6") + 3.0)
        self.play(FadeIn(uses, shift=0.1 * UP), run_time=0.7)
        self.finish()


# ------------------------------------------------------------------ resonator model (analytic round-trip sum) -----
def ring_stored(phi, q):
    """|sum_n (q e^{i phi})^n|^2 / max  (q = r a: what survives one round trip, amplitude)."""
    return (1 - q) ** 2 / np.abs(1 - q * np.exp(1j * np.asarray(phi))) ** 2


def ring_through(phi, r, a):
    """All-pass ring: t = (r - a e^{i phi}) / (1 - r a e^{i phi}); returns |t|^2."""
    e = np.exp(1j * np.asarray(phi))
    return np.abs((r - a * e) / (1 - r * a * e)) ** 2


def ring_schematic(c, R, yb, x0, x1, width=16):
    return VGroup(Circle(radius=R).move_to(c).set_stroke(CORE_FILL, width),
                  Line([x0, yb, 0], [x1, yb, 0]).set_stroke(CORE_FILL, width))


# ============================================================ 4  ring: round trips ==============
class Ch4Ring(EpChapter):
    CH = 4
    TAG = "4\\quad Resonators: round trips"

    def construct(self):
        S = self.S
        self.show_tag()
        RC, RR = np.array([-4.75, 0.75, 0]), 1.2
        YB = RC[1] - RR - 0.32
        BX0, BX1 = -6.9, -2.6
        sch = ring_schematic(RC, RR, YB, BX0, BX1)
        lab_in = tex("in", 28, GREY_A).move_to([BX0 + 0.45, YB + 0.38, 0])
        lab_th = tex("through", 28, GREY_A).next_to([BX1, YB, 0], RIGHT, buff=0.15)
        self.at(S("r1") + 0.2)
        self.play(Create(sch[1]), FadeIn(lab_in), FadeIn(lab_th), run_time=0.6)
        self.play(Create(sch[0]), run_time=0.9)
        # light: pulses along the bus, light circulating in the ring (pure functions of time)
        lop = ValueTracker(0.0)
        t_l = S("r1") + 2.0

        def light():
            o = lop.get_value()
            g = VGroup()
            if o <= 0.003:
                return g
            t = self.now - t_l
            Lb = BX1 - BX0
            for k in range(3):
                x = BX0 + ((1.6 * t + k * Lb / 3) % Lb)
                g.add(Dot([x, YB, 0], radius=0.09, color=C_U).set_opacity(o))
            for k in range(2):
                a = -PI / 2 + 1.4 * t + k * PI
                g.add(Dot(RC + RR * np.array([np.cos(a), np.sin(a), 0]), radius=0.08, color=C_U).set_opacity(0.8 * o))
            return g
        lt = always_redraw(light)
        leak = VGroup(CurvedArrow(np.array([RC[0] - 0.35, YB + 0.05, 0]), np.array([RC[0] + 0.25, RC[1] - RR + 0.08, 0]),
                                  angle=-1.2, color=YELLOW, stroke_width=3, tip_length=0.14),
                      tex("a little leaks across", 26, YELLOW).move_to([RC[0], YB - 0.55, 0]))
        self.at(S("r1") + 2.0)
        self.add(lt)
        self.play(lop.animate.set_value(1.0), run_time=0.6)
        self.at(S("r1") + 4.4)
        self.play(Create(leak[0]), FadeIn(leak[1]), run_time=0.7)

        # ---------------- r2: the phasor chain (each round trip adds a rotated, weaker copy)
        L0, QRA, NMAX = 0.95, 0.86, 44
        ph = ValueTracker(0.6)
        ns = ValueTracker(0.0)
        res_op = ValueTracker(0.0)
        pox, poy, psc = ValueTracker(-0.6), ValueTracker(0.2), ValueTracker(1.0)    # origin and scale of the phasor picture

        def chain():
            phi = ph.get_value()
            n = ns.get_value()
            q = QRA * np.exp(1j * phi)
            sc = psc.get_value()
            PO = np.array([pox.get_value(), poy.get_value(), 0])
            g = VGroup()
            ro = res_op.get_value()
            if ro > 0.003:                              # the total (drawn under the chain)
                tot = sc * L0 / (1 - q)
                g.add(fast_arrow(PO, PO + np.array([tot.real, tot.imag, 0]), YELLOW, 8, 0.28, 0.9 * ro))
            z = 0j
            for k in range(NMAX):
                o = float(np.clip(n - k, 0, 1))
                if o <= 0:
                    break
                dz = sc * L0 * q ** k
                p0 = PO + np.array([z.real, z.imag, 0])
                p1 = p0 + np.array([dz.real, dz.imag, 0])
                g.add(fast_arrow(p0, p1, C_U, 4, min(0.2, 0.45 * abs(dz)), o * (0.4 + 0.6 * QRA ** k)))
                z += dz
            return g
        ch_ = always_redraw(chain)
        phi_lab = mtex("\\varphi", 32, C_KZ)
        arc_op = ValueTracker(0.0)

        def arc():
            o = arc_op.get_value()
            if o <= 0.003:
                return VGroup()
            phi = ph.get_value()
            p1 = np.array([pox.get_value(), poy.get_value(), 0]) + L0 * RIGHT
            g = VGroup(fast_dashed(p1, p1 + 0.7 * RIGHT, 0.04, GREY_B, 1.5, o),
                       Arc(radius=0.45, start_angle=0, angle=phi, arc_center=p1).set_stroke(C_KZ, 3, opacity=o))
            lab = phi_lab.copy().move_to(p1 + 0.75 * np.array([np.cos(phi / 2), np.sin(phi / 2), 0])).set_opacity(o)
            g.add(lab)
            return g
        arcg = always_redraw(arc)
        phi_t = tex("$\\varphi$: round-trip phase", 30, C_KZ).move_to([4.2, 2.75, 0])
        self.at(S("r2"))
        self.add(ch_, arcg)
        self.play(ns.animate.set_value(1.0), run_time=0.6)
        self.at(S("r2") + 1.6)
        self.play(ns.animate.set_value(2.0), arc_op.animate.set_value(1.0), FadeIn(phi_t), run_time=0.8)
        self.at(S("r2") + 3.4)
        self.play(ns.animate.set_value(float(NMAX)), arc_op.animate.set_value(0.0), run_time=2.6, rate_func=rate_functions.ease_in_quad)

        # ---------------- r3: off resonance -> small spiral, mostly cancels
        st_off = tex("off resonance", 32, GREY_A).move_to([4.2, 2.75, 0])
        tot_l = VGroup(Arrow(ORIGIN, 0.7 * RIGHT, buff=0, stroke_width=7, color=YELLOW, max_tip_length_to_length_ratio=0.35),
                       tex("total light in the ring", 28, YELLOW)).arrange(RIGHT, buff=0.15).move_to([4.2, 2.15, 0])
        self.at(S("r3") + 0.1)
        self.play(FadeOut(phi_t), FadeIn(st_off), run_time=0.5)
        self.at(S("r3") + 2.2)
        self.play(res_op.animate.set_value(1.0), FadeIn(tot_l), run_time=0.7)

        # ---------------- r4: on resonance -> one long arrow, light piles up
        st_on = tex("on resonance", 30, YELLOW).move_to(st_off)
        cond = MathTex("\\beta L = 2\\pi m", "\\quad\\Leftrightarrow\\quad", "n_{\\rm eff}\\, L = m\\,\\lambda", font_size=38)
        cond[0].set_color(C_KZ)
        cond.move_to([2.6, -1.15, 0])
        cond_l = tex("($L$: ring circumference)", 24, GREY_B).next_to(cond, DOWN, buff=0.15)
        wv_op = ValueTracker(0.0)
        MW_ = 14

        def ringwave():
            o = wv_op.get_value()
            if o <= 0.003:
                return VGroup()
            phi = ph.get_value()
            amp = 0.03 + 0.11 * np.sqrt(ring_stored(phi, QRA))
            psi = np.linspace(0, TAU, 300)
            rr = RR + amp * np.cos(MW_ * psi - 2.6 * self.now)
            return poly_line(RC[0] + rr * np.cos(psi), RC[1] + rr * np.sin(psi), C_U, 3, o)
        rw = always_redraw(ringwave)
        self.at(S("r4") + 0.1)
        self.play(FadeOut(leak), run_time=0.4)
        self.remove(arcg)
        self.add(rw)
        self.play(ph.animate.set_value(0.0), FadeTransform(st_off, st_on), wv_op.animate.set_value(1.0), lop.animate.set_value(0.0),
                  run_time=2.4, rate_func=smooth)
        self.remove(lt)
        self.play(Write(cond), FadeIn(cond_l), run_time=1.2)

        # ---------------- r5: sweep the colour -> evenly spaced resonance peaks
        AX0, AX1, AY0, AY1 = -1.9, 6.2, -2.95, -1.55
        u = ValueTracker(0.0)
        axes = VGroup(Arrow([AX0, AY0, 0], [AX1 + 0.3, AY0, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.02),
                      Arrow([AX0, AY0, 0], [AX0, AY1 + 0.45, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.08))
        ax_l = VGroup(tex("wavelength", 24, GREY_B).next_to([AX1 + 0.3, AY0, 0], DOWN, buff=0.1).shift(0.5 * LEFT),
                      tex("stored light", 24, GREY_B).next_to([AX0, AY1 + 0.45, 0], RIGHT, buff=0.12))

        def phi_of(uu):
            return 3 * PI - 6 * PI * uu

        def trace():
            uu = u.get_value()
            if uu <= 1e-3:
                return VGroup()
            us = np.linspace(0, uu, max(2, int(500 * uu)))
            yv = ring_stored(phi_of(us), QRA)
            g = VGroup(poly_line(AX0 + (AX1 - AX0) * us, AY0 + (AY1 - AY0) * yv, YELLOW, 3.5))
            g.add(Dot([AX0 + (AX1 - AX0) * uu, AY0 + (AY1 - AY0) * yv[-1], 0], radius=0.06, color=YELLOW))
            return g
        tr = always_redraw(trace)
        self.at(S("r5"))
        self.play(FadeOut(cond), FadeOut(cond_l), FadeOut(st_on), FadeOut(tot_l), GrowArrow(axes[0]), GrowArrow(axes[1]),
                  FadeIn(ax_l), ph.animate.set_value(3 * PI), psc.animate.set_value(0.42), poy.animate.set_value(0.95), run_time=1.0)
        self.add(tr)
        self.play(u.animate.set_value(1.0), ph.animate.set_value(-3 * PI), run_time=3.2, rate_func=linear)
        # the computed ring from the opening (on resonance)
        rf = RingField()
        sc_ring = RR / float(rf.r["R"])
        hgt = (rf.r["y"][-1] - rf.r["y"][0] + float(rf.r["h"])) * sc_ring
        gop = ValueTracker(0.0)
        cimg, cfd = self.live_image(lambda t: rf.rgba(t, {"G": gop.get_value()}, None, 0.8), hgt * rf.shape[1] / rf.shape[0], RC, hgt)
        cfd.set_value(1.0)
        ccap = tex("computed, on resonance", 24, GREY_B).move_to([RC[0], AY0 + 0.05, 0])
        self.at(S("r5") + 4.35)
        rw.clear_updaters()
        self.add(cimg)
        self.bring_to_front(rw)
        self.play(gop.animate.set_value(1.0), FadeOut(sch), FadeOut(rw), FadeOut(lab_in), FadeOut(lab_th), FadeIn(ccap), run_time=0.9)
        self.finish()


# ============================================================ 5  linewidth & coupling ===========
class Ch5Linewidth(EpChapter):
    CH = 5
    TAG = "5\\quad Linewidth and coupling"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- q1: peak width set by loss; round trips survived
        q = ValueTracker(0.85)
        PX0, PX1, PY0, PY1 = -6.3, -0.9, -0.55, 2.35
        RX0, RX1 = 0.9, 6.3
        pax = VGroup(Arrow([PX0, PY0, 0], [PX1 + 0.3, PY0, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                     Arrow([PX0, PY0, 0], [PX0, PY1 + 0.4, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.05))
        pal = VGroup(tex("wavelength", 24, GREY_B).next_to([PX1 + 0.3, PY0, 0], DOWN, buff=0.1).shift(0.4 * LEFT),
                     tex("stored light", 24, GREY_B).next_to([PX0, PY1 + 0.4, 0], RIGHT, buff=0.12))
        rax = VGroup(Arrow([RX0, PY0, 0], [RX1 + 0.3, PY0, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                     Arrow([RX0, PY0, 0], [RX0, PY1 + 0.4, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.05))
        ral = VGroup(tex("round trips", 24, GREY_B).next_to([RX1 + 0.3, PY0, 0], DOWN, buff=0.1).shift(0.4 * LEFT),
                     tex("light left in the ring", 24, GREY_B).next_to([RX0, PY1 + 0.4, 0], RIGHT, buff=0.12))
        PH_ = PI / 2                                         # plotted detuning range: -pi/2 .. pi/2
        phis = np.linspace(-PH_, PH_, 400)

        def peak():
            yv = ring_stored(phis, q.get_value())
            return poly_line(PX0 + (phis + PH_) / (2 * PH_) * (PX1 - PX0), PY0 + (PY1 - PY0) * yv, YELLOW, 4)
        pk = always_redraw(peak)
        NR = 30

        def stems_():
            qq = q.get_value()
            g = VGroup()
            for n in range(NR):
                x = RX0 + 0.12 + n * (RX1 - RX0 - 0.2) / NR
                h = (PY1 - PY0) * qq ** (2 * n)
                if h > 0.004:
                    g.add(Line([x, PY0, 0], [x, PY0 + h, 0]).set_stroke(C_U, 4))
            return g
        st = always_redraw(stems_)
        self.at(S("q1"))
        self.add(pk, st)
        self.play(GrowArrow(pax[0]), GrowArrow(pax[1]), FadeIn(pal), GrowArrow(rax[0]), GrowArrow(rax[1]), FadeIn(ral),
                  FadeIn(pk), FadeIn(st), run_time=0.9)
        self.at(S("q1") + 2.0)
        self.play(q.animate.set_value(0.55), run_time=1.8)
        self.play(q.animate.set_value(0.85), run_time=1.8)

        # ---------------- q2: Q = lambda / delta lambda
        def fwhm():
            qq = q.get_value()
            c = (1 + qq ** 2 - 2 * (1 - qq) ** 2) / (2 * qq)
            ph_ = np.arccos(np.clip(c, -1, 1))
            xa = PX0 + (-ph_ + PH_) / (2 * PH_) * (PX1 - PX0)
            xb = PX0 + (ph_ + PH_) / (2 * PH_) * (PX1 - PX0)
            y = PY0 + 0.5 * (PY1 - PY0)
            return VGroup(fast_arrow([(xa + xb) / 2, y, 0], [xb, y, 0], C_KZ, 3, 0.12),
                          fast_arrow([(xa + xb) / 2, y, 0], [xa, y, 0], C_KZ, 3, 0.12))
        fw = always_redraw(fwhm)
        dl_l = mtex("\\Delta\\lambda", 32, C_KZ)
        def _dl_pos(m):
            qq = q.get_value()
            ph_ = np.arccos(np.clip((1 + qq ** 2 - 2 * (1 - qq) ** 2) / (2 * qq), -1, 1))
            m.next_to([PX0 + (ph_ + PH_) / (2 * PH_) * (PX1 - PX0), PY0 + 0.5 * (PY1 - PY0), 0], RIGHT, buff=0.15)
        dl_l.add_updater(_dl_pos)
        Qf = MathTex("Q", "=", "{\\lambda", "\\over", "\\Delta\\lambda}", font_size=50).move_to([-3.6, -1.85, 0])
        Qf[4].set_color(C_KZ)
        Qt = tex("high $Q$: light stays stored a long time", 30).move_to([3.6, -1.85, 0])
        self.at(S("q2") + 2.6)
        self.add(fw)
        self.play(FadeIn(fw), FadeIn(dl_l), run_time=0.6)
        self.at(S("q2") + 4.0)
        self.play(Write(Qf), run_time=1.2)
        self.at(S("q2") + 6.4)
        self.play(FadeIn(Qt, shift=0.1 * UP), run_time=0.7)

        # ---------------- q3: the light that goes straight past meets the light leaking out (flipped on resonance)
        RC, RR = np.array([-4.85, 1.45, 0]), 0.9
        YB = RC[1] - RR - 0.27
        BX0, BX1 = -6.9, -2.75
        sch = ring_schematic(RC, RR, YB, BX0, BX1, 14)
        lab_in = tex("in", 26, GREY_A).move_to([BX0 + 0.4, YB + 0.33, 0])
        lab_th = tex("through", 26, GREY_A).move_to([BX1 - 0.6, YB - 0.36, 0])
        A_LOSS = 0.85
        k2 = ValueTracker(0.06)
        PO, LS = np.array([0.0, 2.2, 0]), 2.3

        def rr_():
            return np.sqrt(1 - k2.get_value())
        aop = ValueTracker(0.0)
        bop = ValueTracker(0.0)
        cop = ValueTracker(0.0)

        def arrows():
            r = rr_()
            ringv = -k2.get_value() * A_LOSS / (1 - r * A_LOSS)
            g = VGroup()
            tip1 = PO + LS * r * RIGHT
            if aop.get_value() > 0.003:
                g.add(fast_arrow(PO, tip1, GOLD, 6, 0.22, aop.get_value()))
            if bop.get_value() > 0.003:
                g.add(fast_arrow(tip1 + 0.32 * DOWN, tip1 + 0.32 * DOWN + LS * ringv * RIGHT, C_U, 6, 0.22, bop.get_value()))
            if cop.get_value() > 0.003:
                tot = r + ringv
                if abs(tot) > 0.03:
                    g.add(fast_arrow(PO + 0.9 * DOWN, PO + 0.9 * DOWN + LS * tot * RIGHT, WHITE, 6, 0.22, cop.get_value()))
                else:
                    g.add(Dot(PO + 0.9 * DOWN, radius=0.08, color=C_OK).set_opacity(cop.get_value()))
            return g
        arr = always_redraw(arrows)
        la = tex("straight past", 26, GOLD).move_to(PO + 1.15 * RIGHT + 0.35 * UP)
        lb = tex("back out of the ring", 26, C_U)
        lb.add_updater(lambda m: m.next_to(PO + LS * rr_() * RIGHT + 0.32 * DOWN, RIGHT, buff=0.2))
        lc = tex("through port", 26, WHITE).move_to(PO + 1.3 * DOWN + 0.9 * RIGHT)
        flip = tex("on resonance: flipped", 26, C_U)
        flip.add_updater(lambda m: m.next_to(lb, DOWN, buff=0.12).align_to(lb, LEFT))
        dop = ValueTracker(0.0)
        t_d = S("q3") + 0.9

        def dots():
            o = dop.get_value()
            g = VGroup()
            if o <= 0.003:
                return g
            t = self.now - t_d
            Lb = BX1 - BX0
            x = BX0 + (1.5 * t) % Lb
            g.add(Dot([x, YB, 0], radius=0.09, color=GOLD).set_opacity(o))
            if self.now > S("q3") + 3.0:
                tt = self.now - (S("q3") + 3.0)
                a = -PI / 2 + 1.6 * tt
                g.add(Dot(RC + RR * np.array([np.cos(a), np.sin(a), 0]), radius=0.08, color=C_U).set_opacity(o))
                xb = RC[0] + (1.5 * tt) % (BX1 - RC[0])
                g.add(Dot([xb, YB, 0], radius=0.08, color=C_U).set_opacity(o))
            return g
        dts = always_redraw(dots)
        self.at(S("q3"))
        for m in (pk, st, fw):
            m.clear_updaters()
        dl_l.clear_updaters()
        self.play(FadeOut(VGroup(pax, pal, rax, ral, pk, st, fw, dl_l, Qf, Qt)), FadeIn(sch), FadeIn(lab_in), FadeIn(lab_th), run_time=0.8)
        self.add(arr, dts)
        self.play(dop.animate.set_value(1.0), aop.animate.set_value(1.0), FadeIn(la), run_time=0.7)
        self.at(S("q3") + 3.0)
        self.add(lb)
        self.play(bop.animate.set_value(1.0), FadeIn(lb), run_time=0.7)
        self.at(S("q3") + 5.4)
        self.play(FadeIn(flip), run_time=0.5)

        # ---------------- q4: through port dims; critical coupling = zero
        TX0, TX1, TY0, TY1 = -5.6, 2.2, -2.95, -0.75
        tax = VGroup(Arrow([TX0, TY0, 0], [TX1 + 0.3, TY0, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                     Arrow([TX0, TY0, 0], [TX0, TY1 + 0.35, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.06))
        tal = VGroup(tex("wavelength", 24, GREY_B).next_to([TX1 + 0.3, TY0, 0], DOWN, buff=0.08).shift(0.4 * LEFT),
                     tex("through port", 24, GREY_B).next_to([TX0, TY1 + 0.35, 0], RIGHT, buff=0.12))
        phs = np.linspace(-PI, PI, 300)

        def tcurve(kk, col, w=4, o=1.0):
            r = np.sqrt(1 - kk)
            yv = ring_through(phs, r, A_LOSS)
            return poly_line(TX0 + (phs + PI) / TAU * (TX1 - TX0), TY0 + (TY1 - TY0) * yv, col, w, o)
        tc = always_redraw(lambda: tcurve(k2.get_value(), WHITE))
        KX0, KX1, KY = 3.2, 6.2, -1.9
        kcrit = 1 - A_LOSS ** 2
        def k2x(kk):
            return KX0 + (KX1 - KX0) * (kk - 0.03) / (0.66 - 0.03)
        kline = Line([KX0, KY, 0], [KX1, KY, 0]).set_stroke(GREY_B, 3)
        kl = VGroup(tex("coupling", 26, GREY_A).next_to(kline, UP, buff=0.3),
                    tex("weak", 22, GREY_B).next_to([KX0, KY, 0], DOWN, buff=0.2),
                    tex("strong", 22, GREY_B).next_to([KX1, KY, 0], DOWN, buff=0.2))
        kcr = Line([k2x(kcrit), KY - 0.16, 0], [k2x(kcrit), KY + 0.16, 0]).set_stroke(C_OK, 4)
        knob = always_redraw(lambda: Dot([k2x(k2.get_value()), KY, 0], radius=0.1, color=YELLOW))
        crit = tex("critical coupling", 30, C_OK).move_to([(TX0 + TX1) / 2, TY0 + 0.35, 0]).shift(1.9 * RIGHT)
        self.at(S("q4"))
        self.add(tc)
        self.play(cop.animate.set_value(1.0), FadeIn(lc), GrowArrow(tax[0]), GrowArrow(tax[1]), FadeIn(tal), Create(tc), run_time=0.9)
        self.at(S("q4") + 2.0)
        self.add(knob)
        flip.clear_updaters()
        self.play(FadeIn(kline), FadeIn(kl), FadeIn(knob), FadeOut(flip), run_time=0.5)
        self.play(k2.animate.set_value(kcrit), run_time=2.6)
        self.play(FadeIn(kcr), FadeIn(crit), run_time=0.6)

        # ---------------- q5: weaker or stronger coupling -> shallower dip
        ghost = tcurve(kcrit, C_OK, 2.5, 0.6)
        self.at(S("q5"))
        self.add(ghost)
        self.play(k2.animate.set_value(0.07), FadeOut(crit), run_time=1.1)
        self.play(k2.animate.set_value(0.62), run_time=1.5)
        self.finish()


# ============================================================ 6  on a computer ==================
class Ch6Computer(EpChapter):
    CH = 6
    TAG = "6\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        lines = ["beta2, E = eigh(D2/dx**2 + diag(k0**2*n**2))  # modes",
                 "U = ifft(fft(U) * exp(1j*kz*dz))              # BPM: diffract",
                 "U = U * exp(1j*k0*dn*dz)                      #      refract",
                 "e = exp(1j*phi);  t = (r-a*e) / (1-r*a*e)     # ring"]
        card = code_card(lines, font_size=23)
        card.move_to([0.0, 2.0, 0])
        rows = card[1][0]
        hl = ValueTracker(-1.0)

        def hilite():
            i = hl.get_value()
            g = VGroup()
            for j, rw in enumerate(rows):
                o = float(np.clip(1 - abs(i - j), 0, 1)) if j < 3 else float(np.clip(1 - abs(i - j), 0, 1))
                if j in (1, 2) and 1.0 <= i <= 2.0:
                    o = 1.0
                if o > 0.01:
                    g.add(Rectangle(width=card[0].width - 0.2, height=0.36).move_to([card[0].get_center()[0], rw.get_center()[1], 0])
                          .set_stroke(width=0).set_fill(YELLOW, 0.16 * o))
            return g
        hg = always_redraw(hilite)
        self.at(S("a1"))
        self.add(hg)
        self.play(FadeIn(card, shift=0.1 * DOWN), run_time=0.9)

        # ---------------- a2: modes; the window must hold the tails
        wd = np.load(data_path("window.npz"))
        d_ = float(wd["d"])
        def mode_plot(cx, x, pr, half, col, lab, sub):
            sxp = 1.1
            g = VGroup()
            g.add(Rectangle(width=d_ * sxp, height=1.9).move_to([cx, -1.35, 0]).set_stroke(width=0).set_fill(CORE_FILL, 0.6))
            g.add(Line([cx - 2.6, -1.35, 0], [cx + 2.6, -1.35, 0]).set_stroke(GREY_C, 1.5))
            g.add(poly_line(cx + x * sxp, -1.35 + 0.85 * pr, C_U, 4))
            for sgn in (-1, 1):
                g.add(Line([cx + sgn * half * sxp, -2.45, 0], [cx + sgn * half * sxp, -0.25, 0]).set_stroke(col, 5))
            g.add(tex(lab, 28, col).move_to([cx, -0.05, 0]))
            g.add(tex(sub, 24, GREY_B).move_to([cx, -2.75, 0]))
            return g
        wide = mode_plot(3.3, wd["xw"][np.abs(wd["xw"]) <= 2.3], wd["pw"][np.abs(wd["xw"]) <= 2.3], 2.3, C_OK, "wide window",
                         f"$n_{{\\rm eff}} = {float(wd['nw']):.3f}$")
        narrow = mode_plot(-3.3, wd["xn"], wd["pn"], float(wd["xn"][-1]), C_EV, "window too small: squeezed",
                           f"$n_{{\\rm eff}} = {float(wd['nn']):.3f}$ (wrong)")
        # compact recap of the mode solver (same computed matrix and modes as chapter 2)
        sv = np.load(data_path("solver.npz"))
        Hm = sv["D2"] + sv["Dn"]
        v = np.clip(Hm / np.abs(Hm).max() * 0.5 + 0.5, 0, 1)
        himg = np.empty(Hm.shape + (4,), np.uint8)
        himg[..., :3] = DIVERGE[(v * 1023).astype(int)]
        himg[..., 3] = 255
        himg[np.abs(Hm) < 1e-9] = (14, 14, 14, 255)
        him, hfd = self.static_image(np.kron(himg, np.ones((6, 6, 1), np.uint8)), 1.7, [-4.4, -1.35, 0], 1.7, nearest=True)
        hbox = SurroundingRectangle(him, buff=0.03, color=GREY_B, stroke_width=1.5)
        hlab = MathTex("H", font_size=32).next_to(hbox, DOWN, buff=0.12)
        h_ar = Arrow([-3.3, -1.35, 0], [-1.7, -1.35, 0], buff=0, stroke_width=4, color=GREY_A, max_tip_length_to_length_ratio=0.15)
        h_arl = Text("eigh", font=MONO, font_size=24, color=GREY_A).next_to(h_ar, UP, buff=0.1)
        xm_, pr_ = sv["x"], sv["prof"]
        sel_ = np.abs(xm_) <= 1.6
        mini = VGroup()
        for m, y0 in enumerate((-0.55, -1.35, -2.15)):
            mini.add(poly_line(0.9 + 1.25 * xm_[sel_], y0 + 0.3 * pr_[sel_, m], C_U, 3.5))
            mini.add(mtex(f"\\beta_{m}", 28, C_KZ).move_to([3.15, y0, 0]))
        self.at(S("a2"))
        self.play(hl.animate.set_value(0.0), run_time=0.5)
        self.add(him)
        self.play(hfd.animate.set_value(1.0), Create(hbox), FadeIn(hlab), run_time=0.6)
        self.play(GrowArrow(h_ar), FadeIn(h_arl), LaggedStart(*[FadeIn(mm) for mm in mini], lag_ratio=0.12), run_time=1.0)
        self.at(S("a2") + 4.3)
        self.play(hfd.animate.set_value(0.0), FadeOut(VGroup(hbox, hlab, h_ar, h_arl, mini)), run_time=0.4)
        him.clear_updaters()
        self.remove(him)
        self.play(FadeIn(narrow), run_time=0.6)
        self.play(FadeIn(wide), run_time=0.6)

        # ---------------- a3: BPM = diffraction step + refraction step, slice after slice
        cp = np.load(data_path("coupler.npz"))
        xg, zg, I = cp["x"], cp["z"], cp["I"]
        c1, c2, w = float(cp["c1"]), float(cp["c2"]), float(cp["w"])
        MW, MH, MC = 11.4, 1.9, np.array([0.0, -1.35, 0])
        sxm = MH / (xg[-1] - xg[0])
        img = heat_rgba(I, I.max() * 0.9, gamma=0.65)
        mim, mfd, msw = self.sweep_image(img, MW, MC, MH, axis="right")
        guides = VGroup(*[Rectangle(width=MW, height=w * sxm).move_to([MC[0], MC[1] + c * sxm, 0]).set_stroke(width=0).set_fill(CORE_FILL, 1)
                          for c in (c1, c2)])
        frame = Rectangle(width=MW, height=MH).move_to(MC).set_stroke(GREY_C, 1.5)
        NS = 26
        xs_sl = MC[0] - MW / 2 + (np.arange(NS) + 0.5) * MW / NS
        sl_op = ValueTracker(0.0)
        march = ValueTracker(0.0)

        def slices():
            o = sl_op.get_value()
            g = VGroup()
            if o <= 0.003:
                return g
            m = march.get_value()
            for k, x in enumerate(xs_sl):
                hot = 1.0 if (m > 0 and abs((x - (MC[0] - MW / 2)) / MW - m) < 0.5 / NS) else 0.0
                g.add(Line([x, MC[1] - MH / 2, 0], [x, MC[1] + MH / 2, 0]).set_stroke(interpolate_color(GREY_B, YELLOW, hot),
                                                                                      2.5 + 2 * hot, opacity=o * (0.55 + 0.45 * hot)))
            return g
        sls = always_redraw(slices)
        k0 = int(NS * 0.42)
        bx = xs_sl[k0]
        bd = VGroup(Brace(Line([xs_sl[k0 - 1], 0, 0], [bx, 0, 0]), UP, buff=0).next_to([0, MC[1] + MH / 2, 0], UP, buff=0.05),
                    )
        bd[0].set_x((xs_sl[k0 - 1] + bx) / 2)
        diff_l = MathTex("\\text{diffract: FFT},\\ \\times e^{ik_z\\Delta z},\\ \\text{IFFT}", font_size=28).next_to(bd[0], UP, buff=0.08)
        diff_l.shift(2.2 * LEFT)
        refr_ar = Arrow([bx + 1.6, MC[1] + MH / 2 + 0.55, 0], [bx + 0.05, MC[1] + MH / 2 - 0.05, 0], buff=0, stroke_width=3, color=YELLOW,
                        max_tip_length_to_length_ratio=0.12)
        refr_l = MathTex("\\text{refract: }\\times e^{ik_0\\Delta n\\,\\Delta z}", font_size=28, color=YELLOW).next_to(refr_ar.get_start(), RIGHT, buff=0.1)
        self.at(S("a3"))
        self.play(hl.animate.set_value(1.0), FadeOut(narrow), FadeOut(wide), run_time=0.6)
        self.add(sls)
        self.play(FadeIn(guides), Create(frame), sl_op.animate.set_value(1.0), run_time=0.8)
        self.at(S("a3") + 3.6)
        self.play(GrowFromCenter(bd[0]), FadeIn(diff_l), run_time=0.7)
        self.at(S("a3") + 7.0)
        self.play(GrowArrow(refr_ar), FadeIn(refr_l), run_time=0.7)

        # ---------------- a4: repeat slice after slice -> the coupler
        self.at(S("a4"))
        self.add(mim)
        mfd.set_value(1.0)
        self.bring_to_front(frame, sls)
        self.play(FadeOut(bd), FadeOut(diff_l), FadeOut(refr_ar), FadeOut(refr_l), run_time=0.4)
        self.play(msw.animate.set_value(1.05), march.animate.set_value(1.0), run_time=2.9, rate_func=linear)
        march.set_value(0.0)

        # ---------------- a5: resonators: a geometric series
        geo = MathTex("\\sum_{n\\ge 0} \\big(r a\\, e^{i\\varphi}\\big)^n", "=", "\\frac{1}{1 - r a\\, e^{i\\varphi}}", font_size=46)
        geo.move_to([0, -1.25, 0])
        geo_l = tex("every round trip, added up", 26, GREY_B).next_to(geo, DOWN, buff=0.3)
        self.at(S("a5"))
        sls.clear_updaters()
        self.play(hl.animate.set_value(3.0), mfd.animate.set_value(0.0), FadeOut(guides), FadeOut(frame), FadeOut(sls), run_time=0.6)
        mim.clear_updaters()
        self.remove(mim)
        self.play(Write(geo), run_time=1.4)
        self.play(FadeIn(geo_l), run_time=0.5)

        # ---------------- a6: the catch: forward-only, gentle angles
        cimg_cp, cfd_cp = self.static_image(img, 4.8, [-3.3, -1.3, 0], 1.6)
        rf = RingField()
        rh = 2.6
        rimg, rfd = self.live_image(lambda t: rf.rgba(t, {"G": 1.0}, None, 0.8), rh * rf.shape[1] / rf.shape[0], [3.3, -1.35, 0], rh)
        ok_l = tex("couplers: BPM is fine", 28, C_OK).next_to(cimg_cp, UP, buff=0.2)
        bad_l = tex("rings, tight bends: full wave equation", 28, C_EV).move_to([3.3, 0.3, 0])
        nxt = tex("(next episode)", 24, GREY_B).next_to(rimg, DOWN, buff=0.12)
        self.at(S("a6"))
        self.play(FadeOut(geo), FadeOut(geo_l), hl.animate.set_value(1.5), run_time=0.5)
        self.add(cimg_cp)
        self.play(cfd_cp.animate.set_value(1.0), FadeIn(ok_l), run_time=0.7)
        self.at(S("a6") + 3.6)
        self.add(rimg)
        self.play(rfd.animate.set_value(1.0), FadeIn(bad_l), FadeIn(nxt), run_time=0.8)
        self.finish()


# ============================================================ 7  recap ==========================
class Ch7Recap(EpChapter):
    CH = 7
    TAG = "7\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        t1 = tex("Trapped light settles into \\emph{modes},\\\\ the eigenvectors of the cross-section.", 36)
        t2 = tex("A resonator stores light\\\\ where round trips add \\emph{in phase}.", 36)
        t1.move_to([-3.4, 1.9, 0])
        t2.move_to([3.4, 1.9, 0])
        # icons: three mode profiles; aligned phasor arrows
        xs = np.linspace(-1.6, 1.6, 300)
        icon1 = VGroup()
        for m, y0 in zip(range(3), (0.35, -0.65, -1.65)):
            icon1.add(Line([-5.2, y0, 0], [-1.6, y0, 0]).set_stroke(GREY_D, 1))
            icon1.add(poly_line(-3.4 + 1.05 * xs, y0 + 0.36 * mode_profile(m, xs / 1.1), C_U, 3.5))
        icon2 = VGroup()
        L0, q = 0.5, 0.86
        x = 1.0
        for k in range(16):
            l = L0 * q ** k
            icon2.add(fast_arrow([x, -0.65, 0], [x + l, -0.65, 0], C_U, 3.5, min(0.14, 0.45 * l), 0.4 + 0.6 * q ** k))
            x += l
        icon2.add(fast_arrow([1.0, -1.15, 0], [x, -1.15, 0], YELLOW, 6, 0.22))
        icon2.add(tex("in phase: one long arrow", 26, YELLOW).move_to([3.4, -1.65, 0]))
        self.at(S("z1"))
        self.play(FadeIn(t1, shift=0.1 * UP), run_time=0.9)
        self.play(LaggedStart(*[Create(c) for c in icon1], lag_ratio=0.15), run_time=1.2)
        self.at(S("z1") + 3.9)
        self.play(FadeIn(t2, shift=0.1 * UP), run_time=0.9)
        self.play(LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else FadeIn(a) for a in icon2], lag_ratio=0.08), run_time=1.3)

        # ---------------- z2: where it breaks -> Maxwell
        mx = MathTex("\\nabla\\times\\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}", "\\qquad",
                     "\\nabla\\times\\mathbf{H} = \\frac{\\partial \\mathbf{D}}{\\partial t}", font_size=50).move_to([0, 1.1, 0])
        brk = tex("sharp bends $\\cdot$ gratings $\\cdot$ nanostructures", 30, GREY_A).move_to([0, 2.55, 0])
        self.at(S("z2"))
        self.play(VGroup(t1, t2, icon2).animate.set_opacity(0.35), icon1.animate.set_stroke(opacity=0.35), run_time=0.6)
        self.at(S("z2") + 2.6)
        self.play(FadeOut(VGroup(t1, t2, icon1, icon2)), FadeIn(brk, shift=0.1 * UP), run_time=0.7)
        self.at(S("z2") + 6.2)
        self.play(Write(mx), run_time=1.6)

        # ---------------- z3: teaser
        nc = next_card("Maxwell on a grid: FDTD and the perfectly matched layer", y=-1.0)
        self.at(S("z3") + 0.2)
        self.play(FadeIn(nc, shift=0.1 * UP), run_time=0.9)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Modes", "Ch2Eigen", "Ch3Coupler", "Ch4Ring", "Ch5Linewidth", "Ch6Computer", "Ch7Recap"]
