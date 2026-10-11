"""E01 · The Angular Spectrum Method, from scratch -- chapter scenes (Optics in 5 Minutes).

The pilot (angular_spectrum_method_5min/asm5_scenes.py) ported onto the shared series pipeline: o5common replaces
asm5_common, and the computed fields come from physics.py (data/*.npz) instead of being recomputed in every render piece.
Render: ./render.sh --skip-audio ; preview: ./preview.sh Ch3PlaneWaves
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403


def load(name):
    return np.load(data_path(name))


# ---- pilot-only diagram helpers (from asm5_common.py, unchanged) ----
def k_arrow_in_panel(P0, H, kx_t, length=1.5, show_angle=True, comps=False, op=None):
    def build():
        kx = kx_t.get_value()
        o = op.get_value() if op is not None else 1.0
        if kx > 1.0 or o <= 0.002:
            return VGroup()
        th = np.arcsin(np.clip(kx, -1, 1))
        base = P0 + np.array([0, -H + 0.55, 0])
        d = np.array([np.sin(th), np.cos(th), 0])
        tip = base + length * d
        g = VGroup(fast_arrow(base, tip, WHITE, 5, 0.25))
        if show_angle:
            g.add(fast_dashed(base, base + length * UP, 0.06, GREY_B, 1.5))
            if abs(th) > 0.08:
                g.add(Arc(radius=0.7, start_angle=PI / 2, angle=-th, arc_center=base).set_stroke(WHITE, 2))
                g.add(MathTex("\\theta", font_size=30).move_to(base + 0.95 * np.array([np.sin(th / 2), np.cos(th / 2), 0])))
        if comps:
            corner = np.array([tip[0], base[1], 0])
            g.add(Line(base, corner).set_stroke(C_KX, 5), Line(corner, tip).set_stroke(C_KZ, 5))
            g.add(MathTex("k_x", font_size=30, color=C_KX).next_to(Line(base, corner), DOWN, buff=0.08))
            g.add(MathTex("k_z", font_size=30, color=C_KZ).next_to(Line(corner, tip), RIGHT if kx >= 0 else LEFT, buff=0.08))
        return fade_vgroup(g, o)
    return always_redraw(build)


def pipeline(y=0.0, font=40, buff=2.05):
    boxes = VGroup(*[MathTex(s, font_size=font) for s in ("U(x,0)", "A(k_x)", "A(k_x)\\,e^{\\,i k_z z}", "U(x,z)")])
    for b, c in zip(boxes, (C_U, C_KX, C_H, C_U)):
        b.set_color(c)
    boxes.arrange(RIGHT, buff=buff).move_to([0, y, 0])
    frames = VGroup(*[SurroundingRectangle(b, buff=0.2, color=GREY_B, corner_radius=0.08) for b in boxes])
    arrows, labs = VGroup(), VGroup()
    for a, b, s, c in zip(frames[:-1], frames[1:], ("\\text{FFT}", "\\times\\, e^{\\,i k_z z}", "\\text{FFT}^{-1}"),
                          (WHITE, C_H, WHITE)):
        ar = Arrow(a.get_right(), b.get_left(), buff=0.08, stroke_width=4, color=GREY_A)
        arrows.add(ar)
        labs.add(MathTex(s, font_size=30, color=c).next_to(ar, UP, buff=0.12))
    return boxes, frames, arrows, labs


LAM_D = 0.78          # wavelength of the wave panels (scene units)


def phasor_pair(scene, P0, H, kx_t, dz):
    """Phase dials at z = 0 and z = d beside a wave panel (the factor e^{i k_z d})."""
    y_base = P0[1] - H

    def build():
        kx = kx_t.get_value()
        kz = TAU_ / LAM_D * np.sqrt(max(0.0, 1 - kx ** 2))
        d = dz.get_value()
        g = VGroup()
        for z, ang, col in ((0.0, 0.0, C_U), (d, kz * d, C_H)):
            y = y_base + 0.25 + z
            cen = np.array([P0[0] + H + 0.62, y, 0])
            g.add(fast_dashed([P0[0] - H, y, 0], [P0[0] + H, y, 0], 0.08, col, 2, 0.8))
            g.add(Circle(radius=0.36).move_to(cen).set_stroke(GREY_B, 1.5))
            g.add(fast_arrow(cen, cen + 0.36 * np.array([np.cos(ang), np.sin(ang), 0]), col, 4, 0.13))
        return g
    return always_redraw(build)


def panel_frame(P0, H):
    sq = Square(side_length=2 * H).move_to(P0).set_stroke(GREY_B, 2)
    px = MathTex("x", font_size=30, color=GREY_B).next_to(sq, DOWN, buff=0.1).shift((H - 0.2) * RIGHT)
    pz = MathTex("z", font_size=30, color=GREY_B).next_to(sq, LEFT, buff=0.1).shift((H - 0.2) * UP)
    return VGroup(sq, px, pz)


def evanescent_envelope(P0, H, kx_t):
    def build():
        kx = kx_t.get_value()
        if kx <= 1.02:
            return VGroup()
        kap = TAU_ / LAM_D * np.sqrt(kx ** 2 - 1)
        zz = np.linspace(0, 2 * H, 120)
        x0 = P0[0] + H + 0.25
        y0 = P0[1] - H
        return VGroup(VMobject().set_points_as_corners([[x0 + 0.9 * np.exp(-kap * z), y0 + z, 0] for z in zz]).set_stroke(C_EV, 3),
                      Line([x0, y0, 0], [x0, y0 + 2 * H, 0]).set_stroke(GREY_C, 1.5),
                      MathTex("e^{-\\kappa z}", font_size=32, color=C_EV).move_to([x0 + 0.55, y0 + 1.25, 0]))
    return always_redraw(build)


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        nz, nxr = 1215, 702
        A = load("hook.npz")["A"]                       # 8-wavelength slit, |U| for z = 0..240 (physics.py)
        base = heat_rgba(A, 1.15, gamma=0.55, flip=False)
        cz = np.linspace(0, 1, nz)
        sweep = ValueTracker(0.0)
        hw, hh, hc = 9.0, 5.2, np.array([0.35, 0.1, 0])
        z_l = hc[0] - hw / 2

        def heat(t):
            img = base.copy()
            img[..., 3] = (255 * np.clip((sweep.get_value() - cz) / 0.01, 0, 1))[None, :].astype(np.uint8)
            return img
        hm, hm_f = self.live_image(heat, hw, hc, height=hh)
        laser = VGroup(RoundedRectangle(corner_radius=0.08, width=1.1, height=0.5).set_fill(GREY_E, 1).set_stroke(GREY_B, 2),
                       Tex("laser", font_size=26, color=GREY_B)).move_to([-6.05, 0.1, 0])
        laser[1].next_to(laser[0], DOWN, buff=0.12)
        gap = 4.0 / 40 * hh / 2
        barrier = VGroup(Line([z_l, hc[1] + gap, 0], [z_l, hc[1] + hh / 2, 0]),
                         Line([z_l, hc[1] - hh / 2, 0], [z_l, hc[1] - gap, 0])).set_stroke(WHITE, 6)
        beam = Rectangle(width=z_l - (-5.5), height=0.6).set_stroke(width=0).set_fill(BLUE_B, 0.55)
        beam.move_to([0.5 * (z_l - 5.5), hc[1], 0])
        wall = Rectangle(width=0.14, height=hh).move_to([hc[0] + hw / 2 + 0.2, hc[1], 0]).set_fill(GREY_E, 1).set_stroke(GREY_B, 1)
        last = A[:, -1:]
        wall_img = ImageMobject(heat_rgba(np.repeat(last, 8, axis=1), last.max(), gamma=0.6, flip=False))
        wall_img.stretch_to_fit_width(0.14).stretch_to_fit_height(hh).move_to(wall)
        wall_lab = Tex("wall", font_size=26, color=GREY_B).next_to(wall, DOWN, buff=0.12)
        slit_lab = Tex("slit", font_size=26, color=GREY_B).next_to(barrier, DOWN, buff=0.12)
        self.add(hm)
        self.play(FadeIn(laser), FadeIn(barrier), FadeIn(wall), FadeIn(wall_lab), FadeIn(slit_lab), hm_f.animate.set_value(1.0),
                  run_time=0.8)
        self.at(S("h1"))
        self.play(GrowFromEdge(beam, LEFT), run_time=0.7)
        self.play(sweep.animate.set_value(1.0), run_time=3.2, rate_func=linear)
        self.play(FadeIn(wall_img), run_time=0.6)

        # --- h2: what is the field at an arbitrary distance?
        probe = ValueTracker(0.15)
        q = MathTex("U(x,z)", "=", "?", font_size=34)
        q[0].set_color(C_U)
        q[2].set_color(YELLOW)
        q.add_updater(lambda m: m.next_to([z_l + probe.get_value() * hw, hc[1] + hh / 2, 0], UP, buff=0.08))

        def rider():
            f = probe.get_value()
            j = int(np.clip(f, 0, 1) * (nz - 1))
            zl = z_l + f * hw
            ys = hc[1] + hh / 2 - (np.arange(nxr) + 0.5) / nxr * hh
            pts = [[zl + 0.75 * a_, y_, 0] for y_, a_ in zip(ys[::3], A[::3, j])]
            return VGroup(DashedLine([zl, hc[1] - hh / 2, 0], [zl, hc[1] + hh / 2, 0], dash_length=0.08).set_stroke(WHITE, 2),
                          VMobject().set_points_as_corners(pts).set_stroke(YELLOW, 3))
        rid = always_redraw(rider)
        self.at(S("h2") + 0.2)
        self.add(q)
        self.play(FadeIn(rid), FadeIn(q), run_time=0.5)
        self.play(probe.animate.set_value(0.85), run_time=2.4)
        self.play(probe.animate.set_value(0.45), run_time=1.6)

        # --- h3/h4: the method, and what it needs
        title = Tex("The Angular Spectrum Method", font_size=68)
        tback = BackgroundRectangle(title, fill_opacity=0.6, buff=0.25).set_fill(BLACK)
        tg = VGroup(tback, title).move_to([0, 0.9, 0])
        self.at(S("h3"))
        q.clear_updaters()
        self.play(FadeOut(rid), FadeOut(q), hm_f.animate.set_value(0.28), beam.animate.set_fill(opacity=0.2),
                  FadeOut(wall_img), run_time=0.5)
        self.play(FadeIn(tback), Write(title), run_time=1.5)
        b1 = VGroup(MathTex("\\frac{d}{dx}\\,,\\;\\int", font_size=44), Tex("calculus", font_size=30, color=GREY_A)).arrange(DOWN, buff=0.2)
        b2 = VGroup(MathTex("\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}", font_size=40),
                    Tex("linear algebra", font_size=30, color=GREY_A)).arrange(DOWN, buff=0.2)
        badges = VGroup(*[VGroup(SurroundingRectangle(b, buff=0.25, corner_radius=0.12, color=GREY_B).set_fill(BLACK, 0.7), b)
                          for b in (b1, b2)]).arrange(RIGHT, buff=1.4).move_to([0, -1.35, 0])
        self.at(S("h4") + 1.4)
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * UP) for b in badges], lag_ratio=0.35), run_time=1.3)
        self.finish()


# ============================================================ 1  waves =============
class Ch1Waves(Chapter):
    CH = 1
    TAG = "1\\quad Waves"

    def construct(self):
        S = self.S
        self.show_tag()
        lam, y0, amp = 2.4, 1.2, 1.0
        k = TAU_ / lam
        V0, T0s, T1s, DEC = 0.8, self.t0, S("w2"), 0.6        # drift speed, start, stop time, deceleration

        def shift_at(t):
            if t <= T1s:
                return V0 * (t - T0s)
            tau = min(t - T1s, DEC)
            return V0 * (T1s - T0s) + V0 * (tau - tau ** 2 / (2 * DEC))
        shift = type("Shift", (), {"get_value": lambda _self: shift_at(self.now)})()
        axis = Line([-6.4, y0, 0], [6.4, y0, 0]).set_stroke(GREY_C, 2)
        xl = MathTex("x", font_size=34, color=GREY_B).next_to(axis, RIGHT, buff=0.1)
        reveal = ValueTracker(0.0)          # not Create(wave): it would suspend the redraw (wave stalls, then jumps)

        def wave_fn():
            r = reveal.get_value()
            if r < 1e-3:
                return VMobject()
            return graph_line(lambda x: amp * np.cos(k * (x - shift.get_value())), -6.4, -6.4 + 12.8 * r, y0, 1, 1,
                              color=C_U, width=4)
        wave = always_redraw(wave_fn)
        self.at(S("w1"))
        self.play(Create(axis), FadeIn(xl), run_time=0.7)
        self.add(wave)
        self.play(reveal.animate.set_value(1.0), run_time=1.6)

        # --- w2: wavelength
        self.at(S("w2"))
        self.wait(DEC)
        s0 = shift_at(T1s + DEC)
        n0 = np.ceil((-2.6 - s0) / lam)
        x1, x2 = s0 + n0 * lam, s0 + (n0 + 1) * lam
        crest = VGroup(DashedLine([x1, y0 + amp, 0], [x1, y0 - 1.25, 0], dash_length=0.07),
                       DashedLine([x2, y0 + amp, 0], [x2, y0 - 1.25, 0], dash_length=0.07)).set_stroke(GREY_A, 1.5)
        br = BraceBetweenPoints([x1, y0 + amp + 0.12, 0], [x2, y0 + amp + 0.12, 0], UP, color=WHITE)
        lam_l = MathTex("\\lambda", font_size=44).next_to(br, UP, buff=0.08)
        self.play(Create(crest), GrowFromCenter(br), Write(lam_l), run_time=1.0)

        # --- w3: wavenumber
        eq = MathTex("k", "=", "{2\\pi", "\\over", "\\lambda}", font_size=66).move_to([-1.2, -1.15, 0])
        eq_l = Tex("wavenumber", font_size=30, color=GREY_B).next_to(eq, DOWN, buff=0.18)
        self.at(S("w3"))
        self.play(Write(eq), run_time=1.2)
        self.play(FadeIn(eq_l), run_time=0.5)

        # --- w4: phase turns k radians per unit length
        u = ValueTracker(0.0)
        dial_c = np.array([3.4, -1.25, 0])

        def dial():
            a = TAU_ * u.get_value()
            xx = x1 + u.get_value() * lam
            yy = y0 + amp * np.cos(k * (xx - shift.get_value()))
            g = VGroup(Circle(radius=0.75).move_to(dial_c).set_stroke(GREY_B, 2),
                       fast_arrow(dial_c, dial_c + 0.75 * np.array([np.cos(a), np.sin(a), 0]), YELLOW, 5, 0.2),
                       Dot([xx, yy, 0], radius=0.09, color=YELLOW))
            if a > 0.05:
                g.add(Arc(radius=0.35, start_angle=0, angle=a, arc_center=dial_c).set_stroke(YELLOW, 3))
            return g
        dl = always_redraw(dial)
        dial_l = MathTex("\\text{phase} = k\\,x", font_size=36).next_to(dial_c, RIGHT, buff=1.0)
        turn_l = MathTex("\\text{one } \\lambda \\;\\Rightarrow\\; 2\\pi", font_size=34, color=YELLOW).next_to(dial_l, DOWN, buff=0.25)
        self.at(S("w4"))
        self.play(FadeIn(dl), FadeIn(dial_l), run_time=0.6)
        self.play(u.animate.set_value(1.0), run_time=3.2, rate_func=linear)
        self.play(FadeIn(turn_l), run_time=0.5)

        # --- w5: one colour, one k
        cols = [(RED_C, 2.9, 1.95, "red"), (GREEN_C, 2.3, 0.85, "green"), (PURPLE_A, 1.75, -0.25, "violet")]
        waves3 = VGroup(*[graph_line(lambda x, l=l: 0.38 * np.cos(TAU_ / l * x), -5.0, 6.2, yy, 1, 1, color=c, width=3.5)
                          for c, l, yy, _ in cols])
        names = VGroup(*[Tex(nm, font_size=28, color=c).move_to([-5.8, yy, 0]) for c, _, yy, nm in cols])
        rule = Tex("one colour $\\;\\Rightarrow\\;$ one $\\lambda$ $\\;\\Rightarrow\\;$ one fixed $k$", font_size=42)
        rule = VGroup(SurroundingRectangle(rule, buff=0.2, corner_radius=0.1, color=YELLOW), rule).move_to([0.3, -1.85, 0])
        self.at(S("w5"))
        wave.clear_updaters()
        self.play(FadeOut(VGroup(crest, br, lam_l, eq, eq_l, dial_l, turn_l, axis, xl)), FadeOut(dl), run_time=0.5)
        self.remove(dl)
        self.play(ReplacementTransform(wave, waves3[1]), FadeIn(waves3[0]), FadeIn(waves3[2]), FadeIn(names), run_time=1.0)
        self.at(S("w5") + 2.6)
        self.play(Create(rule[0]), Write(rule[1]), run_time=1.1)
        self.finish()


# ============================================================ 2  complex arrows ====
class Ch2Arrows(Chapter):
    CH = 2
    TAG = "2\\quad Complex numbers as arrows"

    def construct(self):
        S = self.S
        self.show_tag()
        C = np.array([-4.35, -0.15, 0])
        R = 1.45
        axes = VGroup(Arrow(C + 1.95 * LEFT, C + 2.05 * RIGHT, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04),
                      Arrow(C + 1.85 * DOWN, C + 1.95 * UP, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04))
        axl = VGroup(Tex("Re", font_size=28, color=GREY_B).next_to(axes[0], RIGHT, buff=0.08),
                     Tex("Im", font_size=28, color=GREY_B).next_to(axes[1], UP, buff=0.06))
        circ = Circle(radius=R).move_to(C).set_stroke(GREY_C, 2)
        th = ValueTracker(0.75)
        rho = ValueTracker(0.0)                   # common rotation (z6)

        def euler_arrow():
            a = th.get_value() + rho.get_value()
            tip = C + R * np.array([np.cos(a), np.sin(a), 0])
            foot = C + R * np.cos(a) * RIGHT
            g = VGroup(Line(C, foot).set_stroke(C_U, 6), fast_dashed(foot, tip, 0.06, C_IM, 4),
                       fast_arrow(C, tip, WHITE, 5, 0.22))
            if a % TAU_ > 0.12 and abs(rho.get_value()) < 1e-3:
                g.add(Arc(radius=0.45, start_angle=0, angle=a % TAU_, arc_center=C).set_stroke(YELLOW, 2.5),
                      MathTex("\\theta", font_size=30, color=YELLOW).move_to(C + 0.72 * np.array([np.cos(a % TAU_ / 2), np.sin(a % TAU_ / 2), 0])))
            return g
        ea = always_redraw(euler_arrow)
        self.at(S("z1"))
        self.play(GrowArrow(axes[0]), GrowArrow(axes[1]), FadeIn(axl), Create(circ), run_time=1.2)

        euler = MathTex("e^{i\\theta}", "=", "\\cos\\theta", "+", "i\\sin\\theta", font_size=50).move_to([-3.9, 2.55, 0])
        euler[2].set_color(C_U)
        euler[4].set_color(C_IM)
        one = MathTex("1", font_size=30)
        self.at(S("z2"))
        self.play(Write(euler), run_time=1.8)
        self.add(ea)
        self.play(FadeIn(ea), run_time=0.4)
        self.play(th.animate.set_value(2.3), run_time=2.0)
        self.play(th.animate.set_value(0.9), run_time=1.6)

        # ---------------- a row of arrows = a field along x
        xb = np.linspace(-0.35, 6.2, 11)
        yb, L = 1.35, 0.6
        kw = TAU_ / 2.2
        m = ValueTracker(0.0)
        n_show = ValueTracker(0.0)
        sh_op = ValueTracker(0.0)
        gr_op = ValueTracker(0.0)

        def amp_ph(x):
            a0 = 0.3 + 0.7 * np.exp(-((x - 2.9) / 1.6) ** 2)
            p0 = 1.2 * np.sin(0.9 * x) + 0.4 * x
            mm = m.get_value()
            return (1 - mm) * a0 + mm, (1 - mm) * p0 + mm * kw * x + rho.get_value()

        base_line = Line([-0.9, yb, 0], [6.75, yb, 0]).set_stroke(GREY_C, 1.5)

        def row():
            g = VGroup()
            ns = n_show.get_value()
            for i, x in enumerate(xb):
                if i >= ns:
                    break
                a, p = amp_ph(x)
                tip = np.array([x + L * a * np.cos(p), yb + L * a * np.sin(p), 0])
                g.add(fast_arrow([x, yb, 0], tip, WHITE, 3.5, 0.13))
                g.add(Dot([x, yb, 0], radius=0.025, color=GREY_B))
                so = sh_op.get_value()
                if so > 0:
                    g.add(Line([x, yb, 0], [x + L * a * np.cos(p), yb, 0]).set_stroke(C_U, 7, opacity=so))
                    g.add(fast_dashed(tip, [tip[0], yb, 0], 0.04, GREY_B, 1.2, so))
            return g
        rw = always_redraw(row)
        yg = -1.35

        def graph():
            o = gr_op.get_value()
            if o <= 0:
                return VGroup()
            xx = np.linspace(-0.6, 6.5, 400)
            a, p = amp_ph(xx)
            g = VGroup(Line([-0.9, yg, 0], [6.75, yg, 0]).set_stroke(GREY_C, 1.5),
                       VMobject().set_points_as_corners([[x, yg + 0.8 * v, 0] for x, v in zip(xx, a * np.cos(p))]).set_stroke(C_U, 4))
            ai, pi_ = amp_ph(xb)
            for x, v in zip(xb, ai * np.cos(pi_)):
                g.add(Dot([x, yg + 0.8 * v, 0], radius=0.05, color=C_U))
            return fade_vgroup(g, o)
        gr = always_redraw(graph)
        lab_len = Tex("length = amplitude,\\; angle = phase", font_size=32).move_to([2.9, 2.45, 0])
        lab_real = Tex("the light we see: each arrow's shadow", font_size=30, color=C_U).move_to([2.9, -0.35, 0])
        self.at(S("z3"))
        self.add(base_line, rw)
        self.play(FadeIn(base_line), n_show.animate.set_value(11.0), run_time=1.8, rate_func=linear)
        self.play(FadeIn(lab_len), run_time=0.6)
        self.at(S("z4"))
        self.add(gr)
        self.play(sh_op.animate.set_value(1.0), run_time=0.8)
        self.play(gr_op.animate.set_value(1.0), FadeIn(lab_real), run_time=1.0)

        # ---------------- z5: the pure wave e^{ikx}
        lab_wave = MathTex("U(x)", "=", "e^{ikx}", font_size=42).move_to([2.9, 2.45, 0])
        lab_wave[0].set_color(C_U)
        self.at(S("z5"))
        self.play(m.animate.set_value(1.0), FadeTransform(lab_len, lab_wave), run_time=2.2)

        # ---------------- z6: shift = rotate every arrow = multiply by e^{i phi}
        payoff = MathTex("\\text{shift the wave}", "\\;=\\;", "\\text{rotate every arrow}", "\\;=\\;",
                         "\\times\\,e^{i\\varphi}", font_size=38).move_to([1.0, -2.65, 0])
        payoff[4].set_color(YELLOW)
        self.at(S("z6"))
        self.play(rho.animate.set_value(-0.75 * TAU_), run_time=3.0)
        self.play(Write(payoff), run_time=1.4)
        self.play(rho.animate.set_value(-1.25 * TAU_), run_time=2.2)
        self.finish()


# ============================================================ 3  plane waves =======
class Ch3PlaneWaves(Chapter):
    CH = 3
    TAG = "3\\quad Plane waves"

    def construct(self):
        S = self.S
        self.show_tag()
        P0, H = np.array([3.55, -0.15, 0]), 2.35
        frame = panel_frame(P0, H)
        fwd = VGroup(Arrow([P0[0] - H - 0.5, P0[1] - H + 0.3, 0], [P0[0] - H - 0.5, P0[1] + H - 0.4, 0], buff=0, stroke_width=3,
                           color=GREY_B, max_tip_length_to_length_ratio=0.06),
                     Tex("forward", font_size=26, color=GREY_B))
        fwd[1].rotate(PI / 2).next_to(fwd[0], LEFT, buff=0.08)
        kx_t = ValueTracker(0.0)
        w_op = ValueTracker(0.0)
        a_op = ValueTracker(0.0)
        wv = wave_panel(self, P0, H, kx_t, LAM_D, op=w_op)
        karr = k_arrow_in_panel(P0, H, kx_t, show_angle=False, comps=True, op=a_op)
        self.at(S("p1"))
        self.play(Create(frame[0]), FadeIn(frame[1:]), FadeIn(fwd), run_time=1.0)

        eq_pw = MathTex("U", "=", "e^{\\,i(", "k_x", "x", "+", "k_z", "z", ")}", font_size=50).move_to([-3.5, 2.75, 0])
        eq_pw[0].set_color(C_U)
        eq_pw[3].set_color(C_KX)
        eq_pw[6].set_color(C_KZ)
        self.at(S("p2"))
        self.add(wv)
        self.play(Write(eq_pw), w_op.animate.set_value(1.0), run_time=1.5)
        self.play(kx_t.animate.set_value(0.35), run_time=1.5)

        self.at(S("p3"))
        self.add(karr)
        self.play(a_op.animate.set_value(1.0), run_time=0.6)
        self.play(kx_t.animate.set_value(-0.3), run_time=2.0)
        self.play(kx_t.animate.set_value(0.35), run_time=1.5)

        helm = MathTex("{\\partial^2 U \\over \\partial x^2}", "+", "{\\partial^2 U \\over \\partial z^2}", "+", "k^2 U", "=", "0",
                       font_size=42).move_to([-3.5, 1.35, 0])
        helm_l = Tex("Helmholtz equation: the rule light obeys", font_size=26, color=GREY_B).next_to(helm, DOWN, buff=0.15)
        self.at(S("p4"))
        self.play(Write(helm), run_time=1.6)
        self.play(FadeIn(helm_l), run_time=0.5)

        L1 = MathTex("{\\partial \\over \\partial x}\\, e^{\\,i(k_x x + k_z z)}", "=", "i k_x", "\\, e^{\\,i(k_x x + k_z z)}",
                     font_size=34).move_to([-3.5, -0.1, 0])
        L1[2].set_color(C_KX)
        L2 = MathTex("\\big[", "(ik_x)^2", "+", "(ik_z)^2", "+", "k^2", "\\big]\\,U", "=", "0", font_size=40).move_to([-3.5, -1.15, 0])
        L2[1].set_color(C_KX)
        L2[3].set_color(C_KZ)
        L3 = MathTex("k_x^2", "+", "k_z^2", "=", "k^2", font_size=54).move_to([-3.5, -2.3, 0])
        L3[0].set_color(C_KX)
        L3[2].set_color(C_KZ)
        box = SurroundingRectangle(L3, buff=0.15, color=YELLOW)
        self.at(S("p5"))
        self.play(FadeOut(helm_l), Write(L1), run_time=1.5)
        self.at(S("p5") + 3.0)
        self.play(Write(L2), run_time=1.3)
        self.at(S("p5") + 5.6)
        self.play(TransformFromCopy(L2[1], L3[0]), TransformFromCopy(L2[3], L3[2]), TransformFromCopy(L2[5], L3[4]),
                  FadeIn(L3[1]), FadeIn(L3[3]), run_time=1.2)
        self.play(Create(box), run_time=0.5)

        # ---------------- p6: the k-circle
        kc, R = np.array([-3.6, -1.6, 0]), 1.2
        circ, kvec = k_circle(kc, R, kx_t)
        kz_eq = MathTex("k_z", "=", "\\sqrt{k^2 - k_x^2}", font_size=44).move_to([-3.5, 0.85, 0])
        kz_eq[0].set_color(C_KZ)
        self.at(S("p6"))
        self.play(FadeOut(VGroup(helm, L1, L2)), VGroup(L3, box).animate.move_to([-3.5, 1.75, 0]), run_time=0.8)
        self.play(Create(circ[0]), GrowArrow(circ[1]), GrowArrow(circ[2]), FadeIn(circ[3:]), run_time=1.0)
        self.add(kvec)
        self.play(kx_t.animate.set_value(-0.6), run_time=1.8)
        self.play(kx_t.animate.set_value(0.5), Write(kz_eq), run_time=2.0)

        # ---------------- p7: the tilt angle
        ang = k_arrow_in_panel(P0, H, kx_t, show_angle=True, comps=False)
        sin_eq = MathTex("\\sin\\theta", "=", "{k_x", "\\over", "k}", font_size=44).next_to(frame[0], UP, buff=0.15)
        sin_eq[2].set_color(C_KX)
        self.at(S("p7"))
        self.add(ang)
        self.play(Write(sin_eq), run_time=1.2)
        self.play(kx_t.animate.set_value(-0.45), run_time=2.2)
        self.finish()


# ============================================================ 4  basis =============
class Ch4Basis(Chapter):
    CH = 4
    TAG = "4\\quad A basis of waves"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- b1: linearity (two plane waves add)
        n = 300
        g = np.linspace(-1, 1, n)
        X, Z = np.meshgrid(g, g[::-1])
        lam_p = 0.42
        kp = TAU_ / lam_p
        th1, th2 = np.radians(28), np.radians(-22)
        E1 = np.exp(1j * kp * (np.sin(th1) * X + np.cos(th1) * Z))
        E2 = np.exp(1j * kp * (np.sin(th2) * X + np.cos(th2) * Z))
        w_om = TAU_ * 0.9

        def field_img(E, scale):
            def fn(t):
                v = np.real(E * np.exp(-1j * w_om * t)) / scale
                rgb = DIVERGE[np.clip((v * 0.5 + 0.5) * 1023, 0, 1023).astype(int)]
                out = np.empty((n, n, 4), np.uint8)
                out[..., :3] = rgb
                out[..., 3] = 255
                return out
            return fn
        pw = 2.7
        cs = [np.array([-4.65, 0.55, 0]), np.array([-0.95, 0.55, 0]), np.array([3.3, 0.55, 0])]
        ims = [self.live_image(field_img(E, s), pw, c) for E, s, c in ((E1, 1.0, cs[0]), (E2, 1.0, cs[1]), (E1 + E2, 2.0, cs[2]))]
        borders = VGroup(*[Square(side_length=pw).move_to(c).set_stroke(GREY_B, 1.5) for c in cs])
        signs = VGroup(MathTex("+", font_size=64).move_to(0.5 * (cs[0] + cs[1])), MathTex("=", font_size=64).move_to([1.17, 0.55, 0]))
        lin = Tex("solution $+$ solution $=$ solution: the equation is \\emph{linear}", font_size=34).move_to([0, -1.55, 0])
        self.at(S("b1"))
        for im, _ in ims:
            self.add(im)
        self.play(LaggedStart(*[f.animate.set_value(1.0) for _, f in ims], lag_ratio=0.35), FadeIn(borders), FadeIn(signs), run_time=1.6)
        self.at(S("b1") + 3.0)
        self.play(Write(lin), run_time=1.2)

        # ---------------- b2: vectors, basis, dot products
        plane = NumberPlane(x_range=[0, 4, 1], y_range=[0, 3, 1], x_length=3.6, y_length=2.7,
                            background_line_style={"stroke_color": GREY_D, "stroke_width": 1}, axis_config={"stroke_color": GREY_B})
        plane.move_to([-4.0, 0.0, 0])
        o = plane.c2p(0, 0)
        v = Arrow(o, plane.c2p(3, 2), buff=0, stroke_width=5, color=WHITE, max_tip_length_to_length_ratio=0.12)
        e1 = Arrow(o, plane.c2p(1, 0), buff=0, stroke_width=6, color=BLUE_C, max_tip_length_to_length_ratio=0.3)
        e2 = Arrow(o, plane.c2p(0, 1), buff=0, stroke_width=6, color=GREEN_C, max_tip_length_to_length_ratio=0.3)
        proj = VGroup(DashedLine(plane.c2p(3, 2), plane.c2p(3, 0), dash_length=0.06),
                      DashedLine(plane.c2p(3, 2), plane.c2p(0, 2), dash_length=0.06)).set_stroke(GREY_A, 1.5)
        labs = VGroup(MathTex("\\vec v", font_size=34).next_to(plane.c2p(3, 2), UR, buff=0.05),
                      MathTex("\\vec e_1", font_size=30, color=BLUE_C).next_to(plane.c2p(1, 0), DOWN, buff=0.12),
                      MathTex("\\vec e_2", font_size=30, color=GREEN_C).next_to(plane.c2p(0, 1), LEFT, buff=0.12))
        dec = MathTex("\\vec v", "=", "3", "\\,\\vec e_1", "+", "2", "\\,\\vec e_2", font_size=42).next_to(plane, UP, buff=0.3)
        dec[3].set_color(BLUE_C)
        dec[6].set_color(GREEN_C)
        dots_eq = MathTex("3 = \\vec v\\cdot\\vec e_1,\\qquad 2 = \\vec v\\cdot\\vec e_2", font_size=36).next_to(plane, DOWN, buff=0.62)
        self.at(S("b2"))
        self.play(*[f.animate.set_value(0.0) for _, f in ims], FadeOut(borders, signs, lin), run_time=0.6)
        for im, _ in ims:
            im.clear_updaters()
            self.remove(im)
        self.play(Create(plane), run_time=0.8)
        self.play(GrowArrow(v), GrowArrow(e1), GrowArrow(e2), FadeIn(labs), run_time=0.8)
        self.play(Create(proj), Write(dec), run_time=1.2)
        self.at(S("b2") + 4.6)
        self.play(Write(dots_eq), run_time=1.2)

        # ---------------- b3: functions are vectors
        xs = np.linspace(0.7, 4.2, 14)
        fvals = 0.25 + 1.1 * np.exp(-((xs - 2.45) / 0.9) ** 4)
        yb3 = -0.9
        stems = VGroup(*[VGroup(Line([x, yb3, 0], [x, yb3 + fv, 0]).set_stroke(C_U, 3), Dot([x, yb3 + fv, 0], radius=0.05, color=C_U))
                         for x, fv in zip(xs, fvals)])
        ax3 = Line([0.4, yb3, 0], [4.5, yb3, 0]).set_stroke(GREY_C, 1.5)
        mat = Matrix([["U(x_1)"], ["U(x_2)"], ["U(x_3)"], ["\\vdots"], ["U(x_N)"]], v_buff=0.65).scale(0.62).move_to([6.05, -0.15, 0])
        mat.get_entries().set_color(C_U)
        to_mat = Arrow([4.55, -0.15, 0], [5.15, -0.15, 0], buff=0, stroke_width=4, color=GREY_A, max_tip_length_to_length_ratio=0.3)
        fl = Tex("a function is a long vector", font_size=32).move_to([3.3, 1.85, 0])
        self.at(S("b3"))
        self.play(Create(ax3), LaggedStart(*[GrowFromEdge(s, DOWN) for s in stems], lag_ratio=0.08), run_time=1.3)
        self.play(GrowArrow(to_mat), FadeIn(mat), FadeIn(fl), run_time=1.0)

        # ---------------- b4: the basis of waves
        rows_y = [2.0, 0.8, -0.4, -1.6]
        xr0, xr1 = 0.9, 5.6
        rows = VGroup()
        rlabs = VGroup()
        dk = TAU_ / (xr1 - xr0)
        for i, yy in enumerate(rows_y):
            rows.add(VGroup(Line([xr0, yy, 0], [xr1, yy, 0]).set_stroke(GREY_D, 1),
                            graph_line(lambda x, i=i: 0.42 * np.cos(i * dk * (x - xr0)), xr0, xr1, yy, 1, 1, color=BLUE_B, width=3)))
            rlabs.add(MathTex("e^{i k_x x}", ",\\;", f"k_x={i}", font_size=28).next_to([xr0, yy, 0], LEFT, buff=0.2))
            rlabs[-1][2].set_color(C_KX)
        field = VGroup(Line([-6.0, 1.0, 0], [-1.4, 1.0, 0]).set_stroke(GREY_C, 1.5),
                       graph_line(lambda x: 1.1 * (np.abs(x + 3.7) < 0.75).astype(float), -6.0, -1.4, 1.0, 1, 1, color=C_U, width=4),
                       MathTex("U(x)", font_size=36, color=C_U).move_to([-3.7, 2.55, 0]))
        rl = Tex("(real parts shown)", font_size=24, color=GREY_B).move_to([3.25, -2.45, 0])
        self.at(S("b4"))
        self.play(FadeOut(VGroup(plane, v, e1, e2, proj, labs, dec, dots_eq, stems, ax3, mat, to_mat, fl)), run_time=0.6)
        self.play(FadeIn(field), LaggedStart(*[Create(r) for r in rows], lag_ratio=0.25), FadeIn(rlabs), FadeIn(rl), run_time=2.0)

        # ---------------- b5: dot product = Fourier transform
        W = 0.75
        cn = [W / np.pi] + [2 * np.sin(i * W) / (i * np.pi) for i in (1, 2, 3)]
        vals = VGroup(*[MathTex(f"{c:.2f}", font_size=32, color=C_KX).next_to([xr1, yy, 0], RIGHT, buff=0.25) for c, yy in zip(cn, rows_y)])
        sum_eq = MathTex("A(k_x)", "=", "\\sum_n", "U(x_n)", "\\,e^{-ik_x x_n}", "\\,\\Delta x", font_size=38).move_to([-3.7, -0.55, 0])
        int_eq = MathTex("A(k_x)", "=", "\\int", "U(x)", "\\,e^{-ik_x x}", "\\,dx", font_size=40).move_to([-3.7, -0.55, 0])
        for e in (sum_eq, int_eq):
            e[0].set_color(C_KX)
            e[3].set_color(C_U)
        conj = Tex("(complex dot product: conjugate the basis wave)", font_size=22, color=GREY_B).move_to([-4.0, -1.45, 0])
        ft = Tex("= the Fourier transform", font_size=34, color=C_KX).move_to([-3.7, -2.15, 0])
        self.at(S("b5"))
        self.play(Write(sum_eq), run_time=1.6)
        self.play(LaggedStart(*[FadeIn(vv, shift=0.1 * LEFT) for vv in vals], lag_ratio=0.25), run_time=1.2)
        self.at(S("b5") + 3.6)
        self.play(TransformMatchingTex(sum_eq, int_eq), run_time=1.2)
        self.play(FadeIn(conj), FadeIn(ft), run_time=0.6)

        # ---------------- b6: spectrum of a slit; low k_x = shape, high k_x = edges
        sx, top_y, amp = 9.0 / TAU_, 1.4, 1.3
        top_axis = Line([-4.5, top_y, 0], [4.5, top_y, 0]).set_stroke(GREY_C, 1.5)
        rect = graph_line(lambda x: (np.abs(x) <= W).astype(float), -np.pi, np.pi, top_y, sx, amp, color=C_U, width=4)
        nsum = ValueTracker(0)

        def series(x, N):
            f = np.full_like(x, W / np.pi)
            for i in range(1, N + 1):
                f += 2 * np.sin(i * W) / (i * np.pi) * np.cos(i * x)
            return f
        psum = always_redraw(lambda: graph_line(lambda x: series(x, int(nsum.get_value())), -np.pi, np.pi, top_y, sx, amp,
                                                color=YELLOW, width=3))
        yb6, sp = -2.35, 0.34
        ns = range(-12, 13)

        def spec_stems():
            g = VGroup(Line([-4.6, yb6, 0], [4.6, yb6, 0]).set_stroke(GREY_C, 1.5))
            N = int(nsum.get_value())
            for i in ns:
                a = W / np.pi if i == 0 else np.sin(abs(i) * W) / (abs(i) * np.pi)
                h = 3.4 * a
                col = YELLOW if abs(i) <= N else GREY_C
                g.add(Line([i * sp, yb6, 0], [i * sp, yb6 + h, 0]).set_stroke(col, 3), Dot([i * sp, yb6 + h, 0], radius=0.04, color=col))
            return g
        spec = always_redraw(spec_stems)
        kxl = MathTex("k_x", font_size=30, color=C_KX).move_to([4.95, yb6, 0])
        low = Tex("low $k_x$: overall shape", font_size=28).move_to([0, -0.55, 0])
        high = VGroup(Tex("high $k_x$: sharp edges", font_size=28).move_to([3.0, -0.95, 0]),
                      Tex("high $k_x$: sharp edges", font_size=28).move_to([-3.0, -0.95, 0]))
        self.at(S("b6"))
        self.play(FadeOut(VGroup(field, rows, rlabs, rl, vals, int_eq, conj, ft)), run_time=0.5)
        self.add(spec, psum)
        self.play(Create(top_axis), FadeIn(rect), FadeIn(kxl), run_time=0.8)
        self.play(nsum.animate.set_value(3), run_time=1.2)
        self.play(FadeIn(low), run_time=0.4)
        self.play(nsum.animate.set_value(12), run_time=2.0)
        self.play(FadeIn(high), run_time=0.5)
        self.finish()


# ============================================================ 5  diagonal ==========
class Ch5Diagonal(Chapter):
    CH = 5
    TAG = "5\\quad Propagation is diagonal"

    def construct(self):
        S = self.S
        self.show_tag()
        P0, H = np.array([3.55, 0.05, 0]), 2.35
        frame = panel_frame(P0, H)
        frame.remove(frame[1])                                   # no x label: the z=0 field sits under the panel
        kx_t = ValueTracker(0.3)
        rev = ValueTracker(0.0)
        wv = wave_panel(self, P0, H, kx_t, LAM_D, reveal=rev)
        speed = 0.75

        def base_wave():
            kx = kx_t.get_value()
            ph = (speed * self.now / LAM_D) % 1.0
            xx = np.linspace(P0[0] - H, P0[0] + H, 300)
            yy = P0[1] - H - 0.45
            return VGroup(Line([P0[0] - H, yy, 0], [P0[0] + H, yy, 0]).set_stroke(GREY_D, 1),
                          VMobject().set_points_as_corners([[x, yy + 0.25 * np.cos(TAU_ / LAM_D * kx * (x - P0[0]) - TAU_ * ph), 0]
                                                            for x in xx]).set_stroke(C_U, 4))
        bw = always_redraw(base_wave)
        kc, R = np.array([-3.9, -1.2, 0]), 1.25
        circ, kvec = k_circle(kc, R, kx_t)
        lab0 = Tex("a basis wave on $z=0$ \\dots", font_size=30).move_to([-3.6, 2.4, 0])
        z0_lab = MathTex("z=0", font_size=28, color=GREY_B).next_to([P0[0] - H, P0[1] - H - 0.45, 0], LEFT, buff=0.15)
        lab1 = Tex("\\dots is the start of a tilted plane wave", font_size=30).move_to([-3.6, 1.8, 0])
        self.at(S("d1"))
        self.play(Create(frame[0]), FadeIn(frame[1:]), FadeIn(circ), run_time=0.8)
        self.add(bw, kvec)
        self.play(FadeIn(bw), FadeIn(lab0), FadeIn(z0_lab), run_time=0.7)
        self.add(wv)
        self.play(rev.animate.set_value(1.0), FadeIn(lab1), run_time=2.0, rate_func=linear)
        self.play(kx_t.animate.set_value(-0.45), run_time=2.0)

        # ---------------- d2: only a phase factor
        dz = ValueTracker(0.0)
        ph = phasor_pair(self, P0, H, kx_t, dz)
        prop = MathTex("e^{ik_x x}", "\\;\\longrightarrow\\;", "e^{ik_x x}", "\\,e^{\\,i k_z z}", font_size=46).move_to([-3.6, 1.85, 0])
        prop[3].set_color(C_H)
        self.at(S("d2"))
        self.play(FadeOut(lab0), FadeOut(lab1), run_time=0.4)
        self.add(ph)
        self.play(Write(prop), run_time=1.3)
        self.play(dz.animate.set_value(3.6), run_time=3.0, rate_func=linear)

        # ---------------- d3: eigenvectors / eigenvalues
        eig = MathTex("\\mathcal{P}_z", "\\big[\\,e^{ik_x x}\\,\\big]", "=", "e^{\\,i k_z z}", "\\,e^{ik_x x}", font_size=46).move_to([-3.5, 2.35, 0])
        eig[3].set_color(C_H)
        ana = MathTex("A", "\\,\\vec v", "=", "\\lambda", "\\,\\vec v", font_size=46).move_to([-3.5, 0.95, 0])
        ana[3].set_color(C_H)
        bv = Brace(eig[1], DOWN, buff=0.08, color=GREY_B)
        bl = Brace(eig[3], DOWN, buff=0.08, color=C_H)
        tv = Tex("eigenvector", font_size=28, color=GREY_A).next_to(bv, DOWN, buff=0.05)
        tl = Tex("eigenvalue", font_size=28, color=C_H).next_to(bl, DOWN, buff=0.05)
        self.at(S("d3"))
        self.play(FadeOut(prop), FadeOut(circ), FadeOut(kvec), run_time=0.5)
        self.remove(kvec)
        self.play(Write(eig), run_time=1.6)
        self.play(GrowFromCenter(bv), FadeIn(tv), GrowFromCenter(bl), FadeIn(tl), run_time=0.8)
        self.play(FadeIn(ana, shift=0.1 * UP), run_time=0.8)

        # ---------------- d4: dense vs diagonal matrix (computed: N = 36 samples, dx = lambda/4, z = 1.5 lambda)
        N, dxs, zz = 36, 0.25, 1.5
        kxs = TAU_ * np.fft.fftfreq(N, d=dxs)
        kzs = np.sqrt((TAU_ ** 2 - kxs ** 2).astype(complex))
        kzs = np.where(kzs.imag < 0, -kzs, kzs)
        Hk = np.exp(1j * kzs * zz)
        Pm = np.fft.ifft(Hk[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)
        order = np.argsort(kxs)

        def mat_img(M, gamma):
            a = np.abs(M) / np.abs(M).max()
            hue = (np.angle(M) % TAU_) / TAU_
            rgb = PHASE_HUE[(hue * 2047).astype(int)].astype(float) * (a ** gamma)[..., None]
            img = np.zeros((N * 10, N * 10, 4), np.uint8)
            img[..., :3] = np.kron(rgb, np.ones((10, 10, 1))).astype(np.uint8)
            img[..., 3] = 255
            return img
        D = np.diag(Hk[order])
        m1 = ImageMobject(mat_img(Pm, 0.5)).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        m2 = ImageMobject(mat_img(D, 0.5)).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        for mm, x in ((m1, -3.3), (m2, 3.3)):
            mm.stretch_to_fit_width(4.0).stretch_to_fit_height(4.0).move_to([x, -0.45, 0])
        f1 = SurroundingRectangle(m1, buff=0.04, color=GREY_B, stroke_width=1.5)
        f2 = SurroundingRectangle(m2, buff=0.04, color=GREY_B, stroke_width=1.5)
        t1 = Tex("basis of points: every point mixes", font_size=30).next_to(m1, UP, buff=0.2)
        t2 = Tex("plane-wave basis: diagonal", font_size=30).next_to(m2, UP, buff=0.2)
        cap = Tex("(computed: 36 samples, colour = phase, brightness = size)", font_size=22, color=GREY_B).move_to([0, -2.85, 0])
        self.at(S("d4"))
        self.play(FadeOut(VGroup(eig, ana, bv, bl, tv, tl, frame, z0_lab)), FadeOut(wv), FadeOut(bw), FadeOut(ph), run_time=0.5)
        self.remove(wv, bw, ph)
        self.play(FadeIn(m1), Create(f1), FadeIn(t1), FadeIn(cap), run_time=0.9)
        self.at(S("d4") + 3.6)
        self.play(FadeIn(m2), Create(f2), FadeIn(t2), run_time=0.9)

        # ---------------- d5: diagonalise
        fz = MathTex("U(z)", "=", "\\mathcal{F}^{-1}", "\\,D\\,", "\\mathcal{F}", "\\,U(0)", font_size=60).move_to([0, 0.55, 0])
        fz[0].set_color(C_U)
        fz[5].set_color(C_U)
        fz[3].set_color(C_H)
        an = MathTex("A", "=", "P", "\\,D\\,", "P^{-1}", font_size=46, color=GREY_A)
        an.shift(fz[3].get_center() - an[3].get_center() + 1.15 * DOWN)
        b_f = Brace(fz[4], DOWN, buff=0.1)
        b_d = Brace(fz[3], DOWN, buff=0.1, color=C_H)
        b_i = Brace(fz[2], DOWN, buff=0.1)
        steps = VGroup(Tex("1. change basis", font_size=30).move_to([4.2, 1.95, 0]),
                       Tex("2. multiply by eigenvalues", font_size=30, color=C_H).move_to([0.3, 1.95, 0]),
                       Tex("3. change back", font_size=30).move_to([-3.7, 1.95, 0]))
        for st, fac in zip(steps, (fz[4], fz[3], fz[2])):
            st.add(Line(st.get_bottom() + 0.06 * DOWN, fac.get_top() + 0.08 * UP).set_stroke(GREY_B, 1.5))
        rl = Arrow([2.6, 2.75, 0], [-2.6, 2.75, 0], buff=0, stroke_width=3, color=GREY_B, max_tip_length_to_length_ratio=0.04)
        rlt = Tex("applied right to left", font_size=24, color=GREY_B).next_to(rl, UP, buff=0.06)
        self.at(S("d5"))
        self.play(FadeOut(Group(m1, m2, f1, f2, t1, t2, cap)), run_time=0.5)
        self.play(Write(fz), run_time=1.4)
        self.play(FadeIn(an, shift=0.1 * UP), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(s, shift=0.1 * DOWN) for s in steps], lag_ratio=0.4), GrowArrow(rl), FadeIn(rlt), run_time=1.8)
        self.finish()


# ============================================================ 6  evanescent ========
class Ch6Evanescent(Chapter):
    CH = 6
    TAG = "6\\quad Evanescent waves"

    def construct(self):
        S = self.S
        self.show_tag()
        P0, H = np.array([3.55, -0.15, 0]), 2.35
        frame = panel_frame(P0, H)
        kx_t = ValueTracker(0.3)
        wv = wave_panel(self, P0, H, kx_t, LAM_D)
        karr = k_arrow_in_panel(P0, H, kx_t)
        kc, R = np.array([-3.9, -0.95, 0]), 1.4
        circ, kvec = k_circle(kc, R, kx_t)
        eq = MathTex("k_z", "=", "\\sqrt{k^2 - k_x^2}", font_size=46).move_to([-3.9, 2.55, 0])
        eq[0].set_color(C_KZ)
        eq2 = MathTex("k_z", "=", "i\\sqrt{k_x^2 - k^2}", font_size=46).move_to(eq).set_color(C_EV)
        self.at(S("e1"))
        self.add(wv, karr, kvec)
        self.play(Create(frame[0]), FadeIn(frame[1:]), FadeIn(circ), FadeIn(eq), run_time=0.8)
        self.at(S("e1") + 2.4)
        self.play(kx_t.animate.set_value(1.0), run_time=2.2, rate_func=rate_functions.ease_in_sine)
        self.play(kx_t.animate.set_value(1.06), TransformMatchingTex(eq, eq2), run_time=1.2, rate_func=rate_functions.ease_out_sine)

        env = evanescent_envelope(P0, H, kx_t)
        ev = Tex("evanescent", font_size=34, color=C_EV).next_to(frame[0], UP, buff=0.15)
        dec = MathTex("e^{\\,i k_z z}", "=", "e^{-\\kappa z}", font_size=44, color=C_EV).move_to([-3.9, 1.7, 0])
        self.at(S("e2"))
        self.add(env)
        self.play(FadeIn(ev), FadeIn(env), run_time=0.8)
        self.play(Write(dec), run_time=1.2)

        # ---------------- e3: the diffraction limit, computed
        # after a few wavelengths only |k_x| < k is left: the best possible image is the field with its
        # spectrum cut at |k_x| = k (inverse FFT of the propagating part)
        dl = load("difflimit.npz")                      # computed in physics.py
        cols = []
        for s_, cx, tag in ((2.0, -3.4, "wide"), (0.3, 3.4, "close")):
            xs_, a0, a3 = dl[f"x_{tag}"], dl[f"a0_{tag}"], dl[f"a3_{tag}"]
            sxs = 2.6 / 3.0
            top = VMobject().set_points_as_corners([[cx + x * sxs, 0.35 + 1.0 * v / a0.max(), 0] for x, v in zip(xs_[::4], a0[::4])]).set_stroke(C_U, 3)
            bot = VMobject().set_points_as_corners([[cx + x * sxs, -2.25 + 1.0 * v / a3.max(), 0] for x, v in zip(xs_[::4], a3[::4])]).set_stroke(C_U, 3)
            axes = VGroup(Line([cx - 2.7, 0.35, 0], [cx + 2.7, 0.35, 0]), Line([cx - 2.7, -2.25, 0], [cx + 2.7, -2.25, 0])).set_stroke(GREY_C, 1.5)
            ttl = Tex(f"two details ${s_:g}\\lambda$ apart", font_size=32).move_to([cx, 2.0, 0])
            cols.append((VGroup(axes[0], top, ttl), VGroup(axes[1], bot)))
        zl = VGroup(MathTex("z=0", font_size=30, color=GREY_B).move_to([0, 0.75, 0]),
                    VGroup(MathTex("|k_x|<k", font_size=30, color=C_KX), Tex("only", font_size=24, color=GREY_B),
                           Tex("survives", font_size=24, color=GREY_B)).arrange(DOWN, buff=0.08).move_to([0, -1.75, 0]))
        zl[1][1].next_to(zl[1][0], UP, buff=0.08)
        ok = Tex("still two: resolved", font_size=30, color=GREEN).move_to([-3.4, -2.75, 0])
        bad = Tex("merged into one blob", font_size=30, color=C_EV).move_to([3.4, -2.75, 0])
        lim = Tex("finer than about $\\lambda/2$ lives only in evanescent waves $\\Rightarrow$ the diffraction limit",
                  font_size=28, color=GREY_A).move_to([0, 2.75, 0])
        self.at(S("e3"))
        self.play(FadeOut(VGroup(frame, circ, eq2, dec, ev)), FadeOut(wv), FadeOut(karr), FadeOut(kvec), FadeOut(env), run_time=0.5)
        self.remove(wv, karr, kvec, env)
        self.play(FadeIn(cols[0][0]), FadeIn(cols[1][0]), FadeIn(zl[0]), run_time=0.8)
        self.play(FadeIn(cols[0][1]), FadeIn(cols[1][1]), FadeIn(zl[1]), run_time=1.0)
        self.play(FadeIn(ok), FadeIn(bad), run_time=0.6)
        self.at(S("e3") + 4.6)
        self.play(FadeIn(lim), run_time=0.7)
        self.finish()


# ============================================================ 7  algorithm =========
class Ch7Algorithm(Chapter):
    CH = 7
    TAG = "7\\quad The algorithm"

    def construct(self):
        S = self.S
        self.show_tag()
        names = ["\\textbf{1}\\quad FFT", "\\textbf{2}\\quad multiply by $H(k_x)$", "\\textbf{3}\\quad inverse FFT"]
        boxes = VGroup()
        for i, nm in enumerate(names):
            t = Tex(nm, font_size=32)
            r = RoundedRectangle(corner_radius=0.12, width=3.7, height=0.72).set_stroke(GREY_B, 2)
            boxes.add(VGroup(r, t).move_to([-4.3 + 4.3 * i, 2.75, 0]))
        self.at(S("a1"))
        self.play(LaggedStart(*[FadeIn(b, shift=0.1 * DOWN) for b in boxes], lag_ratio=0.3), run_time=1.3)

        def hi(i):
            return [boxes[j][0].animate.set_stroke(YELLOW if j == i else GREY_B, 3 if j == i else 2) for j in range(3)]

        # ---------------- a2: step 1
        W = 0.75
        xs = np.linspace(-6.2, -1.7, 28)
        fv = (np.abs((xs + 3.95) / 0.6) <= W).astype(float) * 1.1 + 0.02
        yb = -0.6
        st1 = VGroup(*[VGroup(Line([x, yb, 0], [x, yb + f, 0]).set_stroke(C_U, 3), Dot([x, yb + f, 0], radius=0.04, color=C_U))
                       for x, f in zip(xs, fv)])
        ks = np.arange(-13, 14)
        a_k = np.where(ks == 0, W / np.pi, np.sin(np.abs(ks) * W) / (np.pi * np.where(ks == 0, 1, np.abs(ks))))
        st2 = VGroup(*[VGroup(Line([3.95 + 0.17 * kk, yb, 0], [3.95 + 0.17 * kk, yb + 4.2 * a, 0]).set_stroke(C_KX, 3),
                              Dot([3.95 + 0.17 * kk, yb + 4.2 * a, 0], radius=0.035, color=C_KX)) for kk, a in zip(ks, a_k)])
        ax1 = VGroup(Line([-6.4, yb, 0], [-1.5, yb, 0]), Line([1.6, yb, 0], [6.3, yb, 0])).set_stroke(GREY_C, 1.5)
        lab1 = VGroup(MathTex("U(x_n)", font_size=32, color=C_U).move_to([-3.95, 1.25, 0]),
                      MathTex("A(k_x)", font_size=32, color=C_KX).move_to([3.95, 1.25, 0]))
        fft_ar = Arrow([-1.2, 0.0, 0], [1.3, 0.0, 0], buff=0, stroke_width=5, color=GREY_A)
        fft_t = Tex("FFT", font_size=34).next_to(fft_ar, UP, buff=0.1)
        cost = MathTex("N\\log_2 N \\approx 5\\times10^{4}", "\\quad\\text{vs}\\quad", "N^2 \\approx 1.7\\times10^{7}",
                       "\\qquad(N = 4096)", font_size=32).move_to([0, -2.35, 0])
        cost[3].set_color(GREY_B)
        self.at(S("a2"))
        self.play(*hi(0), Create(ax1[0]), LaggedStart(*[GrowFromEdge(s, DOWN) for s in st1], lag_ratio=0.04), FadeIn(lab1[0]), run_time=1.2)
        self.at(S("a2") + 2.0)
        self.play(GrowArrow(fft_ar), FadeIn(fft_t), Create(ax1[1]), LaggedStart(*[GrowFromEdge(s, DOWN) for s in st2], lag_ratio=0.03),
                  FadeIn(lab1[1]), run_time=1.3)
        self.at(S("a2") + 4.2)
        self.play(Write(cost), run_time=1.2)

        # ---------------- a3: step 2, the transfer function as rotating arrows
        Hq = MathTex("H(k_x)", "=", "e^{\\,i z\\sqrt{k^2-k_x^2}}", font_size=46).move_to([0, 1.45, 0])
        Hq[0].set_color(C_H)
        Hq[2].set_color(C_H)
        zt = ValueTracker(0.0)
        kxa = np.linspace(-1.6, 1.6, 17)
        yr = -0.75

        def h_row():
            Z = zt.get_value()
            g = VGroup(Line([-6.4, yr, 0], [6.4, yr, 0]).set_stroke(GREY_C, 1.5))
            for kk in kxa:
                x = kk / 1.6 * 6.0
                if abs(kk) <= 1:
                    ang, a, col = Z * np.sqrt(1 - kk ** 2), 1.0, C_H
                else:
                    ang, a, col = 0.0, np.exp(-Z * np.sqrt(kk ** 2 - 1)), C_EV
                r = 0.31 * a
                g.add(Circle(radius=0.31).move_to([x, yr, 0]).set_stroke(GREY_C, 1.5))
                if r > 0.01:
                    g.add(fast_arrow([x, yr, 0], [x + r * np.cos(ang), yr + r * np.sin(ang), 0], col, 3.5, 0.1))
            return g
        hr = always_redraw(h_row)
        tk = VGroup(MathTex("-k", font_size=28).move_to([-3.75, yr - 0.55, 0]), MathTex("k", font_size=28).move_to([3.75, yr - 0.55, 0]),
                    MathTex("k_x", font_size=30, color=C_KX).move_to([6.65, yr, 0]))
        hrl = Tex("each frequency turns its own phase; beyond $\\pm k$ it decays", font_size=28, color=GREY_A).move_to([0, -2.0, 0])
        self.at(S("a3"))
        self.play(*hi(1), FadeOut(VGroup(st1, st2, ax1, lab1, fft_ar, fft_t, cost)), run_time=0.5)
        self.play(Write(Hq), run_time=1.2)
        self.add(hr)
        self.play(FadeIn(hr), FadeIn(tk), run_time=0.6)
        self.play(zt.animate.set_value(9.0), FadeIn(hrl), run_time=3.4, rate_func=linear)

        # ---------------- a4: step 3
        alg = load("algorithm.npz")                     # computed in physics.py
        xs4, prof = alg["xs4"], alg["prof"]
        res = VGroup(Line([-5.5, -1.3, 0], [5.5, -1.3, 0]).set_stroke(GREY_C, 1.5),
                     VMobject().set_points_as_corners([[x / 25 * 5.5, -1.3 + 1.6 * a, 0] for x, a in zip(xs4[::3], prof[::3])]).set_stroke(C_U, 4))
        res_l = MathTex("U(x, z)", font_size=40, color=C_U).move_to([0, 1.0, 0])
        self.at(S("a4"))
        self.play(*hi(2), FadeOut(Hq), FadeOut(hr), FadeOut(tk), FadeOut(hrl), run_time=0.5)
        self.remove(hr)
        self.play(Create(res), Write(res_l), run_time=1.2)

        # ---------------- a5: sampling + zero padding (wrap-around computed both ways)
        smp_c = np.array([-3.6, 0.2, 0])
        lam_s = 2.4
        wave_s = graph_line(lambda x: 0.8 * np.cos(TAU_ / lam_s * x), -2.0, 2.0, 0.2, 1, 1, color=C_U, width=3).shift(smp_c[0] * RIGHT)
        dots_s = VGroup(*[Dot([smp_c[0] + x, 0.2 + 0.8 * np.cos(TAU_ / lam_s * x), 0], radius=0.06, color=YELLOW)
                          for x in np.arange(-2.0, 2.01, lam_s / 3)])
        br = BraceBetweenPoints([smp_c[0] - 2.0 + 0 * lam_s, 1.15, 0], [smp_c[0] - 2.0 + lam_s, 1.15, 0], UP)
        brl = MathTex("\\lambda", font_size=34).next_to(br, UP, buff=0.05)
        samp = MathTex("\\Delta x < \\lambda/2", font_size=40, color=YELLOW).move_to([-3.6, -1.25, 0])
        # tilted Gaussian beam in a 20-lambda window: periodic FFT without padding vs zero-padded
        Anp, Ap = alg["Anp"], alg["Ap"]
        imw = []
        for Aw, cx, txt in ((Anp, 1.55, "no padding: light wraps around"), (Ap, 5.0, "zero-padded: correct")):
            im = ImageMobject(heat_rgba(Aw, 1.0, gamma=0.7))
            im.stretch_to_fit_width(2.9).stretch_to_fit_height(3.2).move_to([cx, -0.55, 0])
            imw.append(Group(im, SurroundingRectangle(im, buff=0.02, color=GREY_B, stroke_width=1.5),
                             Tex(txt, font_size=24).next_to(im, UP, buff=0.15)))
        self.at(S("a5"))
        self.play(FadeOut(res), FadeOut(res_l), *[b[0].animate.set_stroke(GREY_B, 2) for b in boxes], run_time=0.5)
        self.play(Create(wave_s), FadeIn(dots_s), GrowFromCenter(br), FadeIn(brl), run_time=1.0)
        self.play(Write(samp), run_time=0.8)
        self.at(S("a5") + 3.6)
        self.play(FadeIn(imw[0]), run_time=0.8)
        self.play(FadeIn(imw[1]), run_time=0.8)

        # ---------------- a6/a7: slit and lens, computed exactly this way
        def heat_demo(A, vmax, label, center=np.array([-0.9, -0.6, 0]), w=9.0, h=4.4):
            base = heat_rgba(A, vmax)
            rows = np.arange(base.shape[0])[::-1] / (base.shape[0] - 1)
            sw = ValueTracker(0.0)

            def fn(t):
                img = base.copy()
                img[..., 3] = (255 * np.clip((sw.get_value() - rows) / 0.012, 0, 1))[:, None].astype(np.uint8)
                return img
            im, fd = self.live_image(fn, w, center, height=h)
            box = Rectangle(width=w, height=h).move_to(center).set_stroke(GREY_B, 1.5)
            lab = Tex(label, font_size=28, color=GREY_A).next_to(box, RIGHT, buff=0.25).align_to(box, UP)
            return im, fd, sw, box, lab

        A6 = alg["A6"]
        im6, fd6, sw6, box6, lab6 = heat_demo(A6, 1.35, "slit,\\\\$12\\lambda$ wide\\\\[4pt]$z$: 0 to $150\\lambda$")
        slit = VGroup(Line(box6.get_corner(DL), box6.get_corner(DL) + (9.0 * 19 / 50) * RIGHT),
                      Line(box6.get_corner(DR) + (9.0 * 19 / 50) * LEFT, box6.get_corner(DR))).set_stroke(WHITE, 6)
        self.at(S("a6"))
        self.play(FadeOut(Group(wave_s, dots_s, br, brl, samp, *imw)), run_time=0.5)
        self.add(im6)
        self.play(fd6.animate.set_value(1.0), FadeIn(box6), Create(slit), FadeIn(lab6), run_time=0.6)
        self.play(sw6.animate.set_value(1.0), run_time=5.6, rate_func=linear)

        A7 = alg["A7"]
        im7, fd7, sw7, box7, lab7 = heat_demo(A7, A7.max() * 0.75, "lens:\\\\curved starting\\\\phase")
        f_mark = DashedLine(box7.get_left() + (55 / 110 * 4.4 - 2.2) * UP, box7.get_right() + (55 / 110 * 4.4 - 2.2) * UP,
                            dash_length=0.08).set_stroke(GREY_A, 1)
        f_lab = MathTex("f", font_size=30, color=GREY_A).next_to(f_mark, LEFT, buff=0.1)
        self.at(S("a7"))
        self.play(fd6.animate.set_value(0.0), FadeOut(slit), FadeOut(lab6), run_time=0.5)
        im6.clear_updaters()
        self.remove(im6, box6)
        self.add(box7, im7)
        self.play(fd7.animate.set_value(1.0), FadeIn(lab7), run_time=0.5)
        self.play(sw7.animate.set_value(1.0), run_time=3.6, rate_func=linear)
        self.play(Create(f_mark), FadeIn(f_lab), run_time=0.5)
        self.finish()


# ============================================================ 8  recap =============
class Ch8Recap(Chapter):
    CH = 8
    TAG = "8\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- r1: exact vs paraxial (k-space)
        sc = 2.9
        o = np.array([0, -2.15, 0])
        ax = VGroup(Arrow(o + 3.6 * LEFT, o + 3.8 * RIGHT, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                    Arrow(o, o + 3.9 * UP, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03))
        axl = VGroup(MathTex("k_x", font_size=30, color=C_KX).next_to(ax[0], RIGHT, buff=0.06),
                     MathTex("k_z", font_size=30, color=C_KZ).next_to(ax[1], UP, buff=0.06))
        kk = np.linspace(-1, 1, 300)
        exact = VMobject().set_points_smoothly([o + sc * np.array([k_, np.sqrt(1 - k_ ** 2), 0]) for k_ in kk]).set_stroke(C_KZ, 5)
        kp = np.linspace(-1.2, 1.2, 300)
        parax = DashedVMobject(VMobject().set_points_smoothly([o + sc * np.array([k_, 1 - k_ ** 2 / 2, 0]) for k_ in kp]),
                               num_dashes=50).set_stroke(C_EV, 3)
        le = MathTex("\\text{exact: } k_z=\\sqrt{k^2-k_x^2}", font_size=32, color=C_KZ).move_to([4.3, 0.9, 0])
        lp = MathTex("\\text{paraxial approx.: } k_z\\approx k-\\frac{k_x^2}{2k}", font_size=30, color=C_EV).move_to([-4.0, 0.9, 0])
        top = Tex("angular spectrum method: exact at any distance and any angle", font_size=34).move_to([0, 2.65, 0])
        self.at(S("r1"))
        self.play(GrowArrow(ax[0]), GrowArrow(ax[1]), FadeIn(axl), run_time=0.7)
        self.play(Create(exact), FadeIn(le), run_time=1.2)
        self.play(Create(parax), FadeIn(lp), run_time=1.2)
        self.play(Write(top), run_time=1.3)

        # ---------------- r2: three ideas
        clock = lambda: self.now
        op2 = ValueTracker(0.0)             # fade the spinning dials via a tracker: FadeIn/FadeOut would freeze them
        icon1 = VGroup(*[graph_line(lambda x, i=i: 0.22 * np.cos((i + 1) * 3.0 * x), -0.8, 0.8, 0.45 - 0.45 * i, 1, 1, color=BLUE_B, width=3)
                         for i in range(3)]).move_to([-4.6, 0.3, 0])

        def dials():
            g = VGroup()
            o = op2.get_value()
            if o <= 0.002:
                return g
            for i, rate in enumerate((1.0, 1.7, 2.6)):
                c = np.array([-0.9 + 0.9 * i, 0.3, 0])
                a = rate * clock()
                g.add(Circle(radius=0.35).move_to(c).set_stroke(GREY_B, 1.5),
                      fast_arrow(c, c + 0.35 * np.array([np.cos(a), np.sin(a), 0]), C_H, 4, 0.12))
            return fade_vgroup(g, o)
        icon2 = always_redraw(dials)
        icon3 = VGroup(MathTex("\\sum", font_size=60), graph_line(lambda x: np.exp(-(x / 0.35) ** 2) * 0.8, -0.9, 0.9, -0.3, 1, 1, color=C_U, width=4))
        icon3.arrange(RIGHT, buff=0.3).move_to([4.6, 0.3, 0])
        l1 = Tex("break into plane waves", font_size=30).move_to([-4.6, -0.95, 0])
        l2 = Tex("advance each phase", font_size=30, color=C_H).move_to([0, -0.95, 0])
        l3 = Tex("add them back up", font_size=30).move_to([4.6, -0.95, 0])
        ars = VGroup(Arrow([-3.2, 0.3, 0], [-1.6, 0.3, 0], buff=0, stroke_width=4, color=GREY_A),
                     Arrow([1.6, 0.3, 0], [3.2, 0.3, 0], buff=0, stroke_width=4, color=GREY_A))
        self.at(S("r2"))
        self.play(FadeOut(VGroup(ax, axl, exact, parax, le, lp, top)), run_time=0.5)
        self.add(icon2)
        self.play(FadeIn(icon1), FadeIn(l1), run_time=0.6)
        self.play(GrowArrow(ars[0]), op2.animate.set_value(1.0), FadeIn(l2), run_time=0.7)
        self.play(GrowArrow(ars[1]), FadeIn(icon3), FadeIn(l3), run_time=0.7)

        # ---------------- r3: final card
        boxes, frames, arrows, labs = pipeline(y=-0.4)
        ttl = Tex("The Angular Spectrum Method", font_size=60).move_to([0, 1.6, 0])
        self.at(S("r3"))
        self.play(FadeOut(VGroup(icon1, icon3, l1, l2, l3, ars)), op2.animate.set_value(0.0), run_time=0.5)
        self.remove(icon2)
        self.play(FadeIn(ttl, shift=0.15 * DOWN), FadeIn(boxes[0]), Create(frames[0]), run_time=0.7)
        for i in range(3):
            self.play(GrowArrow(arrows[i]), FadeIn(labs[i]), FadeIn(boxes[i + 1]), Create(frames[i + 1]), run_time=0.6)
        self.play(Indicate(labs[0], color=YELLOW), Indicate(labs[2], color=YELLOW), Indicate(labs[1], color=C_H), run_time=1.0)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Waves", "Ch2Arrows", "Ch3PlaneWaves", "Ch4Basis", "Ch5Diagonal", "Ch6Evanescent",
            "Ch7Algorithm", "Ch8Recap"]
