"""The Diffraction Limit: why every image is blurry -- chapter scenes (Optics in 5 Minutes, E03).
Render: ./render.sh ; preview: ./preview.sh Ch0Hook
Every physical field is precomputed by physics.py (data/*.npz, numpy FFTs); scenes only load and draw them.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403

C_I = BLUE_A           # intensity profiles (the HEAT map's cyan family)
C_OTF = WHITE          # incoherent OTF curve (the pupil / coherent transfer stays GREEN)


# ================================================================== base & helpers ==
class Ep(Chapter):
    """Chapter base with the whole-frame at() of the pilot (common's at() still waits a float duration, which can
    differ by one frame between workers when the target lies on the frame grid)."""

    def at(self, t):
        x = (t - self.now) * config.frame_rate                # frames still to go
        if x > 0.5 + 1e-6:
            k = int(np.ceil(x - 1e-6))
            self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
        elif x < -0.5 - 1e-6:
            self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


def load(name):
    return np.load(data_path(name))


def frame_rect(c, w, h, color=GREY_B, width=1.5):
    return Rectangle(width=w, height=h).move_to(c).set_stroke(color, width)


def arc_pts(c, r, a0, a1, n=40):
    aa = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(aa), c[1] + r * np.sin(aa), np.zeros(n)], 1)


def poly_pts(pts, color, width, opacity=1.0):
    return VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity=opacity)


def dashed_circle(c, r, n=40, color=WHITE, width=2.0, opacity=1.0, duty=0.55):
    """Cheap dashed circle for per-frame redraws: one VMobject, one short polyline per dash."""
    vm = VMobject().set_stroke(color, width, opacity=opacity)
    if r <= 1e-3 or opacity <= 0.002:
        return vm
    for i in range(n):
        p = arc_pts(c, r, TAU_ * i / n, TAU_ * (i + duty) / n, 4)
        vm.start_new_path(p[0])
        for q in p[1:]:
            vm.add_line_to(q)
    return vm


_BANK = {}


class SlotNum:
    """Live number drawn into the glyph slots of a pre-built template part (e.g. the "0.00" of
    MathTex("NA", "=", "0.00")): each digit is a copy of a pre-built glyph moved onto its slot, so it sits on the
    template's baseline. Never builds Tex per frame."""

    def __init__(self, slots, font_size, color=WHITE):
        key = (font_size, str(color))
        if key not in _BANK:
            _BANK[key] = list(MathTex("0123456789", font_size=font_size, color=color)[0])
        self.bank = _BANK[key]
        self.slots = list(slots)

    def make(self, s):
        g = VGroup()
        for ch, sl in zip(s, self.slots):
            g.add(self.bank[int(ch)].copy().move_to(sl.get_center()) if ch.isdigit() else sl.copy())
        return g


def kspace_axes(kc, R, ext=0.5, up=0.55, fs=32, labels=True):
    """Upper half of the k-circle (forward-going plane waves) with k_x and k_z axes."""
    circ = Arc(radius=R, start_angle=0, angle=PI, arc_center=kc).set_stroke(GREY_A, 2.5)
    kx_ax = Arrow(kc + (R + ext) * LEFT, kc + (R + ext + 0.3) * RIGHT, buff=0, stroke_width=2, color=GREY_B,
                  max_tip_length_to_length_ratio=0.03)
    kz_ax = Arrow(kc + 0.25 * DOWN, kc + (R + up) * UP, buff=0, stroke_width=2, color=GREY_B,
                  max_tip_length_to_length_ratio=0.05)
    g = VGroup(circ, kx_ax, kz_ax)
    if labels:
        g.add(MathTex("k_x", font_size=fs, color=C_KX).next_to(kx_ax, RIGHT, buff=0.08),
              MathTex("k_z", font_size=fs, color=C_KZ).next_to(kz_ax, UP, buff=0.06),
              MathTex("k", font_size=fs).move_to(kc + (R + 0.3) * np.array([-0.72, 0.69, 0])))
    return g


def cap_group(kc, R, na, op=1.0, band=True):
    """The lens's cap on the k-circle: accepted arc, acceptance cone and the band |k_x| <= k NA (GREEN)."""
    g = VGroup()
    if op <= 0.002:
        return g
    th = np.arcsin(np.clip(na, 0, 1))
    a0, a1 = PI / 2 - th, PI / 2 + th
    g.add(poly_pts(arc_pts(kc, R, a0, a1, 50), C_H, 8, op))
    for a in (a0, a1):
        g.add(fast_dashed(kc, kc + R * np.array([np.cos(a), np.sin(a), 0]), 0.07, C_H, 2.5, op))
    if band:
        g.add(Line(kc + R * na * LEFT, kc + R * na * RIGHT).set_stroke(C_H, 9, opacity=op))
    return g


# ===================================================================== 0  hook ======
class Ch0Hook(Ep):
    CH = 0

    def construct(self):
        S, E = self.S, self.E
        d = load("hook.npz")
        PW, PH, PC = 10.4, 5.2, np.array([0.0, 0.45, 0])
        sc = PW / 8.0                                             # scene units per micrometre
        P = lambda x, y: np.array([PC[0] + x * sc, PC[1] + y * sc, 0])
        fr = frame_rect(PC, PW, PH, GREY_D)
        pts, amps = d["pts"], d["amps"]
        dots = VGroup(*[Dot(P(x, y), radius=0.035 + 0.025 * a, color=C_REF) for (x, y), a in zip(pts, amps)])
        cap1 = Tex("tiny glowing beads, each far smaller than the wavelength of light", font_size=32,
                   color=GREY_A).move_to([0, 3.42, 0])
        lam_u = 0.5 * sc
        xw = np.linspace(0, 2 * lam_u, 90)
        wave = poly_line(-5.2 + xw, -2.55 + 0.11 * np.sin(TAU_ * xw / lam_u), C_U, 3)
        wave_l = MathTex("\\lambda = 500\\,\\mathrm{nm}", font_size=28, color=C_U).next_to(wave, RIGHT, buff=0.25)
        bar = Line([5.2 - sc, -2.55, 0], [5.2, -2.55, 0]).set_stroke(WHITE, 3)
        bar_l = MathTex("1\\,\\mu\\mathrm{m}", font_size=28).next_to(bar, LEFT, buff=0.2)
        marks = VGroup(wave, wave_l, bar, bar_l)
        self.play(Create(fr), FadeIn(marks), run_time=0.8)
        self.at(S("h1"))
        self.play(LaggedStart(*[FadeIn(dd, scale=0.4) for dd in dots], lag_ratio=0.06), FadeIn(cap1), run_time=2.0)

        # ---------------- h2: the computed image sweeps in over the beads
        base = heat_rgba(d["field"], 1.0, gamma=0.45)
        fm, ff, fsw = self.sweep_image(base, PW, PC, PH, axis="right")
        cap2 = Tex("seen through a perfect microscope (computed)", font_size=32, color=GREY_A).move_to(cap1)
        self.add(fm)
        ff.set_value(1.0)
        self.at(S("h2") + 0.6)
        self.play(FadeOut(cap1), run_time=0.35)
        self.play(FadeIn(cap2), fsw.animate.set_value(0.28), run_time=0.9, rate_func=linear)
        self.play(fsw.animate.set_value(1.0), run_time=2.3, rate_func=linear)
        self.remove(*dots)                                        # LaggedStart added each dot on its own
        # the most isolated bead: its rings
        dist = np.array([min(np.hypot(*(p - q)) for j, q in enumerate(pts) if j != i) - 0.3 * abs(p[0]) for i, p in enumerate(pts)])
        b = pts[int(np.argmax(dist))]
        ring = Circle(radius=0.62 * sc).move_to(P(*b)).set_stroke(WHITE, 2.5)
        ring_l = Tex("rings", font_size=28).next_to(ring, UP, buff=0.08)
        self.at(S("h2") + 4.8)
        self.play(Create(ring), FadeIn(ring_l), run_time=0.7)

        # ---------------- h3: push two beads together (computed for each separation)
        pair, prof, seps = d["pair"], d["prof"], d["seps"]
        s_end = 0.15
        sep = ValueTracker(0.6)

        def idx():
            return int(np.clip(round((0.6 - sep.get_value()) / 0.005), 0, len(seps) - 1))
        QC, QW = np.array([-3.25, 0.2, 0]), 5.6
        QH = QW * pair.shape[1] / pair.shape[2]
        qm, qf = self.live_image(lambda t: heat_rgba(pair[idx()], 1.4, gamma=0.6), QW, QC, QH)
        qfr = frame_rect(QC, QW, QH)
        cut = DashedLine(QC + QW / 2 * LEFT, QC + QW / 2 * RIGHT, dash_length=0.08).set_stroke(WHITE, 1.2, opacity=0.5)
        GX0, GX1, GY0, GH = 0.85, 6.15, -1.1, 2.5
        upm = (GX1 - GX0) / 2 / 1.1                                # plot units per micrometre (data: +-1.1 um)
        gax = Line([GX0, GY0, 0], [GX1, GY0, 0]).set_stroke(GREY_B, 1.5)
        gl = Tex("brightness along the dashed line", font_size=26, color=GREY_B).move_to([(GX0 + GX1) / 2, GY0 - 0.4, 0])
        xs_p = np.linspace(GX0, GX1, prof.shape[1])

        def prof_fn():
            i = idx()
            g = VGroup(poly_line(xs_p, GY0 + GH * prof[i] / 2.05, C_I, 4))
            for sg in (-1, 1):
                x = (GX0 + GX1) / 2 + sg * sep.get_value() / 2 * upm
                g.add(Line([x, GY0 - 0.13, 0], [x, GY0 + 0.13, 0]).set_stroke(C_REF, 5))
            return g
        prof_m = always_redraw(prof_fn)
        tpl = Tex("separation~~", "000", "~nm", font_size=36, color=GREY_A).move_to([3.5, 2.1, 0])
        st, nmt = tpl[0], tpl[2]
        sn = SlotNum(tpl[1], 36, WHITE)
        rd = always_redraw(lambda: sn.make(f"{int(round(sep.get_value() * 1000)):03d}"))
        merged = Tex("one blob", font_size=34, color=YELLOW).move_to([(GX0 + GX1) / 2 + 1.55, GY0 + 1.85, 0])
        self.at(S("h3"))
        self.play(ff.animate.set_value(0.0), FadeOut(fr), FadeOut(cap2), FadeOut(marks), FadeOut(ring), FadeOut(ring_l),
                  run_time=0.45)
        self.add(qm)
        self.play(qf.animate.set_value(1.0), FadeIn(qfr), FadeIn(cut), FadeIn(gax), FadeIn(gl), FadeIn(st), FadeIn(nmt),
                  run_time=0.55)
        self.add(prof_m, rd)
        self.play(sep.animate.set_value(s_end), run_time=3.6, rate_func=smooth)
        self.play(FadeIn(merged, scale=0.8), run_time=0.4)

        # ---------------- h4: the question, then the title card
        q = Tex("Why can't a perfect lens make a perfect point?", font_size=44).move_to([0, -2.62, 0])
        self.at(S("h4"))
        self.play(qf.animate.set_value(0.0), FadeOut(VGroup(qfr, cut, gax, gl, st, nmt, merged, rd, prof_m)),
                  ff.animate.set_value(0.4), FadeIn(fr), run_time=0.8)
        self.remove(qm, rd, prof_m)
        self.at(S("h4") + 1.2)
        self.play(Write(q), run_time=1.3)
        title = series_title(3, "The Diffraction Limit", "why every image is blurry")
        title.remove(title[0])                                    # no half-opaque box: its edge would cut a bead
        self.at(S("h4") + 3.0)
        self.play(FadeIn(title, shift=0.1 * DOWN), ff.animate.set_value(0.18), run_time=0.9)
        self.finish()


# ===================================================================== 1  stamps ====
class Ch1Stamps(Ep):
    CH = 1
    TAG = "1\\quad Images are stamps"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = load("stamps.npz")
        spts, samps, cum, icon = d["pts"], d["amps"], d["cum"], d["icon"]
        PW = 3.8
        OC, IC = np.array([-4.2, 0.05, 0]), np.array([4.2, 0.05, 0])
        sc = PW / 10.0                                            # units per wavelength (panels show 10 lambda)
        ofr, ifr = frame_rect(OC, PW, PW), frame_rect(IC, PW, PW)
        olab = Tex("object $O$: points of light", font_size=30).next_to(ofr, UP, buff=0.18)
        ilab = Tex("image $I$ (computed)", font_size=30).next_to(ifr, UP, buff=0.18)
        dots = VGroup(*[Dot(OC + sc * np.array([x, y, 0]), radius=0.04 + 0.05 * a, color=C_REF)
                        for (x, y), a in zip(spts, samps)])
        self.at(S("s1") + 0.3)
        self.play(Create(ofr), FadeIn(olab), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(dd, scale=0.3) for dd in dots], lag_ratio=0.15), run_time=1.6)

        # ---------------- s2: one point -> the PSF
        ICC = np.array([0.0, 0.75, 0])
        icm, icf = self.static_image(heat_rgba(icon, 1.0, gamma=0.45), 1.6, ICC)
        icfr = frame_rect(ICC, 1.6, 1.6)
        icl = Tex("PSF", font_size=36).next_to(icfr, UP, buff=0.12)
        icl2 = Tex("point spread function", font_size=24, color=GREY_B).next_to(icfr, DOWN, buff=0.12)
        lens_ar = Arrow([-1.9, -1.0, 0], [1.9, -1.0, 0], buff=0, stroke_width=4, color=GREY_A,
                        max_tip_length_to_length_ratio=0.06)
        lens_l = Tex("lens", font_size=28, color=GREY_A).next_to(lens_ar, DOWN, buff=0.1)
        self.at(S("s2"))
        self.play(GrowArrow(lens_ar), FadeIn(lens_l), Create(ifr), FadeIn(ilab), run_time=0.8)
        self.add(icm)
        self.play(icf.animate.set_value(1.0), FadeIn(icfr), FadeIn(icl), run_time=0.7)
        self.at(S("s2") + 3.0)
        self.play(FadeIn(icl2), run_time=0.5)

        # ---------------- s3: every point stamps a scaled copy; the stamps add
        nst = ValueTracker(0.0)
        n = len(spts)

        def stamp_img(t):
            v = nst.get_value()
            j = int(np.floor(v))
            A = cum[-1] if j >= n else cum[j] + (v - j) * (cum[j + 1] - cum[j])
            return heat_rgba(A, 1.0, gamma=0.55)
        im, imf = self.live_image(stamp_img, PW, IC, PW)

        def rings():
            v = nst.get_value()
            g = VGroup()
            if v <= 1e-3 or v >= n - 1e-3:
                return g
            j = int(np.floor(v))
            o = np.sin(np.pi * (v - j))
            p = np.array([spts[j][0], spts[j][1], 0])
            g.add(Circle(radius=0.27).move_to(OC + sc * p).set_stroke(WHITE, 3, opacity=o),
                  Circle(radius=0.6).move_to(IC + sc * p).set_stroke(WHITE, 3, opacity=o))
            return g
        rings_m = always_redraw(rings)
        self.add(im, rings_m)
        self.at(S("s3"))
        self.play(imf.animate.set_value(1.0), run_time=0.3)
        for j, rt in enumerate((1.0, 0.85, 0.7, 0.6, 0.5, 0.45, 0.45, 0.45, 0.45)):
            self.play(nst.animate.set_value(j + 1), run_time=rt, rate_func=smooth)
        eq1 = MathTex("I(\\mathbf r)", "=", "\\sum_j", "a_j", "\\,\\mathrm{PSF}(\\mathbf r-\\mathbf r_j)", font_size=42)
        eq1[3].set_color(C_REF)
        eq1.move_to([0, -2.45, 0])
        self.at(S("s3") + 6.0)
        self.play(Write(eq1), run_time=1.3)

        # ---------------- s4: that sum is a convolution
        eq2 = MathTex("I", "=", "O", "\\ast", "\\mathrm{PSF}", font_size=54).move_to([-0.9, -2.45, 0])
        conv = Tex("$\\ast$ : convolution", font_size=32, color=YELLOW).next_to(eq2, RIGHT, buff=0.6)
        self.at(S("s4") + 0.2)
        self.play(Indicate(eq1[2:], color=YELLOW, scale_factor=1.08), run_time=1.0)
        self.at(S("s4") + 2.4)
        self.play(ReplacementTransform(eq1, eq2), run_time=1.0)
        self.play(FadeIn(conv, shift=0.1 * LEFT), run_time=0.5)

        # ---------------- s5: a single wave goes in, the same wave comes out, fainter
        kx, Hk = d["kx"], d["H"]
        Hq = lambda v: float(np.interp(v, kx, Hk))
        q = ValueTracker(0.4)                                     # k_x / k
        SW, SH = 3.4, 1.55
        SO, SI = np.array([-4.4, 1.25, 0]), np.array([-4.4, -1.55, 0])
        xs = np.linspace(-6.0, 6.0, 340)                          # swatches show 12 wavelengths

        def stripes(t, amp):
            v = q.get_value()
            row = 0.5 * (1 + amp(v) * np.cos(TAU_ * v * (xs - 0.35 * (t - self.t0))))
            return heat_rgba(np.repeat(row[None, :], 6, axis=0), 1.0, gamma=0.8)
        om, of_ = self.live_image(lambda t: stripes(t, lambda v: 1.0), SW, SO, SH)
        sm, sf = self.live_image(lambda t: stripes(t, Hq), SW, SI, SH)
        sfr = VGroup(frame_rect(SO, SW, SH), frame_rect(SI, SW, SH))
        slab = VGroup(Tex("object: stripes", font_size=30).next_to(sfr[0], UP, buff=0.15),
                      Tex("image: same stripes, fainter", font_size=30).next_to(sfr[1], DOWN, buff=0.15))
        bl_ar = Arrow(SO + (SH / 2 + 0.1) * DOWN, SI + (SH / 2 + 0.1) * UP, buff=0, stroke_width=4, color=GREY_A,
                      max_tip_length_to_length_ratio=0.2)
        bl_l = Tex("blur", font_size=28, color=GREY_A).next_to(bl_ar, RIGHT, buff=0.15)
        self.at(S("s5"))
        self.play(FadeOut(VGroup(ofr, ifr, olab, ilab, dots, icfr, icl, icl2, lens_ar, lens_l, eq2, conv)),
                  imf.animate.set_value(0.0), icf.animate.set_value(0.0), run_time=0.6)
        self.remove(im, icm, rings_m)
        self.add(om)
        self.play(of_.animate.set_value(1.0), FadeIn(sfr[0]), FadeIn(slab[0]), run_time=0.7)
        self.at(S("s5") + 2.9)
        self.add(sm)
        self.play(GrowArrow(bl_ar), FadeIn(bl_l), run_time=0.5)
        self.play(sf.animate.set_value(1.0), FadeIn(sfr[1]), FadeIn(slab[1]), run_time=0.7)

        # ---------------- s6: waves are eigenvectors -> convolution becomes multiplication
        eig = MathTex("\\mathrm{blur}\\big[\\,e^{ik_x x}\\big]", "=", "H(k_x)", "\\,e^{ik_x x}", font_size=46)
        eig.move_to([2.6, 1.85, 0])
        eig[2].set_color(C_H)
        bv = Brace(eig[3], UP, buff=0.1, color=GREY_B)
        bh = Brace(eig[2], DOWN, buff=0.1, color=C_H)
        tv = Tex("eigenvector", font_size=28, color=GREY_A).next_to(bv, UP, buff=0.06)
        th = Tex("eigenvalue", font_size=28, color=C_H).next_to(bh, DOWN, buff=0.06)
        mul = MathTex("\\mathcal{F}[I]", "=", "H(k_x)", "\\,\\mathcal{F}[O]", font_size=52).move_to([2.6, -0.5, 0])
        mul[2].set_color(C_H)
        mul_l = Tex("$\\mathcal{F}$: Fourier transform", font_size=26, color=GREY_B).next_to(mul, DOWN, buff=0.3)
        self.at(S("s6") + 0.3)
        self.play(Write(eig), run_time=1.4)
        self.play(GrowFromCenter(bv), FadeIn(tv), GrowFromCenter(bh), FadeIn(th), run_time=0.8)
        self.at(S("s6") + 5.0)
        self.play(Write(mul), FadeIn(mul_l), run_time=1.3)

        # ---------------- s7/s8: the transfer function H(k_x); fine stripes fade most
        BX0, BY0, BW, BH = -0.4, -2.35, 6.0, 3.1
        X = lambda v: BX0 + v / 1.1 * BW
        Y = lambda h: BY0 + h * BH
        hax = VGroup(Arrow([BX0, BY0, 0], [BX0 + BW + 0.35, BY0, 0], buff=0, stroke_width=2, color=GREY_B,
                           max_tip_length_to_length_ratio=0.03),
                     Arrow([BX0, BY0, 0], [BX0, BY0 + BH + 0.4, 0], buff=0, stroke_width=2, color=GREY_B,
                           max_tip_length_to_length_ratio=0.06))
        hax.add(MathTex("k_x", font_size=32, color=C_KX).next_to(hax[0], RIGHT, buff=0.08),
                MathTex("H", font_size=32, color=C_H).next_to(hax[1], UP, buff=0.06),
                MathTex("1", font_size=26, color=GREY_B).move_to([BX0 - 0.22, Y(1.0), 0]),
                Line([BX0 - 0.06, Y(1.0), 0], [BX0 + 0.06, Y(1.0), 0]).set_stroke(GREY_B, 2))
        hcurve = poly_line(X(kx), Y(Hk), C_H, 5)
        hlab = Tex("$H$ = Fourier transform of the PSF", font_size=30, color=C_H).move_to([3.65, Y(0.92), 0])

        def hdot():
            v = q.get_value()
            p = np.array([X(v), Y(Hq(v)), 0])
            return VGroup(fast_dashed([p[0], BY0, 0], p, 0.06, GREY_B, 1.5), Dot(p, radius=0.08, color=WHITE),
                          Line([p[0], BY0 - 0.1, 0], [p[0], BY0 + 0.1, 0]).set_stroke(C_KX, 5))
        hd = always_redraw(hdot)
        self.at(S("s7"))
        self.play(FadeOut(VGroup(eig, bv, bh, tv, th, mul_l)), mul.animate.scale(0.85).move_to([2.6, 2.35, 0]),
                  run_time=0.7)
        self.play(GrowArrow(hax[0]), GrowArrow(hax[1]), FadeIn(hax[2:]), run_time=0.6)
        self.play(Create(hcurve), run_time=1.2)
        self.add(hd)
        self.play(q.animate.set_value(0.12), run_time=1.8)
        self.at(S("s7") + 4.6)
        self.play(FadeIn(hlab), run_time=0.6)
        self.at(S("s7") + 5.6)
        self.play(q.animate.set_value(0.93), run_time=5.5, rate_func=linear)
        self.finish()


# ===================================================================== 2  pupil =====
class Ch2Pupil(Ep):
    CH = 2
    TAG = "2\\quad The lens is a hole in $k$-space"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        KC, R = np.array([-4.35, -2.1, 0]), 2.1
        axes = kspace_axes(KC, R)
        SRC, LX, LH = np.array([0.55, -0.15, 0]), 4.1, 1.6
        th_max = np.arctan(LH / (LX - SRC[0]))
        na = ValueTracker(float(np.sin(th_max)))                   # NA of the drawn lens (0.41)
        cap_on, fan_op = ValueTracker(0.0), ValueTracker(0.0)
        th_fan = np.radians(np.linspace(-80, 80, 15))

        def fan():
            g = VGroup()
            o = fan_op.get_value()
            if o <= 0.002:
                return g
            c, n_ = cap_on.get_value(), na.get_value()
            for t_ in th_fan:
                tip = KC + R * np.array([np.sin(t_), np.cos(t_), 0])
                inside = abs(np.sin(t_)) <= n_ + 1e-9
                col = WHITE if inside else interpolate_color(WHITE, GREY_D, c)
                g.add(fast_arrow(KC, tip, col, 2.5, 0.13, o * (1.0 if inside else 1 - 0.55 * c)))
            return g
        fan_m = always_redraw(fan)
        cap_m = always_redraw(lambda: cap_group(KC, R, na.get_value(), cap_on.get_value()))
        lab_kna = MathTex("k\\,\\mathrm{NA}", font_size=28, color=C_H)
        tick_op = ValueTracker(0.0)

        def tick():
            o = tick_op.get_value()
            if o <= 0.002:
                return VGroup()
            m = lab_kna.copy().move_to(KC + R * na.get_value() * RIGHT + 0.36 * DOWN)
            return m.set_fill(opacity=o)
        tick_m = always_redraw(tick)

        # ---------------- p1: recall -- light from a point is a sum of plane waves (the k-circle)
        rec = Tex("one arrow per plane wave", font_size=28, color=GREY_B).move_to(KC + (R + 1.3) * UP)
        ds = load("stamps.npz")
        QC2 = np.array([-0.6, 0.35, 0])
        qm_, qf_ = self.static_image(heat_rgba(ds["icon"], 1.0, gamma=0.45), 1.9, QC2)
        qfr_ = frame_rect(QC2, 1.9, 1.9)
        qq = Tex("PSF", font_size=36).next_to(qfr_, UP, buff=0.15)
        qmark = MathTex("?", font_size=96, color=YELLOW).next_to(qfr_, RIGHT, buff=0.45)
        self.add(qm_)
        self.at(S("p1"))
        self.play(qf_.animate.set_value(1.0), FadeIn(qfr_), FadeIn(qq), FadeIn(qmark, scale=0.6), run_time=0.6)
        self.at(S("p1") + 1.7)
        self.play(qf_.animate.set_value(0.0), FadeOut(qfr_), FadeOut(qq), FadeOut(qmark),
                  Create(axes[0]), GrowArrow(axes[1]), GrowArrow(axes[2]), FadeIn(axes[3:]), run_time=1.0)
        self.remove(qm_)
        self.add(fan_m)
        self.at(S("p1") + 3.6)
        self.play(fan_op.animate.set_value(1.0), FadeIn(rec), run_time=1.0)

        # ---------------- p2: each direction has its own k_x = k sin(theta)
        t1 = np.radians(38)
        tip1 = KC + R * np.array([np.sin(t1), np.cos(t1), 0])
        hl = VGroup(Arrow(KC, tip1, buff=0, stroke_width=6, color=WHITE, max_tip_length_to_length_ratio=0.1),
                    Line(KC, [tip1[0], KC[1], 0]).set_stroke(C_KX, 7),
                    DashedLine(tip1, [tip1[0], KC[1], 0], dash_length=0.07).set_stroke(GREY_A, 2),
                    Arc(radius=0.55, start_angle=PI / 2, angle=-t1, arc_center=KC).set_stroke(WHITE, 2.5))
        hl_th = MathTex("\\theta", font_size=32).move_to(KC + 0.82 * np.array([np.sin(t1 / 2), np.cos(t1 / 2), 0]))
        kxe = MathTex("k_x", "=", "k\\,\\sin\\theta", font_size=44).move_to(KC + (R + 1.3) * UP)
        kxe[0].set_color(C_KX)
        self.at(S("p2") + 2.2)
        self.play(FadeOut(rec), FadeIn(hl), FadeIn(hl_th), run_time=0.8)
        self.play(Write(kxe), run_time=1.0)
        self.at(S("p2") + 5.6)
        self.play(FadeOut(hl), FadeOut(hl_th), fan_op.animate.set_value(1.0), run_time=0.6)
        self.play(Indicate(axes[0], color=WHITE, scale_factor=1.03), run_time=0.9)

        # ---------------- p3: a lens has a finite opening (schematic)
        lens = Ellipse(width=0.34, height=2 * LH).move_to([LX, SRC[1], 0]).set_stroke(BLUE_B, 2.5).set_fill(BLUE_E, 0.35)
        oax = DashedLine([0.2, SRC[1], 0], [6.7, SRC[1], 0], dash_length=0.08).set_stroke(GREY_D, 1.2)
        src = Dot(SRC, radius=0.08, color=C_REF)
        IMG = np.array([6.35, SRC[1], 0])
        caught, missed = VGroup(), VGroup()
        for ph in np.radians([0, -8, 8, -16, 16]):
            yh = SRC[1] + (LX - SRC[0]) * np.tan(ph)
            caught.add(VGroup(Line(SRC, [LX, yh, 0]), Line([LX, yh, 0], IMG)).set_stroke(BLUE_B, 2.5))
        for ph in np.radians([-56, 56, -72, 72]):
            seg = clip_to_box(SRC, SRC + 14 * np.array([np.cos(ph), np.sin(ph), 0]), 3.45, -0.45, 3.25, 1.95)
            if seg:
                missed.add(Line(*seg).set_stroke(GREY_D, 1.8))
        marg = VGroup(*[Line(SRC, [LX, SRC[1] + sg * LH, 0]).set_stroke(C_H, 3) for sg in (-1, 1)])
        tharc = Arc(radius=1.1, start_angle=0, angle=th_max, arc_center=SRC).set_stroke(C_H, 2.5)
        thl = MathTex("\\theta_{\\max}", font_size=30, color=C_H).move_to(SRC + 1.55 * np.array([np.cos(0.66), np.sin(0.66), 0]))
        lens_l = Tex("lens", font_size=28, color=BLUE_B).next_to(lens, DOWN, buff=0.12)
        src_l = Tex("point", font_size=28, color=C_REF).next_to(src, LEFT, buff=0.15)
        schem = VGroup(oax, missed, caught, marg, lens, src, tharc, thl, lens_l, src_l)
        self.at(S("p3"))
        self.play(FadeIn(src), FadeIn(src_l), Create(oax), FadeIn(lens), FadeIn(lens_l), run_time=0.7)
        self.play(LaggedStart(*[Create(m) for m in missed], lag_ratio=0.05),
                  LaggedStart(*[Create(c) for c in caught], lag_ratio=0.05), run_time=1.6)
        self.at(S("p3") + 3.4)
        self.play(Create(marg), Create(tharc), FadeIn(thl), run_time=0.9)

        # ---------------- p4: NA; on the k-circle the lens keeps a cap |k_x| <= k NA
        na_eq = MathTex("\\mathrm{NA}", "=", "n\\,\\sin\\theta_{\\max}", font_size=44).move_to([3.0, 2.45, 0])
        na_eq[0].set_color(C_H)
        na_note = Tex("($n = 1$ in air)", font_size=26, color=GREY_B).next_to(na_eq, DOWN, buff=0.12)
        self.at(S("p4") + 0.2)
        self.play(Write(na_eq), FadeIn(na_note), run_time=1.2)
        self.add(cap_m, tick_m)
        self.at(S("p4") + 4.4)
        self.play(FadeOut(kxe), cap_on.animate.set_value(1.0), run_time=1.3)
        self.at(S("p4") + 7.0)
        self.play(tick_op.animate.set_value(1.0), run_time=0.6)

        # ---------------- p5: evanescent waves never leave; the lens is a hole in k-space: the pupil
        ev = VGroup(Line(KC + R * LEFT, KC + (R + 0.45) * LEFT), Line(KC + R * RIGHT, KC + (R + 0.42) * RIGHT)).set_stroke(C_EV, 6)
        ev_l = Tex("evanescent", font_size=26, color=C_EV).move_to(KC + (R + 0.15) * RIGHT + 0.42 * DOWN)
        self.at(S("p5") + 0.4)
        self.play(Create(ev), FadeIn(ev_l), run_time=0.8)
        PUC, RPU = np.array([0.1, -0.35, 0]), 1.15
        pu_bound = DashedVMobject(Circle(radius=RPU).move_to(PUC), num_dashes=40).set_stroke(GREY_B, 1.5)
        pu_ax = VGroup(Line(PUC + (RPU + 0.2) * LEFT, PUC + (RPU + 0.2) * RIGHT),
                       Line(PUC + (RPU + 0.2) * DOWN, PUC + (RPU + 0.2) * UP)).set_stroke(GREY_D, 1.2)
        pu_op = ValueTracker(0.0)

        def pupil_disc():
            o = pu_op.get_value()
            if o <= 0.002:
                return VGroup()
            return Circle(radius=RPU * na.get_value()).move_to(PUC).set_stroke(C_H, 3, opacity=o).set_fill(C_H, 0.45 * o)
        pu_m = always_redraw(pupil_disc)
        pu_l = Tex("pupil", font_size=32, color=C_H).move_to(PUC + (RPU + 0.45) * DOWN)
        pu_k = MathTex("k", font_size=30, color=GREY_B).move_to(PUC + (RPU + 0.25) * np.array([-0.72, 0.69, 0]))
        self.add(pu_m)
        self.at(S("p5") + 5.0)
        self.play(FadeOut(schem), FadeOut(na_eq), FadeOut(na_note), run_time=0.6)
        self.play(Create(pu_bound), FadeIn(pu_ax), pu_op.animate.set_value(1.0), FadeIn(pu_l), FadeIn(pu_k), run_time=1.0)

        # ---------------- p6: add up the waves that pass -> the PSF (computed); the NA slider appears
        dn = load("na.npz")
        nas, stack = dn["nas"], dn["stack"]
        PSC, PSW = np.array([4.45, -0.35, 0]), 3.8
        upl = PSW / (2 * float(dn["half"]) * float(dn["dx"]))      # scene units per wavelength

        def psf_img(t):
            i = int(np.clip(round((na.get_value() - nas[0]) / 0.005), 0, len(nas) - 1))
            return heat_rgba(stack[i].astype(np.float32), 1.0, gamma=0.45)
        pm, pf = self.live_image(psf_img, PSW, PSC, PSW)
        pfr = frame_rect(PSC, PSW, PSW)
        pt = Tex("PSF (computed)", font_size=30).next_to(pfr, UP, buff=0.15)
        lam_bar = VGroup(Line(PSC + (PSW / 2 - 0.2 - upl) * RIGHT + (PSW / 2 + 0.3) * DOWN,
                              PSC + (PSW / 2 - 0.2) * RIGHT + (PSW / 2 + 0.3) * DOWN).set_stroke(WHITE, 3))
        lam_bar.add(MathTex("\\lambda", font_size=28).next_to(lam_bar[0], LEFT, buff=0.15))
        add_ar = Arrow(PUC + (RPU + 0.25) * RIGHT, PSC + (PSW / 2 + 0.12) * LEFT, buff=0, stroke_width=4, color=GREY_A,
                       max_tip_length_to_length_ratio=0.25)
        add_l = Tex("add up", font_size=26, color=GREY_A).next_to(add_ar, UP, buff=0.1)
        self.at(S("p6") + 0.6)
        self.play(GrowArrow(add_ar), FadeIn(add_l), FadeOut(ev_l), run_time=0.7)
        self.add(pm)
        self.play(pf.animate.set_value(1.0), FadeIn(pfr), FadeIn(pt), FadeIn(lam_bar), run_time=0.9)
        # NA slider
        SX0, SX1, SY = 0.9, 5.9, 2.75
        sl = VGroup(Line([SX0, SY, 0], [SX1, SY, 0]).set_stroke(GREY_B, 3))
        for v in (0.0, 0.5, 1.0):
            x = SX0 + v * (SX1 - SX0)
            sl.add(Line([x, SY - 0.1, 0], [x, SY + 0.1, 0]).set_stroke(GREY_B, 2),
                   MathTex(f"{v:g}", font_size=24, color=GREY_B).move_to([x, SY - 0.33, 0]))
        nat = MathTex("\\mathrm{NA}", "=", "0.00", font_size=36).move_to([-0.35, SY, 0])
        nat[0].set_color(C_H)
        nal = VGroup(nat[0], nat[1])
        sn = SlotNum(nat[2], 36, WHITE)

        def knob():
            v = na.get_value()
            return VGroup(Dot([SX0 + v * (SX1 - SX0), SY, 0], radius=0.12, color=C_H), sn.make(f"{v:.2f}"))
        kn = always_redraw(knob)
        self.at(S("p6") + 3.6)
        self.play(FadeIn(sl), FadeIn(nal), run_time=0.6)
        self.add(kn)

        # ---------------- p7: the signature shot -- cap widens, spot shrinks; Airy radius 0.61 lambda/NA
        ring_op = ValueTracker(0.0)
        ring_m = always_redraw(lambda: dashed_circle(PSC, 0.61 / na.get_value() * upl, 44, WHITE, 2.2,
                                                     ring_op.get_value()))
        airy = MathTex("r_{\\text{Airy}}", "=", "0.61\\,\\lambda/\\mathrm{NA}", font_size=40).move_to([0.1, -2.62, 0])
        self.add(ring_m)
        self.at(E("p6") - 0.4)
        self.play(na.animate.set_value(0.25), run_time=1.0)
        self.play(na.animate.set_value(0.92), run_time=3.4, rate_func=rate_functions.ease_in_out_sine)
        self.at(S("p7") + 3.4)
        self.play(ring_op.animate.set_value(1.0), Write(airy), run_time=1.0)
        self.play(na.animate.set_value(0.45), run_time=2.0)
        self.play(na.animate.set_value(0.7), run_time=1.2)

        # ---------------- p8: magnification is not resolution
        dh = load("hook.npz")
        i15 = int(np.argmin(np.abs(dh["seps"] - 0.15)))
        prgba = heat_rgba(dh["pair"][i15], 1.6, gamma=0.6)
        aspect = dh["pair"].shape[1] / dh["pair"].shape[2]
        SMC, SMW = np.array([-4.6, 0.45, 0]), 2.3
        BGC, BGW = np.array([2.1, 0.45, 0]), 6.9
        sm_, smf = self.static_image(prgba, SMW, SMC, SMW * aspect)
        bg_, bgf = self.static_image(prgba, BGW, BGC, BGW * aspect)
        smfr = frame_rect(SMC, SMW, SMW * aspect)
        sml = Tex("two beads, 150 nm apart", font_size=26).next_to(smfr, DOWN, buff=0.22)
        bgfr = frame_rect(BGC, BGW, BGW * aspect)
        bgl = Tex("magnified $\\times 3$: a bigger blur, not more detail", font_size=30).next_to(bgfr, DOWN, buff=0.2)
        zl = VGroup(*[DashedLine(smfr.get_corner(c1), bgfr.get_corner(c2), dash_length=0.08).set_stroke(GREY_C, 1.2)
                      for c1, c2 in ((UR, UL), (DR, DL))])
        self.at(S("p8"))
        self.play(FadeOut(VGroup(axes, ev, pu_bound, pu_ax, pu_l, pu_k, add_ar, add_l, pfr, pt, lam_bar, sl, nal, airy)),
                  FadeOut(fan_m), FadeOut(cap_m), FadeOut(tick_m), FadeOut(pu_m), FadeOut(kn), FadeOut(ring_m),
                  pf.animate.set_value(0.0), run_time=0.6)
        self.remove(fan_m, cap_m, tick_m, pu_m, kn, ring_m, pm)
        self.add(sm_)
        self.play(smf.animate.set_value(1.0), FadeIn(smfr), FadeIn(sml), run_time=0.6)
        self.add(bg_)
        self.play(bgf.animate.set_value(1.0), FadeIn(bgfr), Create(zl), run_time=1.0)
        self.play(FadeIn(bgl), run_time=0.6)
        self.finish()


# ===================================================================== 3  Abbe ======
class Ch3Abbe(Ep):
    CH = 3
    TAG = "3\\quad Abbe's grating test"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        d = load("abbe.npz")
        ab1, ab2, ab3 = d["ab1"], d["ab2"], d["ab3"]
        NA, ST, half = float(d["na"]), float(d["s_tilt"]), float(d["half"])
        u = ValueTracker(0.0)

        def dper(v):
            return 4.0 - 2.6 * v if v <= 1 else (1.4 if v <= 2 else 1.4 - 0.55 * (v - 2))

        def s_in(v):
            return 0.0 if v <= 1 else (ST * (v - 1) if v <= 2 else ST)

        def img_at(v):
            if v <= 1:
                a, i = ab1, round(v * (len(ab1) - 1))
            elif v <= 2:
                a, i = ab2, round((v - 1) * (len(ab2) - 1))
            else:
                a, i = ab3, round(min(v - 2, 1.0) * (len(ab3) - 1))
            return a[int(i)].astype(np.float32)

        # ---------------- a1: Abbe's question; the grating
        qst = Tex("Abbe, 1873: what is the finest grating a lens can image?", font_size=34).move_to([0.9, 2.95, 0])
        GC, GW = np.array([-5.0, 0.45, 0]), 2.5
        xg = np.linspace(-half, half, 300)

        def grating(t):
            row = (0.5 + 0.5 * np.cos(TAU_ * xg / dper(u.get_value()))) ** 2
            return grey_rgba(np.repeat(row[None, :], 6, axis=0), 1.0)
        gm, gf = self.live_image(grating, GW, GC, GW)
        gfr = frame_rect(GC, GW, GW)
        dtp = MathTex("\\text{period }d", "=", "0.00", "\\,\\lambda", font_size=32).move_to([GC[0], 2.05, 0])
        dl = VGroup(dtp[0], dtp[1], dtp[3])
        snd = SlotNum(dtp[2], 32, WHITE)
        dr_m = always_redraw(lambda: snd.make(f"{dper(u.get_value()):.2f}"))
        self.at(S("a1"))
        self.play(Write(qst), run_time=1.5)
        self.add(gm)
        self.at(S("a1") + 3.2)
        self.play(gf.animate.set_value(1.0), FadeIn(gfr), FadeIn(dl), FadeIn(dr_m), run_time=0.8)

        # ---------------- a2: plane wave in; orders out at sin(theta) = +-lambda/d (k-circle)
        KC, R = np.array([-0.35, -2.15, 0]), 2.0
        axes = kspace_axes(KC, R, ext=0.4, up=0.5, labels=False)
        axes.add(MathTex("k_x", font_size=30, color=C_KX).next_to(axes[1], RIGHT, buff=0.08))
        ill_end = GC + (GW / 2 + 0.15) * DOWN

        def illum():
            s = s_in(u.get_value())
            return fast_arrow(ill_end - 1.25 * np.array([s, np.sqrt(1 - s * s), 0]), ill_end, C_REF, 6, 0.22)
        il_m = always_redraw(illum)
        il_l = Tex("light", font_size=28, color=C_REF).move_to(ill_end + 1.0 * DOWN + 0.75 * LEFT)
        olab = {m: Tex(t, font_size=28) for m, t in ((-1, "$-1$"), (0, "$0$"), (1, "$+1$"))}
        k_lab = MathTex("2\\pi/d", font_size=28, color=C_K)
        ord_op = ValueTracker(0.0)

        def orders():
            o = ord_op.get_value()
            g = VGroup()
            if o <= 0.002:
                return g
            v = u.get_value()
            dd, s0 = dper(v), s_in(v)
            for m in (-1, 0, 1):
                s = s0 + m / dd
                inside = abs(s) <= NA + 1e-9
                if abs(s) <= 1:
                    dirn = np.array([s, np.sqrt(1 - s * s), 0])
                    col = (WHITE if m == 0 else C_KX) if inside else GREY_C
                    g.add(fast_arrow(KC, KC + R * dirn, col, 4.5 if inside else 3, 0.18, o * (1 if inside else 0.6)))
                    lp = KC + (R + 0.3) * dirn + (0.24 * LEFT if m == 0 else 0.13 * m * RIGHT)
                    g.add(olab[m].copy().set_color(col).move_to(lp).set_fill(opacity=o))
                else:
                    g.add(Dot([KC[0] + R * np.clip(s, -1.22, 1.22), KC[1], 0], radius=0.08, color=C_EV).set_fill(opacity=o))
            x0, x1 = KC[0] + R * s0, KC[0] + R * (s0 + 1 / dd)
            if x1 <= KC[0] + 1.22 * R:
                y = KC[1] - 0.3
                g.add(Line([x0, y, 0], [x1, y, 0]).set_stroke(C_K, 4, opacity=o),
                      Line([x0, y - 0.09, 0], [x0, y + 0.09, 0]).set_stroke(C_K, 3, opacity=o),
                      Line([x1, y - 0.09, 0], [x1, y + 0.09, 0]).set_stroke(C_K, 3, opacity=o),
                      k_lab.copy().move_to([(x0 + x1) / 2, y - 0.27, 0]).set_fill(opacity=o))
            return g
        ord_m = always_redraw(orders)
        sin_eq = MathTex("\\sin\\theta", "=", "\\pm\\lambda/d", font_size=42).move_to(KC + (R + 1.55) * UP)
        self.at(S("a2"))
        self.add(il_m)
        self.play(FadeIn(il_l), run_time=0.6)
        self.play(Create(axes[0]), GrowArrow(axes[1]), GrowArrow(axes[2]), FadeIn(axes[3:]), run_time=1.0)
        self.add(ord_m)
        self.at(S("a2") + 3.0)
        self.play(ord_op.animate.set_value(1.0), run_time=1.0)
        self.at(S("a2") + 5.6)
        self.play(Write(sin_eq), run_time=1.2)

        # ---------------- a3: the lens must catch the 0th and a 1st order; the image (computed, 4f)
        cap = cap_group(KC, R, NA, 1.0)
        cap_l = Tex("lens: NA = 0.5", font_size=28, color=C_H).move_to(KC + (R + 0.75) * UP)
        IC, IW = np.array([4.7, 0.35, 0]), 3.0
        im, imf = self.live_image(lambda t: heat_rgba(img_at(u.get_value()), 1.0, gamma=0.8), IW, IC, IW)
        ifr = frame_rect(IC, IW, IW)
        il_t = Tex("image (computed)", font_size=30).next_to(ifr, UP, buff=0.15)
        self.at(S("a3"))
        self.play(Create(cap), FadeIn(cap_l), run_time=1.0)
        self.add(im)
        self.at(S("a3") + 2.6)
        self.play(imf.animate.set_value(1.0), FadeIn(ifr), FadeIn(il_t), run_time=0.9)

        # ---------------- a4: finer grating -> the first orders leave the cap -> stripes vanish
        self.at(S("a4") + 0.3)
        self.play(u.animate.set_value(1.0), run_time=5.2, rate_func=linear)

        # ---------------- a5: straight on, d_min = lambda / NA
        f1 = Tex("straight on: $d_{\\min} = \\lambda/\\mathrm{NA}$", font_size=34).move_to([4.7, -1.62, 0])
        self.at(S("a5") + 0.5)
        self.play(Write(f1), run_time=1.3)

        # ---------------- a6: tilt the illumination: the 0th order to one edge, a 1st order fits at the other
        self.at(S("a6") + 0.6)
        self.play(u.animate.set_value(2.0), run_time=5.6, rate_func=smooth)

        # ---------------- a7: twice the reach: d_min = lambda / 2NA -- the diffraction limit
        f2 = Tex("tilted: $d_{\\min} = \\lambda/2\\mathrm{NA}$", font_size=34).move_to([4.7, -2.32, 0])
        f2b = SurroundingRectangle(f2, buff=0.12, color=YELLOW, stroke_width=2.5)
        dlim = Tex("the diffraction limit", font_size=30, color=YELLOW).move_to([4.7, -2.95, 0])
        self.at(S("a7"))
        self.play(u.animate.set_value(2.82), run_time=3.0, rate_func=linear)
        self.play(Write(f2), Create(f2b), run_time=1.0)
        self.at(S("a7") + 4.2)
        self.play(FadeIn(dlim), run_time=0.6)
        self.finish()


# ===================================================================== 4  coherent vs incoherent ==
class Ch4Coherence(Ep):
    CH = 4
    TAG = "4\\quad Coherent vs incoherent"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        dc = load("chart.npz")
        kk, mtf = dc["kk"], dc["mtf"]
        X0, XS, Y0, YS = 3.0, 1.3, -1.85, 2.8
        X = lambda v: X0 + XS * np.asarray(v)
        Y = lambda h: Y0 + YS * np.asarray(h)
        pax = VGroup(Arrow([X(-2.45), Y0, 0], [X(2.5), Y0, 0], buff=0, stroke_width=2, color=GREY_B,
                           max_tip_length_to_length_ratio=0.02),
                     Line([X0, Y0, 0], [X0, Y(1.15), 0]).set_stroke(GREY_B, 2))
        pax.add(MathTex("k_x", font_size=30, color=C_KX).next_to(pax[0], RIGHT, buff=0.06))
        for v in (-2, -1, 1, 2):
            pax.add(Line([X(v), Y0 - 0.08, 0], [X(v), Y0 + 0.08, 0]).set_stroke(GREY_B, 2))
        tk1 = MathTex("k\\,\\mathrm{NA}", font_size=26, color=GREY_A).move_to([X(1), Y0 - 0.33, 0])
        tk2 = MathTex("2k\\,\\mathrm{NA}", font_size=26, color=GREY_A).move_to([X(2), Y0 - 0.33, 0])
        step = poly_line(X([-2.4, -1, -1, 1, 1, 2.4]), Y([0, 0, 1, 1, 0, 0]), C_H, 4)
        step_l = Tex("coherent: the pupil", font_size=28, color=C_H).move_to([X(1.85), Y(1.2), 0])

        # ---------------- c1: laser light -- fields add; the pupil cuts the field at k NA
        dial_c = [np.array([-5.75, 1.3, 0]), np.array([-4.75, 1.3, 0])]
        dial_i = [np.array([-5.75, -1.25, 0]), np.array([-4.75, -1.25, 0])]
        op_c, op_i = ValueTracker(0.0), ValueTracker(0.0)

        def dials():
            t = self.now
            g = VGroup()
            for cs, o, angs in ((dial_c, op_c.get_value(), (2.4 * t, 2.4 * t + 0.7)),
                                (dial_i, op_i.get_value(), (2.4 * t + 2.6 * np.sin(1.3 * t) + 1.6 * np.sin(2.9 * t + 1.0),
                                                            2.1 * t + 2.3 * np.sin(1.7 * t + 2.0) + 1.4 * np.sin(3.7 * t)))):
                if o <= 0.002:
                    continue
                for c, a in zip(cs, angs):
                    g.add(Circle(radius=0.38).move_to(c).set_stroke(GREY_B, 1.5, opacity=o),
                          fast_arrow(c, c + 0.38 * np.array([np.cos(a), np.sin(a), 0]), C_U, 4, 0.12, o))
            return g
        dl_m = always_redraw(dials)
        lc = Tex("laser: fields add", font_size=32).move_to([-4.2, 2.25, 0])
        fc = MathTex("|U_1 + U_2|^2", font_size=42).move_to([-2.4, 1.3, 0])
        li = Tex("glowing beads: intensities add", font_size=32).move_to([-3.75, -0.3, 0])
        fi = MathTex("|U_1|^2 + |U_2|^2", font_size=42).move_to([-2.4, -1.25, 0])
        self.add(dl_m)
        self.at(S("c1"))
        self.play(FadeIn(lc), op_c.animate.set_value(1.0), run_time=0.8)
        self.play(Write(fc), run_time=1.0)
        self.at(S("c1") + 3.4)
        self.play(GrowArrow(pax[0]), Create(pax[1]), FadeIn(pax[2:]), FadeIn(tk1), run_time=0.8)
        self.play(Create(step), FadeIn(step_l), run_time=1.2)

        # ---------------- c2: glowing beads -- random phases -- intensities add
        self.at(S("c2") + 0.2)
        self.play(FadeIn(li), op_i.animate.set_value(1.0), run_time=0.8)
        self.at(S("c2") + 4.2)
        self.play(Write(fi), run_time=1.0)

        # ---------------- c3: the OTF = overlap of the pupil with a shifted copy of itself
        DC, rr = np.array([-4.55, -0.2, 0]), 1.05
        sh = ValueTracker(0.0)
        disc_op = ValueTracker(0.0)
        kx_l = MathTex("k_x", font_size=30, color=C_KX)

        def discs():
            o = disc_op.get_value()
            g = VGroup()
            if o <= 0.002:
                return g
            s = sh.get_value()
            c2 = DC + np.array([s * rr, 0, 0])
            g.add(Circle(radius=rr).move_to(DC).set_stroke(C_H, 3, opacity=o).set_fill(C_H, 0.12 * o),
                  Circle(radius=rr).move_to(c2).set_stroke(C_H, 3, opacity=0.8 * o).set_fill(C_H, 0.12 * o))
            if s < 2 - 1e-3:
                a = np.arccos(np.clip(s / 2, -1, 1))
                lens_ = np.concatenate([arc_pts(DC, rr, -a, a, 36), arc_pts(c2, rr, PI - a, PI + a, 36)])
                g.add(VMobject().set_points_as_corners(np.vstack([lens_, lens_[:1]])).set_stroke(width=0)
                      .set_fill(C_H, 0.7 * o))
            if s > 0.08:
                yb = DC[1] - rr - 0.3
                g.add(fast_arrow([DC[0], yb, 0], [c2[0], yb, 0], C_KX, 4, 0.16, o),
                      kx_l.copy().move_to([(DC[0] + c2[0]) / 2, yb - 0.3, 0]).set_fill(opacity=o))
            return g
        dsc = always_redraw(discs)
        otf_eq = MathTex("\\mathrm{OTF}(k_x)", "\\propto", "\\text{overlap area}", font_size=38).move_to([-3.6, 2.25, 0])
        pu_l = Tex("pupil", font_size=28, color=C_H).move_to(DC + (rr + 0.3) * UP + 0.35 * LEFT)
        self.at(S("c3"))
        self.play(FadeOut(VGroup(lc, fc, li, fi)), op_c.animate.set_value(0.0), op_i.animate.set_value(0.0), run_time=0.6)
        self.remove(dl_m)
        self.add(dsc)
        self.play(disc_op.animate.set_value(1.0), FadeIn(pu_l), run_time=0.7)
        self.play(Write(otf_eq), run_time=1.3)

        # ---------------- c4: slide -> the overlap traces the OTF; zero only at 2 k NA
        o_fill = ValueTracker(0.0)

        def otf_curve():
            s = sh.get_value()
            if s <= 0.01:
                return VGroup()
            m = np.abs(kk) <= s
            xs, ys = X(kk[m]), Y(mtf[m])
            g = VGroup(poly_line(xs, ys, C_OTF, 4))
            poly = np.stack([np.concatenate([xs, xs[::-1]]), np.concatenate([ys, np.full(len(xs), Y0)]), np.zeros(2 * len(xs))], 1)
            g.add(VMobject().set_points_as_corners(np.vstack([poly, poly[:1]])).set_stroke(width=0).set_fill(C_H, 0.22))
            v = float(np.interp(s, kk, mtf))
            g.add(Dot([X(s), Y(v), 0], radius=0.07, color=WHITE), Dot([X(-s), Y(v), 0], radius=0.07, color=WHITE))
            return g
        oc = always_redraw(otf_curve)
        inc_l = Tex("incoherent: OTF", font_size=28, color=C_OTF).move_to([X(2.05), Y(0.55), 0])
        self.add(oc)
        self.at(S("c4") + 0.3)
        self.play(sh.animate.set_value(2.0), run_time=5.2, rate_func=linear)
        self.play(FadeIn(tk2), FadeIn(inc_l), run_time=0.6)

        # ---------------- c5: twice the reach, but fading contrast
        x2 = Tex("$\\times 2$ reach", font_size=30, color=YELLOW).move_to([X(1.5), Y0 - 0.92, 0])
        x2a = DoubleArrow([X(1), Y0 - 0.62, 0], [X(2), Y0 - 0.62, 0], buff=0, stroke_width=3, color=YELLOW,
                          tip_length=0.14, max_tip_length_to_length_ratio=0.2)
        fade_r = Rectangle(width=XS * 0.75, height=YS * 0.2).move_to([X(1.625), Y(0.1), 0]).set_stroke(C_EV, 2).set_fill(C_EV, 0.15)
        fade_l = Tex("contrast fades", font_size=28, color=C_EV).move_to([X(1.95), Y(0.33), 0])
        self.at(S("c5") + 0.2)
        self.play(GrowFromCenter(x2a), FadeIn(x2), run_time=0.8)
        self.at(S("c5") + 4.9)
        self.play(FadeIn(fade_r), FadeIn(fade_l), run_time=0.8)

        # ---------------- c6: the same test chart, both ways (computed)
        PW_, PY = 3.55, -0.25
        pans = []
        for key, x, vmax, lab in (("obj", -4.45, 1.0, "object: test chart"), ("coh", 0.0, 1.25, "coherent (laser)"),
                                  ("inc", 4.45, 1.0, "incoherent (glowing)")):
            m, f = self.static_image(heat_rgba(dc[key].astype(np.float32), vmax, gamma=0.8), PW_, np.array([x, PY, 0]))
            fr_ = frame_rect([x, PY, 0], PW_, PW_)
            pans.append((m, f, fr_, Tex(lab, font_size=30).next_to(fr_, UP, buff=0.15)))
        cap_c = Tex("rings at every edge", font_size=28, color=GREY_A).next_to(pans[1][2], DOWN, buff=0.18)
        cap_i = Tex("smooth, but the finest spokes fade", font_size=28, color=GREY_A).next_to(pans[2][2], DOWN, buff=0.18)
        self.at(S("c6"))
        self.play(FadeOut(VGroup(pax, tk1, tk2, step, step_l, inc_l, x2, x2a, fade_r, fade_l, otf_eq, pu_l)),
                  FadeOut(oc), FadeOut(dsc), run_time=0.6)
        self.remove(oc, dsc)
        for m, f, fr_, lb in pans:
            self.add(m)
        self.play(*[f.animate.set_value(1.0) for _, f, _, _ in pans], *[FadeIn(fr_) for _, _, fr_, _ in pans],
                  *[FadeIn(lb) for _, _, _, lb in pans], run_time=1.0)
        self.at(S("c6") + 2.6)
        self.play(FadeIn(cap_c), run_time=0.6)
        self.at(S("c6") + 4.6)
        self.play(FadeIn(cap_i), run_time=0.6)
        self.finish()


# ===================================================================== 5  on a computer ==
class Ch5Computer(Ep):
    CH = 5
    TAG = "5\\quad On a computer"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        dd = load("deconv.npz")
        cr = lambda A: np.ascontiguousarray(A[76:436, 59:419])     # 360-px crop centred on the drawn content
        obj, blur, meas, naive, wien = (cr(dd[k]) for k in ("obj", "blur", "meas", "naive", "wien"))
        bmax = float(blur.max())

        # ---------------- m1: the forward model = the three steps of episode one
        card = code_card(["F   = fft2(obj)", "F   = F * OTF", "img = ifft2(F)"], font_size=26).move_to([0, 2.2, 0])
        rows = card[1][0]
        hls = [SurroundingRectangle(r, buff=0.06, color=YELLOW, stroke_width=2) for r in rows]
        OC, BC, W = np.array([-3.3, -1.05, 0]), np.array([3.3, -1.05, 0]), 2.9
        om, of_ = self.static_image(heat_rgba(obj, 1.0, gamma=0.8), W, OC)
        bm, bf = self.static_image(heat_rgba(blur, bmax, gamma=0.8), W, BC)
        ofr, bfr = frame_rect(OC, W, W), frame_rect(BC, W, W)
        ol = Tex("object", font_size=28).next_to(ofr, UP, buff=0.12)
        bl = Tex("image (computed)", font_size=28).next_to(bfr, UP, buff=0.12)
        ar = Arrow(OC + (W / 2 + 0.25) * RIGHT, BC + (W / 2 + 0.25) * LEFT, buff=0, stroke_width=4, color=GREY_A,
                   max_tip_length_to_length_ratio=0.08)
        self.at(S("m1"))
        self.play(FadeIn(card), run_time=0.8)
        self.add(om, bm)
        self.play(of_.animate.set_value(1.0), FadeIn(ofr), FadeIn(ol), run_time=0.7)
        for k, dt in enumerate((4.2, 5.5, 7.4)):
            self.at(S("m1") + dt)
            if k == 0:
                self.play(Create(hls[0]), run_time=0.4)
            else:
                self.play(ReplacementTransform(hls[k - 1], hls[k]), run_time=0.4)
        self.play(GrowArrow(ar), bf.animate.set_value(1.0), FadeIn(bfr), FadeIn(bl), run_time=0.8)

        # ---------------- m2: run it backwards -- divide by the OTF
        LC, MC, W2 = np.array([-5.15, -0.35, 0]), np.array([-1.95, -0.35, 0]), 2.7
        mm, mf = self.static_image(heat_rgba(np.clip(meas, 0, None), bmax, gamma=0.8), W2, LC)
        nm, nf = self.static_image(heat_rgba(np.clip(naive, 0, None), 1.0, gamma=0.8), W2, MC)
        wm, wf = self.static_image(heat_rgba(np.clip(wien, 0, None), 1.0, gamma=0.8), W2, MC)
        lfr, mfr = frame_rect(LC, W2, W2), frame_rect(MC, W2, W2)
        ll = Tex("blurred + noise", font_size=28).next_to(lfr, UP, buff=0.12)
        nl = Tex("divide by OTF", font_size=28, color=C_EV).next_to(mfr, UP, buff=0.12)
        inv = code_card(["est = ifft2(fft2(img) / OTF)"], font_size=24).move_to([-3.55, 2.45, 0])
        self.at(S("m2"))
        self.play(FadeOut(VGroup(card, hls[2], ofr, ol, ar, bfr, bl)), of_.animate.set_value(0.0), bf.animate.set_value(0.0),
                  run_time=0.6)
        self.remove(om, bm)
        self.add(mm)
        self.play(mf.animate.set_value(1.0), FadeIn(lfr), FadeIn(ll), run_time=0.8)
        self.at(S("m2") + 3.8)
        self.play(FadeIn(inv), run_time=0.6)
        self.add(nm)
        self.at(E("m2") + 0.1)
        self.play(nf.animate.set_value(1.0), FadeIn(mfr), FadeIn(nl), run_time=0.8)

        # ---------------- m3: why -- 1/OTF explodes near the cutoff (log plot)
        nu, mtf = dd["nu"], dd["mtf"]
        GX0, GW, GY0, GH = 1.25, 5.1, -2.15, 3.6                  # x: 0 .. 1.3 cutoff; y: 1e-3 .. 1e3 (log)
        GX = lambda v: GX0 + np.asarray(v) / 1.3 * GW
        GY = lambda g: GY0 + (np.log10(np.asarray(g)) + 3) / 6 * GH
        gax = VGroup(Arrow([GX0, GY0, 0], [GX0 + GW + 0.25, GY0, 0], buff=0, stroke_width=2, color=GREY_B,
                           max_tip_length_to_length_ratio=0.03),
                     Line([GX0, GY0, 0], [GX0, GY0 + GH, 0]).set_stroke(GREY_B, 2))
        for e, s in ((-3, "10^{-3}"), (0, "1"), (3, "10^{3}")):
            y = GY(10.0 ** e)
            gax.add(Line([GX0 - 0.07, y, 0], [GX0 + 0.07, y, 0]).set_stroke(GREY_B, 2),
                    MathTex(s, font_size=24, color=GREY_B).next_to([GX0 - 0.07, y, 0], LEFT, buff=0.08))
        gax.add(DashedLine([GX(1), GY0, 0], [GX(1), GY0 + GH, 0], dash_length=0.07).set_stroke(GREY_A, 1.5),
                Tex("cutoff", font_size=24, color=GREY_A).move_to([GX(1), GY0 - 0.28, 0]),
                MathTex("k_x", font_size=28, color=C_KX).next_to(gax[0], RIGHT, buff=0.06))
        ok = (nu < 0.999) & (mtf > 1e-3)
        c_otf = poly_line(GX(nu[ok]), GY(mtf[ok]), C_H, 4)
        ok2 = (nu < 0.999) & (mtf > 1e-3)
        c_inv = poly_line(GX(nu[ok2]), GY(np.clip(1 / mtf[ok2], 1, 1e3)), C_EV, 4)
        l_otf = Tex("OTF", font_size=28, color=C_H).move_to([GX(0.32), GY(0.12), 0])
        l_inv = Tex("$1/\\mathrm{OTF}$: noise $\\times$ 1000s", font_size=28, color=C_EV).move_to([GX(0.5), GY(250), 0])
        self.at(S("m3") + 0.6)
        self.play(GrowArrow(gax[0]), Create(gax[1]), FadeIn(gax[2:]), run_time=0.8)
        self.play(Create(c_otf), FadeIn(l_otf), run_time=0.9)
        self.play(Create(c_inv), run_time=1.0)
        self.play(FadeIn(l_inv), run_time=0.5)

        # ---------------- m4: the Wiener filter
        K = float(dd["K"])
        gw = np.where(mtf > 0, mtf / (mtf ** 2 + K), 0.0)
        okw = gw > 1e-3
        c_w = poly_line(GX(nu[okw]), GY(gw[okw]), WHITE, 5)
        wf_eq = MathTex("G", "=", "{H^* \\over |H|^2 + 1/\\mathrm{SNR}}", font_size=40).move_to([2.75, 2.35, 0])
        snr = Tex("SNR: signal-to-\\\\noise ratio", font_size=24, color=GREY_B).next_to(wf_eq, RIGHT, buff=0.35)
        l_w = Tex("Wiener", font_size=28).move_to([GX(0.62), GY(30), 0])
        wl = Tex("Wiener filter", font_size=28, color=WHITE).move_to(nl)
        self.at(S("m4"))
        self.play(FadeOut(inv), Write(wf_eq), run_time=1.1)
        self.play(FadeIn(snr), run_time=0.5)
        self.play(Create(c_w), FadeIn(l_w), c_inv.animate.set_stroke(opacity=0.35), l_inv.animate.set_opacity(0.35), run_time=1.2)
        self.add(wm)
        self.at(S("m4") + 3.6)
        self.play(wf.animate.set_value(1.0), FadeOut(nl), FadeIn(wl), run_time=0.9)
        nf.set_value(0.0)
        self.remove(nm)

        # ---------------- m5: beyond the cutoff the OTF is exactly zero: gone
        gone = Rectangle(width=GX(1.3) - GX(1.0), height=GH).move_to([(GX(1.0) + GX(1.3)) / 2, GY0 + GH / 2, 0])
        gone.set_stroke(width=0).set_fill(C_EV, 0.22)
        gone_l = Tex("OTF\\\\$=0$", font_size=26, color=C_EV).move_to([(GX(1.0) + GX(1.3)) / 2, GY(0.05), 0])
        # finest bar group: columns 308..344, rows 45..200 (from the bottom) of the 360-px crop (see physics.py)
        bx = MC[0] - W2 / 2 + W2 * 0.906
        by = MC[1] - W2 / 2 + W2 * 0.34
        lost = Rectangle(width=W2 * 0.15, height=W2 * 0.5).move_to([bx, by, 0]).set_stroke(YELLOW, 2.5)
        lost_l = Tex("finer than $\\lambda/2\\mathrm{NA}$: lost", font_size=28, color=YELLOW).next_to(mfr, DOWN, buff=0.15)
        self.at(S("m5") + 0.3)
        self.play(FadeIn(gone), FadeIn(gone_l), run_time=0.8)
        self.at(S("m5") + 3.3)
        self.play(Create(lost), FadeIn(lost_l), run_time=0.8)

        # ---------------- m6: only priors or extra measurements -> computational imaging
        t1 = Tex("prior knowledge", font_size=40)
        t2 = Tex("extra measurements", font_size=40)
        orr = Tex("or", font_size=32, color=GREY_B)
        top = VGroup(t1, orr, t2).arrange(RIGHT, buff=0.45).move_to([0, 1.0, 0])
        ci = Tex("computational imaging", font_size=60, color=YELLOW).move_to([0, -1.2, 0])
        dar = Arrow([0, 0.55, 0], [0, -0.6, 0], buff=0, stroke_width=4, color=GREY_A, max_tip_length_to_length_ratio=0.2)
        self.at(S("m6"))
        self.play(FadeOut(VGroup(gax, c_otf, c_inv, l_otf, l_inv, c_w, l_w, wf_eq, snr, gone, gone_l, lost, lost_l, lfr, ll, mfr, wl)),
                  mf.animate.set_value(0.0), wf.animate.set_value(0.0), run_time=0.7)
        self.remove(mm, wm)
        self.play(FadeIn(t1, shift=0.1 * UP), run_time=0.6)
        self.play(FadeIn(orr), FadeIn(t2, shift=0.1 * UP), run_time=0.6)
        self.at(S("m6") + 3.6)
        self.play(GrowArrow(dar), FadeIn(ci, shift=0.15 * DOWN), run_time=1.0)
        self.finish()


# ===================================================================== 6  recap =====
class Ch6Recap(Ep):
    CH = 6
    TAG = "6\\quad Recap"

    def construct(self):
        S, E = self.S, self.E
        self.show_tag()
        # ---------------- r1: imaging is convolution with the PSF
        l1 = Tex("Imaging is convolution with the PSF.", font_size=42).move_to([0, 2.6, 0])
        ds = load("stamps.npz")
        spts, samps = ds["pts"], ds["amps"]
        PW, Y = 2.6, -0.55
        OC, PC_, IC = np.array([-3.4, Y, 0]), np.array([0.0, Y, 0]), np.array([3.4, Y, 0])
        ofr = frame_rect(OC, PW, PW)
        dots = VGroup(*[Dot(OC + PW / 10 * np.array([x, y, 0]), radius=0.03 + 0.035 * a, color=C_REF) for (x, y), a in zip(spts, samps)])
        pm, pf = self.static_image(heat_rgba(ds["icon"], 1.0, gamma=0.45), 1.5, PC_)
        im, imf = self.static_image(heat_rgba(ds["cum"][-1], 1.0, gamma=0.55), PW, IC)
        pfr, ifr = frame_rect(PC_, 1.5, 1.5), frame_rect(IC, PW, PW)
        ops = VGroup(MathTex("\\ast", font_size=60).move_to([-1.55, Y, 0]), MathTex("=", font_size=60).move_to([1.55, Y, 0]))
        pl = Tex("PSF", font_size=28).next_to(pfr, DOWN, buff=0.12)
        self.at(S("r1") + 0.2)
        self.play(FadeIn(l1, shift=0.1 * DOWN), run_time=0.8)
        self.add(pm, im)
        self.play(FadeIn(ofr), FadeIn(dots), FadeIn(ops), pf.animate.set_value(1.0), FadeIn(pfr), FadeIn(pl),
                  imf.animate.set_value(1.0), FadeIn(ifr), run_time=1.0)

        # ---------------- r2: the lens keeps sin(theta) <= NA; finer than lambda/2NA never arrives
        l2 = Tex("A lens only keeps directions with $\\sin\\theta \\le \\mathrm{NA}$,", font_size=34).move_to([0, 1.9, 0])
        l3 = Tex("so details finer than about $\\lambda/2\\mathrm{NA}$ never arrive.", font_size=34).move_to([0, 1.35, 0])
        KC, R = np.array([-3.0, -2.45, 0]), 1.8
        ax = kspace_axes(KC, R, ext=0.35, up=0.4, fs=28)
        nr = ValueTracker(0.6)
        ths = np.radians(np.linspace(-80, 80, 15))

        def fan_cap():
            n_ = nr.get_value()
            g = VGroup()
            for t_ in ths:
                inside = abs(np.sin(t_)) <= n_
                g.add(fast_arrow(KC, KC + R * np.array([np.sin(t_), np.cos(t_), 0]), WHITE if inside else GREY_D, 2.5, 0.12,
                                 1.0 if inside else 0.45))
            return g.add(cap_group(KC, R, n_, 1.0))
        fc_m = always_redraw(fan_cap)
        dn = load("na.npz")
        nas, stack = dn["nas"], dn["stack"]
        PC2 = np.array([2.9, -1.1, 0])
        am, af = self.live_image(lambda t: heat_rgba(stack[int(np.clip(round((nr.get_value() - nas[0]) / 0.005), 0, len(nas) - 1))]
                                                     .astype(np.float32), 1.0, gamma=0.45), 2.9, PC2)
        afr = frame_rect(PC2, 2.9, 2.9)
        arr = Arrow([-0.45, -1.1, 0], [1.25, -1.1, 0], buff=0, stroke_width=4, color=GREY_A, max_tip_length_to_length_ratio=0.15)
        self.at(S("r2"))
        self.play(FadeOut(VGroup(ofr, dots, ops, pfr, pl, ifr)), pf.animate.set_value(0.0), imf.animate.set_value(0.0),
                  FadeIn(l2), run_time=0.8)
        self.remove(pm, im)
        self.add(am)
        self.play(FadeIn(ax), FadeIn(fc_m), run_time=1.0)
        self.play(GrowArrow(arr), af.animate.set_value(1.0), FadeIn(afr), run_time=0.8)
        self.at(S("r2") + 3.4)
        self.play(FadeIn(l3), nr.animate.set_value(0.88), run_time=2.0)
        self.play(nr.animate.set_value(0.45), run_time=2.0)

        # ---------------- r3: where it breaks -- the PSF changes across the field (computed, illustrative)
        da = load("aberr.npz")
        mos = da["mosaic"]
        MC_, MW = np.array([0, -0.95, 0]), 4.8
        MH = MW * mos.shape[0] / mos.shape[1]
        base = heat_rgba(mos, 1.0, gamma=0.5)
        yy, xx = np.mgrid[0:mos.shape[0], 0:mos.shape[1]]
        rn = np.hypot((xx - mos.shape[1] / 2) / (mos.shape[1] / 2), (yy - mos.shape[0] / 2) / (mos.shape[1] / 2))
        rv = ValueTracker(0.0)

        def mos_img(t):
            img = base.copy()
            img[..., 3] = (255 * np.clip((rv.get_value() - rn) / 0.08, 0, 1)).astype(np.uint8)
            return img
        mm, mf = self.live_image(mos_img, MW, MC_, MH)
        mfr = frame_rect(MC_, MW, MH)
        ml = Tex("one lens, across the field (computed, illustrative aberrations)", font_size=26,
                 color=GREY_A).next_to(mfr, DOWN, buff=0.15)
        self.at(S("r3"))
        self.play(FadeOut(VGroup(ax, afr, arr)), FadeOut(fc_m), af.animate.set_value(0.0), run_time=0.6)
        self.remove(am, fc_m)
        self.add(mm)
        mf.set_value(1.0)
        self.play(FadeIn(mfr), FadeIn(ml), rv.animate.set_value(0.12), run_time=0.8)
        self.at(S("r3") + 2.0)
        self.play(rv.animate.set_value(1.25), run_time=3.2, rate_func=linear)

        # ---------------- r4: teaser -- when do fields add, and when do intensities?
        nc = next_card("coherence: when does light interfere?", y=-1.35)
        dop = ValueTracker(0.0)

        def dials():
            o = dop.get_value()
            g = VGroup()
            if o <= 0.002:
                return g
            t = self.now
            for c, a in ((np.array([-0.55, 0.05, 0]), 2.2 * t), (np.array([0.55, 0.05, 0]), 2.55 * t + 0.4)):
                g.add(Circle(radius=0.4).move_to(c).set_stroke(GREY_B, 1.5, opacity=o),
                      fast_arrow(c, c + 0.4 * np.array([np.cos(a), np.sin(a), 0]), C_U, 4, 0.13, o))
            return g
        dl_m = always_redraw(dials)
        self.at(S("r4"))
        self.play(FadeOut(VGroup(mfr, ml)), mf.animate.set_value(0.0), run_time=0.6)
        self.remove(mm)
        self.add(dl_m)
        self.play(dop.animate.set_value(1.0), run_time=0.8)
        self.at(S("r4") + 2.6)
        self.play(FadeIn(nc, shift=0.1 * UP), run_time=0.9)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Stamps", "Ch2Pupil", "Ch3Abbe", "Ch4Coherence", "Ch5Computer", "Ch6Recap"]
