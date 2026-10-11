"""Why a Lens Computes a Fourier Transform -- chapter scenes (Optics in 5 Minutes, E02).
Render: ./render.sh ; preview: ./preview.sh Ch0Hook
Every physical field is precomputed by physics.py (data/*.npz); construct() only loads and draws.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403
from PIL import Image  # noqa: E402


def _at_whole_frames(self, t):
    """Wait until global time t in whole frames (pilot fix, FAST_ANIMATION_PIPELINE.md section 10): a float wait can
    flip len(np.arange(0, d, 1/60)) between a rendered and a skipped clock when t lies on the frame grid."""
    x = (t - self.now) * config.frame_rate                # frames still to go
    if x > 0.5 + 1e-6:
        k = int(np.ceil(x - 1e-6))
        self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
    elif x < -0.5 - 1e-6:
        self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


Chapter.at = _at_whole_frames

# direction tints (kx < 0, kx = 0, kx > 0): the series' diverging colours, used for "colour = direction"
T_NEG, T_ZERO, T_POS = np.array([0.36, 0.80, 1.0]), np.array([0.94, 0.94, 0.94]), np.array([1.0, 0.80, 0.30])
C_NEG, C_ZERO, C_POS = rgb_to_color(T_NEG), rgb_to_color(T_ZERO), rgb_to_color(T_POS)
C_CHIRP = C_EV                       # the troublesome chirp across the slide
C_LENS = C_H                         # the lens phase (a multiplier, like a transfer function)


def load(name):
    return np.load(data_path(name))


def tint_of(s):
    """Colour for a direction with normalised kx = s in [-1, 1] (cyan - white - gold)."""
    s = float(np.clip(s, -1, 1))
    return T_ZERO + (T_POS - T_ZERO) * s if s >= 0 else T_ZERO + (T_NEG - T_ZERO) * (-s)


def log_heat(A, floor=3e-5, top=0.75, gamma=1.4, flip=True):
    L = np.log10(1 + np.asarray(A, float) / floor) / np.log10(1 + 1 / floor) / top
    return heat_rgba(L, 1.0, gamma, flip)


def radial_reveal(base, r_t, soft=0.05):
    """fn(t) for live_image: the image appears from the centre outwards as tracker r_t goes 0 -> 1."""
    h, w = base.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    R = np.hypot((xx - (w - 1) / 2) / (w / 2), (yy - (h - 1) / 2) / (h / 2)) / np.sqrt(2)
    a0 = base[..., 3].astype(np.float32)

    def fn(t):
        img = base.copy()
        img[..., 3] = (a0 * np.clip((r_t.get_value() * (1 + soft) - R) / soft, 0, 1)).astype(np.uint8)
        return img
    return fn


def lens_shape(c, h, w, color=GREY_A, fill=0.22, width=2.5):
    """Biconvex lens outline (side view), centre c, height h, max thickness w."""
    s = w / 2
    R = (h ** 2 / 4 + s ** 2) / (2 * s)
    th = np.arcsin(h / (2 * R))
    a1 = np.linspace(-th, th, 40)
    a2 = np.linspace(np.pi - th, np.pi + th, 40)
    c = np.asarray(c, float)
    right = [c + np.array([-(R - s) + R * np.cos(a), R * np.sin(a), 0]) for a in a1]
    left = [c + np.array([(R - s) + R * np.cos(a), R * np.sin(a), 0]) for a in a2]
    vm = VMobject().set_points_as_corners(right + left + [right[0]])
    return vm.set_stroke(color, width).set_fill(color, fill)


def dots_vm(pts, r=0.04, color=YELLOW, n=10, opacity=1.0):
    """Many small filled dots as one VMobject (cheap enough for per-frame redraws)."""
    vm = VMobject().set_stroke(width=0).set_fill(color, opacity)
    ang = np.linspace(0, TAU_, n, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1) * r
    for p in pts:
        q = np.asarray(p, float) + ring
        vm.start_new_path(q[0])
        for k in range(1, n):
            vm.add_line_to(q[k])
        vm.add_line_to(q[0])
    return vm


def frame_box(center, w, h, color=GREY_B, width=1.5):
    return Rectangle(width=w, height=h).move_to(center).set_stroke(color, width)


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        hk = load("hook.npz")
        ya = -2.0                                            # side-view optical axis
        x_laser, x_slide, x_wall, x_lens, x_foc = -6.25, -4.75, 6.35, -4.15, -1.55
        laser = VGroup(RoundedRectangle(corner_radius=0.08, width=0.95, height=0.42).set_fill(GREY_E, 1).set_stroke(GREY_B, 2),
                       Tex("laser", font_size=26, color=GREY_B)).move_to([x_laser, ya, 0])
        laser[1].next_to(laser[0], DOWN, buff=0.12)
        slide = Rectangle(width=0.07, height=0.8).move_to([x_slide, ya, 0]).set_fill(GREY_C, 0.9).set_stroke(GREY_A, 1)
        slide_l = Tex("slide", font_size=26, color=GREY_B).next_to(slide, DOWN, buff=0.12)
        wall = Rectangle(width=0.13, height=0.9).move_to([x_wall, ya, 0]).set_fill(GREY_D, 1).set_stroke(GREY_B, 1)
        wall_l = Tex("far wall", font_size=26, color=GREY_B).next_to(wall, DOWN, buff=0.12).shift(0.3 * LEFT)
        beam = Rectangle(width=x_slide - (x_laser + 0.48), height=0.18).set_stroke(width=0).set_fill(BLUE_B, 0.6)
        beam.move_to([0.5 * (x_slide + x_laser + 0.48), ya, 0])
        reach, cone_op = ValueTracker(0.0), ValueTracker(0.22)

        def cone():
            r = reach.get_value()
            if r < 1e-3:
                return VMobject()
            xe = x_slide + r * (x_wall - 0.07 - x_slide)
            he = 0.06 + r * 0.34
            return Polygon([x_slide + 0.04, ya + 0.06, 0], [xe, ya + he, 0], [xe, ya - he, 0], [x_slide + 0.04, ya - 0.06, 0]
                           ).set_stroke(width=0).set_fill(BLUE_B, cone_op.get_value())
        cone_m = always_redraw(cone)
        # face-on views (computed)
        c_obj, w_obj = np.array([x_slide, 1.0, 0]), 1.8
        c_foc, w_foc = np.array([-1.0, 1.1, 0]), 2.5
        c_far, w_far = np.array([4.3, 1.15, 0]), 3.9
        obj_m, obj_f = self.static_image(heat_rgba(hk["obj"], 1.0, 1.0), w_obj, c_obj, nearest=True)
        obj_box = frame_box(c_obj, w_obj, w_obj)
        obj_l = Tex("the slide", font_size=26, color=GREY_A).next_to(obj_box, UP, buff=0.12)
        con_obj = fast_dashed([x_slide, ya + 0.45, 0], [x_slide, c_obj[1] - w_obj / 2 - 0.03, 0], 0.05, GREY_C, 1.2)
        r_far, r_foc = ValueTracker(0.0), ValueTracker(0.0)
        far_m, far_f = self.live_image(radial_reveal(log_heat(hk["far"]), r_far), w_far, c_far)
        far_box = frame_box(c_far, w_far, w_far)
        far_l = Tex("on the far wall", font_size=26, color=GREY_A).next_to(far_box, UP, buff=0.12)
        foc_m, foc_f = self.live_image(radial_reveal(log_heat(hk["lens"]), r_foc), w_foc, c_foc)
        foc_box = frame_box(c_foc, w_foc, w_foc)
        foc_l = Tex("at the focal plane", font_size=26, color=GREY_A).next_to(foc_box, UP, buff=0.12)

        # ---------------- lead-in and h1: laser through the slide
        self.play(FadeIn(laser), FadeIn(slide), FadeIn(slide_l), FadeIn(wall), FadeIn(wall_l), run_time=0.7)
        self.at(S("h1") + 0.2)
        self.play(GrowFromEdge(beam, LEFT), run_time=0.7)
        self.add(obj_m)
        self.play(obj_f.animate.set_value(1.0), FadeIn(obj_box), FadeIn(obj_l), Create(con_obj), run_time=0.8)

        # ---------------- h2: the far wall shows streaks, not an F
        self.at(S("h2"))
        self.add(cone_m)
        self.play(reach.animate.set_value(1.0), run_time=1.3, rate_func=linear)
        self.add(far_m)
        far_f.set_value(1.0)
        self.play(r_far.animate.set_value(1.0), FadeIn(far_box), FadeIn(far_l), run_time=1.6, rate_func=rate_functions.ease_out_sine)

        # ---------------- h3: it is the Fourier transform of the F
        ft_l = VGroup(Tex("= the Fourier transform of the F", font_size=30, color=C_KX),
                      Tex("(computed: Fresnel propagation to $10^6\\lambda$)", font_size=22, color=GREY_B)).arrange(DOWN, buff=0.08)
        ft_l.next_to(far_box, DOWN, buff=0.12)
        self.at(S("h3") + 0.6)
        self.play(Write(ft_l[0]), run_time=1.3)
        self.play(FadeIn(ft_l[1]), run_time=0.5)

        # ---------------- h4: a lens behind the slide gives the same pattern one focal length away
        lens = lens_shape([x_lens, ya, 0], 1.1, 0.3)
        lens_l = Tex("lens", font_size=26, color=GREY_B).next_to(lens, UP, buff=0.1)
        screen = Rectangle(width=0.08, height=0.8).move_to([x_foc, ya, 0]).set_fill(GREY_D, 1).set_stroke(GREY_B, 1)
        conv = Polygon([x_lens + 0.05, ya + 0.42, 0], [x_foc - 0.04, ya + 0.08, 0], [x_foc - 0.04, ya - 0.08, 0], [x_lens + 0.05, ya - 0.42, 0]
                       ).set_stroke(width=0).set_fill(BLUE_B, 0.32)
        fbr = BraceBetweenPoints([x_lens, ya - 0.6, 0], [x_foc, ya - 0.6, 0], DOWN, color=GREY_B, buff=0.02)
        fbr_l = MathTex("f", font_size=30, color=GREY_A).next_to(fbr, DOWN, buff=0.04)
        con_foc = fast_dashed([x_foc, ya + 0.45, 0], [x_foc, c_foc[1] - w_foc / 2 - 0.03, 0], 0.05, GREY_C, 1.2)
        self.at(S("h4"))
        self.play(FadeIn(lens, shift=0.2 * DOWN), FadeIn(lens_l), cone_op.animate.set_value(0.07),
                  wall.animate.set_opacity(0.35), run_time=0.8)
        self.play(FadeIn(conv), FadeIn(screen), GrowFromCenter(fbr), FadeIn(fbr_l), run_time=0.8)
        self.add(foc_m)
        foc_f.set_value(1.0)
        self.at(S("h4") + 2.2)
        self.play(r_foc.animate.set_value(1.0), FadeIn(foc_box), FadeIn(foc_l), Create(con_foc), run_time=1.5,
                  rate_func=rate_functions.ease_out_sine)

        # ---------------- h5: the question, then the title card
        dim = Rectangle(width=16, height=9).set_stroke(width=0).set_fill(BLACK, 0.0)
        card = series_title(2, "Why a Lens Computes a Fourier Transform", "far fields, lenses and the 4f machine", y=0.55, size=56)
        self.at(S("h5") + 0.4)
        self.add(dim)
        self.play(dim.animate.set_fill(BLACK, 0.55), FadeIn(card, shift=0.15 * DOWN), run_time=1.0)
        self.finish()


# ============================================================ 1  far field =========
class Ch1FarField(Chapter):
    CH = 1
    TAG = "1\\quad Far away"

    def construct(self):
        S = self.S
        self.show_tag()
        zd = load("zoom.npz")
        maps = zd["maps"].astype(np.float32)
        Zl, rho, D = zd["Zl"], float(zd["rho"]), float(zd["D"])
        nlev, nx, nz = maps.shape
        Z0, Z1 = float(Zl[0]), float(Zl[-1])
        TA = S("f4") + 0.15                                  # the zoom-out is a pure function of time
        TB = TA + 5.0

        def zv():
            u = np.clip((self.now - TA) / (TB - TA), 0, 1)
            return 0.5 - 0.5 * np.cos(np.pi * u)

        def Zcur():
            return Z0 * (Z1 / Z0) ** zv()
        # map panel: z to the right, x up
        mc, mw, mh = np.array([0.1, 0.72, 0]), 9.4, 4.1
        xl, xr, yb, yt = mc[0] - mw / 2, mc[0] + mw / 2, mc[1] - mh / 2, mc[1] + mh / 2
        out_w, out_h = 640, 280

        def zoom_img(t):
            Z = Zcur()
            li = int(min(np.searchsorted(Zl, Z * (1 - 1e-9)), nlev - 1))
            r = Z / Zl[li]
            box = (0.0, nx / 2 * (1 - r), nz * r, nx / 2 * (1 + r))
            a = np.asarray(Image.fromarray(maps[li]).resize((out_w, out_h), Image.BILINEAR, box=box))
            return heat_rgba(np.sqrt(np.maximum(a, 0)), 1.25, 0.6, flip=False)
        zim, zfade = self.live_image(zoom_img, mw, mc, mh)
        zbox = frame_box(mc, mw, mh)

        def slit_bars():
            g = (D / 2) / (rho * Zcur()) * (mh / 2)
            return VGroup(Line([xl, mc[1] + g, 0], [xl, yt, 0]), Line([xl, yb, 0], [xl, mc[1] - g, 0])).set_stroke(WHITE, 5)
        bars = always_redraw(slit_bars)
        # distance read-out under the right edge (pre-built labels, cross-faded with the zoom)
        dists = [150, 1500, 15000, 150000]
        dlab = [MathTex(f"z = {d:,}".replace(",", "\\,") + "\\,\\lambda", font_size=30, color=GREY_A) for d in dists]
        dl_op = ValueTracker(0.0)
        for i, m in enumerate(dlab):
            m.next_to([xr, yb, 0], DOWN, buff=0.12).align_to([xr, 0, 0], RIGHT)
            m.add_updater(lambda m, i=i: m.set_opacity(
                float(np.clip(1.0 - abs(np.log10(Zcur() / dists[i])) / 0.5, 0, 1)) * dl_op.get_value()))
        slit_l = Tex("slit", font_size=26, color=GREY_B).next_to([xl, yb, 0], DOWN, buff=0.12)
        comp1 = Tex("computed: angular spectrum method", font_size=22, color=GREY_B).move_to([xl, yt - 0.22, 0])
        comp1.align_to([xl + 0.22, 0, 0], LEFT)

        # ---------------- f1: plane waves leaving the slit, each in its own direction
        self.add(zim)
        self.play(zfade.animate.set_value(1.0), FadeIn(zbox), FadeIn(bars), FadeIn(slit_l), FadeIn(comp1), run_time=0.9)
        self.add(*dlab)
        self.play(dl_op.animate.set_value(1.0), run_time=0.4)
        apex = np.array([xl, mc[1], 0])
        sx = mw / Z0
        sy = (mh / 2) / (rho * Z0)
        dir_op = ValueTracker(0.0)
        angs = [-30.0, -15.0, 0.0, 15.0, 30.0]

        def dir_arrows():
            o = dir_op.get_value()
            if o < 1e-3:
                return VGroup()
            g = VGroup()
            for a in angs:
                th = np.radians(a)
                v = np.array([np.cos(th) * sx, np.sin(th) * sy, 0])
                v = v / np.linalg.norm(v)
                g.add(fast_arrow(apex + 0.1 * v, apex + 2.43 * v, BLACK, 10, 0.28))           # halo
                g.add(fast_arrow(apex + 0.12 * v, apex + 2.4 * v, rgb_to_color(tint_of(np.sin(th) / 0.5)), 5, 0.2))
            return fade_vgroup(g, o)
        darr = always_redraw(dir_arrows)
        th_d = np.arctan2(np.sin(np.radians(30.0)) * sy, np.cos(np.radians(30.0)) * sx)
        arc = Arc(radius=1.25, start_angle=0, angle=th_d, arc_center=apex).set_stroke(WHITE, 2.5)
        th_l = MathTex("\\theta", font_size=32).move_to(apex + 1.5 * np.array([np.cos(th_d / 2), np.sin(th_d / 2), 0]))
        sin_eq = MathTex("\\sin\\theta", "=", "{k_x", "\\over", "k}", font_size=40).move_to([-2.2, 3.25, 0])
        sin_eq[2].set_color(C_KX)
        pw_l = Tex("plane waves, one per direction", font_size=30).next_to(sin_eq, RIGHT, buff=0.5)
        self.at(S("f1") + 1.6)
        self.add(darr)
        self.play(dir_op.animate.set_value(1.0), run_time=0.9)
        self.play(FadeIn(pw_l), run_time=0.5)
        self.at(S("f1") + 5.6)
        self.play(Write(sin_eq), Create(arc), FadeIn(th_l), run_time=1.2)

        # ---------------- f2: their amplitudes = the spectrum (strip under the map)
        kq, spk = zd["kq"], zd["spec_k"]
        yb0, sh = -3.0, 1.25
        s_max = rho / np.sqrt(1 + rho ** 2)                   # sin(theta) at the map's top edge
        sxk = (mh / 2) / (s_max * TAU_)
        kq, spk = kq[np.abs(kq) <= s_max * TAU_], spk[np.abs(kq) <= s_max * TAU_]
        strip_ax = Line([mc[0] - mh / 2 - 0.15, yb0, 0], [mc[0] + mh / 2 + 0.15, yb0, 0]).set_stroke(GREY_C, 1.5)
        strip_kx = MathTex("k_x", font_size=28, color=C_KX).next_to(strip_ax, RIGHT, buff=0.1)
        spec_curve = poly_line(mc[0] + kq * sxk, yb0 + sh * spk, C_KX, 4)
        spec_l = VGroup(MathTex("\\hat U(k_x)", font_size=34, color=C_KX), Tex("spectrum", font_size=26, color=GREY_A)
                        ).arrange(DOWN, buff=0.08).move_to([3.75, -2.55, 0])
        self.at(S("f2") + 0.2)
        self.play(Create(strip_ax), FadeIn(strip_kx), Create(spec_curve), FadeIn(spec_l), run_time=1.4)

        # ---------------- f3: up close the waves overlap: a shadow
        near_br = BraceBetweenPoints([xl + 0.05, yb - 0.02, 0], [xl + 1.9, yb - 0.02, 0], DOWN, color=GREY_B)
        near_l = Tex("a shadow", font_size=26, color=GREY_A).next_to(near_br, DOWN, buff=0.06)
        self.at(S("f3"))
        self.play(dir_op.animate.set_value(0.0), FadeOut(arc), FadeOut(th_l), FadeOut(pw_l),
                  FadeOut(sin_eq), FadeOut(slit_l), run_time=0.6)
        self.remove(darr)
        self.play(GrowFromCenter(near_br), FadeIn(near_l), run_time=0.6)

        # ---------------- f4: step back 1000 times (zoom runs from TA to TB, driven by the clock)
        self.at(S("f4"))
        self.play(FadeOut(near_br), FadeOut(near_l), run_time=0.3)

        # ---------------- f5: straight lines: direction theta lands at x = z tan(theta)
        ray_t = ValueTracker(0.0)
        r_angs = [-9.0, 0.0, 4.0, 9.0]
        syf = (mh / 2) / (rho * Z1)

        def rays():
            r = ray_t.get_value()
            if r < 1e-3:
                return VGroup()
            g = VGroup()
            for a in r_angs:
                end = apex + np.array([mw, np.tan(np.radians(a)) * Z1 * syf, 0])
                col = rgb_to_color(tint_of(np.sin(np.radians(a)) / 0.16))
                g.add(Line(apex, apex + r * (end - apex)).set_stroke(col, 3.5))
                if r > 0.98:
                    g.add(Dot(end, radius=0.075, color=col))
            return g
        ray_m = always_redraw(rays)
        a9 = np.arctan2(np.tan(np.radians(9.0)) * Z1 * syf, mw)
        arc2 = Arc(radius=2.0, start_angle=0, angle=a9, arc_center=apex).set_stroke(WHITE, 2.5)
        th2 = MathTex("\\theta", font_size=32).move_to(apex + 2.3 * np.array([np.cos(a9 / 2), np.sin(a9 / 2), 0]))
        land = MathTex("x", "=", "z", "\\tan\\theta", font_size=36, color=C_POS)
        land = VGroup(BackgroundRectangle(land, fill_opacity=0.75, buff=0.08), land).next_to(
            [xr, mc[1] + np.tan(np.radians(9.0)) * Z1 * syf, 0], LEFT, buff=0.3).shift(0.42 * UP)
        self.at(S("f5") + 0.2)
        self.add(ray_m)
        self.play(ray_t.animate.set_value(1.0), run_time=2.3, rate_func=linear)
        self.at(S("f5") + 5.6)
        self.play(Create(arc2), FadeIn(th2), run_time=0.6)
        self.at(S("f5") + 6.6)
        self.play(Write(land), run_time=0.9)

        # ---------------- f6: the pattern on the wall is the spectrum itself (Fraunhofer)
        xw, Aw, spw = zd["xw"], zd["Aw"], zd["spec"]
        x0w, dw = xr + 0.12, 1.25
        ys_w = mc[1] + xw * syf
        wall_ln = Line([x0w, yb, 0], [x0w, yt, 0]).set_stroke(GREY_C, 1.5)
        prof = poly_line(x0w + dw * Aw, ys_w, C_U, 8)
        spec_on_wall = DashedVMobject(poly_line(x0w + dw * spw, ys_w, C_KX, 3.5), num_dashes=70, dashed_ratio=0.55)
        prof_l = Tex("wall", font_size=26, color=C_U).next_to(wall_ln, UP, buff=0.1).shift(0.35 * RIGHT)
        frau = VGroup(MathTex("U_{\\text{wall}}(x)", "\;\\propto\;", "\\hat U", "(k\\sin\\theta)", font_size=40),
                      Tex("Fraunhofer diffraction", font_size=30, color=GREY_A)).arrange(RIGHT, buff=0.45)
        frau[0][0].set_color(C_U)
        frau[0][2].set_color(C_KX)
        frau.move_to([0.6, 3.25, 0])
        self.at(S("f6") - 0.85)
        self.play(FadeOut(ray_m), FadeOut(arc2), FadeOut(th2), FadeOut(land), run_time=0.4)
        self.remove(ray_m)
        self.play(Create(wall_ln), Create(prof), FadeIn(prof_l), run_time=0.9)
        self.at(S("f6") + 1.0)
        tgt = spec_curve.copy().rotate(-PI / 2).stretch_to_fit_height(spec_on_wall.height).stretch_to_fit_width(spec_on_wall.width)
        tgt.move_to(spec_on_wall)
        self.play(Transform(spec_curve, tgt, path_arc=-PI / 2), FadeOut(strip_ax), FadeOut(strip_kx), FadeOut(spec_l), run_time=1.5)
        self.remove(spec_curve)
        self.add(spec_on_wall)
        self.at(S("f6") + 3.3)
        self.play(Write(frau[0]), run_time=1.0)
        self.play(FadeIn(frau[1]), run_time=0.5)

        # ---------------- f7: how far is far?  z lambda / D >> D  <=>  z >> D^2 / lambda
        x1 = Z1 / D                                           # first zero at x = z lambda / D
        hb = BraceBetweenPoints([xr - 0.05, mc[1], 0], [xr - 0.05, mc[1] + x1 * syf, 0], LEFT, color=WHITE)
        hb_l = MathTex("{z\\lambda", "\\over", "D}", font_size=34).next_to(hb, LEFT, buff=0.08)
        hb_l = VGroup(BackgroundRectangle(hb_l, fill_opacity=0.75, buff=0.06), hb_l)
        far_eq = MathTex("{z\\lambda \\over D}", "\\gg", "D", "\;\;\\Longleftrightarrow\;\;", "z", "\\gg", "{D^2 \\over \\lambda}",
                         font_size=40).move_to([-0.6, -2.2, 0])
        box = SurroundingRectangle(far_eq[4:], buff=0.12, color=YELLOW)
        ours = Tex("our slit: $D = 12\\lambda$, so the far field starts beyond about $150\\lambda$", font_size=26,
                   color=GREY_A).next_to(far_eq, DOWN, buff=0.22)
        self.at(S("f7") + 1.4)
        self.play(GrowFromCenter(hb), FadeIn(hb_l), run_time=0.8)
        self.at(S("f7") + 4.0)
        self.play(Write(far_eq[:3]), run_time=1.0)
        self.at(S("f7") + 6.5)
        self.play(Write(far_eq[3:]), run_time=1.0)
        self.play(Create(box), FadeIn(ours), run_time=0.7)
        self.finish()


# ============================================================ 2  paraxial ==========
class Ch2Paraxial(Chapter):
    CH = 2
    TAG = "2\\quad The paraxial shortcut"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- p1: one point on the slide, one point on the wall
        xs_, xw_, yp = -6.0, -1.6, -1.0
        zs_ = xw_ - xs_
        slide_ln = Line([xs_, -2.25, 0], [xs_, 2.2, 0]).set_stroke(GREY_B, 3)
        wall_ln = Line([xw_, -2.25, 0], [xw_, 2.2, 0]).set_stroke(GREY_B, 3)
        lab_s = Tex("slide", font_size=28, color=GREY_B).next_to(slide_ln, UP, buff=0.12)
        lab_w = Tex("wall", font_size=28, color=GREY_B).next_to(wall_ln, UP, buff=0.12)
        yq = ValueTracker(1.3)
        p_src = Dot([xs_, yp, 0], radius=0.08, color=C_U)
        l_src = MathTex("x'", font_size=34, color=C_U).next_to(p_src, LEFT, buff=0.12)
        path_op = ValueTracker(0.0)

        def path():
            o = path_op.get_value()
            q = np.array([xw_, yq.get_value(), 0])
            g = VGroup(Line([xs_, yp, 0], q).set_stroke(WHITE, 3.5, opacity=o), Dot(q, radius=0.08, color=WHITE).set_opacity(o),
                       fast_dashed([xw_ + 0.18, yp, 0], [xw_ + 0.18, q[1], 0], 0.06, GREY_A, 2, o))
            return g
        path_m = always_redraw(path)
        l_dst = MathTex("x", font_size=34)
        l_dst.add_updater(lambda m: m.move_to([xw_ - 0.3, yq.get_value() + 0.05, 0]).set_opacity(path_op.get_value()))
        l_dx = MathTex("x - x'", font_size=30, color=GREY_A)
        l_dx.add_updater(lambda m: m.move_to([xw_ + 0.85, 0.5 * (yp + yq.get_value()), 0]).set_opacity(path_op.get_value()))
        zleg = fast_dashed([xs_, yp, 0], [xw_, yp, 0], 0.06, GREY_A, 2)
        l_z = MathTex("z", font_size=32, color=GREY_A).next_to(zleg, DOWN, buff=0.12)
        self.at(S("p1") + 0.2)
        self.play(Create(slide_ln), Create(wall_ln), FadeIn(lab_s), FadeIn(lab_w), run_time=0.9)
        self.at(S("p1") + 3.1)
        self.play(FadeIn(p_src, scale=1.5), FadeIn(l_src), run_time=0.5)
        self.add(path_m, l_dst, l_dx)
        self.at(S("p1") + 5.6)
        self.play(path_op.animate.set_value(1.0), Create(zleg), FadeIn(l_z), run_time=0.8)

        # ---------------- p2: for small angles the path length is nearly a parabola
        r_ex = MathTex("r", "=", "\\sqrt{z^2 + (x - x')^2}", font_size=40).move_to([2.6, 2.55, 0])
        r_ap = MathTex("\\approx", "z", "+", "{(x - x')^2", "\\over", "2z}", font_size=40)
        r_ap.next_to(r_ex[1], DOWN, buff=0.55, aligned_edge=LEFT)
        small = Tex("small angles", font_size=26, color=GREY_A).next_to(r_ap, RIGHT, buff=0.4)
        # mini plot: path length vs (x - x') in units of z: exact (white) vs parabola (dashed)
        pc, pw, ph = np.array([2.85, -1.45, 0]), 5.2, 2.1
        uu = np.linspace(-1.25, 1.25, 241)
        lim_y = (0.95, 1.85)
        ex_c = plot_xy(uu, np.sqrt(1 + uu ** 2), pc, pw, ph, (-1.25, 1.25), lim_y, WHITE, 3.5)
        pa_c = DashedVMobject(plot_xy(uu, np.minimum(1 + uu ** 2 / 2, 1.9), pc, pw, ph, (-1.25, 1.25), lim_y, C_KZ, 3.5),
                              num_dashes=40)
        ax = VGroup(Line(pc + [-pw / 2, -ph / 2, 0], pc + [pw / 2, -ph / 2, 0]), Line(pc + [0, -ph / 2, 0], pc + [0, ph / 2, 0])).set_stroke(GREY_C, 1.5)
        ax_l = VGroup(MathTex("x - x'", font_size=28, color=GREY_A).next_to(ax[0], DOWN, buff=0.1).shift(1.7 * RIGHT),
                      Tex("path length", font_size=26, color=GREY_A).next_to(ax[1], UP, buff=0.08))
        leg = VGroup(Tex("exact", font_size=26), Tex("parabola", font_size=26, color=C_KZ)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        leg.move_to(pc + [-pw / 2 + 0.75, -ph / 2 + 0.45, 0])

        def tracer():
            u = (yq.get_value() - yp) / zs_
            p1 = pc + np.array([u / 1.25 * pw / 2, (np.sqrt(1 + u * u) - lim_y[0]) / (lim_y[1] - lim_y[0]) * ph - ph / 2, 0])
            p2 = pc + np.array([u / 1.25 * pw / 2, (1 + u * u / 2 - lim_y[0]) / (lim_y[1] - lim_y[0]) * ph - ph / 2, 0])
            return VGroup(Dot(p1, radius=0.06, color=WHITE), Dot(p2, radius=0.06, color=C_KZ))
        trc = always_redraw(tracer)
        self.at(S("p2") - 0.1)
        self.play(Write(r_ex), run_time=1.2)
        self.at(S("p2") + 2.1)
        self.play(Create(ax), FadeIn(ax_l), Create(ex_c), Create(pa_c), FadeIn(leg), run_time=1.1)
        self.add(trc)
        self.play(yq.animate.set_value(-0.6), Write(r_ap), FadeIn(small), run_time=2.0)
        self.play(yq.animate.set_value(2.0), run_time=1.6)

        # ---------------- p3: the wall adds up the waves from every point
        fan_op = ValueTracker(0.0)
        ys_src = np.linspace(-1.9, 1.9, 9)

        def fan():
            o = fan_op.get_value()
            if o < 1e-3:
                return VGroup()
            q = np.array([xw_, yq.get_value(), 0])
            g = VGroup(*[Line([xs_, y, 0], q).set_stroke(C_U, 2, opacity=0.65 * o) for y in ys_src])
            g.add(dots_vm([[xs_, y, 0] for y in ys_src], 0.05, C_U, opacity=o))
            return g
        fan_m = always_redraw(fan)
        integ = MathTex("U(x)", "\;\\propto\;", "\\int", "U(x')", "\\,e^{\\,ik(x - x')^2/2z}", "\\,dx'", font_size=42)
        integ[0].set_color(C_U)
        integ[3].set_color(C_U)
        integ.move_to([2.75, 0.45, 0])
        self.at(S("p3"))
        self.play(FadeOut(VGroup(ax, ax_l, ex_c, pa_c, leg)), FadeOut(trc), run_time=0.5)
        self.remove(trc)
        self.add(fan_m)
        self.play(fan_op.animate.set_value(1.0), yq.animate.set_value(1.0), run_time=1.2)
        self.at(S("p3") + 2.9)
        self.play(Write(integ), run_time=1.6)

        # ---------------- p4: expand the square: three factors
        exp_sq = MathTex("(x - x')^2", "=", "x^2", "-", "2xx'", "+", "x'^2", font_size=38).move_to([0, 1.25, 0])
        exp_sq[2].set_color(GREY_A)
        exp_sq[4].set_color(C_KX)
        exp_sq[6].set_color(C_CHIRP)
        full = MathTex("U(x)", "\;\\propto\;", "e^{\\,ikx^2/2z}", "\\int", "U(x')", "\\,e^{-i(kx/z)\\,x'}", "\\,e^{\\,ikx'^2/2z}", "\\,dx'",
                       font_size=46).move_to([0, 2.45, 0])
        full[0].set_color(C_U)
        full[4].set_color(C_U)
        full[2].set_color(GREY_A)
        full[5].set_color(C_KX)
        full[6].set_color(C_CHIRP)
        self.at(S("p4"))
        for m in (l_dst, l_dx, path_m, fan_m):
            m.clear_updaters()
        self.play(FadeOut(VGroup(slide_ln, wall_ln, lab_s, lab_w, p_src, l_src, zleg, l_z, r_ex, r_ap, small)), FadeOut(path_m),
                  FadeOut(fan_m), FadeOut(l_dst), FadeOut(l_dx), integ.animate.move_to([0, 2.45, 0]).scale(1.08), run_time=0.6)
        self.remove(path_m, fan_m, l_dst, l_dx)
        self.play(Write(exp_sq), run_time=0.8)
        self.at(S("p4") + 1.5)
        self.play(FadeTransform(integ, full), run_time=1.0)
        br1 = Brace(full[2], DOWN, buff=0.1, color=GREY_A)
        t1 = VGroup(Tex("a curved phase", font_size=26, color=GREY_A), Tex("on the wall: $|U|^2$", font_size=26, color=GREY_A),
                    Tex("can't see it", font_size=26, color=GREY_A)).arrange(DOWN, buff=0.06).next_to(br1, DOWN, buff=0.1)
        self.at(S("p4") + 2.6)
        self.play(FadeOut(exp_sq), GrowFromCenter(br1), FadeIn(t1), run_time=0.8)

        # ---------------- p5: the cross term is the Fourier kernel, kx = k x / z
        br2 = Brace(full[5], DOWN, buff=0.1, color=C_KX)
        t2 = VGroup(Tex("Fourier kernel", font_size=30, color=C_KX),
                    MathTex("k_x", "=", "k\\,x / z", font_size=34, color=C_KX)).arrange(DOWN, buff=0.12).next_to(br2, DOWN, buff=0.1)
        self.at(S("p5") + 0.2)
        self.play(GrowFromCenter(br2), FadeIn(t2[0]), run_time=0.8)
        self.at(S("p5") + 3.2)
        self.play(Indicate(full[5], color=C_KX, scale_factor=1.12), run_time=1.0)
        self.at(S("p5") + 4.6)
        self.play(Write(t2[1]), run_time=1.0)

        # ---------------- p6: the x'^2 term is a chirp across the slide; far away it flattens to 1
        fz = load("fresnel.npz")
        zsF, kqF, patF, specF, D = fz["zs"], fz["kq"], fz["pat"], fz["spec"], float(fz["D"])
        zF = D ** 2
        br3 = Brace(full[6], DOWN, buff=0.1, color=C_CHIRP)
        t3 = VGroup(Tex("a chirp", font_size=30, color=C_CHIRP), Tex("across the slide", font_size=26, color=C_CHIRP)
                    ).arrange(DOWN, buff=0.08).next_to(br3, DOWN, buff=0.1)
        zl = ValueTracker(np.log10(0.045))                     # log10(z / (D^2/lambda))
        lc, lw, lh = np.array([-3.55, -1.45, 0]), 5.3, 1.9
        rc, rw, rh = np.array([3.4, -1.45, 0]), 5.6, 1.9
        frames = VGroup(frame_box(lc, lw, lh, GREY_D), frame_box(rc, rw, rh, GREY_D))
        ttl_l = MathTex("\\mathrm{Re}\\,", "e^{\\,ikx'^2/2z}", "\\text{ across the slide}", font_size=30).next_to(frames[0], UP, buff=0.1)
        ttl_l[1].set_color(C_CHIRP)
        ttl_r = VGroup(Tex("on the wall, vs $k_x$:", font_size=28, color=C_U), Tex("spectrum", font_size=28, color=C_KX)
                       ).arrange(RIGHT, buff=0.3).next_to(frames[1], UP, buff=0.1)
        xe = 0.62 * D                                           # left panel shows x' in [-xe, xe]
        edges = VGroup(*[fast_dashed([lc[0] + sg * (D / 2) / xe * lw / 2, lc[1] - lh / 2, 0], [lc[0] + sg * (D / 2) / xe * lw / 2, lc[1] + lh / 2, 0],
                                     0.06, GREY_B, 1.5) for sg in (-1, 1)])
        spec_c = DashedVMobject(plot_xy(kqF, specF, rc, rw, rh * 0.92, (kqF[0], kqF[-1]), (0, 1), C_KX, 3.5), num_dashes=60)
        pan_op = ValueTracker(0.0)

        def chirp():
            o = pan_op.get_value()
            if o < 1e-3:
                return VGroup()
            z = zF * 10 ** zl.get_value()
            xx = np.linspace(-D / 2, D / 2, 500)
            yy = np.cos(TAU_ * xx ** 2 / (2 * z))
            c = plot_xy(xx, yy, lc, lw, lh * 0.86, (-xe, xe), (-1, 1), C_CHIRP, 3.5, o)
            mid = Line([lc[0] - lw / 2, lc[1], 0], [lc[0] + lw / 2, lc[1], 0]).set_stroke(GREY_D, 1, opacity=o)
            return VGroup(mid, c)

        def pattern():
            o = pan_op.get_value()
            if o < 1e-3:
                return VGroup()
            lz = np.log10(zF * 10 ** zl.get_value())
            f = np.interp(lz, np.log10(zsF), np.arange(len(zsF)))
            i0 = int(np.clip(np.floor(f), 0, len(zsF) - 2))
            w = f - i0
            pv = (1 - w) * patF[i0] + w * patF[i0 + 1]
            return plot_xy(kqF, pv, rc, rw, rh * 0.92, (kqF[0], kqF[-1]), (0, 1), C_U, 4.5, o)
        chirp_m, pat_m = always_redraw(chirp), always_redraw(pattern)
        # z slider (log)
        sl0, sl1, sly = -2.3, 2.3, -2.82
        lo, hi = np.log10(0.045), np.log10(25.0)
        slider = Line([sl0, sly, 0], [sl1, sly, 0]).set_stroke(GREY_B, 2)
        ticks = VGroup()
        for v, txt in ((0.1, "0.1"), (1.0, "1"), (10.0, "10")):
            xv = sl0 + (np.log10(v) - lo) / (hi - lo) * (sl1 - sl0)
            ticks.add(Line([xv, sly - 0.07, 0], [xv, sly + 0.07, 0]).set_stroke(GREY_B, 2),
                      MathTex(txt, font_size=24, color=GREY_B).move_to([xv, sly - 0.27, 0]))
        sl_l = MathTex("z \\,/\\, (D^2/\\lambda)", font_size=28, color=GREY_A).next_to(slider, LEFT, buff=0.3)
        knob = always_redraw(lambda: Dot([sl0 + (zl.get_value() - lo) / (hi - lo) * (sl1 - sl0), sly, 0], radius=0.09, color=YELLOW))
        self.at(S("p6") + 0.3)
        self.play(GrowFromCenter(br3), FadeIn(t3), run_time=0.8)
        self.add(chirp_m, pat_m)
        self.at(S("p6") + 2.7)
        self.play(pan_op.animate.set_value(1.0), FadeIn(frames), FadeIn(ttl_l), FadeIn(ttl_r), FadeIn(edges), Create(spec_c),
                  FadeIn(slider), FadeIn(ticks), FadeIn(sl_l), FadeIn(knob), run_time=0.9)
        self.at(S("p6") + 6.2)
        self.play(zl.animate.set_value(hi), run_time=3.6, rate_func=rate_functions.ease_in_out_sine)

        # ---------------- p7: closer in, the chirp scrambles it.  Could we cancel it?
        q = MathTex("?", font_size=60, color=YELLOW).next_to(br3, RIGHT, buff=0.35)
        self.at(S("p7"))
        self.play(zl.animate.set_value(np.log10(0.12)), run_time=1.4, rate_func=rate_functions.ease_in_out_sine)
        self.at(S("p7") + 1.9)
        self.play(Circumscribe(full[6], color=C_CHIRP, buff=0.08), FadeIn(q, scale=1.4), run_time=1.0)
        self.finish()


# ============================================================ 3  lens ==============
class Ch3Lens(Chapter):
    CH = 3
    TAG = "3\\quad A lens sorts directions"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- l1: a lens = glass thicker in the middle = an opposite chirp
        big = lens_shape([-4.3, 0.0, 0], 4.2, 1.15, GREY_A, 0.25, 3)
        thick = Tex("thicker in the middle", font_size=30, color=GREY_A).next_to(big, UP, buff=0.2)
        yy = np.linspace(-2.0, 2.0, 120)
        dl_c = poly_line(-2.55 + 1.5 * (1 - (yy / 2.0) ** 2), yy, C_LENS, 4)
        dl_ax = Line([-2.55, -2.1, 0], [-2.55, 2.1, 0]).set_stroke(GREY_C, 1.5)
        dl_l = Tex("delay", font_size=28, color=C_LENS).move_to([-1.55, 1.65, 0])
        ax_x = MathTex("x", font_size=30, color=GREY_B).next_to(dl_ax, UP, buff=0.08)
        lens_eq = VGroup(Tex("lens:", font_size=32, color=GREY_A), MathTex("\\times\; e^{-ikx^2/2f}", font_size=44, color=C_LENS)
                         ).arrange(RIGHT, buff=0.3).move_to([3.3, 0.85, 0])
        chirp_eq = VGroup(Tex("slide's chirp:", font_size=32, color=GREY_A), MathTex("e^{+ikx'^2/2z}", font_size=44, color=C_CHIRP)
                          ).arrange(RIGHT, buff=0.3)
        chirp_eq.next_to(lens_eq, DOWN, buff=0.55).align_to(lens_eq, RIGHT)
        opp = Tex("opposite sign: an opposite chirp", font_size=30, color=YELLOW).next_to(chirp_eq, DOWN, buff=0.45).align_to(lens_eq, RIGHT)
        self.at(S("l1"))
        self.play(DrawBorderThenFill(big), run_time=1.0)
        self.play(FadeIn(thick), run_time=0.5)
        self.at(S("l1") + 2.3)
        self.play(Create(dl_ax), FadeIn(ax_x), Create(dl_c), FadeIn(dl_l), run_time=1.1)
        self.at(S("l1") + 4.6)
        self.play(Write(lens_eq), run_time=1.4)
        self.at(S("l1") + 8.7)
        self.play(FadeIn(chirp_eq), run_time=0.5)
        self.play(FadeIn(opp), run_time=0.5)

        # ---------------- the computed x-z panel (z right, x up), exact ASM behind a thin lens
        ld = load("lens.npz")
        zs, xs, f = ld["zs"], ld["xs"], float(ld["f"])
        cols = zs <= 46.0
        flds = [F[:, cols] for F in ld["sweep"]] + [ld["minus"][:, cols], ld["slit"][:, cols]]
        i_minus, i_slit = len(flds) - 2, len(flds) - 1
        m0 = np.abs(flds[0]).max()
        ms = max(np.abs(flds[i_slit]).max(), 1.5)
        ENV = [((np.abs(F) / (ms if i == i_slit else m0)) ** 0.55).astype(np.float32) for i, F in enumerate(flds)]
        PH = [(F / np.maximum(np.abs(F), 1e-12)).astype(np.complex64) for F in flds]
        jf = int(np.argmin(np.abs(zs[cols] - f)))
        FOC = [np.abs(F[:, jf]) / (ms if i == i_slit else m0) for i, F in enumerate(flds)]
        nsw = len(ld["sweep_deg"])
        sc = 0.16                                                # scene units per wavelength
        pw, ph = 56 * sc, 30 * sc
        pxl = -5.0
        pc = np.array([pxl + pw / 2, 0.15, 0])
        x_lens, x_foc = pxl + 10 * sc, pxl + 50 * sc
        tilt, opA, opB, opC, opS, pan = (ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0),
                                         ValueTracker(0.0), ValueTracker(0.0))
        OM = TAU_ * 1.4

        def idx_tilt():
            return int(round(np.clip(tilt.get_value(), 0, 7) / 7.0 * (nsw - 1)))

        def tint_tilt():
            return tint_of(np.sin(np.radians(tilt.get_value())) / np.sin(np.radians(7.0)))

        def panel_img(t):
            w = np.exp(-1j * OM * t)
            rgb = np.zeros(ENV[0].shape + (3,), np.float32)
            act = [(idx_tilt(), opA.get_value(), tint_tilt()), (0, opB.get_value(), T_ZERO), (i_minus, opC.get_value(), T_NEG),
                   (i_slit, opS.get_value(), T_ZERO)]
            nact = sum(o for _, o, _ in act)
            gain = 1.0 / max(1.0, 0.55 + 0.45 * nact)
            for i, o, tn in act:
                if o < 1e-3:
                    continue
                crest = 0.5 + 0.5 * (PH[i] * w).real
                rgb += (ENV[i] * (0.28 + 0.72 * crest) * o)[..., None] * tn[None, None, :]
            out = np.empty(ENV[0].shape + (4,), np.uint8)
            out[..., :3] = (np.clip(rgb * gain, 0, 1) * 255).astype(np.uint8)
            out[..., 3] = 255
            return out
        pim, pfade = self.live_image(panel_img, pw, pc, ph)
        pbox = frame_box(pc, pw, ph)
        small = lens_shape([x_lens, pc[1], 0], 24 * sc, 0.42, GREY_A, 0.12, 2.5)
        stops = VGroup(Line([x_lens, pc[1] + 12 * sc, 0], [x_lens, pc[1] + ph / 2, 0]),
                       Line([x_lens, pc[1] - 12 * sc, 0], [x_lens, pc[1] - ph / 2, 0])).set_stroke(GREY_B, 6)
        foc_ln = DashedLine([x_foc, pc[1] - ph / 2, 0], [x_foc, pc[1] + ph / 2, 0], dash_length=0.08).set_stroke(GREY_A, 1.5)
        fb = BraceBetweenPoints([x_lens, pc[1] - ph / 2 - 0.05, 0], [x_foc, pc[1] - ph / 2 - 0.05, 0], DOWN, color=GREY_B)
        fb_l = MathTex("f", font_size=32, color=GREY_A).next_to(fb, DOWN, buff=0.06)
        # incoming directions (outside the panel, left)
        def arrows_in():
            g = VGroup()
            for (o, ang, tn, y) in ((opA.get_value(), tilt.get_value(), tint_tilt(), pc[1] + 1.25),
                                    (opB.get_value(), 0.0, T_ZERO, pc[1]),
                                    (opC.get_value(), -7.0, T_NEG, pc[1] - 1.25)):
                if o < 1e-3:
                    continue
                th = np.radians(ang) * 3.0                      # drawn 3x steeper for legibility (labelled)
                d = np.array([np.cos(th), np.sin(th), 0])
                a = np.array([-6.25, y, 0]) - 0.55 * d
                g.add(fast_arrow(a, a + 1.15 * d, rgb_to_color(tn), 5, 0.2, o))
            return g
        arr_m = always_redraw(arrows_in)
        # focal-plane read-out (right of the panel, same x scale)
        rx0, rdw = pxl + pw + 0.25, 1.75
        r_ax = Line([rx0, pc[1] - ph / 2, 0], [rx0, pc[1] + ph / 2, 0]).set_stroke(GREY_C, 1.5)
        r_l = Tex("focal plane", font_size=26, color=GREY_A).next_to(r_ax, DOWN, buff=0.12).shift(0.6 * RIGHT)
        ys_disp = pc[1] + xs * sc

        def readout():
            g = VGroup()
            for i, o, tn in ((idx_tilt(), opA.get_value(), tint_tilt()), (0, opB.get_value(), T_ZERO), (i_minus, opC.get_value(), T_NEG)):
                if o < 1e-3:
                    continue
                g.add(poly_line(rx0 + rdw * FOC[i] ** 2, ys_disp, rgb_to_color(tn), 3.5, o))
            return g
        ro_m = always_redraw(readout)

        # ---------------- l2: a flat wave passes the lens and converges to a point at f
        self.at(S("l2") - 0.1)
        self.play(FadeOut(VGroup(thick, dl_c, dl_ax, dl_l, ax_x, lens_eq, chirp_eq, opp)), ReplacementTransform(big, small), run_time=0.7)
        self.add(pim, stops, small, arr_m, ro_m)
        comp3 = Tex("computed", font_size=22, color=GREY_B)
        comp3 = VGroup(BackgroundRectangle(comp3, fill_opacity=0.7, buff=0.05), comp3).move_to([x_lens + 0.78, pc[1] + ph / 2 - 0.2, 0])
        self.play(pfade.animate.set_value(1.0), FadeIn(pbox), FadeIn(stops), opA.animate.set_value(1.0), FadeIn(comp3), run_time=0.7)
        focus_ring = Circle(radius=0.22).move_to([x_foc, pc[1], 0]).set_stroke(WHITE, 2.5)
        self.at(S("l2") + 3.2)
        self.play(Create(focus_ring), run_time=0.6)
        self.at(S("l2") + 4.5)
        self.play(FadeOut(focus_ring), Create(foc_ln), GrowFromCenter(fb), FadeIn(fb_l), FadeIn(r_ax), FadeIn(r_l), run_time=0.9)

        # ---------------- l3: tilt the wave: the focus slides to x = f tan(theta)
        x_t = f * np.tan(np.radians(7.0))
        land = MathTex("x", "=", "f", "\\tan\\theta", font_size=34, color=C_POS)
        land = VGroup(BackgroundRectangle(land, fill_opacity=0.8, buff=0.08), land)
        land.next_to([x_foc - 0.1, pc[1] + x_t * sc, 0], LEFT, buff=0.15).shift(0.55 * UP)
        steep = Tex("(arrow tilts $\\times 3$)", font_size=22, color=GREY_B).move_to([-5.85, pc[1] - 1.95, 0])
        self.at(S("l3") + 0.1)
        self.play(tilt.animate.set_value(7.0), FadeIn(steep), run_time=2.3, rate_func=rate_functions.ease_in_out_sine)
        self.at(S("l3") + 2.9)
        self.play(Write(land), run_time=0.8)

        # ---------------- (pause) + l4: three directions, three spots: a lens sorts light by direction
        sorts = Tex("a lens sorts light by direction", font_size=40).move_to([0.9, 3.12, 0])
        self.at(S("l3") + E_("l3") - S("l3") + 0.1)
        self.play(opB.animate.set_value(1.0), opC.animate.set_value(1.0), FadeOut(land), run_time=1.1)
        self.at(S("l4") + 2.4)
        self.play(Write(sorts), run_time=1.1)

        # ---------------- l5: a slit sends out a spectrum of directions -> its spectrum in the focal plane
        sl_gap = 3.0 * sc                                        # slit half-width 3 lambda, just before the lens
        slit = VGroup(Line([x_lens - 0.32, pc[1] + sl_gap, 0], [x_lens - 0.32, pc[1] + ph / 2, 0]),
                      Line([x_lens - 0.32, pc[1] - sl_gap, 0], [x_lens - 0.32, pc[1] - ph / 2, 0])).set_stroke(WHITE, 4)
        xf_, prof_, spec_ = ld["xf"], ld["prof"], ld["spec"]
        prof_c = poly_line(rx0 + rdw * prof_, pc[1] + xf_ * sc, C_U, 6)
        spec_c = DashedVMobject(poly_line(rx0 + rdw * spec_, pc[1] + xf_ * sc, C_KX, 3), num_dashes=60, dashed_ratio=0.55)
        sp_l = Tex("spectrum", font_size=26, color=C_KX).next_to(r_l, DOWN, buff=0.08)
        self.at(S("l5"))
        self.play(opA.animate.set_value(0.0), opB.animate.set_value(0.0), opC.animate.set_value(0.0), opS.animate.set_value(1.0),
                  FadeIn(slit), FadeOut(steep), run_time=0.9)
        self.at(S("l5") + 2.7)
        self.play(Create(prof_c), r_l.animate.set_color(C_U), run_time=0.9)
        self.play(Create(spec_c), FadeIn(sp_l), run_time=0.9)

        # ---------------- l6: the lens chirp cancels the slide chirp at z = f
        canc = MathTex("e^{+ikx'^2/2z}", "\\cdot", "e^{-ikx'^2/2f}", "=", "1", font_size=42)
        canc[0].set_color(C_CHIRP)
        canc[2].set_color(C_LENS)
        when = MathTex("\\text{when } z = f", font_size=36, color=YELLOW)
        cg = VGroup(canc, when).arrange(RIGHT, buff=0.6).move_to([0.9, 3.1, 0])
        self.at(S("l6") - 0.1)
        self.play(FadeOut(sorts), run_time=0.4)
        self.play(Write(canc), run_time=1.6)
        self.at(S("l6") + 3.5)
        self.play(FadeIn(when, shift=0.1 * LEFT), run_time=0.6)

        # ---------------- l7: slide one focal length in front -> exact Fourier transform (2f)
        for m in (arr_m, ro_m):
            m.clear_updaters()
        yA = 0.9
        xA, xB, xC = -4.0, 0.0, 4.0
        axis = Line([-5.6, yA, 0], [5.6, yA, 0]).set_stroke(GREY_C, 1.5)
        L1 = lens_shape([xB, yA, 0], 2.4, 0.42, GREY_A, 0.2, 2.5)
        fplane = DashedLine([xC, yA - 1.3, 0], [xC, yA + 1.3, 0], dash_length=0.08).set_stroke(GREY_A, 2)
        slide2 = Rectangle(width=0.08, height=1.5).set_fill(GREY_C, 0.9).set_stroke(GREY_A, 1).move_to([xB - 0.4, yA, 0])
        sl_lab = Tex("slide", font_size=28, color=GREY_B).next_to(slide2, UP, buff=0.12)
        fp_lab = Tex("focal plane", font_size=28, color=GREY_B).next_to(fplane, UP, buff=0.12)
        b1 = BraceBetweenPoints([xA, yA - 1.4, 0], [xB, yA - 1.4, 0], DOWN, color=GREY_B)
        b2 = BraceBetweenPoints([xB, yA - 1.4, 0], [xC, yA - 1.4, 0], DOWN, color=GREY_B)
        b1l = MathTex("f", font_size=32, color=GREY_A).next_to(b1, DOWN, buff=0.06)
        b2l = MathTex("f", font_size=32, color=GREY_A).next_to(b2, DOWN, buff=0.06)
        res = MathTex("U_f(x)", "\;\\propto\;", "e^{\\,ikx^2/2f}", "\\,", "\\hat U\\!\\left(k x / f\\right)", font_size=44).move_to([0, -2.1, 0])
        res[0].set_color(C_U)
        res[2].set_color(GREY_A)
        res[4].set_color(C_KX)
        self.at(S("l7"))
        self.play(FadeOut(Group(pim, pbox, small, stops, slit, foc_ln, fb, fb_l, r_ax, r_l, sp_l, prof_c, spec_c, cg, comp3)), FadeOut(arr_m),
                  FadeOut(ro_m), pfade.animate.set_value(0.0), run_time=0.5)
        self.remove(pim, arr_m, ro_m)
        self.play(Create(axis), FadeIn(L1), Create(fplane), FadeIn(slide2), FadeIn(sl_lab), FadeIn(fp_lab), GrowFromCenter(b2), FadeIn(b2l),
                  run_time=0.7)
        self.play(Write(res), run_time=0.9)
        self.play(slide2.animate.move_to([xA, yA, 0]), sl_lab.animate.next_to([xA, yA + 0.75, 0], UP, buff=0.12), GrowFromCenter(b1),
                  FadeIn(b1l), run_time=1.2)
        strike = Line(res[2].get_corner(DL), res[2].get_corner(UR)).set_stroke(C_EV, 4)
        self.at(S("l7") + 3.5)
        self.play(res[2].animate.set_opacity(0.25), Create(strike), run_time=0.7)
        exact = Tex("exact Fourier transform", font_size=32, color=YELLOW)
        bx = SurroundingRectangle(VGroup(res[0], res[4]), buff=0.15, color=YELLOW)
        exact.next_to(bx, DOWN, buff=0.15)
        self.at(S("l7") + 5.2)
        self.play(Create(bx), FadeIn(exact), run_time=0.8)
        self.finish()


# ============================================================ 4  4f machine ========
def four_f_schematic(ys=2.5, xs=(-5.0, -2.5, 0.0, 2.5, 5.0), lh=1.2):
    """Object plane, lens 1, Fourier plane, lens 2, image plane (side view) + the four f braces."""
    axis = Line([xs[0] - 0.9, ys, 0], [xs[4] + 0.9, ys, 0]).set_stroke(GREY_C, 1.5)
    planes = VGroup(*[DashedLine([x, ys - 0.55, 0], [x, ys + 0.55, 0], dash_length=0.07).set_stroke(GREY_A, 2) for x in (xs[0], xs[2], xs[4])])
    lenses = VGroup(lens_shape([xs[1], ys, 0], lh, 0.32, GREY_A, 0.2, 2.5), lens_shape([xs[3], ys, 0], lh, 0.32, GREY_A, 0.2, 2.5))
    braces = VGroup(*[BraceBetweenPoints([a, ys - 0.8, 0], [b, ys - 0.8, 0], DOWN, color=GREY_B, buff=0.02) for a, b in zip(xs[:-1], xs[1:])])
    flabs = VGroup(*[MathTex("f", font_size=30, color=GREY_A).next_to(br, DOWN, buff=0.04) for br in braces])
    return axis, planes, lenses, braces, flabs


class Ch4FourF(Chapter):
    CH = 4
    TAG = "4\\quad The 4f machine"

    def construct(self):
        S = self.S
        self.show_tag()
        d4 = load("fourf.npz")
        ys, X = 2.5, (-5.0, -2.5, 0.0, 2.5, 5.0)
        axis, planes, lenses, braces, flabs = four_f_schematic(ys, X)
        pw, py = 2.5, -0.55
        pcs = [np.array([x, py, 0]) for x in (X[0], X[2], X[4])]

        def norm_heat(A, g=0.8):
            return heat_rgba(A / A.max(), 1.0, g)
        obj_m, obj_f = self.static_image(heat_rgba(d4["obj"], 1.0, 1.0), pw, pcs[0])
        spec_imgs = {k: log_heat(d4["spec_" + k] / d4["spec_open"].max(), floor=1e-5, top=0.85, gamma=1.3) for k in ("open", "pinhole", "stop")}
        img_imgs = {k: norm_heat(d4["img_" + k], 0.8) for k in ("open", "pinhole", "stop")}
        sp = {k: self.static_image(v, pw, pcs[1]) for k, v in spec_imgs.items()}
        im = {k: self.static_image(v, pw, pcs[2]) for k, v in img_imgs.items()}
        boxes = VGroup(*[frame_box(c, pw, pw) for c in pcs])
        labs = [Tex("object", font_size=28, color=GREY_A), Tex("spectrum", font_size=28, color=GREY_A), Tex("image", font_size=28, color=GREY_A)]
        for l_, c in zip(labs, pcs):
            l_.next_to([c[0], py - pw / 2, 0], DOWN, buff=0.14)
        # ---------------- (recap of the 2f half) and m1: a second lens transforms back
        self.add(obj_m, sp["open"][0])
        self.play(Create(axis), FadeIn(planes[0]), FadeIn(lenses[0]), FadeIn(planes[1]), GrowFromCenter(braces[0]), GrowFromCenter(braces[1]),
                  FadeIn(flabs[:2]), obj_f.animate.set_value(1.0), sp["open"][1].animate.set_value(1.0), FadeIn(boxes[:2]),
                  FadeIn(labs[0]), FadeIn(labs[1]), run_time=0.6)
        self.at(S("m1") + 0.75)
        self.play(FadeIn(lenses[1], shift=0.3 * LEFT), run_time=0.6)
        self.at(S("m1") + 1.6)
        self.play(FadeIn(planes[2]), GrowFromCenter(braces[2]), GrowFromCenter(braces[3]), FadeIn(flabs[2:]), run_time=0.7)
        self.add(im["open"][0])
        self.at(S("m1") + 5.0)
        self.play(im["open"][1].animate.set_value(1.0), FadeIn(boxes[2]), FadeIn(labs[2]), run_time=0.9)

        # ---------------- m2: upside down: two Fourier transforms mirror x -> -x
        h = 0.42
        ray_pts = [[X[0], ys + h, 0], [X[1], ys + h, 0], [X[3], ys - h, 0], [X[4], ys - h, 0]]
        ray = VMobject().set_points_as_corners(ray_pts).set_stroke(C_REF, 3)
        tips = VGroup(Dot(ray_pts[0], radius=0.07, color=C_REF), Dot(ray_pts[-1], radius=0.07, color=C_REF))
        flip_eq = MathTex("\\mathcal{F}\\{\\mathcal{F}\\{U\\}\\}(x)", "\;\\propto\;", "U(-x)", font_size=38).move_to([0, -2.72, 0])
        flip_eq[2].set_color(C_U)
        new_l = Tex("image $= U(-x)$", font_size=28, color=GREY_A).move_to(labs[2])
        self.at(S("m2"))
        self.play(Indicate(boxes[2], color=YELLOW, scale_factor=1.04), run_time=0.8)
        self.play(Create(ray), FadeIn(tips), run_time=1.1)
        self.at(S("m2") + 2.0)
        self.play(Write(flip_eq), run_time=1.2)
        self.at(S("m2") + 3.9)
        self.play(FadeTransform(labs[2], new_l), run_time=0.7)

        # ---------------- m3: the 4f system; the middle plane holds the spectrum -> a mask edits it
        name = Tex("the 4f system", font_size=32, color=YELLOW).move_to([0, 0.95, 0])
        mask_box = SurroundingRectangle(VGroup(boxes[1]), buff=0.08, color=C_H, stroke_width=3)
        mask_l = Tex("mask here", font_size=28, color=C_H).move_to(labs[1])
        self.at(S("m3"))
        self.play(FadeOut(ray), FadeOut(tips), FadeOut(flip_eq), *[Indicate(fl, color=YELLOW, scale_factor=1.5) for fl in flabs], run_time=0.9)
        self.play(FadeIn(name, shift=0.1 * UP), run_time=0.5)
        self.at(S("m3") + 2.1)
        self.play(Create(mask_box), planes[1].animate.set_stroke(C_H, 3), run_time=0.7)
        self.at(S("m3") + 5.3)
        self.play(FadeTransform(labs[1], mask_l), run_time=0.6)

        # ---------------- m4: a pinhole keeps only low frequencies -> blur
        r_pin, r_stop, half = float(d4["r_pin"]), float(d4["r_stop"]), float(d4["spec_half"])
        pin_c = Circle(radius=r_pin / half * pw / 2).move_to(pcs[1]).set_stroke(C_H, 3)
        pin_screen = VGroup(Line([X[2], ys + 0.08, 0], [X[2], ys + 0.62, 0]), Line([X[2], ys - 0.62, 0], [X[2], ys - 0.08, 0])).set_stroke(C_H, 6)
        pin_l = Tex("pinhole: low frequencies only", font_size=28, color=C_H).move_to(labs[1])
        self.at(S("m4"))
        self.add(sp["pinhole"][0])
        self.play(sp["pinhole"][1].animate.set_value(1.0), sp["open"][1].animate.set_value(0.0), Create(pin_c), FadeIn(pin_screen),
                  FadeOut(mask_box), FadeTransform(mask_l, pin_l), run_time=0.9)
        self.add(im["pinhole"][0])
        self.at(S("m4") + 2.4)
        self.play(im["pinhole"][1].animate.set_value(1.0), im["open"][1].animate.set_value(0.0), run_time=0.9)

        # ---------------- m5: a central stop keeps only high frequencies -> edges
        stop_c = Circle(radius=r_stop / half * pw / 2).move_to(pcs[1]).set_stroke(C_H, 3)
        stop_screen = Line([X[2], ys - 0.17, 0], [X[2], ys + 0.17, 0]).set_stroke(C_H, 8)
        stop_l = Tex("central stop: high frequencies only", font_size=28, color=C_H).move_to(labs[1])
        self.at(S("m5"))
        self.add(sp["stop"][0])
        self.play(sp["stop"][1].animate.set_value(1.0), sp["pinhole"][1].animate.set_value(0.0), ReplacementTransform(pin_c, stop_c),
                  ReplacementTransform(pin_screen, stop_screen), FadeTransform(pin_l, stop_l), run_time=0.9)
        self.add(im["stop"][0])
        self.at(S("m5") + 3.6)
        self.play(im["stop"][1].animate.set_value(1.0), im["pinhole"][1].animate.set_value(0.0), run_time=0.9)
        self.finish()


# ============================================================ 5  phase contrast ====
class Ch5PhaseContrast(Chapter):
    CH = 5
    TAG = "5\\quad Phase contrast"

    def construct(self):
        S = self.S
        self.show_tag()
        zd = load("zernike.npz")
        phi, bright, pc_img, spec = zd["phi"], zd["bright"], zd["pc"], zd["spec"]
        N = phi.shape[0]
        c0, hw = N // 2, 200
        crop = (slice(c0 - hw, c0 + hw), slice(c0 - hw, c0 + hw))
        pw, py = 2.5, -0.8
        X = (-5.0, 0.0, 5.0)
        pcs = [np.array([x, py, 0]) for x in X]
        ph_rgb = phase_rgba(np.exp(1j * (phi[crop] + 0.62 * TAU_)), 1.0, 1.0)
        ph_rgb[..., :3] = (ph_rgb[..., :3].astype(np.float32) * 0.85).astype(np.uint8)
        cam_v = 2.4
        bf_rgb = heat_rgba(bright[crop], cam_v, 1.0)
        pc_rgb = heat_rgba(pc_img[crop], cam_v, 1.0)
        sh = 48
        sp_crop = spec[N // 2 - sh:N // 2 + sh, N // 2 - sh:N // 2 + sh]
        sp_rgb = log_heat(sp_crop / sp_crop.max(), floor=1e-7, top=0.9, gamma=1.6)
        obj_m, obj_f = self.static_image(ph_rgb, pw, pcs[0])
        bf_m, bf_f = self.static_image(bf_rgb, pw, pcs[2])
        pc_m, pc_f = self.static_image(pc_rgb, pw, pcs[2])
        sp_m, sp_f = self.static_image(sp_rgb, pw, pcs[1], nearest=False)
        boxes = VGroup(*[frame_box(c, pw, pw) for c in pcs])
        labs = [Tex("cell: delay only", font_size=28, color=GREY_A), Tex("Fourier plane", font_size=28, color=GREY_A),
                Tex("camera", font_size=28, color=GREY_A)]
        for l_, c in zip(labs, pcs):
            l_.next_to([c[0], py - pw / 2, 0], DOWN, buff=0.14)
        # ---------------- z1: a living cell only delays the light
        self.at(S("z1") + 0.3)
        self.add(obj_m)
        self.play(obj_f.animate.set_value(1.0), FadeIn(boxes[0]), FadeIn(labs[0]), run_time=0.9)
        delay_l = MathTex("e^{\\,i\\varphi(x,y)}", ",\;", "\\varphi \\lesssim 0.6\\,\\mathrm{rad}", font_size=32).next_to(boxes[0], UP, buff=0.15)
        delay_l[0].set_color(C_U)
        self.at(S("z1") + 3.6)
        self.play(FadeIn(delay_l), run_time=0.6)

        # ---------------- z2: big arrow + tiny arrow at 90 deg: length barely changes; camera sees nothing
        O = np.array([-0.95, 1.05, 0])
        Ldc = 1.5
        rot = ValueTracker(0.0)                                    # DC phase shift (0 -> pi/2)
        ph_op = ValueTracker(0.0)
        row = phi[c0 + 5, :]                                       # scan line through the nucleus
        xs_scan = np.arange(N)
        T0 = S("z2") + 1.0

        def scan_u():                                              # scan position (pixel), a pure function of time
            return c0 - 5 + 150 * np.sin(0.9 * (self.now - T0))

        def phi_now():
            return float(np.interp(scan_u(), xs_scan, row))

        def phasors():
            o = ph_op.get_value()
            if o < 1e-3:
                return VGroup()
            a = rot.get_value()
            dc = Ldc * np.array([np.cos(a), np.sin(a), 0])
            tiny = np.array([0, Ldc * phi_now(), 0])
            tot = dc + tiny
            g = VGroup(Line(O + [-0.35, 0, 0], O + [2.0, 0, 0]).set_stroke(GREY_D, 1.5),
                       Line(O + [0, -0.3, 0], O + [0, 2.3, 0]).set_stroke(GREY_D, 1.5),
                       Arc(radius=Ldc, start_angle=-0.15, angle=PI / 2 + 0.45, arc_center=O).set_stroke(GREY_C, 1.5),
                       fast_arrow(O, O + tot, C_U, 6, 0.2),
                       fast_arrow(O, O + dc, WHITE, 5, 0.2),
                       fast_arrow(O + dc, O + dc + tiny, C_IM, 5, 0.16) if abs(tiny[1]) > 0.02 else VGroup())
            # brightness bar |U|^2
            I = np.dot(tot, tot) / Ldc ** 2
            bx = O[0] + 3.0
            g.add(Rectangle(width=0.32, height=0.75 * I).set_stroke(width=0).set_fill(YELLOW, 0.85).move_to([bx, O[1] + 0.375 * I, 0]),
                  Line([bx - 0.3, O[1] + 0.75, 0], [bx + 0.3, O[1] + 0.75, 0]).set_stroke(GREY_B, 1.5))
            return fade_vgroup(g, o)
        ph_m = always_redraw(phasors)
        ring = always_redraw(lambda: Circle(radius=0.1).move_to(
            pcs[0] + np.array([(scan_u() - c0) / hw * pw / 2, -(5 / hw) * pw / 2, 0])).set_fill(opacity=0).set_stroke(
            WHITE, 3, opacity=ph_op.get_value()))
        lab_dc = Tex("unscattered", font_size=26)
        lab_tiny = Tex("scattered", font_size=26, color=C_IM)
        lab_b = Tex("brightness", font_size=26, color=YELLOW).move_to([O[0] + 3.0, O[1] - 0.32, 0])

        def place_dc(m):                                           # below the arrow, or left of it once turned
            a = rot.get_value() / (PI / 2)
            m.move_to(O + (1 - a) * np.array([0.78, -0.3, 0]) + a * np.array([-0.95, 0.78, 0]))

        def place_tiny(m):
            a = rot.get_value()
            tip = O + Ldc * np.array([np.cos(a), np.sin(a), 0])
            m.move_to(tip + np.array([0.75, 0.32, 0]))
        lab_dc.add_updater(place_dc)
        lab_tiny.add_updater(place_tiny)
        self.at(S("z2"))
        self.add(ph_m, ring)
        self.play(ph_op.animate.set_value(1.0), run_time=0.8)
        self.play(FadeIn(lab_dc), FadeIn(lab_tiny), FadeIn(lab_b), run_time=0.6)
        self.add(bf_m)
        self.at(S("z2") + 6.3)
        self.play(bf_f.animate.set_value(1.0), FadeIn(boxes[2]), FadeIn(labs[2]), run_time=0.8)
        nothing = Tex("nothing", font_size=30, color=GREY_A).move_to(pcs[2])

        # ---------------- z3: the big arrow is unscattered light -> one central spot in the Fourier plane
        self.add(sp_m)
        dc_ring = Circle(radius=0.17).move_to(pcs[1]).set_stroke(WHITE, 2.5)
        fly = Dot(O + [Ldc / 2, 0, 0], radius=0.09, color=WHITE)
        self.at(S("z3") + 0.3)
        self.play(sp_f.animate.set_value(1.0), FadeIn(boxes[1]), FadeIn(labs[1]), FadeIn(nothing), run_time=0.8)
        self.at(S("z3") + 2.8)
        self.play(fly.animate.move_to(pcs[1]), run_time=1.0, rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeOut(fly), Create(dc_ring), run_time=0.6)

        # ---------------- z4: delay just that spot by a quarter wave (pause: the big arrow turns 90 deg)
        plate = Circle(radius=0.17).move_to(pcs[1]).set_stroke(C_H, 3).set_fill(C_H, 0.45)
        plate_l = Tex("quarter-wave plate", font_size=28, color=C_H).move_to(labs[1])
        self.at(S("z4"))
        self.play(ReplacementTransform(dc_ring, plate), FadeTransform(labs[1], plate_l), run_time=0.7)
        self.at(S("z4") + 2.0)
        self.play(rot.animate.set_value(PI / 2), run_time=1.3, rate_func=rate_functions.ease_in_out_sine)

        # ---------------- z5: the arrows line up and the cell appears
        self.add(pc_m)
        self.at(S("z5") + 1.0)
        self.play(pc_f.animate.set_value(1.0), bf_f.animate.set_value(0.0), FadeOut(nothing), run_time=0.9)

        # ---------------- z6: Zernike, Nobel Prize 1953
        zt = VGroup(Tex("Zernike phase contrast", font_size=36), Tex("Nobel Prize in Physics, 1953", font_size=28, color=GREY_A)
                    ).arrange(DOWN, buff=0.12).move_to([4.6, 2.55, 0])
        self.at(S("z6") + 0.3)
        self.play(FadeIn(zt, shift=0.1 * DOWN), run_time=0.8)
        self.finish()


# ============================================================ 6  on a computer =====
class Ch6Computer(Chapter):
    CH = 6
    TAG = "6\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- c1: three propagators (rows)
        ys_rows = [2.45, 1.05, -0.35]
        names = [Tex("angular spectrum", font_size=32), Tex("single-FFT Fresnel", font_size=32), Tex("Fraunhofer", font_size=32)]
        cards = VGroup()
        for nm, y in zip(names, ys_rows):
            r = RoundedRectangle(corner_radius=0.12, width=13.3, height=1.15).set_stroke(GREY_B, 2).set_fill(GREY_E, 0.45).move_to([0, y, 0])
            nm.move_to([-4.7, y, 0]).align_to([-6.2, 0, 0], LEFT)
            cards.add(VGroup(r, nm))
        rec1 = MathTex("\\text{FFT}", "\\to", "\\times\\, e^{\\,iz\\sqrt{k^2-k_x^2}}", "\\to", "\\text{FFT}^{-1}", font_size=34).move_to([0.1, ys_rows[0], 0])
        rec1[2].set_color(C_H)
        rec2 = MathTex("\\times\\, e^{\\,ikx^2/2z}", "\\to", "\\text{FFT}", font_size=34).move_to([0.1, ys_rows[1], 0])
        rec2[0].set_color(C_CHIRP)
        rec3 = MathTex("\\text{FFT}", font_size=34).move_to([0.1, ys_rows[2], 0])
        note1 = VGroup(MathTex("\\Delta x' = \\Delta x", font_size=30), Tex("exact; best up close", font_size=26, color=C_H)).arrange(DOWN, buff=0.08)
        note1.move_to([4.95, ys_rows[0], 0])
        note2a = MathTex("\\Delta x' = {\\lambda z \\over N\\Delta x}", font_size=30)
        note2b = Tex("best far away", font_size=26, color=C_H)
        note2 = VGroup(note2a, note2b).arrange(DOWN, buff=0.06).move_to([4.95, ys_rows[1], 0])
        note3 = VGroup(MathTex("z \\gg D^2/\\lambda", font_size=30), Tex("very far", font_size=26, color=C_H)).arrange(DOWN, buff=0.08)
        note3.move_to([4.95, ys_rows[2], 0])
        ep1 = Tex("(episode 1)", font_size=24, color=GREY_B).next_to(names[0], DOWN, buff=0.06)
        self.at(S("c1") + 0.1)
        self.play(LaggedStart(*[FadeIn(c, shift=0.1 * DOWN) for c in cards], lag_ratio=0.3), run_time=1.3)

        def hi(i):
            return [cards[j][0].animate.set_stroke(YELLOW if j == i else GREY_B, 3 if j == i else 2) for j in range(3)]
        # ---------------- c2: ASM
        self.at(S("c2"))
        self.play(*hi(0), FadeIn(ep1), run_time=0.5)
        self.at(S("c2") + 2.9)
        self.play(Write(rec1), run_time=1.4)
        self.at(S("c2") + 6.4)
        self.play(FadeIn(note1[0]), run_time=0.5)
        self.at(S("c2") + 8.5)
        self.play(FadeIn(note1[1]), run_time=0.5)

        # ---------------- c3: single-FFT Fresnel: output grid stretches with distance
        zg = ValueTracker(0.45)                                 # z / z_c for the grid demo
        gy_in, gy_out, gx0 = -1.75, -2.55, 0.1
        n_g, dx_g = 9, 0.5
        grid_op = ValueTracker(0.0)

        def grids():
            o = grid_op.get_value()
            if o < 1e-3:
                return VGroup()
            k = np.arange(n_g) - n_g // 2
            g = VGroup(dots_vm([[gx0 + i * dx_g, gy_in, 0] for i in k], 0.06, GREY_A, opacity=o),
                       dots_vm([[gx0 + i * dx_g * zg.get_value(), gy_out, 0] for i in k], 0.07, C_REF, opacity=o))
            return g
        grid_m = always_redraw(grids)
        g_l = VGroup(Tex("input grid", font_size=26, color=GREY_A).move_to([-4.6, gy_in, 0]),
                     Tex("output grid", font_size=26, color=C_REF).move_to([-4.6, gy_out, 0]))
        self.at(S("c3"))
        self.play(*hi(1), run_time=0.5)
        self.at(S("c3") + 2.2)
        self.play(Write(rec2), run_time=1.1)
        self.add(grid_m)
        self.at(S("c3") + 5.3)
        self.play(grid_op.animate.set_value(1.0), FadeIn(g_l), run_time=0.5)
        self.play(zg.animate.set_value(1.6), run_time=2.2, rate_func=rate_functions.ease_in_out_sine)
        self.at(S("c3") + 8.2)
        self.play(Write(note2a), run_time=1.0)
        self.at(S("c3") + 11.2)
        self.play(FadeIn(note2b), run_time=0.5)

        # ---------------- c4: Fraunhofer: drop the chirp, one FFT
        self.at(S("c4"))
        self.play(*hi(2), run_time=0.5)
        self.play(Write(rec3), run_time=0.6)
        self.at(S("c4") + 2.0)
        self.play(FadeIn(note3), run_time=0.6)

        # ---------------- c5: chirps alias (computed sampling, N = 32)
        cd = load("chirp.npz")
        zs_c, zc = cd["zs"], float(cd["zc"])
        t_f, m_s = cd["t"], cd["m"]
        Nn, dxs = len(m_s), float(cd["dx"])
        up = len(t_f) // Nn
        lzs = np.log10(zs_c / zc)
        kx_s, ka_f = TAU_ * m_s / (Nn * dxs), TAU_ * t_f / (Nn * dxs)
        x_s, x_f = m_s * dxs, t_f * dxs
        kz_s, kz_f = np.sqrt(TAU_ ** 2 - kx_s ** 2), np.sqrt(np.maximum(TAU_ ** 2 - ka_f ** 2, 0))

        def recon(samples):                                      # band-limited (trigonometric) interpolation
            c = np.fft.fft(np.fft.ifftshift(samples))
            F = np.zeros(Nn * up, complex)
            F[:Nn // 2] = c[:Nn // 2]
            F[-Nn // 2:] = c[-Nn // 2:]
            return np.fft.fftshift(np.fft.ifft(F)).real * up
        memo = {}

        def curves(z):
            """Same formulas as physics.py part_chirp, evaluated at the continuous z of this frame (no table steps).
            The constant phase e^{ikz} is dropped from the ASM transfer function (it does not change the sampling)."""
            if z not in memo:
                memo.clear()
                Hs = np.exp(1j * z * (kz_s - TAU_))
                hs = np.exp(1j * TAU_ * x_s ** 2 / (2 * z))
                memo[z] = ((np.cos(z * (kz_f - TAU_)), Hs.real, recon(Hs), np.abs(np.diff(z * kz_s)).max() <= np.pi),
                           (np.cos(TAU_ * x_f ** 2 / (2 * z)), hs.real, recon(hs), np.abs(np.diff(TAU_ * x_s ** 2 / (2 * z))).max() <= np.pi))
            return memo[z]
        zt = ValueTracker(np.log10(0.42))                        # log10(z / z_c)

        def zcur():
            return zc * 10 ** zt.get_value()
        cw, chh = 6.1, 2.0
        cL, cR = np.array([-3.45, 0.55, 0]), np.array([3.45, 0.55, 0])
        fr = VGroup(frame_box(cL, cw, chh + 0.3, GREY_D), frame_box(cR, cw, chh + 0.3, GREY_D))
        tL = MathTex("\\text{angular spectrum: }", "e^{\\,iz\\sqrt{k^2-k_x^2}}", "\\text{ on the } k_x \\text{ grid}", font_size=30)
        tL[1].set_color(C_H)
        tR = MathTex("\\text{Fresnel: }", "e^{\\,ikx^2/2z}", "\\text{ on the } x \\text{ grid}", font_size=30)
        tR[1].set_color(C_CHIRP)
        tL.next_to(fr[0], UP, buff=0.12)
        tR.next_to(fr[1], UP, buff=0.12)
        ok_l = [Tex("sampled", font_size=28, color=C_H), Tex("aliased", font_size=28, color=C_EV)]
        ok_r = [Tex("sampled", font_size=28, color=C_H), Tex("aliased", font_size=28, color=C_EV)]
        for grp, cen in ((ok_l, cL), (ok_r, cR)):
            for m in grp:
                m.next_to([cen[0], cen[1] - (chh + 0.3) / 2, 0], DOWN, buff=0.12)
        ch_op = ValueTracker(0.0)

        def stat(m, which, want_ok):
            ok = bool(curves(zcur())[which][3])
            m.set_opacity(ch_op.get_value() if ok == want_ok else 0.0)
        for grp, which in ((ok_l, 0), (ok_r, 1)):
            grp[0].add_updater(lambda m, which=which: stat(m, which, True))
            grp[1].add_updater(lambda m, which=which: stat(m, which, False))

        def panel(cen, which, col):
            cont, samp, rec, _ = curves(zcur())[which]
            o = ch_op.get_value()
            xs_f = cen[0] - cw / 2 + (t_f + Nn / 2) / Nn * cw
            xs_s = cen[0] - cw / 2 + (m_s + Nn / 2) / Nn * cw
            sy = chh / 2 * 0.92
            return VGroup(poly_line(xs_f, cen[1] + sy * rec, C_KX, 3.5, 0.85 * o),
                          poly_line(xs_f, cen[1] + sy * cont, col, 1.8, o),
                          dots_vm([[x, cen[1] + sy * v, 0] for x, v in zip(xs_s, samp)], 0.06, YELLOW, opacity=o))
        pL = always_redraw(lambda: panel(cL, 0, C_H))
        pR = always_redraw(lambda: panel(cR, 1, C_CHIRP))
        leg = VGroup(Tex("samples", font_size=26, color=YELLOW), Tex("and the wave they trace", font_size=26, color=C_KX)
                     ).arrange(RIGHT, buff=0.2).move_to([0, 2.95, 0])
        # z slider (log) with z_c
        s0, s1, sy_ = -4.6, 4.6, -1.75
        lo, hi_ = float(lzs[0]), float(lzs[-1])
        sl = Line([s0, sy_, 0], [s1, sy_, 0]).set_stroke(GREY_B, 2)
        x_c = s0 + (0 - lo) / (hi_ - lo) * (s1 - s0)
        tick = Line([x_c, sy_ - 0.12, 0], [x_c, sy_ + 0.12, 0]).set_stroke(WHITE, 3)
        sl_l = MathTex("z", font_size=30, color=GREY_A).next_to(sl, RIGHT, buff=0.15)
        knob = always_redraw(lambda: Dot([s0 + (zt.get_value() - lo) / (hi_ - lo) * (s1 - s0), sy_, 0], radius=0.1, color=YELLOW
                                         ).set_opacity(ch_op.get_value()))
        zc_eq = MathTex("z_c", "=", "{N\\Delta x^2", "\\over", "\\lambda}", font_size=36).next_to(tick, DOWN, buff=0.12)
        use_l = Tex("$\\leftarrow$ angular spectrum", font_size=28, color=C_H).next_to(zc_eq, LEFT, buff=0.5).shift(0.1 * DOWN)
        use_r = Tex("Fresnel $\\rightarrow$", font_size=28, color=C_CHIRP).next_to(zc_eq, RIGHT, buff=0.5).shift(0.1 * DOWN)
        self.at(S("c5"))
        grid_m.clear_updaters()
        self.play(FadeOut(VGroup(cards, rec1, rec2, rec3, note1, note2, note3, ep1, g_l)), FadeOut(grid_m), run_time=0.5)
        self.remove(grid_m)
        self.add(pL, pR, knob, *ok_l, *ok_r)
        self.play(ch_op.animate.set_value(1.0), FadeIn(fr), FadeIn(tL), FadeIn(tR), FadeIn(sl), FadeIn(sl_l), run_time=0.8)
        self.at(S("c5") + 3.9)
        self.play(FadeIn(leg), Indicate(fr[1], color=C_EV, scale_factor=1.02), run_time=0.9)

        # ---------------- c6: ASM chirp speeds up with z, Fresnel slows down; they swap at z_c
        self.at(S("c6") + 0.1)
        self.play(zt.animate.set_value(np.log10(2.6)), run_time=4.0, rate_func=rate_functions.ease_in_out_sine)
        self.at(S("c6") + 5.4)
        self.play(zt.animate.set_value(0.0), FadeIn(tick), run_time=1.1, rate_func=rate_functions.ease_in_out_sine)
        self.play(Write(zc_eq), run_time=1.2)
        self.at(S("c6") + 10.8)
        self.play(FadeIn(use_l), zt.animate.set_value(np.log10(0.6)), run_time=0.9)
        self.play(FadeIn(use_r), zt.animate.set_value(np.log10(1.7)), run_time=1.0)
        self.finish()


# ============================================================ 7  recap =============
class Ch7Recap(Chapter):
    CH = 7
    TAG = "7\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- r1: far away, or at a lens's focus: each direction -> its own spot
        zd = load("zoom.npz")
        fan = heat_rgba(np.sqrt(np.maximum(zd["maps"][-1].astype(np.float32), 0)), 1.25, 0.6, flip=False)
        cA, wA, hA = np.array([-3.45, 0.85, 0]), 5.6, 3.0
        fan_m, fan_f = self.static_image(fan, wA, cA, hA)
        ld = load("lens.npz")
        zs, xs, f = ld["zs"], ld["xs"], float(ld["f"])
        cols = zs <= 46.0
        trio = [(ld["sweep"][-1][:, cols], T_POS), (ld["sweep"][0][:, cols], T_ZERO), (ld["minus"][:, cols], T_NEG)]
        m0 = np.abs(trio[1][0]).max()
        ENV = [((np.abs(F) / m0) ** 0.55).astype(np.float32) for F, _ in trio]
        PH = [(F / np.maximum(np.abs(F), 1e-12)).astype(np.complex64) for F, _ in trio]
        OM = TAU_ * 1.4

        def trio_img(t):
            w = np.exp(-1j * OM * t)
            rgb = np.zeros(ENV[0].shape + (3,), np.float32)
            for e, p_, (_, tn) in zip(ENV, PH, trio):
                rgb += (e * (0.28 + 0.72 * (0.5 + 0.5 * (p_ * w).real)))[..., None] * tn[None, None, :]
            out = np.empty(ENV[0].shape + (4,), np.uint8)
            out[..., :3] = (np.clip(rgb / 1.9, 0, 1) * 255).astype(np.uint8)
            out[..., 3] = 255
            return out
        cB, wB = np.array([3.45, 0.85, 0]), 5.6
        hB = wB * 30 / 56
        tri_m, tri_f = self.live_image(trio_img, wB, cB, hB)
        lensB = lens_shape([cB[0] - wB / 2 + 10 / 56 * wB, cB[1], 0], 24 / 30 * hB, 0.3, GREY_A, 0.12, 2)
        boxA, boxB = frame_box(cA, wA, hA), frame_box(cB, wB, hB)
        capA = MathTex("\\text{far away: }", "x = z\\tan\\theta", font_size=32).next_to(boxA, DOWN, buff=0.18)
        capB = MathTex("\\text{at a focus: }", "x = f\\tan\\theta", font_size=32).next_to(boxB, DOWN, buff=0.18)
        line = Tex("each direction lands on its own spot $\\Rightarrow$ light lays out its own spectrum", font_size=34,
                   color=YELLOW).move_to([0, -2.45, 0])
        self.at(S("r1"))
        self.add(fan_m, tri_m)
        self.play(fan_f.animate.set_value(1.0), FadeIn(boxA), FadeIn(capA), run_time=0.7)
        self.at(S("r1") + 0.9)
        self.play(tri_f.animate.set_value(1.0), FadeIn(boxB), FadeIn(lensB), FadeIn(capB), run_time=0.7)
        self.at(S("r1") + 2.5)
        self.play(Write(line), run_time=1.6)

        # ---------------- r2: a second lens brings the picture back; a mask in between edits it
        axis, planes, lenses, braces, flabs = four_f_schematic(0.75, (-5.0, -2.5, 0.0, 2.5, 5.0), 1.4)
        F1 = Tex("F", font_size=60).move_to([-5.0, 2.0, 0])
        F2 = Tex("F", font_size=60).rotate(PI).move_to([5.0, 2.0, 0])
        mask = Line([0, 0.55, 0], [0, 0.95, 0]).set_stroke(C_H, 8)
        mask_l = Tex("mask: edit the spectrum", font_size=30, color=C_H).move_to([0, 2.0, 0])
        self.at(S("r2") - 0.1)
        tri_m.clear_updaters()
        self.play(FadeOut(Group(fan_m, tri_m, boxA, boxB, lensB, capA, capB)), fan_f.animate.set_value(0.0), line.animate.set_opacity(0.0),
                  run_time=0.5)
        self.remove(fan_m, tri_m, line)
        self.play(Create(axis), FadeIn(planes), FadeIn(lenses), FadeIn(braces), FadeIn(flabs), FadeIn(F1), run_time=0.8)
        self.play(TransformFromCopy(F1, F2, path_arc=-PI), run_time=0.9)
        self.at(S("r2") + 2.2)
        self.play(FadeIn(mask), FadeIn(mask_l), run_time=0.6)

        # ---------------- r3: where it breaks: paraxial = parabola; exact kz on the circle
        o = np.array([0, -1.05, 0])
        R = 2.55
        ax = VGroup(Arrow(o + 3.4 * LEFT, o + 3.6 * RIGHT, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                    Arrow(o, o + 3.3 * UP, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.04))
        axl = VGroup(MathTex("k_x", font_size=30, color=C_KX).next_to(ax[0], RIGHT, buff=0.08),
                     MathTex("k_z", font_size=30, color=C_KZ).next_to(ax[1], UP, buff=0.08))
        kk = np.linspace(-1, 1, 300)
        circ = VMobject().set_points_smoothly([o + R * np.array([k_, np.sqrt(1 - k_ ** 2), 0]) for k_ in kk]).set_stroke(C_KZ, 5)
        kp = np.linspace(-1.25, 1.25, 300)
        para = DashedVMobject(VMobject().set_points_smoothly([o + R * np.array([k_, 1 - k_ ** 2 / 2, 0]) for k_ in kp]),
                              num_dashes=50).set_stroke(C_CHIRP, 3.5)
        le = MathTex("\\text{exact: } k_z = \\sqrt{k^2 - k_x^2}", font_size=32, color=C_KZ).move_to([4.15, 1.75, 0])
        lp = MathTex("\\text{paraxial: } k_z \\approx k - {k_x^2 \\over 2k}", font_size=32, color=C_CHIRP).move_to([-4.2, 1.75, 0])
        ok_cap = Tex("small angles only", font_size=28, color=GREY_A).next_to(lp, DOWN, buff=0.15)
        self.at(S("r3"))
        self.play(FadeOut(VGroup(axis, planes, lenses, braces, flabs, F1, F2, mask, mask_l)), run_time=0.5)
        self.play(GrowArrow(ax[0]), GrowArrow(ax[1]), FadeIn(axl), run_time=0.6)
        self.play(Create(para), FadeIn(lp), FadeIn(ok_cap), run_time=1.1)
        self.at(S("r3") + 3.1)
        self.play(Create(circ), FadeIn(le), run_time=1.2)

        # ---------------- r4: a real lens is finite: it catches only a cap of directions -> blur (next: E03)
        na = 0.5
        th = np.linspace(np.arccos(na), np.pi - np.arccos(na), 60)
        cap = VMobject().set_points_smoothly([o + R * np.array([np.cos(a), np.sin(a), 0]) for a in th]).set_stroke(C_H, 9)
        cone = VGroup(Line(o, o + R * np.array([na, np.sqrt(1 - na ** 2), 0])), Line(o, o + R * np.array([-na, np.sqrt(1 - na ** 2), 0]))
                      ).set_stroke(C_H, 2.5)
        caught = Tex("directions the lens catches", font_size=28, color=C_H).next_to(o + R * np.array([-na, np.sqrt(1 - na ** 2), 0]),
                                                                                  LEFT, buff=0.25).shift(0.3 * UP)
        nxt = next_card("the diffraction limit: why every image is blurry", y=-2.6)
        self.at(S("r4") + 1.9)
        self.play(FadeOut(para), FadeOut(lp), FadeOut(ok_cap), circ.animate.set_stroke(opacity=0.35), Create(cap), Create(cone),
                  FadeIn(caught), run_time=1.2)
        self.at(S("r4") + 4.6)
        self.play(FadeIn(nxt, shift=0.15 * UP), run_time=0.9)
        self.finish(fade=0.9)


def E_(k):
    return SEG[k]["end"]


CHAPTERS = ["Ch0Hook", "Ch1FarField", "Ch2Paraxial", "Ch3Lens", "Ch4FourF", "Ch5PhaseContrast", "Ch6Computer", "Ch7Recap"]
