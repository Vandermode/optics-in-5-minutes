"""The Phase Problem: holography and phase retrieval -- chapter scenes (Optics in 5 Minutes, E05).
Render: ./render.sh ; preview: ./preview.sh Ch0Hook

Every physical picture is loaded from data/*.npz (physics.py). The only arithmetic done per frame is forming
|U_c + R|^2 for the current reference tilt (chapter 2 and 6) and cross-fading precomputed frames.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403


def _at_whole_frames(self, t):
    """Chapter.at() that waits a whole number of frames (fix from FAST_ANIMATION_PIPELINE.md section 10:
    a float wait can flip one frame at a render seam when t lies exactly on the 1/60 s grid)."""
    x = (t - self.now) * config.frame_rate                # frames still to go
    if x > 0.5 + 1e-6:
        k = int(np.ceil(x - 1e-6))
        self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
    elif x < -0.5 - 1e-6:
        self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


Chapter.at = _at_whole_frames

C_TWIN = PURPLE_A          # the twin term U* R
C_PH = YELLOW              # phase angles and phase curves
NPIX = 512
NATIVE = NPIX / 135.0      # width (scene units) at which a 512-px image is drawn 1:1 in the 1080p film


# ============================================================ helpers ===============
def npz(name):
    return np.load(data_path(name))


def rgba(rgb, alpha=255):
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out


def heat_u8(v, gamma=1.0):
    """uint8 value map (row 0 = top) -> RGBA with the series HEAT map."""
    idx = ((v.astype(np.float32) / 255.0) ** gamma * 1023).astype(int)
    return rgba(HEAT[idx])


def grey_u8(v):
    return rgba(np.repeat(np.asarray(v, np.uint8)[..., None], 3, -1))


HUE_COLS = [rgb_to_color(PHASE_HUE[int(j / 256 * 2047)] / 255.0) for j in range(256)]


def hue_color(theta):
    return HUE_COLS[int(((theta % TAU_) / TAU_) * 256) % 256]


def box(center, w, h=None, color=GREY_B, width=1.5):
    return Rectangle(width=w, height=h if h else w).move_to(center).set_stroke(color, width)


def blend(stack, f):
    """Frame of an image stack at fractional index f (linear cross-fade)."""
    f = float(np.clip(f, 0, len(stack) - 1))
    i = int(np.floor(f))
    w = f - i
    if w < 1e-3 or i + 1 >= len(stack):
        return stack[i]
    return (stack[i].astype(np.float32) * (1 - w) + stack[i + 1].astype(np.float32) * w).astype(np.uint8)


def memo(fn):
    """Cache the last result of a pure function of its (rounded) argument."""
    last = {}

    def g(x):
        k = round(float(x), 5)
        if k not in last:
            last.clear()
            last[k] = fn(x)
        return last[k]
    return g


def hologram(Uc, m, rows=slice(None), cols=slice(None), imax=4.4):
    """|U_c + R|^2 for a unit tilted reference R = exp(2 pi i m x / N) (m = carrier in FFT bins)."""
    x = np.arange(NPIX)[cols]
    R = np.exp(1j * TAU_ * m * x / NPIX).astype(np.complex64)
    I = np.abs(Uc[rows, cols] + R[None, :]) ** 2
    return np.clip(I * (255.0 / imax), 0, 255).astype(np.uint8)


def window_outline(center, P, color=GREEN, width=3, dashed=False):
    """Outline of the lopsided GS window (disk minus a bite) for a P-wide panel centred at center."""
    c = np.asarray(center, float)
    d = Difference(Circle(radius=0.36 * P).move_to(c), Circle(radius=0.15 * P).move_to(c + np.array([0.27 * P, 0.22 * P, 0])))
    d.set_stroke(color, width).set_fill(opacity=0)
    return DashedVMobject(d, num_dashes=60).set_stroke(color, width) if dashed else d


def omega_profile(ta, acc, tb, dec, turns_per_s):
    """Spin angle Omega(t) (pure function of t): accelerate at ta over acc, cruise until tb, decelerate over dec.
    The cruise speed is adjusted so the arrows stop exactly where they started (whole turns)."""
    span = 0.5 * acc + (tb - ta - acc) + 0.5 * dec
    n = max(1, round(turns_per_s * span))
    w = TAU_ * n / span

    def om(t):
        if t <= ta:
            return 0.0
        if t <= ta + acc:
            return 0.5 * w / acc * (t - ta) ** 2
        o1 = 0.5 * w * acc
        if t <= tb:
            return o1 + w * (t - ta - acc)
        o2 = o1 + w * (tb - ta - acc)
        tau = min(t - tb, dec)
        return o2 + w * tau - 0.5 * w / dec * tau ** 2
    return om


def code_card2(lines, font_size=24, color=GREY_A, title=None):
    """Like o5common.code_card, but keeps leading spaces (Text drops them): rows are shifted by the indent."""
    cw = Text("M" * 20, font=MONO, font_size=font_size).width / 20
    rows = VGroup(*[Text(l.strip() if l.strip() else " ", font=MONO, font_size=font_size, color=color) for l in lines])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    for r, l in zip(rows, lines):
        r.shift((len(l) - len(l.lstrip(" "))) * cw * RIGHT)
    g = VGroup(rows)
    if title:
        t = Text(title, font="Lato", font_size=font_size - 2, color=YELLOW, weight=BOLD)
        t.next_to(rows, UP, aligned_edge=LEFT, buff=0.25)
        g.add(t)
    bx = SurroundingRectangle(g, buff=0.3, corner_radius=0.12, color=GREY_B, stroke_width=1.5).set_fill(GREY_E, 0.6)
    return VGroup(bx, g)


def colour_wheel(center, size=1.0):
    m = ImageMobject(colour_wheel_rgba(200))
    m.stretch_to_fit_width(size).stretch_to_fit_height(size).move_to(center)
    return m


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        d = npz("cells.npz")
        W, C = 5.4, np.array([0.0, 0.25, 0])
        cam_m, cam_f = self.static_image(grey_u8(d["camera"]), W, C)
        ph_m, ph_f, ph_sw = self.sweep_image(rgba(d["phase_rgb"]), W, C, axis="up")
        ph_f.set_value(1.0)                                   # visibility comes from the wipe
        bl, br = box(C, W), box(C, W)
        lab_cam = VGroup(Tex("what a camera", font_size=32), Tex("records:", font_size=32),
                         MathTex("|", "U", "|^2", font_size=40)).arrange(DOWN, buff=0.16).move_to([-4.85, 1.0, 0])
        lab_cam[2][1].set_color(C_U)
        self.add(cam_m, ph_m)
        self.play(cam_f.animate.set_value(1.0), FadeIn(bl), FadeIn(br), run_time=0.8)
        self.at(S("h1") + 0.6)
        self.play(FadeIn(lab_cam, shift=0.1 * RIGHT), run_time=0.7)

        # brightness along one line: almost flat
        row = int(d["row"])
        y_row = C[1] + W / 2 - (row + 0.5) / NPIX * W
        scan = DashedLine([C[0] - W / 2, y_row, 0], [C[0] + W / 2, y_row, 0], dash_length=0.1).set_stroke(YELLOW, 2.5)
        PX0, PX1, PY0, PH = 3.3, 6.6, -2.35, 1.55
        xs = np.linspace(PX0, PX1, NPIX)
        cam_row = d["camera"][row].astype(float) / 255 * 1.6
        phi_row = d["phi_row"].astype(float)
        base = Line([PX0, PY0, 0], [PX1, PY0, 0]).set_stroke(GREY_C, 1.5)
        c_bright = poly_line(xs, PY0 + cam_row / 1.6 * PH, GREY_A, 3)
        c_phase = poly_line(xs, PY0 + phi_row / 3.6 * PH, C_PH, 3)
        t_bright = Tex("brightness along the line", font_size=26, color=GREY_A).next_to(base, UP, buff=PH + 0.25)
        t_phase = Tex("phase along the line", font_size=26, color=C_PH).move_to(t_bright)
        self.at(S("h1") + 2.3)
        self.play(Create(scan), FadeIn(base), Create(c_bright), FadeIn(t_bright), run_time=1.0)

        # h2: the hidden phase is wiped in
        wheel = colour_wheel([4.95, 2.05, 0], 1.25)
        wl = Tex("colour = phase", font_size=28, color=GREY_A).next_to(wheel, DOWN, buff=0.15)
        lab_ph = VGroup(Tex("what the light", font_size=32), Tex("carries: phase", font_size=32),
                        MathTex("\\varphi", font_size=44, color=C_PH)).arrange(DOWN, buff=0.16).move_to(lab_cam)
        self.at(S("h2") + 0.5)
        self.play(ph_sw.animate.set_value(1.0), run_time=2.6, rate_func=linear)
        self.play(FadeIn(wheel), FadeIn(wl), FadeTransform(lab_cam, lab_ph), run_time=0.7)
        self.play(Transform(c_bright, c_phase), FadeTransform(t_bright, t_phase), run_time=1.0)

        # h3: the phase problem in one picture
        WS = 4.0
        PL, PR = np.array([-3.35, 0.45, 0]), np.array([3.35, 0.45, 0])
        self.at(S("h3"))
        self.play(FadeOut(VGroup(scan, base, c_bright, t_phase, lab_ph, wl)), FadeOut(wheel), run_time=0.4)
        self.play(ph_m.animate.scale(WS / W).move_to(PL), cam_m.animate.scale(WS / W).move_to(PR),
                  bl.animate.scale(WS / W).move_to(PL), br.animate.scale(WS / W).move_to(PR), run_time=0.9)
        cl = MathTex("\\text{the light: phase }", "\\varphi", font_size=32).next_to(bl, DOWN, buff=0.2)
        cl[1].set_color(C_PH)
        cr = MathTex("\\text{the camera: }", "|", "U", "|^2", font_size=32).next_to(br, DOWN, buff=0.2)
        cr[2].set_color(C_U)
        fw = Arrow([-1.15, 0.95, 0], [1.15, 0.95, 0], buff=0, stroke_width=5, color=WHITE)
        fwl = Tex("record", font_size=28).next_to(fw, UP, buff=0.1)
        bw = DashedLine([1.15, -0.05, 0], [-1.15, -0.05, 0], dash_length=0.1).set_stroke(YELLOW, 4)
        bw.add_tip(tip_length=0.22)
        q = MathTex("?", font_size=72, color=YELLOW).next_to(bw, DOWN, buff=0.15)
        self.play(FadeIn(cl), FadeIn(cr), GrowArrow(fw), FadeIn(fwl), run_time=0.6)
        self.play(Create(bw), FadeIn(q, scale=0.6), run_time=0.7)

        # h4: title
        title = series_title(5, "The Phase Problem", "holography and phase retrieval")
        self.at(S("h4") + 0.1)
        self.play(ph_f.animate.set_value(0.3), cam_f.animate.set_value(0.3), FadeOut(VGroup(fw, fwl, bw, q, cl, cr, bl, br)),
                  run_time=0.6)
        self.play(FadeIn(title[0]), Write(title[1]), run_time=1.6)
        self.finish()


# ============================================================ 1  the camera ========
class Ch1Camera(Chapter):
    CH = 1
    TAG = "1\\quad What a camera throws away"

    def construct(self):
        S = self.S
        self.show_tag()
        d = npz("cells.npz")
        phi_row, BG = d["phi_row"].astype(float), float(d["bg"])
        # the run of pixels covering the central cell (column 258)
        on = phi_row > 0.03
        a = 258
        while a > 0 and on[a - 1]:
            a -= 1
        b = 258
        while b < len(on) - 1 and on[b + 1]:
            b += 1
        mid, half = 0.5 * (a + b), (b - a) * 1.05
        X0, X1 = -6.0, 6.0
        pix = lambda x: mid + x / (X1 - X0) * 2 * half
        phi_at = lambda x: np.interp(pix(np.asarray(x, float)), np.arange(len(phi_row)), phi_row) * on[np.clip(np.round(pix(np.asarray(x, float))).astype(int), 0, len(on) - 1)]
        pmax = phi_row[a:b + 1].max()
        YS, HC = -1.45, 1.45                                    # slide height, cell height scale
        xd = np.linspace(-5.4, 5.4, 13)
        YD, RD = 1.35, 0.34
        phid = phi_at(xd)
        grow, drain, n_op = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)
        om = omega_profile(S("c4") + 1.6, 1.4, S("c5") + 0.5, 1.6, 1.6)

        slide = Line([-6.4, YS, 0], [6.4, YS, 0]).set_stroke(GREY_B, 3)
        lights = VGroup(*[Arrow([x, -2.95, 0], [x, -1.75, 0], buff=0, stroke_width=4, color=C_U,
                                max_tip_length_to_length_ratio=0.2) for x in (-4.5, -1.5, 1.5, 4.5)])
        light_l = Tex("light", font_size=28, color=C_U).next_to(lights[0], LEFT, buff=0.2)
        xs = np.linspace(X0, X1, 300)
        ps = phi_at(xs)

        def cell():
            g = grow.get_value()
            if g < 1e-3:
                return VGroup()
            pts = [[x, YS + g * HC * p / pmax, 0] for x, p in zip(xs, ps) if p > 0]
            if len(pts) < 3:
                return VGroup()
            poly = Polygon(*pts, [pts[-1][0], YS, 0], [pts[0][0], YS, 0])
            return poly.set_stroke(GREY_A, 2).set_fill(GREY_B, 0.35)
        cel = always_redraw(cell)
        dials = VGroup(*[Circle(radius=RD).move_to([x, YD, 0]).set_stroke(GREY_C, 1.5) for x in xd])
        guides = VGroup(*[fast_dashed([x, YS + 0.05, 0], [x, YD - RD - 0.05, 0], 0.05, GREY_D, 1.2) for x in xd])

        def needles():
            g = VGroup()
            if n_op.get_value() < 0.002:
                return g
            o = om(self.now)
            dr = drain.get_value()
            gr = grow.get_value()
            for x, p in zip(xd, phid):
                th = BG + gr * p - o
                col = interpolate_color(hue_color(th), GREY_B, dr) if dr > 0 else hue_color(th)
                c = np.array([x, YD, 0])
                g.add(fast_arrow(c, c + RD * np.array([np.cos(th), np.sin(th), 0]), col, 4.5, 0.13, opacity=n_op.get_value()))
            return g
        nd = always_redraw(needles)

        # ---------------- c1: light is an arrow
        rem = Tex("each arrow: length = amplitude, angle = phase", font_size=32).move_to([1.3, 2.75, 0])
        self.at(S("c1"))
        self.play(Create(slide), FadeIn(lights), FadeIn(light_l), run_time=0.8)
        self.add(nd)
        self.play(LaggedStart(*[Create(c) for c in dials], lag_ratio=0.06), FadeIn(guides), n_op.animate.set_value(1.0), run_time=1.4)
        self.at(S("c1") + 4.0)
        self.play(FadeIn(rem, shift=0.1 * DOWN), run_time=0.8)

        # ---------------- c2: a cell delays the light -> the arrows turn
        self.add(cel)
        self.at(S("c2") + 0.2)
        self.play(FadeOut(rem), run_time=0.4)
        self.play(grow.animate.set_value(1.0), run_time=3.2, rate_func=smooth)
        ipk = int(np.argmax(ps))
        xp = xs[ipk]
        hl = DoubleArrow([xp + 0.35, YS, 0], [xp + 0.35, YS + HC, 0], buff=0, stroke_width=3, color=WHITE,
                         tip_length=0.15, max_tip_length_to_length_ratio=0.2)
        hlab = MathTex("h", font_size=34).next_to(hl, RIGHT, buff=0.08)
        form = MathTex("\\varphi", "=", "k_0", "\\,\\Delta n", "\\,h", font_size=44).move_to([4.75, 0.4, 0])
        form[0].set_color(C_PH)
        fcap = VGroup(Tex("$h$: thickness", font_size=26, color=GREY_A),
                      Tex("$\\Delta n$: index step, cell vs.\\ water", font_size=26, color=GREY_A),
                      Tex("$k_0 = 2\\pi/\\lambda$", font_size=26, color=GREY_A)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        fcap.next_to(form, DOWN, buff=0.2).align_to(form, LEFT).shift(0.35 * LEFT)
        self.at(S("c2") + 3.9)
        self.play(GrowFromCenter(hl), FadeIn(hlab), Write(form), run_time=1.0)
        self.play(FadeIn(fcap), run_time=0.6)

        # ---------------- c3: the camera records squared lengths only
        bars = VGroup(*[Rectangle(width=0.3, height=0.62).move_to([x, 2.32, 0]).set_stroke(width=0).set_fill(GREY_A, 0.9)
                        for x in xd])
        btxt = MathTex("\\text{the camera records }", "|", "U", "|^2", "\\text{: all equal}", font_size=32).move_to([1.3, 3.05, 0])
        btxt[2].set_color(C_U)
        self.at(S("c3") + 2.3)
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in bars], lag_ratio=0.04), FadeIn(btxt), run_time=1.0)
        self.at(S("c3") + 4.6)
        self.play(drain.animate.set_value(1.0), run_time=1.2)

        # ---------------- c4: the arrows spin far too fast to follow
        spin = VGroup(Tex("green light:", font_size=30), MathTex("6\\times 10^{14}\\ \\text{turns per second}", font_size=32)).arrange(DOWN, buff=0.12)
        spin.move_to([4.75, 0.05, 0])
        self.at(S("c4") + 0.6)
        self.play(drain.animate.set_value(0.0), FadeOut(fcap), FadeOut(form), run_time=0.8)
        self.at(S("c4") + 3.4)
        self.play(FadeIn(spin, shift=0.1 * UP), run_time=0.7)

        # ---------------- c5: all the information is in the phase
        self.at(S("c5") + 2.2)
        arcs = VGroup(*[Arc(radius=0.2, start_angle=BG, angle=p, arc_center=[x, YD, 0]).set_stroke(C_PH, 3)
                        for x, p in zip(xd, phid) if p > 0.25])
        info = VGroup(Tex("all the information:", font_size=30),
                      MathTex("\\varphi(x)", font_size=40, color=C_PH)).arrange(DOWN, buff=0.12).move_to(spin)
        self.play(Create(arcs), FadeTransform(spin, info), run_time=0.9)
        self.at(S("c5") + 4.3)
        self.play(bars.animate.set_fill(opacity=0.25), btxt.animate.set_opacity(0.35), run_time=0.8)
        self.finish()


# ============================================================ 2  holography =========
class Ch2Holography(Chapter):
    CH = 2
    TAG = "2\\quad Holography: add a reference"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- r1-r3: interference turns angle into brightness (phasors)
        P, SC = np.array([-3.55, -0.45, 0]), 1.45
        AU, AR, A0 = 0.65, 1.0, 0.35
        axes = VGroup(Line(P + 2.5 * LEFT, P + 2.5 * RIGHT), Line(P + 2.45 * DOWN, P + 2.45 * UP)).set_stroke(GREY_D, 1.5)
        r_on, s_on, a_on, u_on = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)
        dphi = ValueTracker(1.1)
        om = omega_profile(S("r1") - 0.6, 0.8, S("r2") + 0.3, 1.2, 0.4)

        def geom():
            o = om(self.now)
            tr, tu = A0 - o, A0 + dphi.get_value() - o
            Rv = SC * AR * np.array([np.cos(tr), np.sin(tr), 0])
            Uv = SC * AU * np.array([np.cos(tu), np.sin(tu), 0])
            tail = P + r_on.get_value() * Rv
            return tr, tu, Rv, Uv, tail

        def phasors():
            tr, tu, Rv, Uv, tail = geom()
            g = VGroup()
            ra = r_on.get_value()
            if ra > 0.01:
                g.add(fast_arrow(P, P + ra * Rv, C_REF, 6, 0.24))
            if u_on.get_value() > 0.002:
                g.add(fast_arrow(tail, tail + Uv, C_U, 6, 0.22, opacity=u_on.get_value()))
            if s_on.get_value() > 0.01:
                g.add(fast_arrow(P, tail + Uv, WHITE, 4, 0.2, opacity=s_on.get_value()))
            if a_on.get_value() > 0.01:
                dd = (dphi.get_value() + np.pi) % TAU_ - np.pi
                if abs(dd) > 0.05:
                    g.add(Arc(radius=0.42, start_angle=tr, angle=dd, arc_center=tail).set_stroke(C_PH, 3, opacity=a_on.get_value()))
                g.add(fast_dashed(tail, tail + 0.75 * Rv / np.linalg.norm(Rv), 0.05, C_REF, 1.5, a_on.get_value()))
            return g
        ph = always_redraw(phasors)

        def nrm(v):
            n = np.array([-v[1], v[0], 0])
            return n / (np.linalg.norm(n) + 1e-9)
        lu = MathTex("U", font_size=40, color=C_U)
        lr = MathTex("R", font_size=40, color=C_REF)
        ld = MathTex("\\Delta\\varphi", font_size=34, color=C_PH)
        lu.add_updater(lambda m: m.move_to(geom()[4] + 0.5 * geom()[3] + 0.36 * nrm(geom()[3])).set_opacity(u_on.get_value()))
        lr.add_updater(lambda m: m.move_to(P + 0.5 * r_on.get_value() * geom()[2] - 0.36 * nrm(geom()[2])).set_opacity(
            min(1.0, 3 * r_on.get_value())))

        def ld_pos(m):
            tr, tu, Rv, Uv, tail = geom()
            mid = tr + 0.5 * ((dphi.get_value() + np.pi) % TAU_ - np.pi)
            m.move_to(tail + 0.78 * np.array([np.cos(mid), np.sin(mid), 0])).set_opacity(a_on.get_value())
        ld.add_updater(ld_pos)
        legend = VGroup(*[VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(col, 5), MathTex(t, font_size=32, color=col)).arrange(RIGHT, buff=0.15)
                          for col, t in ((C_REF, "R"), (C_U, "U"), (WHITE, "U+R"))]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        legend.move_to([-6.05, 2.35, 0], aligned_edge=LEFT).shift(0.0 * RIGHT)
        legend.align_to([-6.6, 0, 0], LEFT)
        gab = Tex("Dennis Gabor, 1948", font_size=28, color=GREY_B).move_to([-2.9, 2.75, 0])
        self.at(S("r1"))
        self.add(ph, lu)
        self.play(FadeIn(axes), u_on.animate.set_value(1.0), run_time=0.5)
        self.at(S("r1") + 0.7)
        self.play(FadeIn(gab), run_time=0.5)
        self.add(lr)
        self.at(S("r1") + 3.3)
        self.play(r_on.animate.set_value(1.0), run_time=1.2)
        self.add(ld)
        self.at(S("r1") + 7.0)
        self.play(a_on.animate.set_value(1.0), run_time=0.8)

        # r2: the camera sees the sum; change the phase -> the brightness changes
        MX, MY0, MS = 0.3, -2.55, 0.78                          # meter x, base, scale per unit intensity
        Imax = (AR + AU) ** 2

        def inten():
            return AR ** 2 + AU ** 2 + 2 * AR * AU * np.cos(dphi.get_value())

        def meter():
            I = inten() * s_on.get_value()
            g = VGroup(Rectangle(width=0.42, height=Imax * MS).move_to([MX, MY0 + 0.5 * Imax * MS, 0]).set_stroke(GREY_B, 1.5,
                                                                                                                 opacity=s_on.get_value()))
            if I > 1e-3:
                g.add(Rectangle(width=0.42, height=I * MS).move_to([MX, MY0 + 0.5 * I * MS, 0]).set_stroke(width=0).set_fill(WHITE, 0.85))
            return g
        mt = always_redraw(meter)
        mlab = VGroup(Tex("camera", font_size=26, color=GREY_A), MathTex("|U+R|^2", font_size=30)).arrange(DOWN, buff=0.08)
        mlab.next_to([MX, MY0 + Imax * MS, 0], UP, buff=0.12)
        GX0, GX1, GY0, GH = 1.55, 5.95, -2.55, 2.25
        gax = VGroup(Arrow([GX0, GY0, 0], [GX1 + 0.35, GY0, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                     Arrow([GX0, GY0, 0], [GX0, GY0 + GH + 0.4, 0], buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.06))
        gxl = MathTex("\\Delta\\varphi", font_size=32, color=C_PH).next_to(gax[0], RIGHT, buff=0.1)
        gyl = Tex("brightness", font_size=26, color=GREY_A).next_to(gax[1].get_end(), RIGHT, buff=0.12)
        gtk = VGroup(MathTex("0", font_size=26, color=GREY_B).move_to([GX0, GY0 - 0.28, 0]),
                     MathTex("2\\pi", font_size=26, color=GREY_B).move_to([GX1, GY0 - 0.28, 0]),
                     Line([GX1, GY0 - 0.07, 0], [GX1, GY0 + 0.07, 0]).set_stroke(GREY_B, 2))
        trace_on = ValueTracker(0.0)

        def trace():
            if trace_on.get_value() < 0.5:
                return VGroup()
            dv = float(np.clip(dphi.get_value(), 0, TAU_))
            if dv < 0.02:
                return VGroup()
            tt = np.linspace(0, dv, max(2, int(dv / TAU_ * 160)))
            yy = GY0 + (AR ** 2 + AU ** 2 + 2 * AR * AU * np.cos(tt)) / Imax * GH
            g = VGroup(poly_line(GX0 + tt / TAU_ * (GX1 - GX0), yy, WHITE, 4))
            g.add(Dot([GX0 + dv / TAU_ * (GX1 - GX0), yy[-1], 0], radius=0.06, color=C_PH))
            return g
        tr_m = always_redraw(trace)
        self.at(S("r2"))
        lu.clear_updaters()
        lr.clear_updaters()
        ld.clear_updaters()
        self.add(mt)
        self.play(s_on.animate.set_value(1.0), FadeTransform(lr, legend[0]), FadeTransform(lu, legend[1]), FadeIn(legend[2]),
                  FadeOut(ld), FadeIn(mlab), run_time=0.9)
        self.play(FadeIn(gax), FadeIn(gxl), FadeIn(gyl), FadeIn(gtk), run_time=0.6)
        self.add(tr_m)
        self.at(S("r2") + 2.6)
        self.play(dphi.animate.set_value(0.0), run_time=0.5)
        trace_on.set_value(1.0)
        self.play(dphi.animate.set_value(TAU_), run_time=3.1, rate_func=linear)

        # r3: the phase is written in brightness
        msg = MathTex("\\text{phase}", "\;\\longrightarrow\;", "\\text{brightness}", font_size=40).move_to([3.75, 0.75, 0])
        msg[0].set_color(C_PH)
        self.at(S("r3") + 0.2)
        self.play(FadeIn(msg, shift=0.1 * UP), run_time=0.8)

        # ---------------- r4: expand the square
        eq = MathTex("|", "U", "+", "R", "|^2", "=", "|", "U", "|^2", "+", "|", "R", "|^2", "+", "U", "R^*", "+", "U^*", "R",
                     font_size=52).move_to([0.3, 1.25, 0])
        for i in (1, 7, 14, 17):
            eq[i].set_color(C_U)
        for i in (3, 11, 15, 18):
            eq[i].set_color(C_REF)
        b1 = Brace(eq[6:13], DOWN, buff=0.12, color=GREY_B)
        b1l = Tex("two brightnesses", font_size=28, color=GREY_A).next_to(b1, DOWN, buff=0.1)
        b2 = Brace(eq[14:19], DOWN, buff=0.12, color=C_PH)
        b2l = Tex("cross terms: contain $U$,\\\\phase and all", font_size=28, color=C_PH).next_to(b2, DOWN, buff=0.1)
        star = Tex("$^*$ complex conjugate: the arrow mirrored", font_size=26, color=GREY_B).move_to([0.3, -1.6, 0])
        self.at(S("r4"))
        tr_m.clear_updaters()
        self.play(FadeOut(VGroup(axes, gab, mlab, gax, gxl, gyl, gtk, msg, tr_m, legend)), FadeOut(ph), FadeOut(mt), run_time=0.5)
        self.remove(ph, mt)
        self.play(Write(eq), run_time=1.4)
        self.play(GrowFromCenter(b1), FadeIn(b1l), run_time=0.6)
        self.at(S("r4") + 2.7)
        self.play(GrowFromCenter(b2), FadeIn(b2l), run_time=0.7)
        self.at(S("r4") + 4.3)
        self.play(FadeIn(star), run_time=0.6)

        # ---------------- r5-r7: in-line vs off-axis (hologram computed per frame from the camera field U_c)
        h = npz("holo.npz")
        Uc, MF = h["Uc"], int(h["m_final"])
        tilt = ValueTracker(0.0)
        HC = np.array([-3.35, -0.6, 0])
        hfn = memo(lambda tv: grey_u8(hologram(Uc, MF * tv)))
        him, hfd = self.live_image(lambda t: hfn(tilt.get_value()), NATIVE, HC)
        hbox = box(HC, NATIVE)
        # reference icon: a tilted plane wave (gold crests + arrow)
        RC = np.array([-3.35, 1.95, 0])

        def ref_icon():
            th = np.radians(32) * tilt.get_value()
            u = np.array([np.sin(th), np.cos(th), 0])
            n = np.array([np.cos(th), -np.sin(th), 0])
            g = VGroup(fast_arrow(RC - 0.42 * u, RC + 0.5 * u, C_REF, 4, 0.16))
            for s_ in (-0.2, 0.05, 0.3):
                c = RC + s_ * u
                g.add(Line(c - 0.42 * n, c + 0.42 * n).set_stroke(C_REF, 2.5))
            return g
        ric = always_redraw(ref_icon)
        rlab = MathTex("\\text{reference }", "R", font_size=30).next_to(RC, LEFT, buff=0.75)
        rlab[1].set_color(C_REF)
        cap1 = Tex("in-line: reference straight on (Gabor)", font_size=28, color=GREY_A).next_to(hbox, DOWN, buff=0.15)
        cap2 = Tex("off-axis: reference tilted (Leith \\& Upatnieks)", font_size=28, color=GREY_A).next_to(hbox, DOWN, buff=0.15)
        # schematic spectrum: three discs whose centres follow the tilt
        KC = np.array([3.45, -0.6, 0])
        bb = 0.42
        kcf = MF / 32.0 * bb                                     # carrier in disc radii (3.62 B)
        kax = Arrow(KC + 2.95 * LEFT, KC + 3.0 * RIGHT, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03)
        kxl = MathTex("k_x", font_size=30, color=C_KX).next_to(kax, RIGHT, buff=0.06)
        ktl = Tex("its spectrum (schematic)", font_size=28, color=GREY_B).move_to(KC + 2.05 * UP)

        def discs():
            k = kcf * tilt.get_value()
            return VGroup(Circle(radius=2 * bb).move_to(KC).set_stroke(GREY_B, 2).set_fill(GREY_B, 0.25),
                          Circle(radius=bb).move_to(KC + k * LEFT).set_stroke(C_U, 2.5).set_fill(C_U, 0.35),
                          Circle(radius=bb).move_to(KC + k * RIGHT).set_stroke(C_TWIN, 2.5).set_fill(C_TWIN, 0.35))
        dsc = always_redraw(discs)
        pile = Tex("all four terms on top of each other", font_size=28, color=GREY_A).move_to(KC + 1.35 * DOWN)
        self.at(S("r5"))
        self.play(VGroup(eq, b1, b1l, b2, b2l).animate.scale(0.6).move_to([1.6, 2.95, 0]), FadeOut(star), run_time=0.8)
        self.play(FadeOut(VGroup(b1, b1l, b2, b2l)), run_time=0.3)
        self.add(him, ric)
        self.play(hfd.animate.set_value(1.0), FadeIn(hbox), FadeIn(rlab), FadeIn(ric), FadeIn(cap1), run_time=0.9)
        self.at(S("r5") + 2.7)
        self.play(FadeIn(kax), FadeIn(kxl), FadeIn(ktl), FadeIn(dsc), run_time=0.7)
        self.play(FadeIn(pile), run_time=0.5)
        tw = MathTex("U^*", "\\text{ (twin) in here too}", font_size=30)
        tw[0].set_color(C_TWIN)
        tw.move_to(KC + 1.25 * UP)
        twa = Arrow(tw.get_bottom() + 0.05 * DOWN, KC + 0.2 * UP + 0.12 * RIGHT, buff=0.05, stroke_width=3, color=C_TWIN,
                    max_tip_length_to_length_ratio=0.25)
        self.at(S("r5") + 5.4)
        self.play(FadeIn(tw), GrowArrow(twa), run_time=0.6)

        # r6: tilt the reference -> fringes; the discs slide apart
        self.at(S("r6") + 0.2)
        self.play(FadeOut(pile), FadeOut(tw), FadeOut(twa), run_time=0.3)
        self.at(S("r6") + 1.4)
        self.play(tilt.animate.set_value(1.0), FadeTransform(cap1, cap2), run_time=4.4, rate_func=smooth)
        fr = Tex("fringes", font_size=30).move_to([-0.55, 1.0, 0])
        fra = Arrow(fr.get_left() + 0.05 * LEFT, [-1.75, 0.45, 0], buff=0.05, stroke_width=3, color=WHITE, max_tip_length_to_length_ratio=0.25)
        self.play(FadeIn(fr), GrowArrow(fra), run_time=0.5)

        # r7: a tilt is a shift in spatial frequency
        sh_on = ValueTracker(0.0)
        sh = always_redraw(lambda: fast_arrow(KC + 1.45 * DOWN, KC + 1.45 * DOWN + sh_on.get_value() * kcf * tilt.get_value() * LEFT,
                                              C_REF, 4, 0.2))
        shl = MathTex("\\text{tilt} \\to \\text{shift } k_c", font_size=30, color=C_REF).move_to(KC + 1.45 * DOWN + 0.5 * kcf * LEFT + 0.38 * DOWN)
        L_c = MathTex("|U|^2+|R|^2", font_size=26, color=GREY_A)
        L_l = MathTex("U", "R^*", font_size=32)
        L_l[0].set_color(C_U)
        L_l[1].set_color(C_REF)
        L_r = MathTex("U^*", "R", font_size=32)
        L_r[0].set_color(C_TWIN)
        L_r[1].set_color(C_REF)
        L_c.add_updater(lambda m: m.move_to(KC + (2 * bb + 0.32) * UP))
        L_l.add_updater(lambda m: m.move_to(KC + kcf * tilt.get_value() * LEFT + (bb + 0.32) * UP))
        L_r.add_updater(lambda m: m.move_to(KC + kcf * tilt.get_value() * RIGHT + (bb + 0.32) * UP))
        self.add(sh)
        self.at(S("r7") + 0.4)
        self.play(sh_on.animate.set_value(1.0), FadeIn(shl), run_time=0.8)
        self.at(S("r7") + 2.6)
        self.play(tilt.animate.set_value(0.5), run_time=1.2)
        self.add(L_l, L_r, L_c)
        self.play(tilt.animate.set_value(1.0), FadeIn(L_l), FadeIn(L_c), run_time=1.4)
        self.at(S("r7") + 6.4)
        self.play(FadeIn(L_r), run_time=0.5)
        self.finish()


# ============================================================ 3  three islands ======
class Ch3Islands(Chapter):
    CH = 3
    TAG = "3\\quad Three islands"

    def construct(self):
        S = self.S
        self.show_tag()
        h = npz("holo.npz")
        MF, SH, BB = int(h["m_final"]), int(h["spec_half"]), float(h["band_bins"])
        HC = np.array([-4.55, -0.25, 0])
        him, hfd = self.static_image(grey_u8(h["holo"]), NATIVE, HC)
        hbox = box(HC, NATIVE)
        SC, SW = np.array([3.7, -0.25, 0]), 4.8
        u = SW / (2 * SH)                                        # scene units per FFT bin
        spec_rgba = heat_u8(h["spec"], 0.85)
        sim, sfd = self.static_image(spec_rgba, SW, SC)
        isl = spec_rgba.copy()
        isl[..., 3] = (h["island_alpha"] * 255).astype(np.uint8)
        iim, ifd = self.static_image(isl, SW, SC)
        sbox = box(SC, SW)
        CL, CR = SC + MF * u * LEFT, SC + MF * u * RIGHT
        rB = BB * u

        # ---------------- s1: hologram, zoom on the fringes, FFT
        self.at(S("s1"))
        self.add(him)
        self.play(hfd.animate.set_value(1.0), FadeIn(hbox), run_time=0.8)
        r0, c0, zn = 218, 196, 80                                # zoom window (pixels) on a cell
        zc = HC + np.array([-NATIVE / 2 + (c0 + zn / 2) / NPIX * NATIVE, NATIVE / 2 - (r0 + zn / 2) / NPIX * NATIVE, 0])
        zmark = Square(zn / NPIX * NATIVE).move_to(zc).set_stroke(YELLOW, 2.5)
        ZC, ZW = np.array([-0.55, 0.65, 0]), 2.6
        zim, zfd = self.static_image(grey_u8(h["holo"][r0:r0 + zn, c0:c0 + zn]), ZW, ZC, nearest=True)
        zbox = box(ZC, ZW, color=YELLOW, width=2.5)
        zl = VGroup(Line(zmark.get_corner(UR), zbox.get_corner(UL)), Line(zmark.get_corner(DR), zbox.get_corner(DL))).set_stroke(YELLOW, 1.5, 0.7)
        zt = Tex("zoom: fine fringes", font_size=26).next_to(zbox, UP, buff=0.12)
        self.add(zim)
        self.at(S("s1") + 0.9)
        self.play(Create(zmark), Create(zl), zfd.animate.set_value(1.0), Create(zbox), FadeIn(zt), run_time=0.8)
        fft = Arrow([-2.5, -0.25, 0], [1.15, -0.25, 0], buff=0, stroke_width=5, color=GREY_A)
        fftl = Tex("Fourier transform (FFT)", font_size=28).next_to(fft, UP, buff=0.12)
        self.at(S("s1") + 3.0)
        self.play(FadeOut(VGroup(zmark, zl, zbox, zt)), zfd.animate.set_value(0.0), run_time=0.5)
        self.remove(zim)
        self.add(sim)
        self.play(GrowArrow(fft), FadeIn(fftl), sfd.animate.set_value(1.0), FadeIn(sbox), run_time=1.3)

        # ---------------- s2: name the islands
        lc = MathTex("|U|^2+|R|^2", font_size=30, color=GREY_A).move_to(SC + (2 * rB + 0.32) * UP)
        ll = MathTex("U", "R^*", font_size=34).move_to(CL + (rB + 0.35) * UP)
        ll[0].set_color(C_U)
        ll[1].set_color(C_REF)
        lr = VGroup(MathTex("U^*", "R", font_size=34), Tex("(twin)", font_size=24, color=C_TWIN)).arrange(DOWN, buff=0.06)
        lr[0][0].set_color(C_TWIN)
        lr[0][1].set_color(C_REF)
        lr.move_to(CR + (rB + 0.5) * UP)
        self.at(S("s2") + 0.3)
        self.play(FadeIn(lc, shift=0.1 * DOWN), run_time=0.6)
        self.at(S("s2") + 2.9)
        self.play(*[ShowPassingFlash(Circle(radius=1.3 * rB).move_to(c_).set_stroke(YELLOW, 5), time_width=0.5) for c_ in (CL, CR)],
                  run_time=1.3)
        self.at(S("s2") + 5.0)
        self.play(FadeIn(ll, shift=0.1 * DOWN), run_time=0.5)
        self.at(S("s2") + 6.1)
        self.play(FadeIn(lr, shift=0.1 * DOWN), run_time=0.5)

        # ---------------- s3: they must not overlap -> k_c >= 3B
        circ = VGroup(DashedVMobject(Circle(radius=2 * rB).move_to(SC), num_dashes=36).set_stroke(GREY_A, 2),
                      DashedVMobject(Circle(radius=rB).move_to(CL), num_dashes=22).set_stroke(C_KX, 2),
                      DashedVMobject(Circle(radius=rB).move_to(CR), num_dashes=22).set_stroke(C_KX, 2))
        rl1 = Line(CL, CL + rB * DOWN).set_stroke(C_KX, 3)
        rl1l = MathTex("B", font_size=30, color=C_KX).next_to(rl1, RIGHT, buff=0.08)
        rl2 = Line(SC, SC + 2 * rB * DOWN).set_stroke(GREY_A, 3)
        rl2l = MathTex("2B", font_size=30, color=GREY_A).next_to(rl2, RIGHT, buff=0.08)
        kca = DoubleArrow(CL + 1.55 * DOWN, SC + 1.55 * DOWN, buff=0, stroke_width=3, color=C_REF, tip_length=0.16,
                          max_tip_length_to_length_ratio=0.2)
        kcl = MathTex("k_c", font_size=32, color=C_REF).next_to(kca, DOWN, buff=0.08)
        rule = MathTex("k_c", "\\;\\ge\\;", "3B", font_size=44)
        rule[0].set_color(C_REF)
        rule[2].set_color(C_KX)
        rule = VGroup(SurroundingRectangle(rule, buff=0.14, color=YELLOW, corner_radius=0.08), rule).move_to(SC + 1.85 * UP)
        self.at(S("s3") + 0.2)
        self.play(Create(circ), run_time=0.9)
        self.at(S("s3") + 1.5)
        self.play(Create(rl1), FadeIn(rl1l), Create(rl2), FadeIn(rl2l), run_time=0.8)
        self.at(S("s3") + 3.6)
        self.play(GrowFromCenter(kca), FadeIn(kcl), run_time=0.7)
        self.at(S("s3") + 5.2)
        self.play(FadeIn(rule, scale=0.9), run_time=0.6)

        # ---------------- s4: crop one island, re-centre it, transform back
        crop = Circle(radius=1.04 * rB).move_to(CL).set_stroke(C_H, 4)
        self.at(S("s4"))
        self.play(FadeOut(VGroup(circ, rl1, rl1l, rl2, rl2l, kca, kcl, rule, lc, lr)), Create(crop), run_time=0.6)
        self.add(iim)
        self.play(sfd.animate.set_value(0.15), ifd.animate.set_value(1.0), run_time=0.7)
        self.at(S("s4") + 1.5)
        self.play(iim.animate.shift(MF * u * RIGHT), crop.animate.shift(MF * u * RIGHT), ll.animate.shift(MF * u * RIGHT),
                  run_time=1.0)
        ifft = Arrow([1.15, -0.25, 0], [-2.5, -0.25, 0], buff=0, stroke_width=5, color=GREY_A)
        ifftl = Tex("inverse FFT", font_size=28).next_to(ifft, UP, buff=0.12)
        rec = npz("holo.npz")["refocus"]
        zs = h["zs"]
        zt_ = ValueTracker(0.0)
        dz = float(zs[1] - zs[0])
        rfn = memo(lambda z: rgba(blend(rec, z / dz)))
        fim, ffd = self.live_image(lambda t: rfn(zt_.get_value()), NATIVE, HC)
        self.at(S("s4") + 2.6)
        self.play(FadeOut(fft), FadeOut(fftl), run_time=0.3)
        self.play(GrowArrow(ifft), FadeIn(ifftl), run_time=0.5)
        self.add(fim)
        self.play(hfd.animate.set_value(0.0), ffd.animate.set_value(1.0), run_time=0.8)
        self.remove(him)

        # ---------------- s5: the full complex field
        wheel = colour_wheel([-0.65, 1.45, 0], 1.2)
        wl = VGroup(Tex("colour = phase", font_size=28), Tex("brightness = amplitude", font_size=28)).arrange(DOWN, buff=0.12)
        wl.next_to(wheel, DOWN, buff=0.2)
        self.at(S("s5") + 0.2)
        self.play(FadeOut(VGroup(ifft, ifftl, crop, ll, sbox)), sfd.animate.set_value(0.0), ifd.animate.set_value(0.0), run_time=0.6)
        self.remove(sim, iim)
        self.at(S("s5") + 2.1)
        self.play(FadeIn(wheel), FadeIn(wl[1]), run_time=0.6)
        self.at(S("s5") + 3.7)
        self.play(FadeIn(wl[0]), run_time=0.5)

        # ---------------- s6: recorded out of focus; propagate with the ASM (episode 1)
        FC, FW = np.array([-2.1, -0.3, 0]), 4.8
        cap_a = Tex("camera plane: out of focus", font_size=28, color=GREY_A).move_to([-2.1, 2.45, 0])
        cap_b = Tex("cell plane: in focus", font_size=28, color=C_H).move_to(cap_a)
        self.at(S("s6") + 0.2)
        wl2 = Tex("phase", font_size=26, color=GREY_A).move_to([5.75, 1.55, 0])
        self.play(fim.animate.scale(FW / NATIVE).move_to(FC), hbox.animate.scale(FW / NATIVE).move_to(FC),
                  wheel.animate.scale(0.8).move_to([5.75, 2.25, 0]), FadeTransform(wl, wl2), run_time=1.0)
        self.play(FadeIn(cap_a), run_time=0.5)
        ZX, ZT, ZB = 1.55, 2.0, -2.45
        zmax = float(zs[-1])
        zrec = float(h["zrec"])
        zy = lambda z: ZT + (ZB - ZT) * z / zmax
        zax = Line([ZX, ZT, 0], [ZX, ZB, 0]).set_stroke(GREY_B, 2)
        tk = VGroup(Line([ZX - 0.12, zy(0), 0], [ZX + 0.12, zy(0), 0]).set_stroke(GREY_B, 2),
                    Line([ZX - 0.12, zy(zrec), 0], [ZX + 0.12, zy(zrec), 0]).set_stroke(C_H, 3))
        tkl = VGroup(Tex("camera", font_size=26, color=GREY_A).next_to(tk[0], RIGHT, buff=0.12),
                     Tex("cells", font_size=26, color=C_H).next_to(tk[1], RIGHT, buff=0.12))
        zlab = MathTex("\\text{propagate back by } z", font_size=28).move_to([ZX + 0.5, ZT + 0.4, 0])

        def marker():
            y = zy(zt_.get_value())
            return VGroup(Line([ZX - 0.22, y, 0], [ZX + 0.22, y, 0]).set_stroke(YELLOW, 4), Dot([ZX, y, 0], radius=0.08, color=YELLOW))
        mk = always_redraw(marker)
        asm_l = VGroup(Tex("angular spectrum method", font_size=28, color=GREY_A),
                       MathTex("\\text{FFT}", "\\;\\to\\;", "\\times\\,e^{-i k_z z}", "\\;\\to\\;", "\\text{inverse FFT}", font_size=30))
        asm_l[1][2].set_color(C_H)
        asm_l.arrange(DOWN, buff=0.15).move_to([4.35, -0.35, 0])
        self.at(S("s6") + 3.4)
        self.add(mk)
        self.play(Create(zax), FadeIn(tk), FadeIn(tkl), FadeIn(zlab), FadeIn(mk), run_time=0.8)
        self.at(S("s6") + 4.6)
        self.play(FadeIn(asm_l, shift=0.1 * UP), run_time=0.8)

        # ---------------- s7: sweep z -> the cells snap into focus
        self.at(S("s7") + 0.2)
        self.play(zt_.animate.set_value(zrec + 16), run_time=2.3, rate_func=smooth)
        self.play(zt_.animate.set_value(zrec), FadeTransform(cap_a, cap_b), run_time=0.9, rate_func=smooth)
        self.play(Indicate(tkl[1], color=C_H), run_time=0.8)
        self.finish()


# ============================================================ 4  phase retrieval ====
class Ch4Retrieval(Chapter):
    CH = 4
    TAG = "4\\quad Phase retrieval"

    def construct(self):
        S = self.S
        self.show_tag()
        g = npz("gs.npz")
        PS = 3.2
        OBJ, FAR = np.array([-4.35, 0.5, 0]), np.array([4.35, 0.5, 0])
        W = g["window"]
        win_rgba = rgba(np.repeat(W[..., None], 3, -1))
        truth = rgba(g["truth"])
        snaps, snap_it, err = g["snaps"], g["snap_it"], g["err"]
        t_im, t_fd = self.static_image(truth, PS, OBJ)
        w_im, w_fd = self.static_image(win_rgba, PS, OBJ)
        gi = ValueTracker(0.0)
        gfn = memo(lambda f: rgba(blend(snaps, f)))
        e_im, e_fd = self.live_image(lambda t: gfn(gi.get_value()), PS, OBJ)
        f_im, f_fd = self.static_image(heat_u8(g["far"], 0.8), PS, FAR)
        ob, fb = box(OBJ, PS), box(FAR, PS)
        t_obj = Tex("object plane", font_size=30).next_to(ob, UP, buff=0.18)
        t_far = Tex("far field (lens focus)", font_size=30).next_to(fb, UP, buff=0.18)

        # ---------------- g1: no reference -> what do we know?
        qm = MathTex("?", font_size=110, color=YELLOW).move_to(OBJ)
        self.at(S("g1"))
        self.add(t_im)
        self.play(t_fd.animate.set_value(0.45), Create(ob), FadeIn(t_obj), FadeIn(qm, scale=0.7), run_time=0.8)

        # ---------------- g2: a window of known shape, a transparent sample -> |U| known
        outline = window_outline(OBJ, PS, WHITE, 3)
        kn = VGroup(Tex("known: ", "lengths", font_size=28), Tex("(1 inside, 0 outside)", font_size=24, color=GREY_B)).arrange(DOWN, buff=0.08)
        kn[0][1].set_color(C_U)
        kn.next_to(ob, DOWN, buff=0.15)
        self.at(S("g2") + 1.2)
        self.add(w_im)
        self.play(FadeOut(qm), t_fd.animate.set_value(0.0), w_fd.animate.set_value(1.0), Create(outline), run_time=1.0)
        self.remove(t_im)
        self.at(S("g2") + 5.6)
        self.play(FadeIn(kn, shift=0.1 * UP), run_time=0.7)

        # ---------------- g3: a lens shows the far field; a camera measures its lengths
        lens = Ellipse(width=0.42, height=2.6).move_to([0, 0.5, 0]).set_stroke(GREY_A, 2).set_fill(BLUE_E, 0.35)
        lensl = Tex("lens", font_size=26, color=GREY_A).next_to(lens, DOWN, buff=0.1)
        rays = VGroup(*[Line([-2.6, 0.5 + y, 0], [0, 0.5 + y, 0]).set_stroke(C_U, 2, 0.6) for y in (-0.8, 0, 0.8)],
                      *[Line([0, 0.5 + y, 0], [2.6, 0.5 - 0.3 * y, 0]).set_stroke(C_U, 2, 0.6) for y in (-0.8, 0, 0.8)])
        ms = VGroup(Tex("measured: ", "lengths", font_size=28), Tex("(camera at the focus)", font_size=24, color=GREY_B)).arrange(DOWN, buff=0.08)
        ms[0][1].set_color(C_KX)
        ms.next_to(fb, DOWN, buff=0.15)
        self.at(S("g3"))
        self.add(f_im)
        self.play(FadeIn(lens), FadeIn(lensl), Create(rays), run_time=0.8)
        self.play(f_fd.animate.set_value(1.0), Create(fb), FadeIn(t_far), run_time=0.8)
        self.at(S("g3") + 2.4)
        self.play(FadeIn(ms, shift=0.1 * UP), run_time=0.6)
        ang = Tex("angles: unknown in both planes", font_size=30, color=YELLOW).move_to([0, -2.55, 0])
        self.at(S("g3") + 5.4)
        self.play(FadeIn(ang), run_time=0.6)

        # ---------------- g4: Gerchberg-Saxton: random angles, back and forth with FFTs
        gs_t = Tex("Gerchberg--Saxton (1972)", font_size=34).move_to([0.9, 3.05, 0])
        top = Arrow([-2.55, 1.6, 0], [2.55, 1.6, 0], buff=0, stroke_width=4, color=GREY_A)
        topl = Tex("FFT", font_size=28).next_to(top, UP, buff=0.08)
        bot = Arrow([2.55, -0.6, 0], [-2.55, -0.6, 0], buff=0, stroke_width=4, color=GREY_A)
        botl = Tex("inverse FFT", font_size=28).next_to(bot, DOWN, buff=0.08)
        self.at(S("g4"))
        self.play(FadeIn(gs_t), FadeOut(VGroup(lens, lensl, rays, ang)), run_time=0.6)
        self.add(e_im)
        self.at(S("g4") + 2.1)
        self.play(w_fd.animate.set_value(0.0), e_fd.animate.set_value(1.0), FadeOut(outline), run_time=0.8)
        self.remove(w_im)
        rnd = Tex("random angles", font_size=26, color=GREY_A).next_to(ob, DOWN, buff=0.15)
        self.play(FadeOut(kn), FadeIn(rnd), run_time=0.4)
        self.at(S("g4") + 3.7)
        self.play(GrowArrow(top), FadeIn(topl), GrowArrow(bot), FadeIn(botl), run_time=0.9)

        # ---------------- g5: keep the angles, swap in the known lengths (a dot runs round the loop)
        loop_t = ValueTracker(0.0)

        def loop_pt(s):
            s = s % 1.0
            pts = [np.array([-2.55, 1.6, 0]), np.array([2.55, 1.6, 0]), np.array([2.55, -0.6, 0]), np.array([-2.55, -0.6, 0])]
            seg = [5.1, 2.2, 5.1, 2.2]
            L = sum(seg)
            d_ = s * L
            for i in range(4):
                if d_ <= seg[i]:
                    return pts[i] + (pts[(i + 1) % 4] - pts[i]) * d_ / seg[i]
                d_ -= seg[i]
            return pts[0]
        dot = always_redraw(lambda: Dot(loop_pt(loop_t.get_value()), radius=0.1, color=YELLOW).set_opacity(
            min(1.0, 10 * min(loop_t.get_value(), 1.0 - loop_t.get_value())) if 0 < loop_t.get_value() < 1.0 else 0.0))
        k_far = Tex("keep angles,\\\\lengths $\\to$ measured", font_size=26, color=C_KX).next_to(fb, DOWN, buff=0.15)
        k_obj = Tex("keep angles,\\\\lengths $\\to$ 1 in the window", font_size=26, color=C_U).next_to(ob, DOWN, buff=0.15)
        self.add(dot)
        self.at(S("g5") + 0.1)
        self.play(loop_t.animate.set_value(0.36), FadeTransform(ms, k_far), run_time=1.25, rate_func=linear)
        self.play(loop_t.animate.set_value(0.86), FadeTransform(rnd, k_obj), run_time=1.75, rate_func=linear)
        self.play(loop_t.animate.set_value(1.0), run_time=0.5, rate_func=linear)

        # ---------------- g6: bouncing between two constraints (schematic)
        Oc, Rc = np.array([-0.25, -3.1, 0]), 4.0                   # constraint 1: an arc (non-convex, like |U| = 1)
        xs_ = -1.0
        pstar = np.array([Oc[0] + xs_, Oc[1] + np.sqrt(Rc ** 2 - xs_ ** 2), 0])
        lang = np.radians(12)                                      # constraint 2: a line through the answer
        ld = np.array([np.cos(lang), np.sin(lang), 0])
        a0, a1 = np.arccos(2.35 / Rc), np.arccos(-2.35 / Rc)
        arc = Arc(radius=Rc, start_angle=a0, angle=a1 - a0, arc_center=Oc).set_stroke(C_U, 4)
        lin = Line(pstar - 1.4 * ld, pstar + 3.45 * ld).set_stroke(C_KX, 4)
        x = Oc + Rc * np.array([np.cos(np.arccos(2.1 / Rc)), np.sin(np.arccos(2.1 / Rc)), 0])
        pts = [x]
        for _ in range(7):
            y = pstar + np.dot(x - pstar, ld) * ld
            pts.append(y)
            x = Oc + Rc * (y - Oc) / np.linalg.norm(y - Oc)
            pts.append(x)
        nseg = ValueTracker(0.0)

        def zig():
            n = nseg.get_value()
            gz = VGroup()
            if n < 0.005:
                return gz
            gz.add(Dot(pts[0], radius=0.07, color=WHITE).set_opacity(min(1.0, 10 * n)))
            for i in range(len(pts) - 1):
                f = float(np.clip(n - i, 0, 1))
                if f <= 0:
                    break
                q = pts[i] + f * (pts[i + 1] - pts[i])
                gz.add(Line(pts[i], q).set_stroke(WHITE, 2.5), Dot(q, radius=0.055, color=WHITE))
            return gz
        zz = always_redraw(zig)
        l_arc = Tex("known window", font_size=28, color=C_U).move_to([1.25, -0.3, 0])
        l_lin = Tex("measured far field", font_size=28, color=C_KX).move_to([0.9, 1.9, 0])
        ans = VGroup(Dot(pstar, radius=0.1, color=C_H), Tex("answer", font_size=28, color=C_H).move_to(pstar + np.array([-0.55, 0.42, 0])))
        sch = Tex("(schematic)", font_size=24, color=GREY_B).move_to([-0.25, -1.1, 0])
        self.at(S("g6"))
        dot.clear_updaters()
        self.play(FadeOut(VGroup(top, topl, bot, botl, dot)), run_time=0.4)
        self.remove(dot)
        self.play(Create(arc), Create(lin), FadeIn(l_arc), FadeIn(l_lin), FadeIn(sch), run_time=0.9)
        self.add(zz)
        self.play(nseg.animate.set_value(len(pts) - 1), run_time=3.6, rate_func=rate_functions.ease_out_sine)
        self.play(FadeIn(ans), run_time=0.4)

        # ---------------- g7: the computed run: the error falls, the cells emerge
        EX0, EX1, EY0, EY1 = -1.7, 2.4, -1.55, 1.75
        it_max = float(snap_it[-1])
        lerr = np.log10(err)
        ey = lambda v: EY0 + (v - (-4.5)) / (0.5 - (-4.5)) * (EY1 - EY0)
        eax = VGroup(Line([EX0, EY0, 0], [EX1, EY0, 0]), Line([EX0, EY0, 0], [EX0, EY1, 0])).set_stroke(GREY_B, 2)
        exl = Tex("iteration", font_size=26, color=GREY_A).next_to(eax[0], DOWN, buff=0.1)
        eyl = Tex("error (log)", font_size=26, color=GREY_A).next_to(eax[1], UP, buff=0.1)
        etk = VGroup(*[VGroup(Line([EX0 - 0.08, ey(v), 0], [EX0, ey(v), 0]).set_stroke(GREY_B, 2),
                              MathTex(f"10^{{{v}}}", font_size=24, color=GREY_B).next_to([EX0 - 0.08, ey(v), 0], LEFT, buff=0.06))
                       for v in (0, -2, -4)])

        def ecurve():
            itc = float(np.interp(gi.get_value(), np.arange(len(snap_it)), snap_it))
            n = int(itc) + 1
            if n < 2:
                return VGroup()
            xs_ = EX0 + np.arange(n) / it_max * (EX1 - EX0)
            ys_ = ey(lerr[:n])
            return VGroup(poly_line(xs_, ys_, C_EV, 3.5), Dot([xs_[-1], ys_[-1], 0], radius=0.06, color=C_EV))
        ec = always_redraw(ecurve)
        itl = [MathTex(f"\\text{{iteration }}{int(i)}", font_size=26, color=GREY_A) for i in snap_it]
        for m in itl:
            m.next_to(ob, DOWN, buff=0.15)
        it_op = ValueTracker(0.0)
        itm = always_redraw(lambda: itl[int(round(gi.get_value()))].copy().set_opacity(it_op.get_value()))
        self.at(S("g7") - 0.35)
        self.play(FadeOut(VGroup(arc, lin, l_arc, l_lin, sch, ans, k_obj, k_far)), FadeOut(zz), FadeIn(eax), FadeIn(exl), FadeIn(eyl), FadeIn(etk),
                  run_time=0.5)
        self.remove(zz)
        self.add(ec, itm)
        self.play(gi.animate.set_value(len(snap_it) - 1), it_op.animate.set_value(1.0), run_time=3.7, rate_func=rate_functions.ease_in_out_sine)

        # ---------------- g8: blind spots: shifted copy, flipped twin, constant phase -> identical far field
        bl = g["blind"]
        sh = g["shift"]
        BS, BY = 2.5, 1.05
        bx = [-5.1, -1.7, 1.7, 5.1]
        names = ["found", "shifted copy", "flipped twin", "constant phase"]
        bims = [self.static_image(rgba(bl[0] if i else snaps[-1]), BS, [bx[i], BY, 0]) for i in range(4)]
        btit = VGroup(*[Tex(n, font_size=28).move_to([bx[i], BY + BS / 2 + 0.3, 0]) for i, n in enumerate(names)])
        for i in range(1, 4):
            bims[i] = self.static_image(rgba(bl[i]), BS, [bx[i], BY, 0])
        bbx = VGroup(*[box([x, BY, 0], BS) for x in bx])
        FS, FY = 1.25, -1.2
        fims = [self.static_image(heat_u8(g["far"], 0.8), FS, [x, FY, 0]) for x in bx]
        eqs = VGroup(*[MathTex("=", font_size=44).move_to([0.5 * (bx[i] + bx[i + 1]), FY, 0]) for i in range(3)])
        same = Tex("the same far-field brightness", font_size=28, color=C_KX).move_to([0, FY - FS / 2 - 0.35, 0])
        self.at(S("g8"))
        e_im.clear_updaters()
        self.play(FadeOut(VGroup(eax, exl, eyl, etk, ec, itm, gs_t, t_obj, t_far, fb, ob)), f_fd.animate.set_value(0.0),
                  e_im.animate.scale(BS / PS).move_to([bx[0], BY, 0]), run_time=0.8)
        self.remove(ec, itm, f_im)
        self.add(bims[0][0])
        bims[0][1].set_value(1.0)
        self.remove(e_im)
        self.add(fims[0][0])
        self.play(FadeIn(btit[0]), Create(bbx[0]), fims[0][1].animate.set_value(1.0), run_time=0.6)
        for i, tt in ((1, 2.8), (2, 3.9), (3, 5.0)):
            self.at(S("g8") + tt)
            self.add(bims[i][0], fims[i][0])
            self.play(bims[i][1].animate.set_value(1.0), FadeIn(btit[i]), Create(bbx[i]), fims[i][1].animate.set_value(1.0),
                      FadeIn(eqs[i - 1]), run_time=0.6)
        self.at(S("g8") + 6.2)
        self.play(FadeIn(same), run_time=0.6)

        # ---------------- g9: the lopsided window rules out two; diverse data helps
        kw = VGroup(*[window_outline([bx[i], BY, 0], BS, C_H, 3, dashed=True) for i in (1, 2)])
        xs_ = VGroup(*[Cross(scale_factor=0.35, stroke_color=C_EV, stroke_width=8).move_to([bx[i] + BS / 2 - 0.3, BY + BS / 2 - 0.3, 0])
                       for i in (1, 2)])
        kwl = Tex("dashed: the known window", font_size=26, color=C_H).move_to([0, -2.75, 0])
        harm = Tex("harmless", font_size=26, color=GREY_A).move_to([bx[3], -2.75, 0])
        self.at(S("g9") + 0.6)
        self.play(Create(kw), FadeIn(kwl), run_time=0.8)
        self.play(FadeIn(xs_, scale=0.6), FadeIn(harm), run_time=0.5)
        # diverse data: several distances, overlapping scans (schematic icons)
        icon1 = VGroup(*[Polygon([-0.65, -0.22, 0], [0.45, -0.22, 0], [0.65, 0.22, 0], [-0.45, 0.22, 0]).move_to([-3.2, -1.5 + 0.38 * k, 0])
                         for k in range(3)])
        icon1.set_stroke(C_U, 2).set_fill(C_U, 0.15)
        i1l = Tex("several distances", font_size=28).next_to(icon1, RIGHT, buff=0.3)
        icon2 = VGroup(*[Circle(radius=0.42).move_to([1.6 + 0.42 * k, -1.2 - 0.2 * (k % 2), 0]) for k in range(4)]).set_stroke(C_KX, 2.5)
        i2l = Tex("overlapping scans (ptychography)", font_size=28).next_to(icon2, RIGHT, buff=0.3)
        g_icons = VGroup(icon1, i1l, icon2, i2l)
        g_icons.move_to([0, -1.6, 0])
        self.at(S("g9") + 3.0)
        self.play(FadeOut(VGroup(eqs, same, kwl)), harm.animate.move_to([bx[3], BY - BS / 2 - 0.3, 0]),
                  *[f[1].animate.set_value(0.0) for f in fims], run_time=0.5)
        for f in fims:
            self.remove(f[0])
        self.play(FadeIn(icon1), FadeIn(i1l), run_time=0.6)
        self.at(S("g9") + 6.1)
        self.play(FadeIn(icon2), FadeIn(i2l), run_time=0.6)
        self.finish()


# ============================================================ 5  designing a hologram
class Ch5Design(Chapter):
    CH = 5
    TAG = "5\\quad Run it backwards: design"

    def construct(self):
        S = self.S
        self.show_tag()
        c = npz("cgh.npz")
        PS = 3.5
        MC, FC = np.array([-3.75, 0.3, 0]), np.array([3.75, 0.3, 0])
        masks, fars, snap_it, eff = c["masks"], c["fars"], c["snap_it"], c["eff"]
        ci = ValueTracker(0.0)
        mfn = memo(lambda f: rgba(blend(masks, f)))
        ffn = memo(lambda f: heat_u8(blend(fars, f)))
        m_im, m_fd = self.live_image(lambda t: mfn(ci.get_value()), PS, MC, nearest=True)
        f_im, f_fd = self.live_image(lambda t: ffn(ci.get_value()), PS, FC)
        t_im, t_fd = self.static_image(grey_u8(c["target"]), PS, FC)
        mb, fb = box(MC, PS), box(FC, PS)
        t_mask = Tex("phase-only mask", font_size=30).next_to(mb, UP, buff=0.18)
        t_tgt = Tex("target far field", font_size=30).next_to(fb, UP, buff=0.18)
        t_far = Tex("far field of the mask", font_size=30).next_to(fb, UP, buff=0.18)

        # ---------------- d1: ask for a far field
        qm = MathTex("?", font_size=110, color=YELLOW).move_to(MC)
        fwd = Arrow([-1.75, 0.3, 0], [1.75, 0.3, 0], buff=0, stroke_width=4, color=GREY_A)
        fwdl = Tex("far field", font_size=28).next_to(fwd, UP, buff=0.08)
        self.at(S("d1") + 0.1)
        self.play(Create(mb), Create(fb), FadeIn(t_mask), FadeIn(t_tgt), GrowArrow(fwd), FadeIn(fwdl), run_time=0.9)
        self.play(FadeIn(qm, scale=0.7), run_time=0.5)
        self.at(S("d1") + 2.9)
        self.add(t_im)
        self.play(t_fd.animate.set_value(1.0), run_time=0.9)

        # ---------------- d2: a phase-only mask; same loop
        dev = Tex("liquid-crystal screen,\\\\or etched glass", font_size=26, color=GREY_A).next_to(mb, DOWN, buff=0.15)
        self.at(S("d2"))
        self.add(m_im)
        self.play(m_fd.animate.set_value(1.0), FadeOut(qm), run_time=0.9)
        self.at(S("d2") + 2.1)
        self.play(FadeIn(dev), run_time=0.6)
        top = Arrow([-1.8, 1.25, 0], [1.8, 1.25, 0], buff=0, stroke_width=4, color=GREY_A)
        topl = Tex("FFT", font_size=28).next_to(top, UP, buff=0.08)
        bot = Arrow([1.8, -0.65, 0], [-1.8, -0.65, 0], buff=0, stroke_width=4, color=GREY_A)
        botl = Tex("inverse FFT", font_size=28).next_to(bot, DOWN, buff=0.08)
        k_m = Tex("keep angles,\\\\lengths $\\to$ 1", font_size=26, color=C_U).next_to(mb, DOWN, buff=0.15)
        k_f = Tex("keep angles,\\\\lengths $\\to$ target", font_size=26, color=C_KX).next_to(fb, DOWN, buff=0.15)
        self.at(S("d2") + 4.9)
        self.play(FadeOut(fwd), FadeOut(fwdl), GrowArrow(top), FadeIn(topl), GrowArrow(bot), FadeIn(botl), run_time=0.8)
        self.at(S("d2") + 6.5)
        self.play(FadeOut(dev), FadeIn(k_m), FadeIn(k_f), run_time=0.6)

        # ---------------- d3: iterate: the mask scrambles, the far field snaps into the lambda
        itl = [MathTex(f"\\text{{iteration }}{int(i)}", font_size=28, color=GREY_A).move_to([0, 0.3, 0]) for i in snap_it]
        itm = always_redraw(lambda: itl[int(round(ci.get_value()))].copy())
        EB0, EBW, EBY = -1.6, 3.2, -1.95
        effl = Tex("light inside the target", font_size=24, color=GREY_A).move_to([0, EBY + 0.35, 0])

        def effbar():
            itc = float(np.interp(ci.get_value(), np.arange(len(snap_it)), snap_it))
            e = float(np.interp(itc, np.arange(len(eff)), eff))
            return VGroup(Rectangle(width=EBW, height=0.22).move_to([0, EBY, 0]).set_stroke(GREY_B, 1.5),
                          Rectangle(width=max(e * EBW, 0.01), height=0.22).move_to([EB0 + 0.5 * e * EBW, EBY, 0]).set_stroke(width=0).set_fill(C_H, 0.9))
        eb = always_redraw(effbar)
        self.at(S("d3"))
        self.add(f_im, itm)
        self.play(t_fd.animate.set_value(0.0), f_fd.animate.set_value(1.0), FadeTransform(t_tgt, t_far), FadeIn(itm), run_time=0.5)
        self.remove(t_im)
        self.add(eb)
        self.play(FadeIn(eb), FadeIn(effl), run_time=0.3)
        self.play(ci.animate.set_value(len(snap_it) - 1), run_time=3.6, rate_func=rate_functions.ease_in_out_sine)

        # ---------------- d4: speckle; the far-field phase is free (random)
        tg = c["target"].astype(float)
        zn = 34
        cs = np.cumsum(np.cumsum(np.pad(tg, ((1, 0), (1, 0))), 0), 1)
        win = cs[zn:, zn:] - cs[:-zn, zn:] - cs[zn:, :-zn] + cs[:-zn, :-zn]
        r0, c0 = np.unravel_index(np.argmax(win), win.shape)
        half = tg.shape[0]
        zcen = FC + np.array([-PS / 2 + (c0 + zn / 2) / half * PS, PS / 2 - (r0 + zn / 2) / half * PS, 0])
        zmark = Square(zn / half * PS).move_to(zcen).set_stroke(YELLOW, 2.5)
        ZS = 2.45
        Z1, Z2 = np.array([-4.55, 0.25, 0]), np.array([-1.55, 0.25, 0])
        sp_im, sp_fd = self.static_image(heat_u8(fars[-1][r0:r0 + zn, c0:c0 + zn]), ZS, Z1, nearest=True)
        fp_im, fp_fd = self.static_image(rgba(c["far_phase"][r0:r0 + zn, c0:c0 + zn]), ZS, Z2, nearest=True)
        zb1, zb2 = box(Z1, ZS, color=YELLOW, width=2.5), box(Z2, ZS, color=YELLOW, width=2.5)
        z1t = Tex("speckle (zoom)", font_size=28).next_to(zb1, UP, buff=0.15)
        z2t = Tex("far-field phase: random", font_size=28).next_to(zb2, UP, buff=0.15)
        z2c = Tex("neighbours interfere at random", font_size=26, color=GREY_A).move_to([-3.05, -1.55, 0])
        self.at(S("d4"))
        itm.clear_updaters()
        self.play(FadeOut(VGroup(top, topl, bot, botl, k_m, k_f, t_mask, mb, itm, effl)), FadeOut(eb), m_fd.animate.set_value(0.0),
                  Create(zmark), run_time=0.6)
        self.remove(m_im, eb, itm)
        self.add(sp_im)
        self.play(sp_fd.animate.set_value(1.0), Create(zb1), FadeIn(z1t), run_time=0.7)
        self.at(S("d4") + 4.4)
        self.add(fp_im)
        self.play(fp_fd.animate.set_value(1.0), Create(zb2), FadeIn(z2t), run_time=0.7)
        self.at(S("d4") + 6.3)
        self.play(FadeIn(z2c), run_time=0.6)

        # ---------------- d5: gradient descent (episode 10)
        GC = np.array([-3.05, 0.1, 0])
        valley = lambda x: 0.28 * x ** 2
        vcurve = poly_line(GC[0] + np.linspace(-2.2, 2.2, 120), GC[1] - 1.0 + valley(np.linspace(-2.2, 2.2, 120)), GREY_A, 3)
        t_ball = S("d5") + 1.3

        def ball():
            tau = float(np.clip((self.now - t_ball) / 2.6, 0, 1))
            x = -1.95 * (1 - smoothstep(tau))
            return Dot([GC[0] + x, GC[1] - 1.0 + valley(x) + 0.13, 0], radius=0.13, color=C_ADJ)
        bl_ = always_redraw(ball)
        gdl = Tex("gradient descent (episode 10)", font_size=28).move_to(GC + 1.05 * UP)
        self.at(S("d5"))
        self.play(FadeOut(VGroup(zb1, zb2, z1t, z2t, z2c, zmark)), sp_fd.animate.set_value(0.0), fp_fd.animate.set_value(0.0), run_time=0.5)
        self.remove(sp_im, fp_im)
        self.add(bl_)
        self.play(Create(vcurve), FadeIn(gdl), FadeIn(bl_), run_time=0.6)
        self.finish()


# ============================================================ 6  on a computer ======
class Ch6Computer(Chapter):
    CH = 6
    TAG = "6\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        c1 = code_card2(["S = fft2(hologram)",
                        "U = ifft2(recentre(S * island))",
                        "U = asm(U, -z)          # refocus (episode 1)"], font_size=24, title="digital holography")
        c2 = code_card2(["for it in range(200):",
                        "    F = fft2(U);  F = A_far * exp(1j*angle(F))",
                        "    U = ifft2(F); U = A_obj * exp(1j*angle(U))"], font_size=24,
                       title="phase retrieval & design (Gerchberg-Saxton)")
        c1.move_to([0.4, 1.55, 0])
        c2.move_to([0.4, -1.3, 0])

        def hl(card, i, color=YELLOW):
            row = card[1][0][i]
            return SurroundingRectangle(row, buff=0.07, color=color, stroke_width=2.5, corner_radius=0.05)
        self.at(S("a1") + 0.1)
        self.play(FadeIn(c1, shift=0.1 * UP), run_time=0.7)
        self.play(FadeIn(c2, shift=0.1 * UP), run_time=0.7)
        h1, h2, h3 = hl(c1, 0), hl(c1, 1), hl(c1, 2)
        self.at(S("a2") + 1.0)
        self.play(Create(h1), run_time=0.5)
        self.at(S("a2") + 2.7)
        self.play(ReplacementTransform(h1, h2), run_time=0.5)
        self.at(S("a2") + 5.3)
        self.play(ReplacementTransform(h2, h3), run_time=0.5)
        g1, g2 = hl(c2, 0, C_H), VGroup(hl(c2, 1, C_H), hl(c2, 2, C_H))
        self.at(S("a3") + 1.5)
        self.play(FadeOut(h3), Create(g1), run_time=0.5)
        self.at(S("a3") + 2.9)
        self.play(ReplacementTransform(g1, g2), run_time=0.5)

        # ---------------- a4: >= 2 pixels per fringe, or the island wraps around (computed)
        h = npz("holo.npz")
        Uc, MF = h["Uc"], int(h["m_final"])
        wrap, m_wrap = h["wrap_spec"], h["m_wrap"]
        mt = ValueTracker(float(MF))
        r0, c0, zn = 12, 8, 24
        zfn = memo(lambda m: grey_u8(hologram(Uc, m, slice(r0, r0 + zn), slice(c0, c0 + zn))))
        PC, PW = np.array([-3.7, 0.35, 0]), 3.6
        p_im, p_fd = self.live_image(lambda t: zfn(mt.get_value()), PW, PC, nearest=True)
        sfn = memo(lambda m: heat_u8(blend(wrap, float(np.interp(m, m_wrap, np.arange(len(m_wrap))))), 0.85))
        SC, SWd = np.array([3.4, 0.35, 0]), NATIVE
        s_im, s_fd = self.live_image(lambda t: sfn(mt.get_value()), SWd, SC)
        pb, sb = box(PC, PW), box(SC, SWd)
        pt = Tex("camera pixels (zoom)", font_size=28).next_to(pb, UP, buff=0.15)
        st = Tex("spectrum: the full camera band", font_size=28).next_to(sb, UP, buff=0.15)
        # pixels-per-fringe gauge
        GX0, GX1, GY = PC[0] - 1.7, PC[0] + 1.7, -2.15
        gv = lambda p: GX0 + (p - 1.2) / (4.6 - 1.2) * (GX1 - GX0)
        gauge = VGroup(Line([GX0, GY, 0], [GX1, GY, 0]).set_stroke(GREY_B, 2),
                       Line([gv(1.2), GY, 0], [gv(2.0), GY, 0]).set_stroke(C_EV, 6))
        for p in (2, 3, 4):
            gauge.add(Line([gv(p), GY - 0.08, 0], [gv(p), GY + 0.08, 0]).set_stroke(GREY_B, 2),
                      MathTex(str(p), font_size=26, color=GREY_B).move_to([gv(p), GY - 0.3, 0]))
        gl = Tex("pixels per fringe", font_size=26, color=GREY_A).move_to([PC[0], GY + 0.42, 0])
        gm = always_redraw(lambda: Triangle(fill_opacity=1).scale(0.1).rotate(PI).set_stroke(width=0).set_fill(
            C_EV if NPIX / mt.get_value() < 2.0 else YELLOW).move_to([gv(np.clip(NPIX / mt.get_value(), 1.2, 4.6)), GY + 0.15, 0]))
        self.at(S("a4"))
        self.play(FadeOut(VGroup(c1, c2, g2)), run_time=0.5)
        self.add(p_im, s_im)
        self.play(p_fd.animate.set_value(1.0), s_fd.animate.set_value(1.0), FadeIn(pb), FadeIn(sb), FadeIn(pt), FadeIn(st), run_time=0.8)
        self.add(gm)
        self.play(FadeIn(gauge), FadeIn(gl), FadeIn(gm), run_time=0.5)
        self.at(S("a4") + 2.2)
        self.play(mt.animate.set_value(float(m_wrap[-1])), run_time=4.6, rate_func=rate_functions.ease_in_sine)
        wl = Tex("wrapped around!", font_size=30, color=C_EV).move_to(SC + (NATIVE / 2 + 0.0) * DOWN + 0.32 * DOWN)
        self.play(FadeIn(wl), run_time=0.4)

        # ---------------- a4b: off-axis holography spends pixels on fringes
        uI = NATIVE / NPIX
        isl_c = SC + MF * uI * LEFT
        used = Circle(radius=float(h["band_bins"]) * uI * 1.05).move_to(isl_c).set_stroke(C_H, 3)
        usedl = Tex("the image uses one island", font_size=26, color=C_H).move_to(SC + (NATIVE / 2 + 0.32) * DOWN)
        self.at(S("a4b"))
        self.play(FadeOut(wl), mt.animate.set_value(float(MF)), run_time=1.6, rate_func=smooth)
        self.play(Create(used), FadeIn(usedl), run_time=0.7)

        # ---------------- a5: GS can stall -> random restarts (computed: 8 starts)
        g = npz("gs.npz")
        stall = g["stall"]
        X0, X1, Y0, Y1 = -4.0, 4.4, -2.3, 1.9
        ny = lambda v: Y0 + (v + 6.0) / (0.5 + 6.0) * (Y1 - Y0)
        ax = VGroup(Line([X0, Y0, 0], [X1, Y0, 0]), Line([X0, Y0, 0], [X0, Y1, 0])).set_stroke(GREY_B, 2)
        axl = VGroup(Tex("iteration", font_size=26, color=GREY_A).next_to([0.5 * (X0 + X1), Y0, 0], DOWN, buff=0.12),
                     Tex("error (log)", font_size=26, color=GREY_A).next_to([X0, Y1, 0], UP, buff=0.12))
        for v in (0, -3, -6):
            axl.add(MathTex(f"10^{{{v}}}", font_size=24, color=GREY_B).next_to([X0, ny(v), 0], LEFT, buff=0.1))
        ttl = Tex("the same problem from 8 random starts (computed)", font_size=28).move_to([0.2, 2.45, 0])
        rv = ValueTracker(0.0)
        nit = stall.shape[1]

        def curves():
            n = int(rv.get_value() * (nit - 1)) + 1
            gg = VGroup()
            if n < 2:
                return gg
            xs = X0 + np.arange(n) / (nit - 1) * (X1 - X0)
            for e in stall:
                col = C_H if e[-1] < 0.01 else (C_EV if e[-1] > 0.05 else GOLD)
                gg.add(poly_line(xs, ny(np.log10(np.maximum(e[:n], 1e-6))), col, 3))
            return gg
        cv = always_redraw(curves)
        sl = Tex("stalled", font_size=30, color=C_EV).move_to([X1 - 0.6, ny(np.log10(0.13)) + 0.35, 0])
        cl = Tex("converged: restart until one does", font_size=28, color=C_H).move_to([-1.6, ny(-5.0), 0])
        self.at(S("a5"))
        self.play(FadeOut(VGroup(pb, sb, pt, st, gauge, gl, used, usedl)), FadeOut(gm), p_fd.animate.set_value(0.0), s_fd.animate.set_value(0.0),
                  run_time=0.5)
        self.remove(p_im, s_im, gm)
        self.add(cv)
        self.play(FadeIn(ax), FadeIn(axl), FadeIn(ttl), run_time=0.4)
        self.play(rv.animate.set_value(1.0), run_time=2.6, rate_func=linear)
        self.play(FadeIn(sl), FadeIn(cl), run_time=0.5)
        self.finish()


# ============================================================ 7  recap =============
class Ch7Recap(Chapter):
    CH = 7
    TAG = "7\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        cells = npz("cells.npz")
        h = npz("holo.npz")
        g = npz("gs.npz")
        c = npz("cgh.npz")
        TS, TY = 2.5, 1.0
        tx = [-5.1, -1.7, 1.7, 5.1]
        imgs = [grey_u8(cells["camera"]), grey_u8(h["holo"][214:278, 150:214]), rgba(g["snaps"][-1]), heat_u8(c["fars"][-1])]
        ims = [self.static_image(im, TS, [x, TY, 0], nearest=(i == 1)) for i, (im, x) in enumerate(zip(imgs, tx))]
        bxs = VGroup(*[box([x, TY, 0], TS) for x in tx])
        caps = [Tex("cameras lose phase", font_size=26), Tex("fringes record it", font_size=26),
                Tex("constraints recover it", font_size=26), Tex("\\dots or design it", font_size=26)]
        for cp, x in zip(caps, tx):
            cp.move_to([x, TY - TS / 2 - 0.3, 0])
        arrs = VGroup(*[Arrow([tx[i] + TS / 2 + 0.05, TY, 0], [tx[i + 1] - TS / 2 - 0.05, TY, 0], buff=0, stroke_width=3,
                              color=GREY_B, max_tip_length_to_length_ratio=0.35) for i in range(3)])
        for i, tt in enumerate((0.1, 1.8, 4.0, 6.0)):
            self.at(S("e1") + tt)
            self.add(ims[i][0])
            anims = [ims[i][1].animate.set_value(1.0), Create(bxs[i]), FadeIn(caps[i])]
            if i:
                anims.append(GrowArrow(arrs[i - 1]))
            self.play(*anims, run_time=0.6)

        # e2: where it breaks
        br = VGroup(Tex("holography needs steady, coherent light (episode 4)", font_size=30),
                    Tex("retrieval needs diverse data", font_size=30)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        br.move_to([0, -1.55, 0])
        self.at(S("e2") + 1.2)
        self.play(FadeIn(br[0], shift=0.1 * UP), run_time=0.6)
        self.at(S("e2") + 3.6)
        self.play(FadeIn(br[1], shift=0.1 * UP), run_time=0.6)

        # e3: teaser
        nxt = next_card("flat optics: a lens one micron thick")
        mask = rgba(c["masks"][-1])
        mk = self.static_image(mask, TS, [tx[3], TY, 0], nearest=True)
        self.at(S("e3") + 0.6)
        self.add(mk[0])
        self.play(mk[1].animate.set_value(1.0), FadeTransform(caps[3], Tex("a phase mask", font_size=26).move_to(caps[3])),
                  FadeOut(br), run_time=0.8)
        self.at(S("e3") + 2.5)
        self.play(FadeIn(nxt, shift=0.1 * UP), run_time=0.8)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Camera", "Ch2Holography", "Ch3Islands", "Ch4Retrieval", "Ch5Design", "Ch6Computer", "Ch7Recap"]
