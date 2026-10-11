"""Coherence: when does light interfere? -- chapter scenes (Optics in 5 Minutes, E04).

Render: ./render.sh ; preview: ./preview.sh Ch0Hook Ch1Arrows ...
Every chapter renders exactly its window of the narration timeline (o5common.WIN). All physical fields come from
physics.py (data/*.npz); construct() only loads them. Scene state is a pure function of self.now and the plays.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403

C_A1, C_A2 = C_U, C_REF                      # the two interfering waves: BLUE and GOLD; their sum WHITE
WORDS = json.load(open(os.path.join(HERE, "audio", "words.json")))


def W(seg, word, k=0, fallback=0.0):
    """Start time of the k-th spoken occurrence of `word` inside segment `seg` (Whisper word timestamps)."""
    s = SEG[seg]
    hits = [w["s"] for w in WORDS if s["start"] - 0.3 <= w["s"] < s["end"] + 0.1
            and re.sub(r"[^a-z0-9]", "", w["w"].lower()) == word.lower()]
    if len(hits) > k:
        return max(hits[k], s["start"])
    print(f"W(): word '{word}' not heard in segment {seg}; using its start", file=sys.stderr)
    return s["start"] + fallback


def load(name):
    return np.load(data_path(name))


def wl_color(lam):
    """Wavelength (microns) -> a display colour (lightened for a black background)."""
    w = 1000 * lam
    if w < 440:
        r, g, b = (440 - w) / 60, 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, (510 - w) / 20
    elif w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, (645 - w) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    return ManimColor.from_rgb(tuple(float(min(1.0, 0.22 + 0.78 * c)) for c in (r, g, b)))


def stripe_rgba(I, vmax, H=64, gamma=0.8, edge=0.8):
    """1D intensity profile -> 2D 'camera' image of vertical fringes with soft top/bottom edges."""
    row = heat_rgba(np.asarray(I, float)[None, :], vmax, gamma, flip=False)[0]
    y = np.linspace(-1, 1, H)
    e = np.exp(-(np.abs(y) / edge) ** 8)
    img = np.empty((H, row.shape[0], 4), np.uint8)
    img[..., :3] = (row[None, :, :3] * e[:, None, None]).astype(np.uint8)
    img[..., 3] = 255
    return img


class Digits:
    """Numbers assembled from pre-built glyphs: safe to rebuild every frame (copies of pre-built Tex only)."""

    def __init__(self, font_size=32, color=WHITE):
        self.gl = {c: MathTex(c, font_size=font_size, color=color) for c in "0123456789."}
        self.w = max(self.gl[c].width for c in "0123456789")

    def make(self, text, left, y_base, opacity=1.0):
        g, x = VGroup(), left
        for ch in text:
            m = self.gl[ch].copy()
            cw = self.w if ch.isdigit() else m.width + 0.02
            m.move_to([x + cw / 2, 0, 0])
            m.shift(UP * (y_base - m.get_bottom()[1]))
            g.add(m)
            x += cw + 0.025
        if opacity < 0.999:
            g.set_opacity(opacity)
        return g


def dial_static(c, r, label="V", size=32):
    """Visibility gauge: half circle from 0 (left) to 1 (right)."""
    c = np.asarray(c, float)
    arc = Arc(radius=r, start_angle=0, angle=PI, arc_center=c).set_stroke(GREY_B, 3)
    ticks = VGroup(*[Line(c + 0.84 * r * np.array([np.cos(a), np.sin(a), 0]), c + r * np.array([np.cos(a), np.sin(a), 0]))
                     .set_stroke(GREY_B, 2) for a in np.linspace(0, PI, 5)])
    l0 = MathTex("0", font_size=26, color=GREY_B).next_to(c + r * LEFT, DOWN, buff=0.08)
    l1 = MathTex("1", font_size=26, color=GREY_B).next_to(c + r * RIGHT, DOWN, buff=0.08)
    lab = MathTex(label, font_size=size, color=GREY_A).move_to(c + 0.42 * r * UP)
    return VGroup(arc, ticks, l0, l1, lab)


def dial_needle(c, r, v, color=YELLOW, opacity=1.0):
    c = np.asarray(c, float)
    a = PI * (1 - float(np.clip(v, 0, 1)))
    return VGroup(fast_arrow(c, c + 0.8 * r * np.array([np.cos(a), np.sin(a), 0]), color, 5, 0.15, opacity),
                  Dot(c, radius=0.045, color=color).set_opacity(opacity))


def follow(mob, pos_fn, op=None):
    """Keep a pre-built mobject at pos_fn() with opacity op (tracker) -- never FadeIn such a mobject."""
    def upd(m):
        m.move_to(pos_fn())
        if op is not None:
            m.set_opacity(op.get_value())
    mob.add_updater(upd)
    upd(mob)
    return mob


def opaque(fn, op):
    """always_redraw whose drawing is multiplied by the opacity tracker op (empty when invisible)."""
    def build():
        o = op.get_value()
        if o <= 0.002:
            return VGroup()
        return fade_vgroup(fn(), o)
    return always_redraw(build)


def camera_icon(c, s=1.0):
    c = np.asarray(c, float)
    body = RoundedRectangle(corner_radius=0.08 * s, width=0.95 * s, height=0.62 * s).move_to(c).set_fill(GREY_E, 1).set_stroke(GREY_B, 2)
    lens = Circle(radius=0.19 * s).move_to(c).set_fill(BLACK, 1).set_stroke(GREY_B, 2)
    bump = Rectangle(width=0.3 * s, height=0.12 * s).next_to(body, UP, buff=0).shift(0.2 * s * LEFT).set_fill(GREY_E, 1).set_stroke(GREY_B, 2)
    return VGroup(bump, body, lens)


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        h = load("hook.npz")
        sl, se = h["spot_laser"], h["spot_led"]
        n = sl.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        rr = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)

        def radial(rgba):
            base = np.ascontiguousarray(rgba)
            rev = ValueTracker(0.0)

            def fn(t):
                img = base.copy()
                img[..., 3] = (255 * np.clip((1.45 * rev.get_value() - rr) / 0.08, 0, 1)).astype(np.uint8)
                return img
            return fn, rev
        size = 4.5
        cL, cR = np.array([-3.3, -0.05, 0]), np.array([3.3, -0.05, 0])
        fnL, revL = radial(heat_rgba(sl, np.percentile(sl, 99.7), gamma=0.75, flip=False))
        fnR, revR = radial(heat_rgba(se, se.max() * 1.03, gamma=0.75, flip=False))
        mL, fL = self.live_image(fnL, size, cL)
        mR, fR = self.live_image(fnR, size, cR)
        labL = Tex("laser", font_size=42, color=GREY_A).move_to(cL + 2.65 * UP)
        labR = Tex("LED", font_size=42, color=GREY_A).move_to(cR + 2.65 * UP)
        capL = Tex("speckle", font_size=30, color=C_EV).move_to(cL + 2.62 * DOWN)
        capR = Tex("smooth", font_size=30, color=GREY_B).move_to(cR + 2.62 * DOWN)
        cap0 = Tex("(computed)", font_size=24, color=GREY_B).move_to([0, -2.95, 0])

        self.add(mL, mR)
        self.at(S("h1"))
        self.play(FadeIn(labL), fL.animate.set_value(1.0), revL.animate.set_value(1.0), run_time=1.6)
        self.play(FadeIn(capL), run_time=0.5)
        self.at(W("h1", "an") - 0.1)
        self.play(FadeIn(labR), fR.animate.set_value(1.0), revR.animate.set_value(1.0), FadeIn(cap0), run_time=1.6)
        self.play(FadeIn(capR), run_time=0.5)

        # ---------------- h2: two slits -- laser fringes, LED none, LED + pinhole: fringes return
        s = h["s"]
        pick = slice(None, None, 2)
        fr = {k: h[k][pick] for k in ("fr_laser", "fr_led", "fr_pin")}
        y1, y2 = 1.35, -1.55
        img_w, img_h, img_cx = 7.9, 1.45, 2.45

        def stripes(I):
            return stripe_rgba(I, 1.0, H=72, gamma=0.75)
        im_las, f_las, sw_las = self.sweep_image(stripes(fr["fr_laser"]), img_w, [img_cx, y1, 0], img_h, axis="right")
        im_led, f_led, sw_led = self.sweep_image(stripes(fr["fr_led"]), img_w, [img_cx, y2, 0], img_h, axis="right")
        im_pin, f_pin = self.static_image(stripes(fr["fr_pin"]), img_w, [img_cx, y2, 0], img_h)
        xb = -2.2

        def barrier(y):
            g = 0.2
            return VGroup(Line([xb, y + g + 0.06, 0], [xb, y + 0.78, 0]), Line([xb, y + g - 0.06, 0], [xb, y - g + 0.06, 0]),
                          Line([xb, y - g - 0.06, 0], [xb, y - 0.78, 0])).set_stroke(WHITE, 5)
        laser = VGroup(RoundedRectangle(corner_radius=0.07, width=0.95, height=0.42).set_fill(GREY_E, 1).set_stroke(GREY_B, 2),
                       Tex("laser", font_size=26, color=GREY_B)).move_to([-5.95, y1, 0])
        laser[1].next_to(laser[0], DOWN, buff=0.1)
        beam = Rectangle(width=xb - (-5.47), height=0.34).set_stroke(width=0).set_fill(BLUE_B, 0.45).move_to([0.5 * (xb - 5.47), y1, 0])
        led = VGroup(Circle(radius=0.2).set_fill(GREY_E, 1).set_stroke(GREY_B, 2), Tex("LED", font_size=26, color=GREY_B)).move_to([-5.95, y2, 0])
        led[1].next_to(led[0], DOWN, buff=0.1)
        led_c = np.array([-5.95, y2, 0])
        rays = VGroup(*[Line(led_c + 0.2 * RIGHT, [xb, y2 + dy, 0]).set_stroke(GREY_A, 1.5, opacity=0.55)
                        for dy in np.linspace(-0.7, 0.7, 7)])
        xp = -4.25
        pin = VGroup(Line([xp, y2 + 0.04, 0], [xp, y2 + 0.72, 0]), Line([xp, y2 - 0.04, 0], [xp, y2 - 0.72, 0])).set_stroke(WHITE, 5)
        pin_l = Tex("pinhole", font_size=26, color=GREY_A).next_to(pin, UP, buff=0.12)
        rays_pin = VGroup(*[Line(led_c + 0.2 * RIGHT, [xp, y2 + dy, 0]).set_stroke(GREY_A, 1.5, opacity=0.55) for dy in (-0.25, 0, 0.25)],
                          *[Line([xp, y2, 0], [xb, y2 + dy, 0]).set_stroke(GREY_A, 1.5, opacity=0.55) for dy in (-0.25, 0.25)])
        scr = Tex("screen", font_size=26, color=GREY_B).move_to([img_cx, y1 + 1.0, 0])
        self.at(S("h2"))
        self.play(fL.animate.set_value(0.0), fR.animate.set_value(0.0), FadeOut(VGroup(labL, labR, capL, capR, cap0)), run_time=0.5)
        for m in (mL, mR):
            m.clear_updaters()
        self.remove(mL, mR)
        self.add(im_las)
        bar1, bar2 = barrier(y1), barrier(y2)
        self.play(FadeIn(laser), FadeIn(bar1), GrowFromEdge(beam, LEFT), FadeIn(scr), f_las.animate.set_value(1.0), run_time=0.8)
        self.play(sw_las.animate.set_value(1.0), run_time=1.2, rate_func=linear)
        self.at(W("h2", "led") - 0.15)
        self.add(im_led)
        self.play(FadeIn(led), FadeIn(bar2), FadeIn(rays), f_led.animate.set_value(1.0), run_time=0.7)
        self.play(sw_led.animate.set_value(1.0), run_time=1.0, rate_func=linear)
        self.at(W("h2", "pinhole") - 0.6)
        self.add(im_pin)
        self.play(FadeIn(pin, shift=0.4 * DOWN), FadeIn(pin_l), FadeOut(rays), FadeIn(rays_pin),
                  f_pin.animate.set_value(1.0), run_time=1.2)
        im_led.clear_updaters()
        self.remove(im_led)

        # ---------------- h3: the question, the title
        ttl = series_title(4, "Coherence", "when does light interfere?", y=-0.1)
        dim = VGroup(laser, bar1, beam, scr, led, bar2, pin, pin_l, rays_pin)
        self.at(S("h3") + 0.1)
        self.play(f_las.animate.set_value(0.28), f_pin.animate.set_value(0.28), dim.animate.set_opacity(0.22), run_time=0.7)
        self.play(FadeIn(ttl, shift=0.15 * DOWN), run_time=0.9)
        self.at(W("h3", "coherence") - 0.15)
        self.play(Indicate(ttl[1][1], color=YELLOW, scale_factor=1.08), run_time=0.8)
        self.finish()


# ============================================================ 1  arrows ============
class Ch1Arrows(Chapter):
    CH = 1
    TAG = "1\\quad Interference is arrow addition"

    def construct(self):
        S = self.S
        self.show_tag()
        jd = load("jitter.npz")
        sig_tab, gam_tab, noise = jd["sig"], jd["gam"], jd["noise"]

        def gam(sg):
            return complex(np.interp(sg, sig_tab, gam_tab.real), np.interp(sg, sig_tab, gam_tab.imag))

        def nz(t):                                   # seeded jitter, one sample per 1/60 s of film time
            return float(noise[int(round(t * 60)) % len(noise)])

        # ---- the common spin (c2-c3), a pure function of film time, coming to rest at angle 0
        th1 = 0.95                                   # a1's angle after c1
        T = np.arange(self.t0, self.t_end + 0.5, 0.002)
        c2s, stop = S("c2"), W("c3", "spin") + 1.5
        rate = (0.3 + 2.4 * smoothstep((T - (c2s + 0.5)) / 1.5) * (1 - smoothstep((T - (c2s + 4.2)) / 1.5)))
        rate *= smoothstep((T - c2s) / 0.6) * (1 - smoothstep((T - stop) / 1.4))
        spin = TAU_ * np.cumsum(rate) * 0.002
        m = np.round((th1 + spin[-1]) / TAU_)
        spin *= (m * TAU_ - th1) / spin[-1]          # rest exactly at a multiple of 2 pi

        th = ValueTracker(0.35)
        r1 = ValueTracker(1.0)                       # |a1| (c1 only)
        cx, cy, R = ValueTracker(-2.7), ValueTracker(-0.25), ValueTracker(1.45)
        dphi, sig, px = ValueTracker(1.15), ValueTracker(0.0), ValueTracker(0.0)
        op_ax, op1, op2, op_arc, op_dp, op_bar, op_lab = (ValueTracker(0.0) for _ in range(7))
        NF = 4.0                                      # fringes across the screen panel

        def C():
            return np.array([cx.get_value(), cy.get_value(), 0.0])

        def rho(t):
            return th.get_value() + float(np.interp(t, T, spin))

        def dph(t):
            return dphi.get_value() + TAU_ * NF * px.get_value() + sig.get_value() * nz(t)

        def tips(t):
            c, r = C(), R.get_value()
            a = rho(t)
            p1 = c + r * r1.get_value() * np.array([np.cos(a), np.sin(a), 0])
            b = a + dph(t)
            return p1, p1 + r * np.array([np.cos(b), np.sin(b), 0])

        def axes():
            c, r = C(), R.get_value()
            return VGroup(Line(c + 1.3 * r * LEFT, c + 2.1 * r * RIGHT).set_stroke(GREY_D, 2),
                          Line(c + 1.2 * r * DOWN, c + 1.35 * r * UP).set_stroke(GREY_D, 2))
        ax_m = opaque(axes, op_ax)

        def arrows():
            t, c = self.now, C()
            g = VGroup()
            o1, o2 = op1.get_value(), op2.get_value()
            p1, p2 = tips(t)
            fast = abs(rho(t) - rho(t - 1 / 60)) > 0.1
            jit = sig.get_value() > 0.03 and o2 > 0.01
            if fast or jit:
                for k in range(1, 5):
                    q1, q2 = tips(t - k / 60)
                    oo = 0.3 * (1 - k / 5)
                    if fast:
                        g.add(fast_arrow(c, q1, C_A1, 4, 0.16, oo * o1))
                        if o2 > 0.01:
                            g.add(fast_arrow(q1, q2, C_A2, 4, 0.16, oo * o2))
                    else:
                        g.add(fast_arrow(p1, p1 + (q2 - q1), C_A2, 4, 0.16, oo * o2))
            if o2 > 0.01:
                g.add(fast_arrow(c, p2, WHITE, 4, 0.2, o2))
            g.add(fast_arrow(c, p1, C_A1, 6, 0.22, o1))
            if o2 > 0.01:
                g.add(fast_arrow(p1, p2, C_A2, 6, 0.22, o2))
            g.add(Dot(c, radius=0.05, color=GREY_B).set_opacity(o1))
            return g
        arr = always_redraw(arrows)

        def angle_arc():                                    # c1: the phase angle of a1
            c = C()
            a = rho(self.now) % TAU_
            return VGroup(Arc(radius=0.5, start_angle=0, angle=a, arc_center=c).set_stroke(YELLOW, 3),
                          fast_dashed(c, c + 1.2 * R.get_value() * RIGHT, 0.05, GREY_B, 1.5))
        arc_m = opaque(angle_arc, op_arc)
        phi_l = follow(MathTex("\\varphi", font_size=34, color=YELLOW),
                       lambda: C() + 0.78 * np.array([np.cos(rho(self.now) / 2), np.sin(rho(self.now) / 2), 0]), op_arc)

        def dphi_arc():                                     # angle between the arrows (shown while it is fixed)
            p1, p2 = tips(self.now)
            a = rho(self.now)
            b = ((dph(self.now) + PI) % TAU_) - PI
            u = (p1 - C()) / max(np.linalg.norm(p1 - C()), 1e-6)
            return VGroup(fast_dashed(p1, p1 + 0.85 * u, 0.05, GREY_B, 1.5),
                          Arc(radius=0.42, start_angle=a, angle=b, arc_center=p1).set_stroke(YELLOW, 3))
        dp_m = opaque(dphi_arc, op_dp)

        def dp_pos():
            p1, _ = tips(self.now)
            a = rho(self.now) + 0.5 * (((dph(self.now) + PI) % TAU_) - PI)
            return p1 + 0.8 * np.array([np.cos(a), np.sin(a), 0])
        dp_l = follow(MathTex("\\Delta\\varphi", font_size=30, color=YELLOW), dp_pos, op_dp)

        def side():                                         # +1 if a2 turns counter-clockwise from a1
            return 1.0 if ((dph(self.now) + PI) % TAU_) - PI >= 0 else -1.0

        def a1_pos():                                       # below a1 (away from the resultant)
            p1, _ = tips(self.now)
            a = rho(self.now)
            return 0.5 * (C() + p1) + 0.34 * side() * np.array([np.sin(a), -np.cos(a), 0])
        a1_l = follow(MathTex("a_1", font_size=32, color=C_A1), a1_pos, op_lab)

        def a2_pos():                                       # outside the angle arc, near the wide end of the triangle
            p1, p2 = tips(self.now)
            d = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-6)
            nrm = side() * np.array([-d[1], d[0], 0])
            return p1 + 0.36 * np.linalg.norm(p2 - p1) * d + 0.3 * nrm
        a2_l = follow(MathTex("a_2", font_size=32, color=C_A2), a2_pos, op_lab)

        # ---------------- c1: one arrow
        txt_cx = Tex("arrow $=$ a complex number $U$", font_size=36).move_to([2.7, 1.55, 0])
        txt_cx[0][-1].set_color(C_U)
        txt_amp = Tex("length $=$ amplitude", font_size=36).move_to([2.7, 0.6, 0])
        txt_ph = Tex("angle $=$ phase $\\varphi$", font_size=36).move_to([2.7, -0.35, 0])
        txt_ph[0][-1].set_color(YELLOW)
        self.add(ax_m, arc_m, phi_l, arr)
        self.at(S("c1"))
        self.play(op_ax.animate.set_value(1.0), op1.animate.set_value(1.0), run_time=0.7)
        self.at(W("c1", "arrow") - 0.1)
        self.play(FadeIn(txt_cx), run_time=0.7)
        self.at(W("c1", "length") - 0.15)
        self.play(FadeIn(txt_amp), r1.animate.set_value(1.3), run_time=0.7)
        self.play(r1.animate.set_value(1.0), run_time=0.5)
        self.at(W("c1", "angle") - 0.2)
        self.play(FadeIn(txt_ph), op_arc.animate.set_value(1.0), th.animate.set_value(2.0), run_time=0.7)
        self.play(th.animate.set_value(th1), run_time=0.6)

        # ---------------- c2: it spins too fast; the camera sees |U|^2
        spins = Tex("$\\approx 5\\times10^{14}$ turns per second", font_size=34).move_to([2.7, 1.0, 0])
        cam = camera_icon([1.15, -0.85, 0])
        cam_t = MathTex("I", "=", "|U|^2", font_size=48).move_to([3.2, -0.85, 0])
        cam_l = Tex("what a camera records", font_size=28, color=GREY_B).next_to(VGroup(cam, cam_t), DOWN, buff=0.3)
        circ = opaque(lambda: Circle(radius=R.get_value()).move_to(C()).set_stroke(C_A1, 2, opacity=0.35), op_bar)
        self.at(S("c2"))
        self.play(FadeOut(VGroup(txt_cx, txt_amp, txt_ph)), op_arc.animate.set_value(0.0), run_time=0.5)
        self.play(FadeIn(spins), run_time=0.6)
        self.add(circ)
        self.at(W("c2", "camera") - 0.2)
        self.play(FadeIn(cam), op_bar.animate.set_value(1.0), run_time=0.6)
        self.at(W("c2", "squared") - 0.2)
        self.play(FadeIn(cam_t), FadeIn(cam_l), run_time=0.7)

        # ---------------- c3: two arrows add; only the angle between them matters
        self.at(S("c3"))
        self.play(FadeOut(VGroup(spins, cam, cam_t, cam_l)), op_bar.animate.set_value(0.0),
                  cx.animate.set_value(-4.75), cy.animate.set_value(-0.65), R.animate.set_value(1.35), run_time=0.9)
        self.remove(circ, arc_m, phi_l)
        self.add(a1_l, a2_l)
        self.play(op2.animate.set_value(1.0), op_lab.animate.set_value(1.0), run_time=0.8)
        self.add(dp_m, dp_l)
        self.at(W("c3", "angle") - 0.1)
        self.play(op_dp.animate.set_value(1.0), run_time=0.8)

        # ---------------- c4: |a1 + a2|^2 = |a1|^2 + |a2|^2 + 2|a1||a2| cos(dphi)
        eq = MathTex("|", "a_1", "+", "a_2", "|^2", "=", "|a_1|^2", "+", "|a_2|^2", "+",
                     "2|a_1||a_2|\\cos\\Delta\\varphi", font_size=42).move_to([0.8, 2.85, 0])
        eq[1].set_color(C_A1)
        eq[3].set_color(C_A2)
        eq[6].set_color(C_A1)
        eq[8].set_color(C_A2)
        eq[10].set_color(YELLOW)
        cross_box = SurroundingRectangle(eq[10], buff=0.08, color=YELLOW, stroke_width=2)
        cross_l = Tex("interference", font_size=28, color=YELLOW).next_to(cross_box, DOWN, buff=0.1)
        avg_l = Tex("averages to 0", font_size=28, color=GREY_A).next_to(cross_box, DOWN, buff=0.1)
        xb_, yb_, sb = -1.55, -2.3, 0.5

        def bar():
            _, p2 = tips(self.now)
            I = (np.linalg.norm(p2 - C()) / R.get_value()) ** 2
            return VGroup(Rectangle(width=0.32, height=max(sb * I, 1e-3)).move_to([xb_, yb_ + sb * I / 2, 0])
                          .set_stroke(width=0).set_fill(YELLOW, 0.85),
                          Line([xb_ - 0.25, yb_, 0], [xb_ + 0.25, yb_, 0]).set_stroke(GREY_B, 2))
        op_b2 = ValueTracker(0.0)
        bar_m = opaque(bar, op_b2)
        bar_l = MathTex("I", font_size=32, color=YELLOW).move_to([xb_, yb_ - 0.3, 0])
        self.at(S("c4"))
        self.play(Write(eq[:5]), run_time=1.0)
        self.add(bar_m)
        self.play(Write(eq[5:9]), op_b2.animate.set_value(1.0), FadeIn(bar_l), run_time=1.1)
        self.at(W("c4", "cross") - 0.2)
        self.play(Write(eq[9:]), Create(cross_box), op_dp.animate.set_value(0.0), op_lab.animate.set_value(0.0), run_time=1.0)
        self.remove(dp_m, dp_l, a1_l, a2_l)
        self.play(dphi.animate.set_value(1.15 + TAU_), FadeIn(cross_l), run_time=2.0, rate_func=smooth)

        # ---------------- c5: across the screen, dphi grows -> fringes
        x0s, x1s = -0.75, 5.95
        Wsc = x1s - x0s
        ys0, sy = -2.55, 0.5
        st_c, st_h = np.array([0.5 * (x0s + x1s), 1.1, 0]), 0.8
        xs = np.linspace(0, 1, 801)
        phase0 = 1.15                                          # dphi at x = 0 (continuous with c4)

        def I_avg(sg):
            return 2 + 2 * np.real(gam(sg) * np.exp(1j * (phase0 + TAU_ * NF * xs)))
        rv = ValueTracker(0.0)

        def stripe_fn(t):
            img = stripe_rgba(I_avg(sig.get_value()), 4.0, H=48, gamma=0.8)
            m_ = (np.linspace(0, 1, img.shape[1]) <= rv.get_value() + 1e-9)
            img[..., 3] = (255 * m_)[None, :]
            return img
        st_m, st_f = self.live_image(stripe_fn, Wsc, st_c, st_h)
        st_box = Rectangle(width=Wsc, height=st_h).move_to(st_c).set_stroke(GREY_B, 1.5)
        st_lab = Tex("camera", font_size=28, color=GREY_B).next_to(st_box, LEFT, buff=0.2)
        base = VGroup(Line([x0s, ys0, 0], [x1s + 0.15, ys0, 0]).set_stroke(GREY_B, 2),
                      Line([x0s, ys0, 0], [x0s, ys0 + 4 * sy + 0.35, 0]).set_stroke(GREY_B, 2),
                      MathTex("x", font_size=30, color=GREY_B).move_to([x1s + 0.15, ys0 - 0.3, 0]),
                      MathTex("I", font_size=30, color=GREY_B).move_to([x0s - 0.25, ys0 + 4 * sy + 0.2, 0]))

        def curves():
            g = VGroup()
            r = rv.get_value()
            if r <= 1e-3:
                return g
            k = max(2, int(r * (len(xs) - 1)) + 1)
            Ia = I_avg(sig.get_value())
            g.add(poly_line(x0s + Wsc * xs[:k], ys0 + sy * Ia[:k], WHITE, 4))
            return g
        cv_m = always_redraw(curves)
        op_inst = ValueTracker(0.0)

        def inst():
            Ii = 2 + 2 * np.cos(phase0 + TAU_ * NF * xs + sig.get_value() * nz(self.now))
            return poly_line(x0s + Wsc * xs, ys0 + sy * Ii, C_A2, 2.5)
        inst_m = opaque(inst, op_inst)

        def probe():
            p = px.get_value()
            x = x0s + Wsc * p
            Ia = 2 + 2 * np.real(gam(sig.get_value()) * np.exp(1j * (phase0 + TAU_ * NF * p)))
            yb0 = st_c[1] - st_h / 2
            return VGroup(fast_dashed([x, ys0, 0], [x, ys0 + 4 * sy + 0.15, 0], 0.06, GREY_A, 1.5),
                          Line([x, yb0 - 0.16, 0], [x, yb0 + 0.06, 0]).set_stroke(YELLOW, 4),
                          Dot([x, ys0 + sy * Ia, 0], radius=0.07, color=YELLOW))
        op_pr = ValueTracker(0.0)
        pr_m = opaque(probe, op_pr)
        self.at(S("c5"))
        self.add(st_m, cv_m, pr_m)
        self.play(FadeIn(base), Create(st_box), FadeIn(st_lab), st_f.animate.set_value(1.0), op_pr.animate.set_value(1.0), run_time=0.8)
        self.play(px.animate.set_value(1.0), rv.animate.set_value(1.0), run_time=3.6, rate_func=linear)

        # ---------------- c6: jitter -> the fringes dance; the camera's average flattens
        leg = VGroup(VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(C_A2, 3), Tex("each instant", font_size=26, color=C_A2)).arrange(RIGHT, buff=0.15),
                     VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(WHITE, 4), Tex("camera (average)", font_size=26)).arrange(RIGHT, buff=0.15)
                     ).arrange(RIGHT, buff=0.6).move_to([2.6, 0.12, 0])
        self.at(S("c6"))
        self.play(px.animate.set_value(0.37), run_time=0.8)
        self.add(inst_m)
        self.at(W("c6", "jitter") - 0.1)
        self.play(op_inst.animate.set_value(0.85), FadeIn(leg), run_time=0.5)
        self.play(sig.animate.set_value(1.2), run_time=1.7, rate_func=linear)
        self.play(sig.animate.set_value(3.5), run_time=2.2, rate_func=linear)

        # ---------------- c7: on average the cross term vanishes
        flat_l = MathTex("|a_1|^2", "+", "|a_2|^2", font_size=32).move_to([4.75, ys0 + sy * 2 + 0.42, 0])
        flat_l[0].set_color(C_A1)
        flat_l[2].set_color(C_A2)
        flat_bg = BackgroundRectangle(flat_l, fill_opacity=0.85, buff=0.08).set_fill(BLACK)
        self.at(S("c7"))
        self.play(eq[10].animate.set_opacity(0.3), cross_box.animate.set_stroke(opacity=0.3), FadeOut(cross_l), FadeIn(avg_l),
                  op_inst.animate.set_value(0.35), run_time=0.8)
        self.at(W("c7", "intensities") - 0.2)
        self.play(FadeIn(flat_bg), FadeIn(flat_l, shift=0.1 * UP), run_time=0.7)

        # ---------------- c8: visibility
        Vq = MathTex("V", "=", "{I_{\\max}-I_{\\min}", "\\over", "I_{\\max}+I_{\\min}}", font_size=40).move_to([-4.3, 1.2, 0])
        dc, dr = np.array([-4.3, -1.55, 0]), 1.05
        dstat = dial_static(dc, dr)
        dig = Digits(30)
        op_d = ValueTracker(0.0)

        def vis():
            return abs(gam(sig.get_value()))

        def dial_dyn():
            v = vis()
            return VGroup(dial_needle(dc, dr, v), dig.make(f"{v:.2f}", dc[0] - 0.36, dc[1] - 0.62))
        dial_m = opaque(dial_dyn, op_d)

        def minmax():
            v = vis()
            g = VGroup()
            for I_ in (2 + 2 * v, 2 - 2 * v):
                y = ys0 + sy * I_
                g.add(fast_dashed([x0s, y, 0], [x1s, y, 0], 0.07, YELLOW, 1.5, 0.8))
            return g
        mm_m = opaque(minmax, op_d)
        def mm_op():                                            # hide the labels once the two lines merge
            return op_d.get_value() * float(np.clip((vis() - 0.12) / 0.25, 0, 1))
        imax_l = MathTex("I_{\\max}", font_size=26, color=YELLOW)
        imax_l.add_updater(lambda m: m.move_to([x1s + 0.5, ys0 + sy * (2 + 2 * vis()), 0]).set_opacity(mm_op()))
        imin_l = MathTex("I_{\\min}", font_size=26, color=YELLOW)
        imin_l.add_updater(lambda m: m.move_to([x1s + 0.5, ys0 + sy * (2 - 2 * vis()) + 0.12, 0]).set_opacity(mm_op()))
        imax_l.update()
        imin_l.update()
        self.at(S("c8") - 0.3)
        self.play(op1.animate.set_value(0.0), op2.animate.set_value(0.0), op_ax.animate.set_value(0.0),
                  op_b2.animate.set_value(0.0), FadeOut(bar_l), op_inst.animate.set_value(0.0), FadeOut(leg), FadeOut(flat_l), FadeOut(flat_bg),
                  run_time=0.6)
        self.remove(arr, ax_m, bar_m, inst_m)
        self.add(dial_m, mm_m, imax_l, imin_l)
        self.play(Write(Vq), FadeIn(dstat), op_d.animate.set_value(1.0), run_time=1.0)
        self.at(W("c8", "one") - 0.1)
        self.play(sig.animate.set_value(0.0), eq[10].animate.set_opacity(1.0), cross_box.animate.set_stroke(opacity=1.0),
                  FadeOut(avg_l), run_time=1.2)
        self.at(W("c8", "zero") - 0.35)
        self.play(sig.animate.set_value(3.5), run_time=1.15)
        self.finish()


# ============================================================ 2  colour ============
class Ch2Colour(Chapter):
    CH = 2
    TAG = "2\\quad Colour: temporal coherence"

    def construct(self):
        S = self.S
        self.show_tag()
        td = load("temporal.npz")
        dL, dlam, Itab, envtab, lctab = td["dL"], td["dlam"], td["I"], td["env"], td["lc"]
        lam_ax, spectab, cols = td["lam_ax"], td["spec"], td["cols"]
        lam0 = float(td["lam0"])
        sel = slice(None, None, 3)
        dLs = dL[sel]
        yb, sy = -2.72, 1.22                                # merged plot: baseline, scale (I in [0, 2])
        GREEN_L = wl_color(0.55)

        def row(lam, base, scale, color, width=3.5, opacity=1.0):
            return poly_line(dLs, base + scale * (1 + np.cos(TAU_ * dLs / lam)), color, width, opacity)

        # ---------------- t1: two copies of a beam, one delayed by dL
        y1, y2, amp = 2.45, 1.2, 0.27
        dl = ValueTracker(0.0)
        rv1, rv2 = ValueTracker(0.0), ValueTracker(0.0)
        op_w = ValueTracker(1.0)
        xw = np.linspace(-6.0, 6.0, 700)

        def waves():
            g = VGroup()
            o = op_w.get_value()
            if o <= 0.002:
                return g
            for rv, y, sh, col in ((rv1, y1, 0.0, C_A1), (rv2, y2, dl.get_value(), C_A2)):
                r = rv.get_value()
                if r > 1e-3:
                    k = max(2, int(r * len(xw)))
                    g.add(poly_line(xw[:k], y + amp * np.cos(TAU_ * (xw[:k] - sh) / lam0), col, 3.5, o))
            return g
        wv = always_redraw(waves)
        op_m = ValueTracker(0.0)

        def marker():
            d = dl.get_value()
            g = VGroup(fast_dashed([0, y2 - 0.42, 0], [0, y1 + 0.4, 0], 0.05, GREY_A, 1.5),
                       fast_dashed([d, y2 - 0.42, 0], [d, y2 + 0.4, 0], 0.05, C_A2, 1.5))
            if d > 0.12:
                ym = y2 + 0.42
                g.add(fast_arrow([d / 2, ym, 0], [0, ym, 0], WHITE, 2.5, 0.1), fast_arrow([d / 2, ym, 0], [d, ym, 0], WHITE, 2.5, 0.1))
            return g
        mk = opaque(marker, op_m)
        dl_l = follow(MathTex("\\Delta L", font_size=30), lambda: [max(dl.get_value() / 2, 0.0), y2 + 0.73, 0], op_m)
        n1 = Tex("beam", font_size=28, color=C_A1).move_to([-6.45, y1 + 0.42, 0])
        n2 = Tex("delayed copy", font_size=28, color=C_A2).move_to([-5.95, y2 - 0.5, 0])
        self.add(wv)
        self.at(S("t1") + 0.3)
        self.play(rv1.animate.set_value(1.0), FadeIn(n1), run_time=1.2)
        self.at(W("t1", "split") - 0.2)
        self.play(rv2.animate.set_value(1.0), FadeIn(n2), run_time=0.9)
        self.add(mk, dl_l)
        self.at(W("t1", "delay") - 0.1)
        self.play(op_m.animate.set_value(1.0), dl.animate.set_value(3 * lam0), run_time=1.8)

        # ---------------- t2: one colour -> fringes forever
        axes = VGroup(Line([-6.25, yb, 0], [6.35, yb, 0]).set_stroke(GREY_B, 2))
        for v, s_ in ((-5, "-5\\,\\mu m"), (0, "0"), (5, "5\\,\\mu m")):
            axes.add(Line([v, yb - 0.07, 0], [v, yb + 0.07, 0]).set_stroke(GREY_B, 2),
                     MathTex(s_, font_size=24, color=GREY_B).move_to([v, yb - 0.3, 0]))
        axes.add(MathTex("\\Delta L", font_size=30, color=GREY_B).move_to([6.6, yb + 0.05, 0]))
        rvg = ValueTracker(0.0)
        green_live = always_redraw(lambda: VGroup() if rvg.get_value() < 1e-3 else
                                   poly_line(dLs[:max(2, int(rvg.get_value() * len(dLs)))],
                                             yb + sy * (1 + np.cos(TAU_ * dLs[:max(2, int(rvg.get_value() * len(dLs)))] / 0.55)),
                                             GREEN_L, 3.5))
        op_pd = ValueTracker(0.0)

        def probe():
            d = dl.get_value()
            return VGroup(fast_dashed([d, y2 - 0.42, 0], [d, yb, 0], 0.06, GREY_C, 1.2),
                          Dot([d, yb + sy * (1 + np.cos(TAU_ * d / 0.55)), 0], radius=0.07, color=YELLOW))
        pd = opaque(probe, op_pd)
        self.at(S("t2"))
        self.add(green_live, pd)
        self.play(FadeIn(axes), rvg.animate.set_value(1.0), run_time=1.3, rate_func=linear)
        self.play(op_pd.animate.set_value(1.0), dl.animate.set_value(4.95), run_time=1.3)

        # ---------------- t3: each colour its own spacing
        rows_y = [0.75, -0.6, -1.95]
        rsc = 0.5
        specs = [(0.65, wl_color(0.65), "red"), (0.55, GREEN_L, "green"), (0.45, wl_color(0.45), "blue")]
        rows = [row(l, yy, rsc, c) for (l, c, _), yy in zip(specs, rows_y)]
        names = VGroup(*[Tex(nm, font_size=28, color=c).move_to([-6.45, yy + 0.5, 0]) for (_, c, nm), yy in zip(specs, rows_y)])
        z_line = DashedLine([0, -2.05, 0], [0, 1.95, 0], dash_length=0.08).set_stroke(GREY_A, 1.5)
        z_lab = Tex("all agree", font_size=26, color=GREY_A).move_to([0, 2.2, 0])
        o_line = DashedLine([2.0, -2.05, 0], [2.0, 1.95, 0], dash_length=0.08).set_stroke(C_EV, 1.5)
        o_lab = Tex("out of step", font_size=26, color=C_EV).move_to([2.2, 2.2, 0])
        green_static = poly_line(dLs, yb + sy * (1 + np.cos(TAU_ * dLs / 0.55)), GREEN_L, 3.5)
        self.at(S("t3"))
        self.play(op_w.animate.set_value(0.0), op_m.animate.set_value(0.0), op_pd.animate.set_value(0.0), FadeOut(VGroup(n1, n2)),
                  FadeOut(axes), run_time=0.5)
        self.remove(wv, mk, dl_l, pd, green_live)
        self.add(green_static)
        self.play(ReplacementTransform(green_static, rows[1]), FadeIn(names[1]), run_time=0.6)
        self.play(Create(rows[0]), Create(rows[2]), FadeIn(names[0]), FadeIn(names[2]), run_time=0.9)
        self.at(W("t3", "zero") - 0.2)
        self.play(Create(z_line), FadeIn(z_lab), run_time=0.6)
        self.at(W("t3", "further") - 0.1)
        self.play(Create(o_line), FadeIn(o_lab), run_time=0.6)

        # ---------------- t4: a whole spectrum -> fringes only near zero delay
        merged = [row(l, yb, sy, c, 2.5, 0.75) for (l, c, _) in specs]
        fan = VGroup(*[row(l, yb, sy, wl_color(l), 2.0, 0.55) for l in cols])
        dlam_t = ValueTracker(0.100)
        li = np.log(dlam)

        def tab(tbl, d):
            x = np.interp(np.log(d), li, np.arange(len(li)))
            i = int(np.clip(np.floor(x), 0, len(li) - 2))
            f = x - i
            return (1 - f) * tbl[i] + f * tbl[i + 1]
        op_sum, op_env, op_lc = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)
        sum_m = opaque(lambda: poly_line(dLs, yb + sy * tab(Itab, dlam_t.get_value())[sel], WHITE, 4), op_sum)

        def envelope():
            e = tab(envtab, dlam_t.get_value())[sel]
            return VGroup(poly_line(dLs, yb + sy * (1 + e), YELLOW, 2.5), poly_line(dLs, yb + sy * (1 - e), YELLOW, 2.5))
        env_m = opaque(envelope, op_env)

        def lc_mark():
            lc = float(np.interp(np.log(dlam_t.get_value()), li, lctab))
            g = VGroup()
            if lc < 11.8:
                yl = yb + sy * 2.12
                a, b = -lc / 2, lc / 2
                g.add(fast_arrow([0, yl, 0], [a, yl, 0], YELLOW, 3, 0.14), fast_arrow([0, yl, 0], [b, yl, 0], YELLOW, 3, 0.14),
                      fast_dashed([a, yl, 0], [a, yb + sy * (1 + np.exp(-1)), 0], 0.05, YELLOW, 1.2),
                      fast_dashed([b, yl, 0], [b, yb + sy * (1 + np.exp(-1)), 0], 0.05, YELLOW, 1.2))
            return g
        lc_m = opaque(lc_mark, op_lc)
        lc_tex = MathTex("\\ell_c", "\\approx", "{\\lambda^2", "\\over", "\\Delta\\lambda}", font_size=40, color=YELLOW)
        lc_name = Tex("coherence length", font_size=30, color=YELLOW)
        lc_grp = VGroup(lc_name, lc_tex).arrange(RIGHT, buff=0.3).move_to([2.9, 0.62, 0])
        self.at(S("t4"))
        self.play(FadeOut(VGroup(z_line, z_lab, o_line, o_lab, names)), FadeIn(axes),
                  *[Transform(r_, m_) for r_, m_ in zip(rows, merged)], run_time=1.0)
        self.play(LaggedStart(*[FadeIn(f) for f in fan], lag_ratio=0.08), run_time=1.1)
        self.add(sum_m)
        self.at(W("t4", "only") - 0.2)
        self.play(op_sum.animate.set_value(1.0), fan.animate.set_stroke(opacity=0.16), *[r_.animate.set_stroke(opacity=0.16) for r_ in rows],
                  run_time=1.2)
        self.add(env_m, lc_m)
        self.at(W("t4", "window") - 0.2)
        self.play(op_env.animate.set_value(1.0), op_lc.animate.set_value(1.0), fan.animate.set_stroke(opacity=0.0),
                  *[r_.animate.set_stroke(opacity=0.0) for r_ in rows], run_time=0.8)
        self.remove(fan, *rows)
        self.at(W("t4", "coherence") - 0.1)
        self.play(FadeIn(lc_name), run_time=0.5)
        self.at(W("t4", "roughly") - 0.1)
        self.play(Write(lc_tex), run_time=1.0)

        # ---------------- t5: the envelope is the Fourier transform of the spectrum
        sx0, sx1, syb, sh = -6.2, -2.4, 0.95, 1.25
        lx = lambda l: sx0 + (l - 0.36) / (0.80 - 0.36) * (sx1 - sx0)
        sp_ax = VGroup(Line([sx0, syb, 0], [sx1 + 0.1, syb, 0]).set_stroke(GREY_B, 2),
                       MathTex("\\lambda", font_size=30, color=GREY_B).move_to([sx1 + 0.3, syb, 0]),
                       Tex("spectrum", font_size=28, color=GREY_A).move_to([0.5 * (sx0 + sx1), syb + sh + 0.45, 0]))
        for l, s_ in ((0.4, "400"), (0.55, "550"), (0.7, "700\\,nm")):
            sp_ax.add(MathTex(s_, font_size=22, color=GREY_B).move_to([lx(l), syb - 0.25, 0]))
        lines_l = np.linspace(0.38, 0.78, 70)

        def spectrum():
            S_ = tab(spectab, dlam_t.get_value())
            g = VGroup()
            hv = np.interp(lines_l, lam_ax, S_)
            for l, v in zip(lines_l, hv):
                if v > 0.01:
                    g.add(Line([lx(l), syb, 0], [lx(l), syb + sh * v, 0]).set_stroke(wl_color(l), 3.2, opacity=0.8))
            g.add(poly_line(lx(lam_ax), syb + sh * S_, WHITE, 2.5))
            half = lam_ax[S_ >= 0.5]
            if half.size > 1 and lx(half.max()) - lx(half.min()) > 0.12:
                yh = syb + 0.5 * sh
                g.add(fast_arrow([lx(lam0), yh, 0], [lx(half.min()), yh, 0], WHITE, 2, 0.08),
                      fast_arrow([lx(lam0), yh, 0], [lx(half.max()), yh, 0], WHITE, 2, 0.08))
            return g
        op_sp = ValueTracker(0.0)
        sp_m = opaque(spectrum, op_sp)
        def dl_pos():
            S_ = tab(spectab, dlam_t.get_value())
            half = lam_ax[S_ >= 0.5]
            right = lx(half.max()) if half.size else lx(lam0)
            return [right + 0.38, syb + 0.5 * sh + 0.02, 0]
        dl_lab = follow(MathTex("\\Delta\\lambda", font_size=28), dl_pos, op_sp)
        ft = CurvedArrow([-2.2, 1.45, 0], [-0.95, 0.12, 0], angle=-PI / 3, color=GREY_A, stroke_width=3)
        ft_l = Tex("Fourier\\\\transform", font_size=28, color=GREY_A).move_to([-0.3, 1.38, 0])
        self.at(S("t5"))
        self.add(sp_m, dl_lab)
        self.play(FadeIn(sp_ax), op_sp.animate.set_value(1.0), run_time=0.8)
        self.play(Create(ft), FadeIn(ft_l), run_time=0.8)
        self.at(W("t5", "narrow") - 0.2)
        self.play(dlam_t.animate.set_value(0.025), run_time=2.2)

        # ---------------- t6: laser, LED, sunlight
        lab = [Tex("laser: $\\ell_c\\sim$ cm to km", font_size=32), Tex("LED: $\\ell_c\\approx 10\\,\\mu$m", font_size=32),
               Tex("sunlight: $\\ell_c\\approx 1\\,\\mu$m", font_size=32)]
        for i, l_ in enumerate(lab):
            l_.move_to([3.6, 2.8 - 0.56 * i, 0])
        pre = [(W("t6", "laser"), 0.0012), (W("t6", "led"), 0.040), (W("t6", "sunlight"), 0.300)]
        for i, (tt, d) in enumerate(pre):
            self.at(tt - 0.15)
            anims = [dlam_t.animate.set_value(d), FadeIn(lab[i], shift=0.1 * LEFT)]
            if i:
                anims.append(lab[i - 1].animate.set_color(GREY_B))
            else:
                anims.append(FadeOut(VGroup(ft, ft_l)))
            self.play(*anims, run_time=1.2)
        self.finish()


# ============================================================ 3  size ==============
class Ch3Size(Chapter):
    CH = 3
    TAG = "3\\quad Size: spatial coherence"

    def construct(self):
        S = self.S
        self.show_tag()
        sd = load("spatial.npz")
        K = 11
        uk = (np.arange(K) - (K - 1) / 2) / K                 # source points, fractions of the source width
        xs_, xb, xsc, yg = -6.2, -3.7, -2.1, 0.35
        xp_ = xs_ + 0.5                                        # pinhole plate, just in front of the LED
        w = ValueTracker(0.0)                                  # effective source width, units of lambda/d (fringes vanish at 1)
        dsep = ValueTracker(1.0)                               # slit separation (units of the initial d)
        single = ValueTracker(1.0)                             # 1: show one point only
        ys1 = ValueTracker(0.0)                                # its offset (s2)
        op_pts, op_geo, op_pin = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)
        hs = 0.85

        def src_y():
            if single.get_value() > 0.5:
                return np.array([yg + ys1.get_value()])
            return yg + 2 * hs * w.get_value() * uk

        def slits():
            d = 0.5 * dsep.get_value()
            return yg + d, yg - d

        def geometry():
            g = VGroup()
            o = op_geo.get_value()
            sa, sb_ = slits()
            gap = 0.07
            for seg in ((2.15, sa + gap), (sa - gap, sb_ + gap), (sb_ - gap, -1.45)):
                if seg[0] > seg[1]:
                    g.add(Line([xb, seg[0], 0], [xb, seg[1], 0]).set_stroke(WHITE, 5, opacity=o))
            g.add(Line([xsc, -1.45, 0], [xsc, 2.15, 0]).set_stroke(GREY_B, 3, opacity=o))
            po, pp = op_pts.get_value(), op_pin.get_value()
            x0 = xs_ + pp * (xp_ - xs_)                        # rays start at the source, or at the pinhole's opening
            for y in src_y():
                for s_ in (sa, sb_):
                    g.add(Line([x0, y, 0], [xb, s_, 0]).set_stroke(C_REF, 1.5, opacity=0.35 * po))
                g.add(Dot([x0, y, 0], radius=0.07 * (1 - 0.5 * pp), color=C_REF).set_opacity(po))
            if pp > 0.002:                                     # the LED stays big behind the plate
                for y in yg + 2 * hs * uk:
                    g.add(Dot([xs_, y, 0], radius=0.07, color=C_REF).set_opacity(po * pp))
                h = 0.91 * hs * w.get_value() + 0.04
                g.add(Line([xp_, yg + h, 0], [xp_, yg + 1.4, 0]).set_stroke(WHITE, 4, opacity=pp),
                      Line([xp_, yg - h, 0], [xp_, yg - 1.4, 0]).set_stroke(WHITE, 4, opacity=pp))
            return g
        geo = always_redraw(geometry)
        lab_src = Tex("source", font_size=26, color=C_REF).move_to([xs_, -1.75, 0])
        lab_sl = Tex("two slits", font_size=26, color=GREY_A).move_to([xb, -1.75, 0])
        lab_sc = Tex("screen", font_size=26, color=GREY_B).move_to([xsc, -1.75, 0])

        # right: the copies (waterfall) and their sum
        xw0, xw1 = -1.05, 6.45
        Lw = (xw1 - xw0) / 3.0                                 # three fringe periods
        xc = 0.5 * (xw0 + xw1)
        xg = np.linspace(xw0, xw1, 300)
        ytop, dy, ac = 2.85, 0.27, 0.17
        ysum, asum = -2.62, 0.6
        col = ValueTracker(0.0)                                # 0: waterfall, 1: copies collapsed onto the sum
        op_cp, op_sum = ValueTracker(0.0), ValueTracker(0.0)

        def shifts():                                          # fringe shift (periods) = d * tilt / lambda
            if single.get_value() > 0.5:
                return np.array([-ys1.get_value() / (2 * hs)])
            return -w.get_value() * dsep.get_value() * uk

        def copy_curve(sh):
            return 1 + np.cos(TAU_ * ((xg - xc) / Lw - sh))

        def copies():
            g = VGroup()
            o = op_cp.get_value()
            if o <= 0.002:
                return g
            c = col.get_value()
            pk = []
            for k, sh in enumerate(shifts()):
                yk = (1 - c) * (ytop - dy * k) + c * ysum
                sc = (1 - c) * ac + c * asum
                g.add(poly_line(xg, yk + sc * copy_curve(sh), C_REF, 2.0 - 0.6 * c, o * (0.85 - 0.6 * c)))
                pk.append([xc + Lw * sh, yk + 2 * sc, 0])
            if c < 0.5 and len(pk) > 1:
                g.add(VMobject().set_points_as_corners(pk).set_stroke(WHITE, 1.5, opacity=0.5 * o * (1 - 2 * c)))
                for p in pk:
                    g.add(Dot(p, radius=0.035, color=WHITE).set_opacity(0.8 * o * (1 - 2 * c)))
            return g
        cp_m = always_redraw(copies)

        def vis():
            return abs(np.mean(np.exp(1j * TAU_ * shifts())))

        def sumcurve():
            I = np.mean([copy_curve(sh) for sh in shifts()], axis=0)
            return VGroup(Line([xw0, ysum, 0], [xw1, ysum, 0]).set_stroke(GREY_B, 2),
                          poly_line(xg, ysum + asum * I, WHITE, 4.5))
        sum_m = opaque(sumcurve, op_sum)
        cam_l = Tex("camera", font_size=26, color=GREY_A).move_to([xw0 + 0.55, ysum + 1.45, 0])
        sum_l = Tex("camera: the sum of the copies", font_size=26, color=GREY_A).move_to([xw0 + 1.85, ysum + 1.45, 0])
        dc, dr = np.array([-4.9, -2.55, 0]), 0.6
        dstat = dial_static(dc, dr, size=28)
        dig = Digits(26)
        op_d = ValueTracker(0.0)
        dial_m = opaque(lambda: VGroup(dial_needle(dc, dr, vis()), dig.make(f"{vis():.2f}", dc[0] + 0.85, dc[1] - 0.03)), op_d)

        # ---------------- s1: one point vs a crowd
        self.add(geo)
        self.at(S("s1"))
        self.play(op_geo.animate.set_value(1.0), FadeIn(VGroup(lab_sl, lab_sc)), run_time=0.8)
        self.add(sum_m)
        self.at(W("s1", "laser") - 0.1)
        self.play(op_pts.animate.set_value(1.0), op_sum.animate.set_value(1.0), FadeIn(lab_src), FadeIn(cam_l), run_time=0.9)
        crowd = VGroup(*[Dot([xs_, yg + 2 * hs * 0.6 * u, 0], radius=0.07, color=C_REF) for u in uk])
        self.at(W("s1", "crowd") - 0.3)
        self.play(LaggedStart(*[FadeIn(d_, scale=1.6) for d_ in crowd], lag_ratio=0.08), run_time=1.0)
        self.play(FadeOut(crowd), run_time=0.5)

        # ---------------- s2: one point off centre -> tilted wave -> shifted fringes
        def wavefronts():
            g = VGroup()
            if single.get_value() < 0.5:
                return g
            p = np.array([xs_, yg + ys1.get_value(), 0])
            tgt = np.array([xb, yg, 0])
            dist = np.linalg.norm(tgt - p)
            a0 = np.arctan2(tgt[1] - p[1], tgt[0] - p[0])
            for j in range(3):
                r = dist - 0.12 - 0.42 * j
                g.add(Arc(radius=r, start_angle=a0 - 0.32, angle=0.64, arc_center=p).set_stroke(C_REF, 2.5, opacity=0.8 - 0.2 * j))
            return g
        op_wf = ValueTracker(0.0)
        wf_m = opaque(wavefronts, op_wf)
        ref = DashedLine([xc, ysum, 0], [xc, ysum + 2 * asum + 0.1, 0], dash_length=0.06).set_stroke(GREY_A, 1.5)
        shift_ar = always_redraw(lambda: fast_arrow([xc, ysum + 2 * asum + 0.25, 0],
                                                    [xc + Lw * shifts()[0], ysum + 2 * asum + 0.25, 0], YELLOW, 3, 0.12)
                                 if single.get_value() > 0.5 and abs(shifts()[0]) > 0.04 else VGroup())
        self.at(S("s2"))
        self.add(wf_m, ref, shift_ar)
        self.play(op_wf.animate.set_value(1.0), FadeIn(ref), run_time=0.6)
        self.at(W("s2", "tilted") - 0.4)
        self.play(ys1.animate.set_value(0.55), run_time=1.6)
        self.play(ys1.animate.set_value(-0.35), run_time=1.4)

        # ---------------- s3: many independent points -> a sum of shifted copies
        self.at(S("s3"))
        self.play(op_wf.animate.set_value(0.0), ys1.animate.set_value(0.0), FadeOut(ref), run_time=0.6)
        self.remove(wf_m, shift_ar)
        single.set_value(0.0)                                  # w = 0: all copies coincide, nothing jumps
        self.add(cp_m)
        self.play(op_cp.animate.set_value(1.0), w.animate.set_value(0.32), run_time=1.2)
        self.add(dial_m)
        self.at(W("s3", "sum") - 0.4)
        self.play(FadeIn(dstat), op_d.animate.set_value(1.0), FadeTransform(cam_l, sum_l), run_time=0.9)

        # ---------------- s4: the source grows -> copies fill each other's dark bands -> flat grey (signature)
        self.at(S("s4") + 0.2)
        self.play(w.animate.set_value(1.0), run_time=4.4, rate_func=smooth)
        self.play(col.animate.set_value(1.0), run_time=1.4)

        # ---------------- s5: theta_s ~ lambda/d
        def ang():
            ys = src_y()
            m_ = np.array([xb, yg, 0])
            x0 = xs_ + op_pin.get_value() * (xp_ - xs_)
            top, bot = np.array([x0, ys.max(), 0]), np.array([x0, ys.min(), 0])
            a1, a2 = np.arctan2(top[1] - yg, top[0] - xb), np.arctan2(bot[1] - yg, bot[0] - xb)
            g = VGroup(fast_dashed(m_, top, 0.06, YELLOW, 1.5), fast_dashed(m_, bot, 0.06, YELLOW, 1.5))
            span = (a2 - a1) % TAU_
            if 1e-3 < span < PI:
                g.add(Arc(radius=1.0, start_angle=a1, angle=span, arc_center=m_).set_stroke(YELLOW, 2.5))
            return g
        op_an = ValueTracker(0.0)
        an_m = opaque(ang, op_an)
        th_l = follow(MathTex("\\theta_s", font_size=30, color=YELLOW), lambda: [xb - 1.27, yg + 0.02, 0], op_an)

        def dmark():
            sa, sb_ = slits()
            return VGroup(fast_arrow([xb + 0.28, yg, 0], [xb + 0.28, sa, 0], WHITE, 2.5, 0.1),
                          fast_arrow([xb + 0.28, yg, 0], [xb + 0.28, sb_, 0], WHITE, 2.5, 0.1))
        dm_m = opaque(dmark, op_an)
        d_l = follow(MathTex("d", font_size=30), lambda: [xb + 0.52, yg + 0.02, 0], op_an)
        rule = VGroup(Tex("fringes vanish when", font_size=32), MathTex("\\theta_s", "\\approx", "{\\lambda", "\\over", "d}", font_size=40))
        rule.arrange(RIGHT, buff=0.25).move_to([-4.15, 2.85, 0])
        rule[1][0].set_color(YELLOW)
        self.at(S("s5"))
        self.add(an_m, th_l, dm_m, d_l)
        self.play(op_an.animate.set_value(1.0), run_time=0.8)
        self.at(W("s5", "lambda") - 0.3)
        self.play(Write(rule), run_time=1.2)

        # ---------------- s6: a pinhole (or distance) shrinks the source -> fringes return
        pin_l = Tex("pinhole", font_size=26, color=GREY_A).move_to([xp_ + 0.1, yg + 1.65, 0])
        star = Star(n=5, outer_radius=0.22, inner_radius=0.09, color=C_REF).set_fill(C_REF, 1).set_stroke(width=0).move_to([xs_, yg, 0])
        star_l = Tex("a distant star", font_size=26, color=C_REF).move_to([xs_ + 0.35, yg + 0.55, 0])
        self.at(S("s6"))
        self.play(op_pin.animate.set_value(1.0), FadeIn(pin_l), FadeOut(rule), run_time=0.5)
        self.play(w.animate.set_value(0.06), run_time=1.3)
        self.at(W("s6", "distance") - 0.2)
        self.play(op_pin.animate.set_value(0.0), FadeOut(pin_l), FadeIn(star, scale=0.5), FadeIn(star_l), run_time=0.8)

        # ---------------- s7: van Cittert-Zernike: visibility vs slit separation = |FT of the source shape|
        q, Vs = sd["q"], sd["V_strip"]
        px0, px1, pyb, ph = -0.6, 6.2, -2.45, 2.55
        qmax = 2.5
        qx = lambda v: px0 + v / qmax * (px1 - px0)
        pax = VGroup(Line([px0, pyb, 0], [px1 + 0.15, pyb, 0]).set_stroke(GREY_B, 2), Line([px0, pyb, 0], [px0, pyb + ph + 0.2, 0]).set_stroke(GREY_B, 2),
                     MathTex("V", font_size=30, color=GREY_B).move_to([px0 - 0.3, pyb + ph, 0]),
                     MathTex("1", font_size=24, color=GREY_B).move_to([px0 - 0.22, pyb + ph * 0.92, 0]))
        tick_s = VGroup(Tex("slit separation $d$", font_size=26, color=GREY_B).move_to([px1 - 1.1, pyb - 0.35, 0]),
                        MathTex("\\lambda/\\theta_s", font_size=26, color=GREY_B).move_to([qx(1.0), pyb - 0.35, 0]),
                        Line([qx(1.0), pyb - 0.07, 0], [qx(1.0), pyb + 0.07, 0]).set_stroke(GREY_B, 2))
        sel = q <= qmax
        vcurve = poly_line(qx(q[sel]), pyb + ph * 0.92 * Vs[sel], WHITE, 4)
        q_now = lambda: w.get_value() * dsep.get_value()
        op_q = ValueTracker(0.0)
        qdot = opaque(lambda: Dot([qx(min(q_now(), qmax)), pyb + ph * 0.92 * float(np.interp(q_now(), q, Vs)), 0], radius=0.08, color=YELLOW), op_q)
        src_box = VGroup(Line([0.3, 1.2, 0], [0.7, 1.2, 0]), Line([0.7, 1.2, 0], [0.7, 1.95, 0]), Line([0.7, 1.95, 0], [1.6, 1.95, 0]),
                         Line([1.6, 1.95, 0], [1.6, 1.2, 0]), Line([1.6, 1.2, 0], [2.0, 1.2, 0])).set_stroke(C_REF, 3.5)
        src_l = Tex("source shape", font_size=26, color=C_REF).move_to([1.15, 0.9, 0])
        ftar = Arrow([2.2, 1.55, 0], [3.5, 1.55, 0], buff=0, stroke_width=4, color=GREY_A, max_tip_length_to_length_ratio=0.18)
        ft_l = Tex("Fourier transform", font_size=26, color=GREY_A).next_to(ftar, UP, buff=0.1)
        vabs = Tex("visibility vs $d$", font_size=28, color=WHITE).next_to(ftar, RIGHT, buff=0.2)
        vcz = Tex("van Cittert--Zernike theorem", font_size=34, color=YELLOW).move_to([3.0, 2.85, 0])
        self.at(S("s7"))
        self.play(op_cp.animate.set_value(0.0), op_sum.animate.set_value(0.0), FadeOut(sum_l), FadeOut(star), FadeOut(star_l),
                  w.animate.set_value(0.5), run_time=0.8)
        self.remove(cp_m, sum_m)
        self.add(qdot)
        self.play(FadeIn(pax), FadeIn(tick_s), Create(vcurve), op_q.animate.set_value(1.0), run_time=1.0)
        self.play(dsep.animate.set_value(2.0), run_time=2.2)
        self.play(FadeIn(src_box), FadeIn(src_l), GrowArrow(ftar), FadeIn(ft_l), FadeIn(vabs), run_time=0.8)
        self.at(W("s7", "van") - 0.2)
        self.play(Write(vcz), run_time=1.0)

        # ---------------- s8: Michelson & Pease, 1920: Betelgeuse
        dm, Vd, d0 = sd["dm"], sd["V_disk"], float(sd["d0"])
        dmax = 4.6
        mx = lambda v: px0 + v / dmax * (px1 - px0)
        dsel = dm <= dmax
        vdisk = poly_line(mx(dm[dsel]), pyb + ph * 0.92 * Vd[dsel], WHITE, 4)
        mt = VGroup(*[VGroup(Line([mx(v), pyb - 0.07, 0], [mx(v), pyb + 0.07, 0]).set_stroke(GREY_B, 2),
                             MathTex(f"{v}\\,\\mathrm{{m}}", font_size=24, color=GREY_B).move_to([mx(v), pyb - 0.35, 0])) for v in (1, 2, 3)])
        mtl = Tex("mirror separation $d$", font_size=26, color=GREY_B).move_to([px1 - 1.0, pyb + 0.75, 0])
        dd = ValueTracker(0.0)
        op_m = ValueTracker(0.0)
        mdot = opaque(lambda: Dot([mx(dd.get_value()), pyb + ph * 0.92 * float(np.interp(dd.get_value(), dm, Vd)), 0], radius=0.08,
                                  color=YELLOW), op_m)
        betel = VGroup(Circle(radius=0.2).set_fill(ORANGE, 1).set_stroke(width=0).move_to([xs_, yg, 0]),
                       Tex("Betelgeuse", font_size=26, color=ORANGE).move_to([xs_ + 0.25, yg + 0.5, 0]))
        mich = Tex("Michelson \\& Pease, 1920", font_size=32, color=YELLOW).move_to([3.0, 2.85, 0])
        res = VGroup(Tex("fringes vanish at $d\\approx 3$ m", font_size=28),
                     Tex("$\\Rightarrow$ angular size $\\approx 0.047''$", font_size=28)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        res.move_to([3.25, 1.05, 0])
        mir_l = Tex("two mirrors", font_size=26, color=GREY_A).move_to(lab_sl)
        self.at(S("s8"))
        self.play(FadeOut(VGroup(src_box, src_l, ftar, ft_l, vabs, vcz, tick_s)), op_q.animate.set_value(0.0), op_pts.animate.set_value(0.0),
                  op_an.animate.set_value(0.0), op_d.animate.set_value(0.0), FadeOut(dstat), FadeOut(lab_src),
                  ReplacementTransform(vcurve, vdisk), FadeTransform(lab_sl, mir_l), dsep.animate.set_value(1.0), run_time=0.9)
        self.remove(qdot, an_m, th_l, dm_m, d_l, dial_m)
        self.add(mdot)
        self.play(FadeIn(mich), FadeIn(mt), FadeIn(mtl), FadeIn(betel), op_m.animate.set_value(1.0), run_time=0.8)
        self.at(W("s8", "apart") - 0.3)
        self.play(dd.animate.set_value(d0), dsep.animate.set_value(1.8), run_time=2.8, rate_func=smooth)
        self.play(FadeIn(res, shift=0.1 * UP), run_time=0.8)
        self.finish()


# ============================================================ 4  speckle ===========
class Ch4Speckle(Chapter):
    CH = 4
    TAG = "4\\quad Speckle"

    def construct(self):
        S = self.S
        self.show_tag()
        sp = load("speckle.npz")
        I1, avg, keepN, Cn = sp["I1"], sp["avg"].astype(np.float32), sp["keep_N"], sp["C"]
        hist, edges, pos, ph0, u = sp["hist"], sp["edges"], sp["pos"], sp["ph"], sp["u"]
        U = float(u[-1])
        rng = np.random.default_rng(21)

        # ---------------- p1: a rough wall
        xwall = -6.2
        yw = np.linspace(-2.3, 2.5, 90)
        hw = 0.07 * ndimage_smooth(rng.standard_normal(90))
        wall = VGroup(poly_line(xwall + hw, yw, GREY_B, 3),
                      *[Line([xwall + hw[i], yw[i], 0], [xwall - 0.28, yw[i] - 0.22, 0]).set_stroke(GREY_D, 1.5) for i in range(0, 90, 6)])
        wall_l = Tex("rough wall", font_size=26, color=GREY_B).move_to([xwall + 0.3, -2.65, 0])
        spot = (-1.2, 1.6)
        bumps_y = np.sort(rng.uniform(*spot, 40))
        bumps = [np.array([xwall + np.interp(y, yw, hw), y, 0]) for y in bumps_y]
        Ppix = np.array([-2.95, 0.2, 0])
        beam = VGroup(*[Line([-3.4, y, 0], [xwall + 0.15, y, 0]).set_stroke(C_U, 2, opacity=0.6) for y in np.linspace(spot[0], spot[1], 7)])
        beam_t = Tex("laser", font_size=26, color=C_U).move_to([-3.75, spot[1] + 0.3, 0])
        self.at(S("p1"))
        self.play(Create(wall[0]), FadeIn(wall[1:]), FadeIn(wall_l), run_time=1.0)
        self.play(LaggedStart(*[GrowFromEdge(b, RIGHT) for b in beam], lag_ratio=0.1), FadeIn(beam_t), run_time=1.0)

        # ---------------- p2: at a pixel, arrows with random phases add like a random walk (signature)
        # 40 scatterers of the computed speckle pattern: arrow j at direction (u, v) has phase ph_j - 2 pi (xi_j u + eta_j v)
        def phases(uv):
            return ph0 - TAU_ * (pos[:, 0] * uv[0] + pos[:, 1] * uv[1])
        n = I1.shape[0]
        vmax_img = np.percentile(I1, 99.6)
        g_ = np.linspace(-U, U, n)
        cen = slice(int(0.3 * n), int(0.7 * n))
        sub = I1[cen, cen]
        mx_ = (sub == ndimage_max(sub)) & (sub > 3.0) & (sub < 4.5)            # grain peaks of moderate brightness
        cand = np.argwhere(mx_)
        ib = cand[np.argmin(np.hypot(*(cand - np.array(sub.shape) / 2).T))]
        bright = np.array([g_[cen][ib[1]], g_[cen][ib[0]]])
        dcand = np.argwhere(sub < 0.01)
        dd_ = np.hypot(*(dcand - ib).T)
        dk = dcand[np.argmin(np.abs(dd_ - 16))]                                # a dark spot a grain or so away
        dark = np.array([g_[cen][dk[1]], g_[cen][dk[0]]])
        mcand = np.argwhere((sub > 0.9) & (sub < 1.2))
        mk_ = mcand[np.argmin(np.hypot(*(mcand - (ib + np.array([-20, 14]))).T))]
        mid = np.array([g_[cen][mk_[1]], g_[cen][mk_[0]]])
        wand = np.array([0.16, 0.0])
        path = ValueTracker(0.0)                               # 0 mid -> 1 dark -> 2 bright -> 3 wander

        def uv_at():
            p = path.get_value()
            pts = [mid, dark, bright, wand]
            i = int(np.clip(np.floor(p), 0, 2))
            f = smoothstep(p - i)
            return (1 - f) * pts[i] + f * pts[i + 1]
        La = 0.25
        Wc = np.array([-0.15, -0.2, 0])                        # the walk is kept centred on Wc
        nshow = ValueTracker(0.0)
        op_walk = ValueTracker(1.0)

        def walk():
            g = VGroup()
            o = op_walk.get_value()
            k = int(nshow.get_value())
            if k < 1 or o <= 0.002:
                return g
            z = np.exp(1j * phases(uv_at()))
            cs = np.concatenate([[0], np.cumsum(z)])
            pts = La * np.stack([cs.real, cs.imag], 1)
            ctr = 0.5 * (pts.min(0) + pts.max(0))
            W0 = Wc - np.array([ctr[0], ctr[1], 0])
            for j in range(k):
                p_ = W0 + np.array([pts[j, 0], pts[j, 1], 0])
                q_ = W0 + np.array([pts[j + 1, 0], pts[j + 1, 1], 0])
                hue = (np.angle(z[j]) % TAU_) / TAU_
                g.add(fast_arrow(p_, q_, rgb_to_color(PHASE_HUE[int(hue * 2047)] / 255), 3.5, 0.09, o))
            if k >= 40:
                g.add(fast_arrow(W0, W0 + np.array([pts[-1, 0], pts[-1, 1], 0]), YELLOW, 5, 0.2, o))
            g.add(Dot(W0, radius=0.05, color=WHITE).set_opacity(o))
            return g
        wk_m = always_redraw(walk)
        wk_l = Tex("sum of 40 arrows", font_size=28, color=GREY_A).move_to([-0.15, 2.75, 0])

        def pix_at():
            return Ppix + np.array([0, 1.4 * uv_at()[1] / U, 0])
        op_rays = ValueTracker(0.0)
        rays = opaque(lambda: VGroup(*[Line(b_, pix_at()).set_stroke(GREY_A, 1, opacity=0.45) for b_ in bumps],
                                     Square(side_length=0.16).move_to(pix_at()).set_stroke(WHITE, 2)), op_rays)
        pix_l = follow(Tex("pixel", font_size=26, color=GREY_A), lambda: pix_at() + 0.32 * UP, op_rays)
        self.at(S("p2"))
        self.add(rays, pix_l)
        self.play(op_rays.animate.set_value(1.0), run_time=1.0)
        self.add(wk_m)
        self.at(W("p2", "arrows") - 0.3)
        self.play(nshow.animate.set_value(40.0), FadeIn(wk_l), run_time=2.3, rate_func=linear)

        # ---------------- p3: cancel / line up (other pixels) -> the computed speckle image; the probe is the pixel
        ic, isz = np.array([4.6, 0.15, 0]), 4.0
        im, im_f, im_sw = self.sweep_image(heat_rgba(I1, vmax_img, gamma=0.8), isz, ic, isz, axis="right")
        im_box = Square(side_length=isz).move_to(ic).set_stroke(GREY_B, 1.5)
        im_l = Tex("speckle (computed)", font_size=28, color=GREY_A).next_to(im_box, UP, buff=0.12)

        def to_scene(uv):
            return ic + np.array([uv[0] / U * isz / 2, uv[1] / U * isz / 2, 0])
        op_pb = ValueTracker(0.0)
        probe = opaque(lambda: Circle(radius=0.17).move_to(to_scene(uv_at())).set_stroke(YELLOW, 3), op_pb)
        self.at(W("p3", "cancel") - 0.45)
        self.play(path.animate.set_value(1.0), run_time=1.0)
        self.at(W("p3", "line") - 0.45)
        self.play(path.animate.set_value(2.0), run_time=1.0)
        self.add(im, probe)
        self.at(W("p3", "dark") - 0.25)
        self.play(im_f.animate.set_value(1.0), FadeIn(im_box), op_pb.animate.set_value(1.0), run_time=0.4)
        self.play(im_sw.animate.set_value(1.0), run_time=0.8, rate_func=linear)
        self.at(W("p3", "speckle") - 0.2)
        self.play(FadeIn(im_l), run_time=0.5)

        # ---------------- p4: contrast = std / mean = 1 (computed histogram vs exponential)
        hx0, hy0, hw_, hh = -2.25, -2.5, 4.1, 3.0
        bins = VGroup(*[Rectangle(width=hw_ / 5 * (b1 - b0) * 0.92, height=max(hh * v, 1e-3)).set_stroke(width=0).set_fill(C_U, 0.75)
                        .move_to([hx0 + hw_ / 5 * 0.5 * (b0 + b1), hy0 + hh * v / 2, 0]) for b0, b1, v in zip(edges[:-1], edges[1:], hist)])
        xe = np.linspace(0, 5, 200)
        expc = poly_line(hx0 + hw_ / 5 * xe, hy0 + hh * np.exp(-xe), YELLOW, 3.5)
        hax = VGroup(Line([hx0, hy0, 0], [hx0 + hw_ + 0.15, hy0, 0]).set_stroke(GREY_B, 2),
                     Line([hx0, hy0, 0], [hx0, hy0 + hh + 0.15, 0]).set_stroke(GREY_B, 2),
                     MathTex("I/\\langle I\\rangle", font_size=26, color=GREY_B).move_to([hx0 + hw_ + 0.15, hy0 - 0.62, 0]),
                     MathTex("e^{-I/\\langle I\\rangle}", font_size=30, color=YELLOW).move_to([hx0 + 1.85, hy0 + 1.25, 0]))
        for v in (1, 2, 3, 4):
            hax.add(MathTex(str(v), font_size=22, color=GREY_B).move_to([hx0 + hw_ / 5 * v, hy0 - 0.25, 0]))
        cform = MathTex("\\text{contrast}", "=", "{\\sigma_I", "\\over", "\\langle I\\rangle}", "=", "1", font_size=38).move_to([-0.1, 1.85, 0])
        cform[6].set_color(YELLOW)
        self.at(S("p4"))
        self.play(op_walk.animate.set_value(0.0), FadeOut(wk_l), path.animate.set_value(3.0), run_time=0.6)
        self.remove(wk_m)
        self.play(FadeIn(hax[:2]), LaggedStart(*[GrowFromEdge(b_, DOWN) for b_ in bins], lag_ratio=0.04), FadeIn(hax[2:]),
                  run_time=1.2)
        self.play(Create(expc), Write(cform), run_time=1.2)

        # ---------------- p5: grain ~ lambda / NA ; NA = sin(theta_max)
        lam_na = 1.0 / float(sp["A"])                             # lambda/NA in direction-sine units (NA = A/z, lambda = 1)
        blen = lam_na / U * isz / 2
        gr = np.array([bright[0], bright[1]])
        gp = to_scene(gr)
        sbar = VGroup(Line(gp + np.array([-blen / 2, -0.32, 0]), gp + np.array([blen / 2, -0.32, 0])).set_stroke(WHITE, 4),
                      Line(gp + np.array([-blen / 2, -0.4, 0]), gp + np.array([-blen / 2, -0.24, 0])).set_stroke(WHITE, 3),
                      Line(gp + np.array([blen / 2, -0.4, 0]), gp + np.array([blen / 2, -0.24, 0])).set_stroke(WHITE, 3))
        sbar_l = MathTex("\\lambda/\\mathrm{NA}", font_size=30).next_to(sbar, DOWN, buff=0.1)
        sbar_l = VGroup(BackgroundRectangle(sbar_l, fill_opacity=0.75, buff=0.06).set_fill(BLACK), sbar_l)
        top_, bot_ = np.array([xwall, spot[1], 0]), np.array([xwall, spot[0], 0])
        cone = VGroup(DashedLine(Ppix, top_, dash_length=0.07).set_stroke(YELLOW, 2), DashedLine(Ppix, bot_, dash_length=0.07).set_stroke(YELLOW, 2))
        a_t, a_b = np.arctan2(top_[1] - Ppix[1], top_[0] - Ppix[0]), np.arctan2(bot_[1] - Ppix[1], bot_[0] - Ppix[0])
        mid_a = np.arctan2(0.5 * (top_[1] + bot_[1]) - Ppix[1], xwall - Ppix[0])
        th_arc = Arc(radius=0.75, start_angle=mid_a, angle=a_t - mid_a, arc_center=Ppix).set_stroke(YELLOW, 3)
        th_lab = MathTex("\\theta_{\\max}", font_size=28, color=YELLOW).move_to(Ppix + 1.05 * np.array([np.cos(mid_a + 0.25), np.sin(mid_a + 0.25), 0]))
        na_t = MathTex("\\mathrm{NA}", "=", "\\sin\\theta_{\\max}", font_size=34).move_to([-4.6, -2.55, 0])
        na_t[2].set_color(YELLOW)
        self.at(S("p5"))
        self.play(FadeOut(VGroup(bins, expc, hax, cform)), op_rays.animate.set_value(0.0), FadeOut(wall_l), op_pb.animate.set_value(0.0),
                  run_time=0.5)
        self.remove(probe, rays, pix_l)
        pix = Square(side_length=0.16).move_to(Ppix).set_stroke(WHITE, 2)
        self.add(pix)
        self.play(Create(sbar), FadeIn(sbar_l), run_time=0.8)
        self.at(W("p5", "numerical") - 0.2)
        self.play(Create(cone), Create(th_arc), FadeIn(th_lab), run_time=1.0)
        self.play(Write(na_t), run_time=0.9)

        # ---------------- p6: averaging N independent patterns -> contrast 1/sqrt(N)
        rgbs = [heat_rgba(a_, vmax_img, gamma=0.8) for a_ in avg]
        nf = ValueTracker(0.0)

        def avg_fn(t):
            x = nf.get_value()
            i = int(np.clip(np.floor(x), 0, len(rgbs) - 2))
            f = float(np.clip(x - i, 0, 1))
            if f < 1e-3:
                return rgbs[i]
            out = rgbs[i].copy()
            out[..., :3] = ((1 - f) * rgbs[i][..., :3] + f * rgbs[i + 1][..., :3]).astype(np.uint8)
            return out
        av_m, av_f = self.live_image(avg_fn, isz, ic, isz)
        Nnow = lambda: float(np.interp(nf.get_value(), np.arange(len(keepN)), keepN))
        cx0, cy0, cw, ch = -2.3, -2.45, 4.3, 3.6
        cxN = lambda N: cx0 + np.log2(N) / 6 * cw
        cax = VGroup(Line([cx0, cy0, 0], [cx0 + cw + 0.15, cy0, 0]).set_stroke(GREY_B, 2),
                     Line([cx0, cy0, 0], [cx0, cy0 + ch * 1.05, 0]).set_stroke(GREY_B, 2),
                     Tex("contrast", font_size=26, color=GREY_B).move_to([cx0 + 0.75, cy0 + ch * 1.05 + 0.15, 0]),
                     MathTex("1", font_size=24, color=GREY_B).move_to([cx0 - 0.2, cy0 + ch * 0.95, 0]))
        for N_ in (1, 4, 16, 64):
            cax.add(MathTex(str(N_), font_size=24, color=GREY_B).move_to([cxN(N_), cy0 - 0.27, 0]))
        cax.add(MathTex("N", font_size=28, color=GREY_B).move_to([cx0 + cw + 0.35, cy0, 0]))
        Ns = np.arange(1, 65)
        theo = poly_line(cxN(Ns), cy0 + ch * 0.95 / np.sqrt(Ns), YELLOW, 3)
        theo_l = MathTex("1/\\sqrt{N}", font_size=32, color=YELLOW).move_to([cxN(4) + 0.55, cy0 + ch * 0.95 / 2 + 0.45, 0])
        op_pts = ValueTracker(0.0)
        pts_m = opaque(lambda: VGroup(*[Dot([cxN(N_), cy0 + ch * 0.95 * Cn[N_ - 1], 0], radius=0.05, color=WHITE)
                                        for N_ in Ns if N_ <= Nnow() + 1e-6]), op_pts)
        dig = Digits(30)
        nlab = MathTex("N", "=", font_size=32).move_to([ic[0] - 0.55, ic[1] - isz / 2 - 0.32, 0])
        n_m = opaque(lambda: dig.make(f"{int(round(Nnow()))}", nlab.get_right()[0] + 0.15, nlab.get_bottom()[1]), op_pts)
        led_t = Tex("LED: many colours and points $\\Rightarrow$ smooth", font_size=28, color=GREY_A).move_to([0.1, 2.75, 0])
        self.at(S("p6"))
        self.play(FadeOut(VGroup(sbar, sbar_l, cone, th_arc, th_lab, na_t)), run_time=0.5)
        self.add(av_m, pts_m, n_m)
        self.play(av_f.animate.set_value(1.0), FadeIn(cax), FadeIn(nlab), op_pts.animate.set_value(1.0), run_time=0.6)
        im.clear_updaters()
        self.remove(im)
        self.play(nf.animate.set_value(len(keepN) - 1.0), run_time=3.0, rate_func=linear)
        self.play(Create(theo), FadeIn(theo_l), run_time=0.6)
        self.at(W("p6", "many") - 0.9)
        self.play(FadeIn(led_t), run_time=0.7)
        self.finish()


def ndimage_max(a, r=4):
    """Local maximum filter (square window of half-size r) without scipy."""
    p = np.pad(a, r, mode="edge")
    out = np.full_like(a, -np.inf)
    for dy_ in range(-r, r + 1):
        for dx_ in range(-r, r + 1):
            out = np.maximum(out, p[r + dy_:r + dy_ + a.shape[0], r + dx_:r + dx_ + a.shape[1]])
    return out


def ndimage_smooth(x, s=2.0):
    k = np.exp(-0.5 * (np.arange(-6, 7) / s) ** 2)
    return np.convolve(x, k / k.sum(), mode="same")


# ============================================================ 5  modes =============
class Ch5Modes(Chapter):
    CH = 5
    TAG = "5\\quad Coherent modes"

    def construct(self):
        S = self.S
        self.show_tag()
        md = load("modes.npz")
        x, ells, Js, lams, vecs, coef = md["x"], md["ells"], md["J"], md["lams"], md["vecs"], md["coef"]
        i0 = 26                                               # ~4 significant modes
        N = len(x)
        li = ValueTracker(float(i0))

        def idx():
            return int(np.clip(round(li.get_value()), 0, len(ells) - 1))

        # ---------------- m1: a partially coherent field, live; two probe points
        # layout 0: big and centred (m1); layout 1: left column next to J (m2 on)
        LB = dict(sx0=-4.7, sx1=4.7, ys=1.6, sh=0.75, dy=-0.3, dr=0.46)
        LS = dict(sx0=-6.2, sx1=-0.5, ys=2.35, sh=0.5, dy=1.05, dr=0.42)
        lay = ValueTracker(0.0)

        def Lv(k):
            f = lay.get_value()
            return (1 - f) * LB[k] + f * LS[k]

        def X(v):
            return Lv("sx0") + (v + 1) / 2 * (Lv("sx1") - Lv("sx0"))
        xf = np.linspace(-1, 1, 480)
        lam0, vec0 = lams[i0], vecs[i0]

        def field_at(t, xi=None):
            fi = t * 24.0
            k = int(np.floor(fi))
            f = fi - k
            c = (1 - f) * coef[:, k % coef.shape[1]] + f * coef[:, (k + 1) % coef.shape[1]]
            v = vec0[:8] if xi is None else np.array([np.interp(xi, x, vv) for vv in vec0[:8]])     # (8, n)
            return (np.sqrt(lam0[:8])[:, None] * c[:, None] * v).sum(0)

        def strip(t):
            Uf = field_at(t)
            Ui = np.interp(xf, x, Uf.real) + 1j * np.interp(xf, x, Uf.imag)
            row = phase_rgba(Ui[None, :], vmax=1.5, gamma=0.7, flip=False)[0]
            return np.repeat(row[None], 36, axis=0)

        def strip_set(L):
            c_ = [0.5 * (L["sx0"] + L["sx1"]), L["ys"], 0]
            m_, f_ = self.live_image(strip, L["sx1"] - L["sx0"], c_, L["sh"])
            box = Rectangle(width=L["sx1"] - L["sx0"], height=L["sh"]).move_to(c_).set_stroke(GREY_B, 1.5)
            lab = Tex("a partially coherent field (colour $=$ phase)", font_size=28 if L is LB else 26,
                      color=GREY_B).next_to(box, UP, buff=0.12)
            return m_, f_, box, lab
        stB_m, stB_f, stB_box, stB_l = strip_set(LB)
        st_m, st_f, st_box, st_l = strip_set(LS)
        x1t, x2t = ValueTracker(-0.3), ValueTracker(-0.08)
        op_pr = ValueTracker(0.0)

        def probes():
            g = VGroup()
            t = self.now
            ys_, sh_, dy_, dr = Lv("ys"), Lv("sh"), Lv("dy"), Lv("dr")
            for tr, col in ((x1t, C_A1), (x2t, C_A2)):
                xv = tr.get_value()
                c = np.array([X(xv), dy_, 0])
                u_ = complex(field_at(t, np.array([xv]))[0])
                g.add(fast_dashed([X(xv), ys_ - sh_ / 2, 0], [X(xv), dy_ + dr, 0], 0.05, col, 1.5),
                      Circle(radius=dr).move_to(c).set_stroke(GREY_B, 1.5),
                      fast_arrow(c, c + dr / 1.2 * min(abs(u_), 1.25) * np.array([np.cos(np.angle(u_)), np.sin(np.angle(u_)), 0]), col, 4, 0.12))
            return g
        pr_m = opaque(probes, op_pr)
        x1_l = follow(MathTex("x_1", font_size=30, color=C_A1), lambda: [X(x1t.get_value()), Lv("dy") - Lv("dr") - 0.28, 0], op_pr)
        x2_l = follow(MathTex("x_2", font_size=30, color=C_A2), lambda: [X(x2t.get_value()), Lv("dy") - Lv("dr") - 0.28, 0], op_pr)
        self.at(S("m1"))
        self.add(stB_m)
        self.play(stB_f.animate.set_value(1.0), FadeIn(stB_box), FadeIn(stB_l), run_time=0.8)
        self.add(pr_m, x1_l, x2_l)
        self.at(W("m1", "pair") - 0.3)
        self.play(op_pr.animate.set_value(1.0), run_time=0.7)
        self.play(x2t.animate.set_value(0.62), run_time=2.2)

        # ---------------- m2: J(x1, x2) = <U(x1) U*(x2)> ; diagonal = intensity
        jc, js = np.array([3.95, -0.6, 0]), 3.9
        jmax = float(Js.max())
        jrgb = [diverge_rgba(J_, jmax, flip=True) for J_ in Js]
        J_m, J_f = self.live_image(lambda t: jrgb[idx()], js, jc, js, nearest=True)
        J_box = Square(side_length=js).move_to(jc).set_stroke(GREY_B, 1.5)
        J_ax = VGroup(MathTex("x_1", font_size=28, color=C_A1).move_to(jc + np.array([0, -js / 2 - 0.3, 0])),
                      MathTex("x_2", font_size=28, color=C_A2).move_to(jc + np.array([-js / 2 - 0.32, 0, 0])))
        Jq = MathTex("J(x_1,x_2)", "=", "\\langle\\, U(x_1)\\, U^*(x_2)\\,\\rangle", font_size=36).move_to([3.95, 2.95, 0])
        Jq[0].set_color(YELLOW)
        jpos = lambda a, b: jc + np.array([a * js / 2, b * js / 2, 0])
        op_mk = ValueTracker(0.0)
        mk_m = opaque(lambda: Square(side_length=0.2).move_to(jpos(x1t.get_value(), x2t.get_value())).set_stroke(YELLOW, 3), op_mk)
        diag = fast_dashed(jpos(-1, -1), jpos(1, 1), 0.07, WHITE, 2.0)
        Iy0 = jc[1] + js / 2 + 0.12
        I_curve = poly_line(jc[0] + x * js / 2, Iy0 + 0.75 * np.diag(Js[i0]) / jmax, WHITE, 3.5)
        I_l = Tex("intensity", font_size=26, color=GREY_A).move_to([jc[0] + js / 2 + 0.15, Iy0 + 0.55, 0])
        self.at(S("m2"))
        self.add(st_m)
        self.play(lay.animate.set_value(1.0), stB_f.animate.set_value(0.0), FadeOut(stB_box), FadeOut(stB_l),
                  st_f.animate.set_value(1.0), FadeIn(st_box), FadeIn(st_l), run_time=0.9)
        stB_m.clear_updaters()
        self.remove(stB_m)
        self.add(J_m, mk_m)
        self.play(J_f.animate.set_value(1.0), FadeIn(J_box), FadeIn(J_ax), op_mk.animate.set_value(1.0), run_time=0.8)
        self.play(Write(Jq), run_time=1.4)
        self.at(W("m2", "diagonal") - 0.2)
        self.play(Create(diag), run_time=0.6)
        self.play(TransformFromCopy(diag, I_curve), FadeIn(I_l), run_time=1.0)

        # ---------------- m3: off the diagonal: would x1 and x2 interfere?
        off_t = Tex("$|J(x_1,x_2)|$ large $\\Leftrightarrow$ $x_1$ and $x_2$ interfere", font_size=28).move_to([-3.35, -0.25, 0])
        self.at(S("m3"))
        self.play(x2t.animate.set_value(-0.12), run_time=1.3)
        self.play(FadeIn(off_t), run_time=0.6)
        self.play(x2t.animate.set_value(0.55), run_time=1.6)

        # ---------------- m4: J(x2, x1) = J(x1, x2)*  -> Hermitian -> diagonalizable
        mk2 = opaque(lambda: Square(side_length=0.2).move_to(jpos(x2t.get_value(), x1t.get_value())).set_stroke(YELLOW, 3), op_mk)
        herm = MathTex("J(x_2,x_1)", "=", "J(x_1,x_2)^*", font_size=34).move_to([-3.35, -1.1, 0])
        herm_l = Tex("Hermitian $\\Rightarrow$ diagonalize it", font_size=30, color=YELLOW).move_to([-3.35, -1.85, 0])
        self.at(S("m4"))
        self.add(mk2)
        self.play(Write(herm), run_time=1.2)
        self.at(W("m4", "hermitian") - 0.2)
        self.play(FadeIn(herm_l), run_time=0.7)

        # ---------------- m5: eigenvectors = coherent modes, eigenvalues = their powers
        rows_y = [2.2, 1.45, 0.7, -0.05]
        mx0, mx1 = -5.9, -2.2
        MX = lambda v: mx0 + (v + 1) / 2 * (mx1 - mx0)
        mcol = [BLUE_A, BLUE_B, BLUE_C, TEAL_B]
        op_md = ValueTracker(0.0)

        def modes_rows():
            g = VGroup()
            i = idx()
            v, l_ = vecs[i], lams[i]
            for n_, y in enumerate(rows_y):
                ph_ = v[n_] / max(np.abs(vecs[i0][0]).max(), 1e-9)
                g.add(Line([mx0, y, 0], [mx1, y, 0]).set_stroke(GREY_D, 1),
                      poly_line(MX(x), y + 0.27 * ph_, mcol[n_], 3.5))
            return g
        md_m = opaque(modes_rows, op_md)
        m_labs = VGroup(*[MathTex(f"\\phi_{n_ + 1}", font_size=30, color=mcol[n_]).move_to([mx0 - 0.42, y, 0]) for n_, y in enumerate(rows_y)])

        def bars():
            g = VGroup()
            l_ = lams[idx()]
            tot = l_.sum()
            bx0 = -1.55
            for n_ in range(10):
                f = l_[n_] / tot
                yb_ = 2.3 - 0.33 * n_
                if f > 0.002:
                    g.add(Rectangle(width=max(2.6 * f, 1e-3), height=0.22).set_stroke(width=0).set_fill(YELLOW, 0.85)
                          .move_to([bx0 + 1.3 * f, yb_, 0]))
            return g
        op_b = ValueTracker(0.0)
        bar_m = opaque(bars, op_b)
        bar_l = Tex("eigenvalues $\\lambda_n$ (power)", font_size=26, color=YELLOW).move_to([-0.55, 2.85, 0])
        dec = MathTex("J", "=", "\\sum_n", "\\lambda_n", "\\,\\phi_n", "\\phi_n^{\\dagger}", font_size=40).move_to([-4.1, -1.05, 0])
        dec[3].set_color(YELLOW)
        dec[4].set_color(C_U)
        dec[5].set_color(C_U)
        self.at(S("m5"))
        self.play(FadeOut(VGroup(st_box, st_l, off_t, herm, herm_l)), st_f.animate.set_value(0.0), op_pr.animate.set_value(0.0), run_time=0.6)
        st_m.clear_updaters()
        self.remove(st_m, pr_m, x1_l, x2_l)
        self.add(md_m, bar_m)
        self.play(op_md.animate.set_value(1.0), FadeIn(m_labs), FadeIn(dec), run_time=1.2)
        self.at(W("m5", "eigenvalues") - 0.2)
        self.play(op_b.animate.set_value(1.0), FadeIn(bar_l), run_time=0.9)

        # ---------------- m6: intensities of the modes add up to I(x)
        iy0, ih = -2.85, 0.9
        op_st = ValueTracker(0.0)
        nst = ValueTracker(0.0)                                # how many modes are stacked so far

        def stack():
            g = VGroup()
            i = idx()
            v, l_ = vecs[i], lams[i]
            I0 = np.diag(Js[i]).max()
            cum = np.zeros_like(x)
            for n_ in range(16):
                f = float(np.clip(nst.get_value() - n_, 0, 1))
                if f <= 0:
                    break
                nxt = cum + f * l_[n_] * v[n_] ** 2
                pts = [[MX(a), iy0 + ih * b / I0, 0] for a, b in zip(x, nxt)] + [[MX(a), iy0 + ih * b / I0, 0] for a, b in zip(x[::-1], cum[::-1])]
                g.add(Polygon(*pts).set_stroke(width=0).set_fill(mcol[min(n_, 3)] if n_ < 4 else GREY_D, 0.85))
                cum = nxt
            if nst.get_value() >= 15.99:
                g.add(poly_line(MX(x), iy0 + ih * np.diag(Js[i]) / I0, WHITE, 2.5))
            return g
        st2 = opaque(stack, op_st)
        sum_t = MathTex("I(x)", "=", "\\sum_n", "\\lambda_n", "|\\phi_n(x)|^2", font_size=36).move_to([0.05, -2.35, 0])
        sum_t[3].set_color(YELLOW)
        self.at(S("m6"))
        self.add(st2)
        op_st.set_value(1.0)
        self.play(nst.animate.set_value(4.0), Write(sum_t), run_time=1.6, rate_func=linear)
        self.at(W("m6", "intensities") - 0.3)
        self.play(nst.animate.set_value(16.0), run_time=0.8)

        # ---------------- m7: one mode for a laser; more for a bigger source
        cnt_l = Tex("modes for 95\\% of the power:", font_size=26, color=GREY_A).move_to([-0.35, -1.2, 0])
        dig = Digits(28, YELLOW)
        frac = lams / lams.sum(1, keepdims=True)
        need = 1 + (np.cumsum(frac, 1) < 0.95).sum(1)
        cnt_m = always_redraw(lambda: dig.make(str(int(need[idx()])), cnt_l.get_right()[0] + 0.15, cnt_l.get_bottom()[1]))
        self.at(S("m7"))
        self.play(FadeIn(cnt_l), run_time=0.4)
        self.add(cnt_m)
        self.play(li.animate.set_value(0.0), run_time=1.0)
        self.at(W("m7", "bigger") - 0.2)
        self.play(li.animate.set_value(len(ells) - 1.0), run_time=2.0, rate_func=linear)
        self.finish()


# ============================================================ 6  on a computer =====
class Ch6Computer(Chapter):
    CH = 6
    TAG = "6\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        dd = load("doe.npz")
        card = code_card(["coherent    I = abs(prop(U))**2",
                          "incoherent  I = conv(I_src, blur)",
                          "partial     I = sum(lam[n] * abs(prop(phi[n]))**2)"], font_size=28, title="three recipes")
        card.move_to([0, 0.35, 0])
        rows = card[1][0]
        self.at(S("k1"))
        self.play(FadeIn(card[0]), FadeIn(card[1][1]), run_time=0.6)
        self.at(W("k1", "coherent") - 0.2)
        self.play(FadeIn(rows[0], shift=0.1 * RIGHT), run_time=0.7)
        self.at(S("k2"))
        self.play(FadeIn(rows[1], shift=0.1 * RIGHT), run_time=0.7)
        self.at(W("k2", "partially") - 0.2)
        self.play(FadeIn(rows[2], shift=0.1 * RIGHT), run_time=0.7)
        notes = VGroup(Tex("prop: propagate a field (e.g.\\ the angular spectrum method)", font_size=26, color=GREY_B),
                       Tex("blur: the image of a single point;\\quad phi[n], lam[n]: the modes and their powers", font_size=26,
                           color=GREY_B)).arrange(DOWN, buff=0.12)
        notes.next_to(card, DOWN, buff=0.35)
        self.play(FadeIn(notes), run_time=0.7)

        # ---------------- k3: a diffractive element under a laser
        size = 3.9
        cL, cR = np.array([-2.75, -0.55, 0]), np.array([3.55, -0.55, 0])
        las = dd["laser"]
        vL = np.percentile(las, 99.85)
        mL, fL, swL = self.sweep_image(heat_rgba(las, vL, gamma=0.7, flip=False), size, cL, size, axis="right")
        boxL = Square(side_length=size).move_to(cL).set_stroke(GREY_B, 1.5)
        labL = Tex("laser", font_size=32, color=GREY_A).next_to(boxL, UP, buff=0.12)
        ph = dd["phase"]
        doe_m, doe_f = self.static_image(phase_rgba(np.exp(1j * ph), vmax=1.0, gamma=1.0, flip=False), 1.05, [-6.15, -0.55, 0], 1.05,
                                         nearest=True)
        doe_l = Tex("diffractive\\\\element", font_size=24, color=GREY_A).move_to([-6.15, -1.45, 0])
        doe_ar = Arrow([-5.55, -0.55, 0], [-4.8, -0.55, 0], buff=0, stroke_width=4, color=GREY_A, max_tip_length_to_length_ratio=0.3)
        ff_l = Tex("far field (computed)", font_size=24, color=GREY_B).next_to(boxL, DOWN, buff=0.12)
        self.at(S("k3"))
        self.play(FadeOut(card), FadeOut(notes), run_time=0.5)
        self.add(doe_m, mL)
        self.play(doe_f.animate.set_value(1.0), FadeIn(doe_l), GrowArrow(doe_ar), fL.animate.set_value(1.0), FadeIn(boxL), FadeIn(labL),
                  run_time=0.8)
        self.play(swL.animate.set_value(1.0), FadeIn(ff_l), run_time=1.0, rate_func=linear)
        self.at(W("k3", "speckled") - 0.3)
        zoom = Circle(radius=0.5).move_to(cL + np.array([0.0, 1.43, 0]) * size / 3.9).set_stroke(YELLOW, 3)
        self.play(Create(zoom), run_time=0.6)

        # ---------------- k4: LED = shifted copies (source points) + stretched copies (colours)
        stages = [dd["laser"], dd["k7"], dd["k19"], dd["k61"], dd["led"]]
        vR = [np.percentile(s_, 99.85) for s_ in stages]
        rgbR = [heat_rgba(s_, v_, gamma=0.7, flip=False) for s_, v_ in zip(stages, vR)]
        stg = ValueTracker(0.0)

        def led_fn(t):
            xg = stg.get_value()
            i = int(np.clip(np.floor(xg), 0, len(rgbR) - 2))
            f = float(np.clip(xg - i, 0, 1))
            if f < 1e-3:
                return rgbR[i]
            out = rgbR[i].copy()
            out[..., :3] = ((1 - f) * rgbR[i][..., :3] + f * rgbR[i + 1][..., :3]).astype(np.uint8)
            return out
        mR, fR = self.live_image(led_fn, size, cR, size)
        boxR = Square(side_length=size).move_to(cR).set_stroke(GREY_B, 1.5)
        labR = Tex("LED", font_size=32, color=GREY_A).next_to(boxR, UP, buff=0.12)
        sc_c, sc_r = np.array([0.4, 0.75, 0]), 0.55
        src = dd["src"] / float(dd["Rs"]) * sc_r
        nsrc = ValueTracker(1.0)
        counts = [1, 7, 19, 61, 61]
        src_m = always_redraw(lambda: VGroup(Circle(radius=sc_r * 1.12).move_to(sc_c).set_stroke(C_REF, 1.5, opacity=0.6),
                                             *[Dot(sc_c + np.array([p[0], p[1], 0]), radius=0.028, color=C_REF)
                                               for p in src[:int(nsrc.get_value())]]))
        src_l = Tex("source points", font_size=24, color=C_REF).move_to(sc_c + np.array([0, -sc_r - 0.32, 0]))
        lam_s = dd["lams"]
        wl_ = dd["wl"]
        spx = lambda l: 0.4 + (l - 0.530) / 0.04 * 0.75
        spec_ic = VGroup(*[Line([spx(l), -1.35, 0], [spx(l), -1.35 + 0.65 * w_ / wl_.max(), 0]).set_stroke(wl_color(l), 4)
                           for l, w_ in zip(lam_s, wl_)])
        spec_l = Tex("colours", font_size=24, color=GREY_A).move_to([0.4, -1.65, 0])
        plus = MathTex("+", font_size=40).move_to([0.4, -0.45, 0])
        self.at(S("k4"))
        self.add(mR, src_m)
        self.play(FadeOut(zoom), fR.animate.set_value(1.0), FadeIn(boxR), FadeIn(labR), FadeIn(src_l), run_time=0.7)
        self.at(W("k4", "tilts") - 0.3)
        for k_ in (1, 2, 3):
            self.play(stg.animate.set_value(float(k_)), nsrc.animate.set_value(float(counts[k_])), run_time=0.65)
        self.at(W("k4", "color") - 0.2)
        self.play(FadeIn(plus), FadeIn(spec_ic), FadeIn(spec_l), stg.animate.set_value(4.0), run_time=1.2)
        sm = VGroup(Tex("blurred, but smooth", font_size=28, color=GREY_A)).next_to(boxR, DOWN, buff=0.12)
        self.at(W("k4", "blurred") - 0.3)
        self.play(FadeIn(sm), run_time=0.6)

        # ---------------- k5: the shortcut and its limit
        form = Tex("small angles: LED $=$ laser pattern $\\circledast$ source image, averaged over colours",
                   font_size=28).move_to([0.3, 2.85, 0])
        lim = Tex("large angles: propagate a few coherent modes instead", font_size=28, color=YELLOW).move_to([0.3, 2.3, 0])
        self.at(S("k5"))
        self.play(FadeIn(form), run_time=0.8)
        self.at(W("k5", "few") - 0.4)
        self.play(FadeIn(lim), run_time=0.8)
        self.finish()


# ============================================================ 7  recap =============
class Ch7Recap(Chapter):
    CH = 7
    TAG = "7\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        noise = load("jitter.npz")["noise"]
        nz = lambda t: float(noise[int(round(t * 60)) % len(noise)])
        cs = [np.array([-3.4, 0.9, 0]), np.array([3.4, 0.9, 0])]
        op = ValueTracker(0.0)
        dphi = 0.9

        def pair(c, jitter):
            t = self.now
            a = 1.6 * t
            p1 = c + 0.75 * np.array([np.cos(a), np.sin(a), 0])
            b = a + dphi + (2.8 * nz(t) if jitter else 0.0)
            p2 = p1 + 0.75 * np.array([np.cos(b), np.sin(b), 0])
            return VGroup(fast_arrow(c, p2, WHITE, 3.5, 0.15), fast_arrow(c, p1, C_A1, 5, 0.17), fast_arrow(p1, p2, C_A2, 5, 0.17),
                          Dot(c, radius=0.04, color=GREY_B))
        arrows = opaque(lambda: VGroup(pair(cs[0], False), pair(cs[1], True)), op)
        xs = np.linspace(0, 1, 500)
        fr = stripe_rgba(2 + 2 * np.cos(TAU_ * 3.5 * xs), 4.0, H=40)
        fl = stripe_rgba(2 + 0 * xs, 4.0, H=40)
        m1, f1 = self.static_image(fr, 4.6, cs[0] + 2.05 * DOWN, 0.7)
        m2, f2 = self.static_image(fl, 4.6, cs[1] + 2.05 * DOWN, 0.7)
        t1 = Tex("phases in step: fields add", font_size=34).move_to(cs[0] + 2.0 * UP)
        t2 = Tex("phases wander: intensities add", font_size=34).move_to(cs[1] + 2.0 * UP)
        e1 = MathTex("|a_1+a_2|^2", font_size=34).move_to(cs[0] + 2.85 * DOWN)
        e2 = MathTex("|a_1|^2+|a_2|^2", font_size=34).move_to(cs[1] + 2.85 * DOWN)
        self.add(arrows, m1, m2)
        self.at(S("r1"))
        self.play(op.animate.set_value(1.0), run_time=0.6)
        self.at(W("r1", "fields") - 0.2)
        self.play(FadeIn(t1), f1.animate.set_value(1.0), FadeIn(e1), run_time=0.8)
        self.at(W("r1", "wander") - 0.4)
        self.play(FadeIn(t2), f2.animate.set_value(1.0), FadeIn(e2), run_time=0.8)

        # ---------------- r2: partially coherent = a sum of coherent patterns
        row_items = VGroup()
        for i, s_ in enumerate((-0.18, 0.0, 0.18)):
            row_items.add(poly_line(np.linspace(-0.95, 0.95, 120), 0.3 * (1 + np.cos(TAU_ * (np.linspace(-0.95, 0.95, 120) / 0.63 - s_))),
                                    C_REF, 2.5))
        tot = poly_line(np.linspace(-0.95, 0.95, 120),
                        np.mean([0.3 * (1 + np.cos(TAU_ * (np.linspace(-0.95, 0.95, 120) / 0.63 - s_))) for s_ in (-0.18, 0.0, 0.18)], 0),
                        WHITE, 3.5)
        parts = [row_items[0], MathTex("+", font_size=40), row_items[1], MathTex("+", font_size=40), row_items[2],
                 MathTex("+\\;\\cdots\\;=", font_size=40), tot]
        eqrow = VGroup(*parts).arrange(RIGHT, buff=0.3).move_to([0, -0.2, 0])
        r2t = Tex("partially coherent light $=$ a sum of coherent patterns", font_size=36).move_to([0, 1.3, 0])
        self.at(S("r2"))
        self.play(op.animate.set_value(0.0), f1.animate.set_value(0.0), f2.animate.set_value(0.0), FadeOut(VGroup(t1, t2, e1, e2)), run_time=0.5)
        for m_ in (m1, m2):
            m_.clear_updaters()
        self.remove(arrows, m1, m2)
        self.play(FadeIn(r2t), LaggedStart(*[FadeIn(p_) for p_ in parts], lag_ratio=0.15), run_time=1.4)

        # ---------------- r3: next time
        nxt = next_card("the phase problem: holography and phase retrieval", y=-2.2)
        ttl = Tex("Coherence", font_size=56).move_to([0, 2.45, 0])
        self.at(S("r3"))
        self.play(FadeIn(ttl, shift=0.1 * DOWN), run_time=0.7)
        self.at(W("r3", "record") - 0.6)
        self.play(FadeIn(nxt, shift=0.1 * UP), run_time=0.9)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Arrows", "Ch2Colour", "Ch3Size", "Ch4Speckle", "Ch5Modes", "Ch6Computer", "Ch7Recap"]
