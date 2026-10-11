"""Periodic Structures: Gratings, Bloch Waves & Band Gaps -- chapter scenes (Optics in 5 Minutes, E07).

Render: ./render.sh ; preview: ./preview.sh Ch0Hook
Every physical field is precomputed by physics.py (data/*.npz); construct() only loads and draws.
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


def _at_whole_frames(self, t):
    """Frame-exact sync (FAST_ANIMATION_PIPELINE.md section 10): wait exactly k whole frames, so a target that lies
    on the 1/60 s grid gives the same frame on every worker (the common/ at() waits a float duration)."""
    x = (t - self.now) * config.frame_rate                # frames still to go
    if x > 0.5 + 1e-6:
        k = int(np.ceil(x - 1e-6))
        self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
    elif x < -0.5 - 1e-6:
        self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


Chapter.at = _at_whole_frames

# ------------------------------------------------------------------ word-level sync ---------
_WORDS = json.load(open(os.path.join(HERE, "audio", "words.json")))


def _norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def WT(seg, word, nth=0, default=None):
    """Start time of the nth occurrence of `word` (prefix match; a tuple gives alternative spellings, e.g. Whisper's
    "kopies" or "color") inside narration segment `seg`.  Falls back to the segment start (+ default) with a warning."""
    s = SEG[seg]
    ws = [w for w in _WORDS if s["start"] - 0.15 <= w["s"] < s["end"] + 0.1]
    alts = word if isinstance(word, (tuple, list)) else (word,)
    hits = sorted(w["s"] for w in ws if any(_norm(w["w"]).startswith(_norm(a)) for a in alts))
    if len(hits) > nth:
        return max(hits[nth], s["start"])
    print(f"WT WARNING: '{word}' not found in {seg}", file=sys.stderr)
    return s["start"] + (default if default is not None else 0.0)


def load(name):
    return np.load(data_path(name))


def rgba(img, alpha=255):
    """RGB uint8 image -> RGBA (opaque)."""
    out = np.empty(img.shape[:2] + (4,), np.uint8)
    out[..., :3] = img[..., :3]
    out[..., 3] = alpha
    return out


def hexrgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c[:3])


def mix(c, w=0.0):
    """Blend an RGB triple (0..1) with white by w and return a hex colour."""
    c = np.asarray(c, float)
    return hexrgb((1 - w) * c + w)


def tx(s, size=30, color=GREY_A, **kw):
    return Tex(s, font_size=size, color=color, **kw)


def mt(s, size=34, color=WHITE, **kw):
    return MathTex(s, font_size=size, color=color, **kw)


def faded(fn, op):
    """always_redraw of fn() whose opacity is multiplied by the tracker op (never FadeIn a moving redraw)."""
    def build():
        o = op.get_value()
        if o <= 0.003:
            return VGroup()
        return fade_vgroup(fn(), o)
    return always_redraw(build)


def half_circle(c, R, color=GREY_A, width=2.5, opacity=1.0):
    return Arc(radius=R, start_angle=0, angle=PI, arc_center=c).set_stroke(color, width, opacity=opacity)


def kaxes(O, xl, xr, top, labels=True, size=30):
    ax = VGroup(Arrow(O + xl * LEFT, O + xr * RIGHT, buff=0, stroke_width=2, color=GREY_B,
                      max_tip_length_to_length_ratio=0.03),
                Arrow(O + 0.0 * DOWN, O + top * UP, buff=0, stroke_width=2, color=GREY_B,
                      max_tip_length_to_length_ratio=0.05))
    if labels:
        ax.add(mt("k_x", size, C_KX).next_to(ax[0], RIGHT, buff=0.06),
               mt("k_z", size, C_KZ).next_to(ax[1], UP, buff=0.06))
    return ax


STACK = load("stack.npz")
LAMV, LAMRGB = STACK["lam_v"], STACK["lam_rgb"]


def spec_hex(lam_nm, w=0.0):
    i = int(np.clip(np.searchsorted(LAMV, lam_nm), 0, len(LAMV) - 1))
    return mix(LAMRGB[i], w)


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        fan = load("fan.npz")
        # ---- left: white light through a grating (computed: RCWA orders + angular spectrum, 31 colours)
        W, Hh = 6.3, 6.3 * 45.0 / 64.0
        C = np.array([-3.25, -0.05, 0])
        hm, hm_f, hm_s = self.sweep_image(rgba(fan["hook"]), W, C, Hh, axis="up", soft=0.02)
        y_g = C[1] - Hh / 2 + Hh * 5.0 / 45.0
        frame = Rectangle(width=W, height=Hh).move_to(C).set_stroke(GREY_C, 1.2)
        xx = np.linspace(C[0] - W / 2, C[0] + W / 2, 900)
        grating = poly_line(xx, y_g + 0.03 * np.cos(TAU * (xx - C[0]) / (W * 1.5 / 64.0)), GREY_A, 2)
        g_lab = tx("grating", 26, GREY_B).move_to([C[0] - W / 2 + 0.75, y_g + 0.3, 0])
        w_lab = tx("white light", 26, GREY_B).next_to(frame, DOWN, buff=0.14)
        comp = tx("(computed)", 22, GREY_B).move_to([C[0] + W / 2 - 0.75, C[1] + Hh / 2 - 0.25, 0])
        self.add(hm)
        self.play(FadeIn(frame), FadeIn(grating), FadeIn(g_lab), FadeIn(w_lab), hm_f.animate.set_value(1.0),
                  run_time=0.8)
        self.at(S("h1") + 0.2)
        self.play(hm_s.animate.set_value(1.0), run_time=4.6, rate_func=linear)
        self.play(FadeIn(comp), run_time=0.4)

        # ---- right: a Bragg stack reflects one colour (computed: transfer matrices, 10 pairs)
        c_ref, c_tr = hexrgb(STACK["c_ref"]), hexrgb(STACK["c_tr"])
        sx, sy, sw, sh = 3.75, 0.95, 2.3, 1.5
        layers = VGroup()
        for i in range(20):
            r = Rectangle(width=sw, height=sh / 20).move_to([sx, sy + sh / 2 - (i + 0.5) * sh / 20, 0])
            r.set_stroke(width=0).set_fill(GREY_B if i % 2 == 0 else GREY_D, 1.0)
            layers.add(r)
        top = sy + sh / 2
        a_in = Arrow([1.45, 2.75, 0], [3.35, top + 0.02, 0], buff=0, stroke_width=7, color=WHITE,
                     max_tip_length_to_length_ratio=0.1)
        a_ref = Arrow([3.55, top + 0.02, 0], [5.45, 2.75, 0], buff=0, stroke_width=7, color=c_ref,
                      max_tip_length_to_length_ratio=0.1)
        a_tr = Arrow([3.9, sy - sh / 2 - 0.02, 0], [4.75, sy - sh / 2 - 1.0, 0], buff=0, stroke_width=5,
                     color=c_tr, max_tip_length_to_length_ratio=0.15)
        l_in = tx("white light", 26, GREY_B).next_to(a_in.get_start(), LEFT, buff=0.12)
        l_st = tx("10 pairs of layers", 24, GREY_B).next_to(layers, LEFT, buff=0.2)
        # reflectance spectrum with a rainbow fill
        px0, px1, py0, ph = 1.35, 6.15, -2.55, 1.15
        lam_v, Rv = STACK["lam_v"], STACK["Rv"]
        nx_img, ny_img = 480, 120
        lam_i = np.linspace(lam_v[0], lam_v[-1], nx_img)
        R_i = np.interp(lam_i, lam_v, Rv)
        img = np.zeros((ny_img, nx_img, 4), np.uint8)
        cols = (255 * LAMRGB[np.clip(np.searchsorted(lam_v, lam_i), 0, len(lam_v) - 1)]).astype(np.uint8)
        hrow = (np.arange(ny_img)[::-1] + 0.5) / ny_img
        fill = hrow[:, None] <= R_i[None, :]
        img[..., :3] = cols[None, :, :]
        img[..., 3] = np.where(fill, 235, 38).astype(np.uint8)
        sp_img, sp_f, sp_s = self.sweep_image(img, px1 - px0, [0.5 * (px0 + px1), py0 + ph / 2, 0], ph, axis="right")
        sp_curve = plot_xy(lam_v, Rv, [0.5 * (px0 + px1), py0 + ph / 2, 0], px1 - px0, ph, (lam_v[0], lam_v[-1]),
                           (0, 1), WHITE, 2.5)
        sp_ax = Line([px0, py0, 0], [px1, py0, 0]).set_stroke(GREY_B, 1.5)
        sp_l = VGroup(tx("400 nm", 22, GREY_B).move_to([px0 + (400 - lam_v[0]) / (lam_v[-1] - lam_v[0]) * (px1 - px0),
                                                         py0 - 0.22, 0]),
                      tx("700 nm", 22, GREY_B).move_to([px0 + (700 - lam_v[0]) / (lam_v[-1] - lam_v[0]) * (px1 - px0),
                                                         py0 - 0.22, 0]),
                      tx("reflected (computed)", 24, GREY_A).move_to([0.5 * (px0 + px1), py0 + ph + 0.22, 0]))
        self.at(S("h2"))
        self.play(LaggedStart(*[FadeIn(r) for r in layers], lag_ratio=0.04), FadeIn(l_st), run_time=1.1)
        self.play(GrowArrow(a_in), FadeIn(l_in), run_time=0.6)
        self.play(GrowArrow(a_ref), run_time=0.6)
        self.play(GrowArrow(a_tr), run_time=0.4)
        self.add(sp_img)
        self.play(sp_f.animate.set_value(1.0), FadeIn(sp_ax), FadeIn(sp_l), run_time=0.4)
        self.play(sp_s.animate.set_value(1.0), Create(sp_curve), run_time=1.6, rate_func=linear)

        # ---- h3: the question
        q1 = tx("direction?", 34, YELLOW).move_to([C[0], C[1] + Hh / 2 + 0.35, 0])
        q2 = tx("colour?", 34, YELLOW).next_to(a_ref.get_end(), UP, buff=0.12).shift(0.2 * LEFT)
        self.at(WT("h3", "direction") - 0.2)
        self.play(FadeIn(q1, shift=0.1 * DOWN), run_time=0.5)
        self.at(WT("h3", "col") - 0.15)
        self.play(FadeIn(q2, shift=0.1 * DOWN), run_time=0.5)

        # ---- title card
        title = series_title(7, "Periodic Structures", "gratings, Bloch waves and band gaps")
        dimmables = VGroup(frame, grating, g_lab, w_lab, comp, layers, a_in, a_ref, a_tr, l_in, l_st, sp_curve, sp_ax,
                           sp_l, q1, q2)
        self.at(S("h4") - 0.2)
        self.play(hm_f.animate.set_value(0.3), sp_f.animate.set_value(0.3), dimmables.animate.set_opacity(0.3),
                  FadeIn(title, shift=0.1 * DOWN), run_time=1.0)
        self.finish()


# ============================================================ 1  flat boundary =====
N1 = 1.5                      # glass index (Ch1)
LAM = 0.8                     # vacuum wavelength in the real-space panels (scene units)
F_T = 0.6                     # temporal frequency of the live crests (Hz)


def crest_lines(kx, kz, xc, yb, box, t, color, width, opacity, n_max=40):
    """Crests (phase = 2 pi j) of exp(i(kx X + kz Y - w t)) with X = x - xc, Y = y - yb, clipped to box
    (cx, cy, hx, hy).  kx, kz in rad / scene unit."""
    g = VGroup()
    kk = np.hypot(kx, kz)
    if kk < 1e-9 or opacity <= 0.003:
        return g
    cx, cy, hx, hy = box
    w_t = TAU * F_T * t
    corners = [(cx + sx * hx - xc, cy + sy * hy - yb) for sx in (-1, 1) for sy in (-1, 1)]
    ph = [kx * X + kz * Y - w_t for X, Y in corners]
    j0, j1 = int(np.floor(min(ph) / TAU)), int(np.ceil(max(ph) / TAU))
    d = np.array([kz, -kx, 0.0]) / kk
    for j in range(max(j0, j1 - n_max), j1 + 1):
        s = (TAU * j + w_t) / kk ** 2
        p0 = np.array([xc + s * kx, yb + s * kz, 0.0])
        seg = clip_to_box(p0 - 30 * d, p0 + 30 * d, cx, cy, hx, hy)
        if seg:
            g.add(Line(*seg).set_stroke(color, width, opacity=opacity))
    return g


def evanescent_crests(kx, kap, xc, yb, box, t, color=C_EV, width=4, opacity=1.0, nseg=14):
    """Vertical crests of exp(i(kx X - w t)) exp(-kap Y) above y = yb (opacity decays with the amplitude)."""
    g = VGroup()
    if opacity <= 0.003:
        return g
    cx, cy, hx, hy = box
    w_t = TAU * F_T * t
    top = cy + hy
    ymax = min(top - yb, 4.0 / max(kap, 1e-3))
    j0 = int(np.floor((kx * (cx - hx - xc) - w_t) / TAU))
    j1 = int(np.ceil((kx * (cx + hx - xc) - w_t) / TAU))
    for j in range(j0, j1 + 1):
        x = xc + (TAU * j + w_t) / kx
        if abs(x - cx) > hx:
            continue
        for i in range(nseg):
            y0, y1 = ymax * i / nseg, ymax * (i + 1) / nseg
            a = np.exp(-kap * 0.5 * (y0 + y1))
            if a < 0.03:
                break
            g.add(Line([x, yb + y0, 0], [x, yb + y1, 0]).set_stroke(color, width, opacity=opacity * a ** 0.7))
    return g


def fresnel_te(kx, n1, n2):
    """TE amplitude reflection for incidence from n1 into n2 at transverse wavenumber kx (units of k0)."""
    kz1 = np.sqrt(complex(n1 ** 2 - kx ** 2))
    kz2 = np.sqrt(complex(n2 ** 2 - kx ** 2))
    if kz2.imag < 0:
        kz2 = -kz2
    return (kz1 - kz2) / (kz1 + kz2)


class Ch1Boundary(Chapter):
    CH = 1
    TAG = "1\\quad A flat boundary"

    def construct(self):
        S = self.S
        self.show_tag()
        k0s = TAU / LAM
        # ---------------- real-space panel (right): glass below, air above
        P0, H = np.array([3.55, -0.35, 0]), 2.45
        yb = P0[1] - 0.2
        xc = P0[0]
        box_g = (P0[0], 0.5 * (P0[1] - H + yb), H, 0.5 * (yb - (P0[1] - H)))
        box_a = (P0[0], 0.5 * (P0[1] + H + yb), H, 0.5 * (P0[1] + H - yb))
        sq = Square(side_length=2 * H).move_to(P0).set_stroke(GREY_B, 2)
        glass = Rectangle(width=2 * H, height=yb - (P0[1] - H)).set_stroke(width=0).set_fill("#2a3036", 0.85)
        glass.move_to([P0[0], 0.5 * (yb + P0[1] - H), 0])
        bnd = Line([P0[0] - H, yb, 0], [P0[0] + H, yb, 0]).set_stroke(GREY_A, 3)
        lab_g = tx("glass, $n_1 = 1.5$", 26, GREY_A).move_to([P0[0] - H + 1.15, P0[1] - H + 0.28, 0])
        lab_a = tx("air, $n_2 = 1$", 26, GREY_A).move_to([P0[0] - H + 0.9, P0[1] + H - 0.28, 0])
        lab_bg_g = BackgroundRectangle(lab_g, fill_opacity=0.75, buff=0.06).set_fill("#2a3036")
        lab_bg_a = BackgroundRectangle(lab_a, fill_opacity=0.75, buff=0.06)
        kx_t = ValueTracker(N1 * np.sin(np.radians(25)))
        w_op = ValueTracker(0.0)          # crest visibility
        rays_op = ValueTracker(0.0)       # rays (incident / transmitted / reflected) in the panel

        def waves():
            kx = kx_t.get_value()
            o = w_op.get_value()
            if o <= 0.003:
                return VGroup()
            t = self.now
            g = VGroup()
            kz1 = np.sqrt(max(N1 ** 2 - kx ** 2, 0.0))
            g.add(crest_lines(kx * k0s, kz1 * k0s, xc, yb, box_g, t, C_U, 3.5, o))
            r = fresnel_te(kx, N1, 1.0)
            # reflected crests in the glass, weighted by |r|^2
            g.add(crest_lines(kx * k0s, -kz1 * k0s, xc, yb, box_g, t, BLUE_E, 3.0, o * 0.85 * abs(r) ** 2))
            if kx < 1.0:
                kz2 = np.sqrt(1.0 - kx ** 2)
                g.add(crest_lines(kx * k0s, kz2 * k0s, xc, yb, box_a, t, C_U, 3.5, o))
            else:
                kap = np.sqrt(kx ** 2 - 1.0) * k0s
                g.add(evanescent_crests(kx * k0s, kap, xc, yb, box_a, t, C_EV, 5, o))
            return g
        wv = always_redraw(waves)

        th1_l = mt("\\theta_1", 30)
        th2_l = mt("\\theta_2", 30)
        ang_op = ValueTracker(0.0)

        def rays():
            o = rays_op.get_value()
            if o <= 0.003:
                return VGroup()
            kx = kx_t.get_value()
            th1 = np.arcsin(np.clip(kx / N1, -1, 1))
            L = 1.45
            p = np.array([xc - 0.35, yb, 0])
            r2 = abs(fresnel_te(kx, N1, 1.0)) ** 2
            g = VGroup()
            oa = ang_op.get_value() * o
            if oa > 0.003:                                    # the normal and the two angles
                g.add(fast_dashed(p + 1.25 * DOWN, p + 1.25 * UP, 0.06, GREY_A, 2, oa))
                g.add(Arc(radius=0.62, start_angle=-PI / 2, angle=-th1, arc_center=p).set_stroke(WHITE, 2.5, opacity=oa))
                g.add(th1_l.copy().move_to(p + 0.9 * np.array([-np.sin(th1 / 2), -np.cos(th1 / 2), 0])).set_opacity(oa))
            a = p - L * np.array([np.sin(th1), np.cos(th1), 0])
            g.add(Line(a, p).set_stroke(BLACK, 11, opacity=0.6 * o), fast_arrow(a, p, WHITE, 6, 0.22, o))
            if r2 > 0.01:
                b = p + L * np.array([np.sin(th1), -np.cos(th1), 0])
                g.add(Line(p, b).set_stroke(BLACK, 11, opacity=0.6 * o * r2),
                      fast_arrow(p, b, WHITE, 6, 0.22, o * min(1.0, 0.25 + r2)))
            if kx < 1.0:
                th2 = np.arcsin(kx)
                c = p + L * np.array([np.sin(th2), np.cos(th2), 0])
                g.add(Line(p, c).set_stroke(BLACK, 11, opacity=0.6 * o), fast_arrow(p, c, WHITE, 6, 0.22, o * (1 - r2) ** 0.5))
                if oa > 0.003:
                    g.add(Arc(radius=0.62, start_angle=PI / 2, angle=-th2, arc_center=p).set_stroke(WHITE, 2.5, opacity=oa))
                    g.add(th2_l.copy().move_to(p + 0.92 * np.array([np.sin(th2 / 2), np.cos(th2 / 2), 0])).set_opacity(oa))
            return g
        rv = always_redraw(rays)
        self.at(S("s1"))
        self.play(Create(sq), FadeIn(glass), Create(bnd), run_time=0.9)
        self.add(wv)
        self.play(FadeIn(lab_bg_g), FadeIn(lab_g), FadeIn(lab_bg_a), FadeIn(lab_a), w_op.animate.set_value(1.0),
                  run_time=1.0)
        # labels stay on top of the crests
        self.bring_to_front(lab_bg_g, lab_g, lab_bg_a, lab_a)

        # ---------------- k-space (left): air circle first (recall of E01)
        O = np.array([-3.55, -2.5, 0])
        R0 = 1.5
        R1 = N1 * R0
        ax = kaxes(O, 2.85, 2.95, 2.62)
        c_air = half_circle(O, R0, GREY_A, 3)
        c_gl = half_circle(O, R1, "#8fb3c9", 3)
        l_air = mt("k", 32).move_to(O + (R0 + 0.28) * np.array([np.cos(2.45), np.sin(2.45), 0]))
        l_air2 = mt("k_0", 32).move_to(l_air)
        l_gl = mt("n_1 k_0", 32, "#8fb3c9").move_to(O + (R1 + 0.42) * np.array([np.cos(2.5), np.sin(2.5), 0]))
        t_air = tx("air", 24, GREY_B).move_to(O + np.array([-0.55, 0.32, 0]))
        t_gl = tx("glass", 24, "#8fb3c9").move_to(O + np.array([-1.85, 0.32, 0]))
        a2_op = ValueTracker(0.0)          # transmitted (air) k arrow
        a1_op = ValueTracker(0.0)          # incident (glass) k arrow
        ln_op = ValueTracker(0.0)          # shared-k_x vertical line
        th_op = ValueTracker(0.0)          # angle arcs

        def kvecs():
            kx = kx_t.get_value()
            g = VGroup()
            xk = O + R0 * kx * RIGHT
            o2, o1, ol, ot = a2_op.get_value(), a1_op.get_value(), ln_op.get_value(), th_op.get_value()
            if ol > 0.003:
                g.add(fast_dashed(xk, xk + (R1 + 0.35) * UP, 0.07, C_KX, 3, ol))
            if o1 > 0.003:
                p1 = O + R1 * np.array([kx / N1, np.sqrt(max(0, 1 - (kx / N1) ** 2)), 0])
                g.add(fast_arrow(O, p1, "#8fb3c9", 5, 0.2, o1), Dot(p1, 0.055, color="#8fb3c9", fill_opacity=o1))
            if o2 > 0.003:
                if kx < 1.0:
                    p2 = O + R0 * np.array([kx, np.sqrt(1 - kx ** 2), 0])
                    g.add(fast_arrow(O, p2, WHITE, 5, 0.2, o2), Dot(p2, 0.055, color=WHITE, fill_opacity=o2))
                    if ot > 0.003:
                        th2 = np.arcsin(kx)
                        g.add(Arc(radius=0.45, start_angle=PI / 2, angle=-th2, arc_center=O).set_stroke(WHITE, 2.5,
                                                                                                       opacity=ot))
                else:
                    g.add(Dot(xk, 0.08, color=C_EV, fill_opacity=o2))
            if max(o1, o2) > 0.003:
                g.add(Line(O, xk).set_stroke(C_KX, 7, opacity=max(o1, o2)))
            return g
        kv = always_redraw(kvecs)
        sin_eq = MathTex("\\sin\\theta", "=", "{k_x", "\\over", "k}", font_size=42)
        sin_eq[2].set_color(C_KX)
        sin_eq.move_to([-3.6, 2.3, 0])
        th_l = mt("\\theta", 30).move_to(O + np.array([0.18, 0.85, 0]))
        self.at(S("s2") + 0.3)
        self.play(GrowArrow(ax[0]), GrowArrow(ax[1]), FadeIn(ax[2:]), Create(c_air), FadeIn(l_air), FadeIn(t_air),
                  run_time=1.2)
        self.add(kv)
        self.play(a2_op.animate.set_value(1.0), run_time=0.6)
        self.at(WT("s2", "sin") - 0.3)
        self.play(Write(sin_eq), th_op.animate.set_value(1.0), FadeIn(th_l), run_time=1.2)
        self.play(kx_t.animate.set_value(0.35), run_time=1.0)
        self.play(kx_t.animate.set_value(N1 * np.sin(np.radians(25))), run_time=0.8)

        # ---------------- s3: glass has a bigger circle, k = n k0
        kn = MathTex("k", "=", "n", "\\,k_0", font_size=42).move_to([-3.6, 1.3, 0])
        kn_l = tx("$n$: refractive index", 26, GREY_B).next_to(kn, DOWN, buff=0.12)
        self.at(S("s3") + 0.2)
        self.play(FadeOut(th_l), th_op.animate.set_value(0.0), run_time=0.4)
        self.at(WT("s3", "grows") - 0.2)
        self.play(Write(kn), run_time=0.9)
        self.play(FadeIn(kn_l), Create(c_gl), Transform(l_air, l_air2), FadeIn(l_gl), FadeIn(t_gl), run_time=1.3)
        self.at(WT("s3", "each") - 0.1)
        self.play(Indicate(c_gl, color="#8fb3c9", scale_factor=1.0), Indicate(c_air, color=WHITE, scale_factor=1.0),
                  run_time=1.2)

        # ---------------- s4: crests meet at the boundary -> one shared k_x
        def meet_dots():
            kx = kx_t.get_value() * k0s
            g = VGroup()
            if kx < 1e-6:
                return g
            w_t = TAU * F_T * self.now
            j0 = int(np.floor((kx * (-H) - w_t) / TAU))
            j1 = int(np.ceil((kx * H - w_t) / TAU))
            for j in range(j0, j1 + 1):
                x = xc + (TAU * j + w_t) / kx
                if abs(x - xc) < H - 0.04:
                    g.add(Dot([x, yb, 0], radius=0.07, color=YELLOW))
            return g
        dots_op = ValueTracker(0.0)
        md = faded(meet_dots, dots_op)
        lam_x = TAU / (kx_t.get_value() * k0s)
        xb0 = P0[0] - H + 0.35
        spacing = VGroup(Line([xb0, yb - 0.2, 0], [xb0 + lam_x, yb - 0.2, 0]).set_stroke(YELLOW, 3),
                         Line([xb0, yb - 0.29, 0], [xb0, yb - 0.11, 0]).set_stroke(YELLOW, 3),
                         Line([xb0 + lam_x, yb - 0.29, 0], [xb0 + lam_x, yb - 0.11, 0]).set_stroke(YELLOW, 3))
        sp_l = mt("2\\pi / k_x", 28, YELLOW).next_to(spacing, DOWN, buff=0.08)
        sp_bg = BackgroundRectangle(sp_l, fill_opacity=0.8, buff=0.05).set_fill("#2a3036")
        cons = VGroup(tx("the boundary looks the same at every $x$", 26, GREY_A),
                      tx("$\\Rightarrow$ $k_x$ is conserved", 34, WHITE)).arrange(DOWN, buff=0.12)
        cons = VGroup(SurroundingRectangle(cons, buff=0.14, color=C_KX, corner_radius=0.08), cons).move_to([-3.6, 1.05, 0])
        self.at(WT("s4", "crests") - 0.2)
        self.add(md)
        self.play(dots_op.animate.set_value(1.0), run_time=0.8)
        self.at(WT("s4", "share") - 0.3)
        self.play(Create(spacing), FadeIn(sp_bg), FadeIn(sp_l), a1_op.animate.set_value(1.0), run_time=0.9)
        self.at(WT("s4", "conserved") - 0.2)
        self.play(FadeOut(kn), FadeOut(kn_l), run_time=0.3)
        self.play(FadeIn(cons, shift=0.1 * UP), run_time=0.6)

        # ---------------- s5: one vertical line through both circles -> Snell's law
        snell = MathTex("n_1", "\\sin\\theta_1", "=", "{k_x", "\\over", "k_0}", "=", "n_2", "\\sin\\theta_2", font_size=40)
        snell[3].set_color(C_KX)
        snell.move_to([-3.6, 2.32, 0])
        snell_l = tx("Snell's law", 30, YELLOW).next_to(snell, UP, buff=0.1)
        self.at(S("s5") + 0.1)
        self.play(FadeOut(VGroup(spacing, sp_l, sp_bg)), dots_op.animate.set_value(0.0), run_time=0.5)
        self.remove(md)
        self.at(WT("s5", "vertical") - 0.2)
        self.play(ln_op.animate.set_value(1.0), run_time=0.8)
        self.at(WT("s5", "crosses") - 0.1)
        self.add(rv)
        self.play(rays_op.animate.set_value(1.0), run_time=1.0)
        self.at(WT("s5", "snell") - 0.5)
        self.play(FadeOut(sin_eq), run_time=0.3)
        self.play(Write(snell), FadeIn(snell_l), ang_op.animate.set_value(1.0), run_time=1.3)

        # ---------------- s6: tilt further -> the line leaves the air circle
        self.at(S("s6") + 0.4)
        self.play(ang_op.animate.set_value(0.0), run_time=0.4)
        self.play(kx_t.animate.set_value(0.985), run_time=3.2, rate_func=rate_functions.ease_in_out_sine)
        self.at(WT("s6", "misses") - 0.2)
        self.play(kx_t.animate.set_value(1.12), run_time=1.6, rate_func=rate_functions.ease_out_sine)

        # ---------------- s7: total internal reflection
        tir = tx("total internal reflection", 36, YELLOW).move_to([-3.6, 1.15, 0])
        ev_l = tx("evanescent tail", 28, C_EV).move_to([P0[0] + H - 1.15, yb + 0.95, 0])
        ev_bg = BackgroundRectangle(ev_l, fill_opacity=0.8, buff=0.06)
        kz_im = tx("$k_x > k_0$: no real $k_z$ in air", 26, C_EV).move_to([O[0] + 0.9, O[1] - 0.38, 0])
        self.at(S("s7") + 0.2)
        self.play(FadeIn(kz_im), run_time=0.6)
        self.at(WT("s7", "evanescent") - 0.2)
        self.play(FadeIn(ev_bg), FadeIn(ev_l), run_time=0.6)
        self.at(WT("s7", "total") - 0.2)
        self.play(FadeOut(cons), FadeIn(tir, shift=0.1 * UP), run_time=0.8)
        self.finish()


# ============================================================ 2  gratings ==========
class Ch2Grating(Chapter):
    CH = 2
    TAG = "2\\quad A periodic boundary"

    def construct(self):
        S = self.S
        self.show_tag()
        fan = load("fan.npz")
        periods = fan["periods"]
        K0, K1 = 0.53 / periods[0], 0.53 / periods[-1]          # grating vector in units of the green k0
        nfr = len(periods)
        rgb_r, rgb_g, rgb_b = [mix(c, 0.0) for c in fan["rgb"]]
        # ---------------- right: the computed field map (glass below, air above, grating at z = 0)
        W, Hh = 6.0, 6.0 * 45.0 / 64.0
        C = np.array([3.55, -0.32, 0])
        y_g = C[1] - Hh / 2 + Hh * 5.0 / 45.0
        frame = Rectangle(width=W, height=Hh).move_to(C).set_stroke(GREY_C, 1.2)
        gm, gm_f, gm_s = self.sweep_image(rgba(fan["green"]), W, C, Hh, axis="up", soft=0.02)
        K_t = ValueTracker(K0)
        sweep = fan["sweep"].astype(np.float32)

        def white_fn(t):
            f = (K_t.get_value() - K0) / (K1 - K0) * (nfr - 1)
            i0 = int(np.clip(np.floor(f), 0, nfr - 2))
            a = float(np.clip(f - i0, 0, 1))
            return rgba(((1 - a) * sweep[i0] + a * sweep[i0 + 1] + 0.5).astype(np.uint8))
        wm, wm_f = self.live_image(white_fn, W, C, Hh)

        def grating_line():
            Lam = 0.53 / K_t.get_value()                       # um
            per = W * Lam / 64.0
            xx = np.linspace(C[0] - W / 2, C[0] + W / 2, 1400)
            return poly_line(xx, y_g + 0.03 * np.cos(TAU * (xx - C[0]) / per), GREY_A, 2)
        gl = always_redraw(grating_line)
        cap = tx("(computed)", 22, GREY_B).move_to([C[0] + W / 2 - 0.72, C[1] + Hh / 2 - 0.24, 0])
        g_lab = tx("grating", 26, GREY_B).move_to([C[0] - W / 2 + 0.72, y_g + 0.28, 0])
        self.at(S("g1"))
        self.add(gm)
        self.play(FadeIn(frame), gm_f.animate.set_value(1.0), gm_s.animate.set_value(5.0 / 45.0), run_time=0.8)
        self.add(gl)
        self.play(FadeIn(g_lab), FadeIn(cap), run_time=0.5)

        # ---------------- left: zoomed grating, period Lambda
        x0, x1 = -5.05, -0.95
        Lz = (x1 - x0) / 3.0
        y0 = 2.05
        xs = np.linspace(x0, x1, 400)
        prof = y0 + 0.22 * np.cos(TAU * (xs - x0) / Lz)
        relief = Polygon(*[[x, y, 0] for x, y in zip(xs, prof)], [x1, y0 - 0.55, 0], [x0, y0 - 0.55, 0])
        relief.set_stroke(GREY_A, 2).set_fill("#2a3036", 1.0)
        br = BraceBetweenPoints([x0 + Lz, y0 + 0.3, 0], [x0 + 2 * Lz, y0 + 0.3, 0], UP, color=GREY_A)
        br_l = mt("\\Lambda", 34).next_to(br, UP, buff=0.06)
        zoom_l = tx("zoomed in", 24, GREY_B).move_to([x0 - 0.85, y0 - 0.3, 0])
        self.at(WT("g1", "period") - 0.6)
        self.play(FadeIn(relief), FadeIn(zoom_l), run_time=0.8)
        self.play(GrowFromCenter(br), FadeIn(br_l), run_time=0.7)

        # ---------------- g2: anything that repeats every Lambda = sum of e^{imKx}
        rows = VGroup()
        rlabs = VGroup()
        ry = [0.95, 0.22, -0.51]
        for m, yy in zip((1, 2, 3), ry):
            rows.add(graph_line(lambda x, m=m: 0.28 * np.cos(m * TAU * (x - x0) / Lz), x0, x1, yy, 1, 1, color=C_K, width=3))
            rlabs.add(mt("e^{i%sKx}" % ("" if m == 1 else str(m)), 30, C_K).move_to([x0 - 0.65, yy, 0]))
        vlines = VGroup(*[fast_dashed([x0 + j * Lz, -0.8, 0], [x0 + j * Lz, y0 - 0.55, 0], 0.06, GREY_C, 1.5)
                          for j in range(4)])
        fs = MathTex("\\text{pattern}", "=", "\\sum_m c_m", "\\,e^{imKx}", ",\\qquad", "K", "=", "{2\\pi", "\\over", "\\Lambda}",
                     font_size=36)
        fs[3].set_color(C_K)
        fs[5].set_color(C_K)
        fs.move_to([-3.4, -1.68, 0])
        self.at(WT("g2", "fourier") - 0.2)
        rp = tx("(real parts shown)", 22, GREY_B).move_to([x1 - 0.85, ry[-1] - 0.47, 0])
        self.play(FadeIn(vlines), LaggedStart(*[Create(r) for r in rows], lag_ratio=0.3), FadeIn(rlabs), FadeIn(rp), run_time=1.35)
        self.at(WT("g2", "sum") - 0.1)
        self.play(Write(fs[:4]), run_time=1.2)
        self.at(WT("g2", "here") - 0.1)
        self.play(Write(fs[4:]), run_time=1.0)

        # ---------------- g3: multiplying adds mK to k_x
        mul = MathTex("e^{i", "k_x", "x}", "\\cdot", "e^{i", "mK", "x}", "=", "e^{i(", "k_x", "+", "mK", ")x}", font_size=42)
        for i in (1, 9):
            mul[i].set_color(C_KX)
        for i in (5, 11):
            mul[i].set_color(C_K)
        mul.move_to([-3.4, -2.62, 0])
        self.at(WT("g3", "multiplies") - 0.2)
        self.play(Write(mul[:7]), run_time=1.2)
        self.at(WT("g3", "shifts") - 0.2)
        self.play(Write(mul[7:]), run_time=1.0)

        # ---------------- g4: k-space, copies of the line spaced by K
        O = np.array([-3.65, -2.45, 0])
        RG = 1.72
        radii = {"r": RG * 530 / 650, "g": RG, "b": RG * 530 / 450}
        cols = {"r": rgb_r, "g": rgb_g, "b": rgb_b}
        ax = kaxes(O, 2.95, 3.05, 2.3)
        circ = {c: half_circle(O, radii[c], cols[c], 3) for c in "rgb"}
        rule = MathTex("k_{x,m}", "=", "k_{x,\\text{in}}", "+", "m", "K", font_size=44)
        rule[0].set_color(C_KX)
        rule[2].set_color(C_KX)
        rule[5].set_color(C_K)
        rule.move_to([-3.6, 2.3, 0])
        grat_eq = MathTex("\\sin\\theta_m", "=", "\\sin\\theta_{\\text{in}}", "+", "m", "{\\lambda", "\\over", "\\Lambda}",
                          font_size=40).move_to([-3.6, 1.35, 0])
        grat_l = tx("the grating equation", 24, GREY_B).next_to(grat_eq, DOWN, buff=0.1)
        ln_op = ValueTracker(0.0)
        ar_op = {c: ValueTracker(0.0) for c in "rgb"}
        nshow = ValueTracker(0.0)                              # copies grow outward

        def klines():
            K = K_t.get_value() * RG
            g = VGroup()
            o = ln_op.get_value()
            if o <= 0.003:
                return g
            ns = nshow.get_value()
            for m in range(-8, 9):
                x = m * K
                if abs(x) > 2.95 or abs(m) > ns + 1e-6:
                    continue
                col = C_KX if m == 0 else C_K
                g.add(fast_dashed(O + x * RIGHT, O + x * RIGHT + 2.3 * UP, 0.07, col, 3 if m == 0 else 2.5, o))
            return g
        kl = always_redraw(klines)

        def karrows():
            K = K_t.get_value() * RG
            g = VGroup()
            for c in "bgr":
                o = ar_op[c].get_value()
                if o <= 0.003:
                    continue
                R = radii[c]
                for m in range(-8, 9):
                    x = m * K
                    if abs(x) >= R - 1e-3:
                        continue
                    tip = O + np.array([x, np.sqrt(R ** 2 - x ** 2), 0])
                    g.add(fast_arrow(O, tip, cols[c], 4, 0.17, o), Dot(tip, 0.05, color=cols[c], fill_opacity=o))
            return g
        ka = always_redraw(karrows)
        kb_y = O[1] - 0.22
        kb = VGroup(Line(O + kb_y * 0 + np.array([0, -0.22, 0]), O + np.array([K0 * RG, -0.22, 0])).set_stroke(C_K, 3),
                    Line(O + np.array([0, -0.32, 0]), O + np.array([0, -0.12, 0])).set_stroke(C_K, 3),
                    Line(O + np.array([K0 * RG, -0.32, 0]), O + np.array([K0 * RG, -0.12, 0])).set_stroke(C_K, 3),
                    mt("K", 30, C_K).move_to(O + np.array([0.5 * K0 * RG, -0.5, 0])))
        self.at(S("g4"))
        self.play(FadeOut(VGroup(relief, zoom_l, br, br_l, rows, rlabs, vlines, fs, rp)), mul.animate.move_to([-3.6, 2.3, 0]),
                  run_time=0.8)
        self.play(GrowArrow(ax[0]), GrowArrow(ax[1]), FadeIn(ax[2:]), Create(circ["g"]), run_time=0.9)
        self.add(kl)
        self.play(ln_op.animate.set_value(1.0), run_time=0.4)
        self.play(FadeTransform(mul, rule), run_time=1.0)
        sym = tx("the pattern repeats only every $\\Lambda$", 26, GREY_A).move_to([-3.6, 1.7, 0])
        self.play(FadeIn(sym), run_time=0.5)
        self.at(WT("g4", ("copies", "kopies", "kop")) - 0.2)
        self.play(nshow.animate.set_value(4.0), run_time=1.6, rate_func=linear)
        self.play(FadeIn(kb), run_time=0.5)

        # ---------------- g5: every copy that hits the circle is an order (computed beams appear)
        mlabs = VGroup()
        for m in (-2, -1, 0, 1, 2):
            x = m * K0 * RG
            tip = O + np.array([x, np.sqrt(RG ** 2 - x ** 2), 0])
            d = (tip - O) / np.linalg.norm(tip - O)
            mlabs.add(mt(f"{m}", 26, rgb_g).move_to(tip + 0.28 * d + (0.12 * UP + 0.2 * RIGHT if m == 0 else 0)))
        self.at(S("g5"))
        self.add(ka)
        self.play(ar_op["g"].animate.set_value(1.0), FadeOut(kb), FadeOut(sym), run_time=0.8)
        self.play(gm_s.animate.set_value(1.0), FadeIn(mlabs), run_time=2.2, rate_func=linear)
        self.play(Write(grat_eq), FadeIn(grat_l), run_time=1.0)

        # ---------------- g6: colour sets the circle -> rainbow (white light)
        self.at(S("g6"))
        self.add(wm)
        self.bring_to_front(frame, gl, g_lab, cap)
        self.play(Create(circ["r"]), Create(circ["b"]), FadeOut(mlabs), run_time=0.9)
        self.at(WT("g6", "red") - 0.1)
        self.play(ar_op["r"].animate.set_value(1.0), ar_op["b"].animate.set_value(1.0), wm_f.animate.set_value(1.0),
                  gm_f.animate.set_value(0.0), run_time=1.6)
        self.remove(gm)
        red_l = tx("same $K$: the small red circle gives the widest angle", 24, GREY_A).move_to([O[0], O[1] - 0.42, 0])
        self.at(WT("g6", "bends") - 0.2)
        self.play(FadeIn(red_l), run_time=0.6)

        # ---------------- g7: shrink the period -> orders leave one by one
        lam_bar_s = 2.2
        bars_x0 = -3.9
        by1, by2 = 1.3, 0.85

        def bars():
            Lam = 0.53 / K_t.get_value()
            g = VGroup(Line([bars_x0, by1, 0], [bars_x0 + Lam * lam_bar_s, by1, 0]).set_stroke(GREY_A, 7),
                       Line([bars_x0, by2, 0], [bars_x0 + 0.53 * lam_bar_s, by2, 0]).set_stroke(rgb_g, 7))
            return g
        bv = always_redraw(bars)
        bl = VGroup(tx("period $\\Lambda$", 28, GREY_A).next_to([bars_x0, by1, 0], LEFT, buff=0.15),
                    tx("wavelength $\\lambda$", 28, rgb_g).next_to([bars_x0, by2, 0], LEFT, buff=0.15))
        self.at(S("g7"))
        self.play(FadeOut(red_l), FadeOut(grat_l), FadeOut(grat_eq), run_time=0.4)
        self.add(bv)
        self.play(FadeIn(bl), FadeIn(bv), run_time=0.4)
        self.play(K_t.animate.set_value(K1), run_time=5.0, rate_func=rate_functions.ease_in_out_sine)

        # ---------------- g8: sub-wavelength -> zeroth order only = effective medium
        sub = tx("$\\Lambda < \\lambda$: only $m = 0$ survives", 32, WHITE).move_to([-3.6, 2.3, 0])
        film = tx("acts like a uniform film", 32, GREY_A).move_to([-3.6, 1.35, 0])
        meta = tx("metasurfaces live here", 32, YELLOW)
        meta = VGroup(SurroundingRectangle(meta, buff=0.14, color=YELLOW, corner_radius=0.08), meta).move_to([-3.6, 0.65, 0])
        self.at(S("g8"))
        self.play(FadeOut(rule), FadeIn(sub, shift=0.1 * UP), run_time=0.8)
        self.at(WT("g8", "uniform") - 0.5)
        self.play(FadeOut(bl), FadeOut(bv), run_time=0.4)
        self.remove(bv)
        self.play(FadeIn(film), run_time=0.6)
        self.at(WT("g8", "meta") - 0.3)
        self.play(FadeIn(meta, shift=0.1 * UP), run_time=0.7)
        self.finish()


# ============================================================ 3  Bloch waves & band gaps ==
class Ch3Bands(Chapter):
    CH = 3
    TAG = "3\\quad Bloch waves and band gaps"

    def construct(self):
        S = self.S
        self.show_tag()
        st = STACK
        dns, nus, half, Rg = st["dns"], st["nus"], st["half"], st["R"]
        nbar, npairs = float(st["nbar"]), int(st["npairs"])
        DN_MAX = float(dns[-1])
        sub = slice(None, None, 2)
        nus_s, half_s, R_s = nus[sub], half[:, sub], Rg[:, sub]
        dn_t = ValueTracker(DN_MAX)

        def interp(arr):
            f = dn_t.get_value() / DN_MAX * (len(dns) - 1)
            i0 = int(np.clip(np.floor(f), 0, len(dns) - 2))
            a = float(np.clip(f - i0, 0, 1))
            return (1 - a) * arr[i0] + a * arr[i0 + 1]

        # ---------------- the strip: layers along z (light travels to the right)
        SC, XZ0 = 0.714, -3.571                 # scene units per period; x of z = 0
        XW0, XW1 = -5.0, 5.0
        ys0, ys1 = 1.85, 2.75
        ysc = 0.5 * (ys0 + ys1)
        shift_t = ValueTracker(0.0)
        fin_t = ValueTracker(0.0)               # 0: infinite lattice, 1: finite stack of 10 periods
        lay_op = ValueTracker(0.0)

        def grey(n):
            v = float(np.clip(0.2 + 0.52 * (n - 1.42) / 0.95, 0, 1))
            return rgb_to_color([v, v, v * 1.04])

        def strip():
            o = lay_op.get_value()
            g = VGroup()
            if o <= 0.003:
                return g
            dn = dn_t.get_value()
            nl, nh = nbar - dn / 2, nbar + dn / 2
            sh = shift_t.get_value()
            fo = fin_t.get_value()
            zmin, zmax = (XW0 - XZ0) / SC, (XW1 - XZ0) / SC
            g.add(Rectangle(width=XW1 - XW0, height=ys1 - ys0).move_to([0.5 * (XW0 + XW1), ysc, 0])
                  .set_stroke(width=0).set_fill(grey(nbar), o))
            for p_ in range(int(np.floor(zmin - sh)) - 1, int(np.ceil(zmax - sh)) + 1):
                for a, b, n in ((p_ + sh, p_ + sh + 0.5, nl), (p_ + sh + 0.5, p_ + sh + 1.0, nh)):
                    za, zb = max(a, zmin), min(b, zmax)
                    if zb <= za:
                        continue
                    inside = 1.0 if (p_ >= 0 and p_ < npairs) else 1.0 - fo
                    if inside <= 0.003:
                        continue
                    x0_, x1_ = XZ0 + za * SC, XZ0 + zb * SC
                    g.add(Rectangle(width=x1_ - x0_, height=ys1 - ys0).move_to([0.5 * (x0_ + x1_), ysc, 0])
                          .set_stroke(width=0).set_fill(grey(n), o * inside))
            return g
        stv = always_redraw(strip)
        light = Arrow([XW0 - 0.95, ysc, 0], [XW0 - 0.1, ysc, 0], buff=0, stroke_width=6, color=C_U,
                      max_tip_length_to_length_ratio=0.3)
        light_l = tx("light", 24, C_U).next_to(light, UP, buff=0.08)
        brl = BraceBetweenPoints([XZ0 + 6 * SC, ys1 + 0.05, 0], [XZ0 + 7 * SC, ys1 + 0.05, 0], UP, color=GREY_A)
        brl_l = mt("\\Lambda", 30).next_to(brl, UP, buff=0.05)
        hl = VGroup(tx("high $n$", 22, GREY_A), tx("low $n$", 22, GREY_A))
        self.at(S("b1"))
        self.add(stv)
        self.play(lay_op.animate.set_value(1.0), GrowArrow(light), FadeIn(light_l), run_time=1.0)
        self.at(WT("b1", "repeating") - 0.2)
        self.play(GrowFromCenter(brl), FadeIn(brl_l), run_time=0.8)

        # ---------------- b2: shift by one period -> nothing changes; a fixed phase per period
        sh_l = tx("shift by $\\Lambda$: the same structure", 28, GREY_A).move_to([0, 1.25, 0])
        self.at(S("b2") + 0.1)
        self.play(FadeIn(sh_l), run_time=0.5)
        self.play(shift_t.animate.set_value(1.0), run_time=1.6, rate_func=rate_functions.ease_in_out_sine)
        self.remove(stv)
        shift_t.set_value(0.0)
        self.add(stv)
        Eb = st["Eb"]
        zb = st["zb"]
        kapb = float(st["kapb"])
        per = zb < 1.0
        z1, E1 = zb[per], Eb[per]
        emax = np.abs(Eb).max()
        w_b = TAU * 0.45

        def psi(z):
            p_ = np.floor(z)
            frac = z - p_
            base = np.interp(frac, z1, E1.real) + 1j * np.interp(frac, z1, E1.imag)
            return base * np.exp(1j * kapb * p_) / emax
        zz = np.linspace((XW0 - XZ0) / SC, (XW1 - XZ0) / SC, 500)
        psi_z = psi(zz)
        wv_op = ValueTracker(0.0)
        env_op = ValueTracker(0.0)

        def bloch_curve():
            ph = np.exp(-1j * w_b * self.now)
            g = VGroup(poly_line(XZ0 + zz * SC, ysc + 0.38 * np.real(psi_z * ph), C_U, 4))
            eo = env_op.get_value()
            if eo > 0.003:
                g.add(poly_line(XZ0 + zz * SC, ysc + 0.38 * np.abs(psi_z), YELLOW, 2.5, eo),
                      poly_line(XZ0 + zz * SC, ysc - 0.38 * np.abs(psi_z), YELLOW, 2.5, eo))
            return g
        bc = faded(bloch_curve, wv_op)
        dial_z = np.arange(2, 9)
        dial_y, dial_r = 0.95, 0.27
        dial_op = ValueTracker(0.0)
        psi_d = psi(dial_z + 0.25)

        def dials():
            ph = np.exp(-1j * w_b * self.now)
            g = VGroup()
            for zd, v in zip(dial_z, psi_d):
                c = np.array([XZ0 + (zd + 0.25) * SC, dial_y, 0])
                u = v * ph / np.abs(v)
                g.add(Circle(radius=dial_r).move_to(c).set_stroke(GREY_B, 1.5),
                      fast_arrow(c, c + dial_r * np.array([u.real, u.imag, 0]), YELLOW, 3.5, 0.11),
                      fast_dashed(c + dial_r * UP, [c[0], ys0, 0], 0.05, GREY_C, 1.2))
            return g
        dv = faded(dials, dial_op)
        step = MathTex("\\times\\, e^{i\\kappa\\Lambda}", font_size=30, color=YELLOW)
        step.move_to([XZ0 + 2.75 * SC, dial_y - 0.55, 0])
        step_ar = CurvedArrow(np.array([XZ0 + 2.35 * SC, dial_y - 0.33, 0]), np.array([XZ0 + 3.15 * SC, dial_y - 0.33, 0]),
                              angle=TAU / 6, color=YELLOW, stroke_width=2.5, tip_length=0.12)
        self.at(WT("b2", "waves") - 0.2)
        self.play(FadeOut(sh_l), run_time=0.3)
        self.add(bc, dv)
        self.play(wv_op.animate.set_value(1.0), run_time=0.9)
        self.at(WT("b2", "phase") - 0.3)
        self.play(dial_op.animate.set_value(1.0), run_time=1.0)
        self.play(Create(step_ar), FadeIn(step), run_time=0.7)

        # ---------------- b3: Bloch waves; kappa only matters modulo K
        bl_eq = MathTex("\\psi(z)", "=", "e^{i\\kappa z}", "\\,u(z)", ",\\qquad", "u(z+\\Lambda)", "=", "u(z)", font_size=40)
        bl_eq[0].set_color(C_U)
        bl_eq[2].set_color(C_KZ)
        bl_eq[3].set_color(YELLOW)
        bl_eq.move_to([0, -0.25, 0])
        bl_l = tx("Bloch wave", 30, C_U).next_to(bl_eq, LEFT, buff=0.5)
        mod_eq = MathTex("e^{i(\\kappa + K)\\Lambda}", "=", "e^{i\\kappa\\Lambda}", "\\, e^{2\\pi i}", "=", "e^{i\\kappa\\Lambda}",
                         font_size=40)
        mod_eq[0][4:6].set_color(C_K)
        mod_eq.move_to([0, -1.35, 0])
        mod_l = tx("$\\kappa$ only matters up to multiples of $K$", 30, GREY_A).next_to(mod_eq, DOWN, buff=0.22)
        self.at(S("b3"))
        self.play(Write(bl_eq[:4]), FadeIn(bl_l), run_time=1.3)
        self.at(WT("b3", "function") - 0.2)
        self.play(env_op.animate.set_value(1.0), Write(bl_eq[4:]), run_time=1.2)
        self.at(WT("b3", "full") - 0.2)
        self.play(Write(mod_eq), run_time=1.5)
        self.at(WT("b3", "matters") - 0.6)
        self.play(FadeIn(mod_l), run_time=0.7)

        # ---------------- b4: band diagram, light line + copies shifted by K
        OB = np.array([-3.3, -2.6, 0])          # (kappa = 0, omega = 0)
        XK = 2.2                                 # scene units per K
        YN = 3.2 / 0.42                          # scene units per nu
        NUMAX = 0.42

        def P(x, nu):
            return OB + np.array([x * XK, nu * YN, 0])
        bax = VGroup(Arrow(P(-1.05, 0), P(1.1, 0), buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                     Arrow(P(0, 0), P(0, NUMAX + 0.03), buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.05))
        bax_l = VGroup(mt("\\kappa", 32, C_KZ).next_to(bax[0], RIGHT, buff=0.08),
                       mt("\\omega", 32).next_to(bax[1], UP, buff=0.06))
        ticks = VGroup()
        for xv, lab in ((-1, "-K"), (-0.5, "-\\tfrac{K}{2}"), (0.5, "\\tfrac{K}{2}"), (1, "K")):
            ticks.add(Line(P(xv, 0) + 0.07 * DOWN, P(xv, 0) + 0.07 * UP).set_stroke(GREY_B, 2),
                      mt(lab, 26, GREY_B).next_to(P(xv, 0), DOWN, buff=0.12))
        freq_l = tx("frequency", 24, GREY_B).next_to(bax_l[1], LEFT, buff=0.12)
        freq_l.shift(0.25 * LEFT)

        def vline(shift, lo=-1.05, hi=1.05):
            xs = np.linspace(lo, hi, 400)
            nu = np.abs(xs - shift) / nbar
            ok = nu <= NUMAX
            pts = [P(x, n) for x, n in zip(xs[ok], nu[ok])]
            return VMobject().set_points_as_corners(pts)
        main_v = VGroup(vline(0, -0.8, 0), vline(0, 0, 0.8)).set_stroke(C_KZ, 4)
        copy_p = VGroup(vline(1, 0.2, 1.0), vline(1, 1.0, 1.05)).set_stroke(C_KZ, 3.5, opacity=0.8)
        copy_m = VGroup(vline(-1, -1.0, -0.2), vline(-1, -1.05, -1.0)).set_stroke(C_KZ, 3.5, opacity=0.8)
        sh_ar = VGroup(Arrow(P(0.35, 0.35 / nbar), P(1.35, 0.35 / nbar) + 0.0 * RIGHT, buff=0.05, stroke_width=3, color=C_K,
                             max_tip_length_to_length_ratio=0.12),
                       Arrow(P(-0.35, 0.35 / nbar), P(-1.35, 0.35 / nbar), buff=0.05, stroke_width=3, color=C_K,
                             max_tip_length_to_length_ratio=0.12))
        sh_ar_l = VGroup(mt("+K", 28, C_K).next_to(sh_ar[0], UP, buff=0.04), mt("-K", 28, C_K).next_to(sh_ar[1], UP, buff=0.04))
        zone = VGroup(fast_dashed(P(-0.5, 0), P(-0.5, NUMAX), 0.07, GREY_A, 2), fast_dashed(P(0.5, 0), P(0.5, NUMAX), 0.07, GREY_A, 2))
        zone_l = tx("one zone", 24, GREY_A).move_to(P(0, 0) + 0.36 * DOWN)
        self.at(S("b4"))
        self.play(FadeOut(VGroup(bl_eq, bl_l, mod_eq, mod_l, step, step_ar, brl, brl_l)), dial_op.animate.set_value(0.0),
                  env_op.animate.set_value(0.0), wv_op.animate.set_value(0.0), run_time=0.6)
        self.remove(dv, bc)
        self.play(GrowArrow(bax[0]), GrowArrow(bax[1]), FadeIn(bax_l), FadeIn(ticks), FadeIn(freq_l), run_time=0.9)
        self.at(WT("b4", "contrast") - 0.4)
        self.play(dn_t.animate.set_value(0.0), Create(main_v), run_time=1.4)
        self.at(WT("b4", ("copies", "kop")) - 0.1)
        self.play(GrowArrow(sh_ar[0]), GrowArrow(sh_ar[1]), FadeIn(sh_ar_l), run_time=0.7)
        self.play(Create(copy_p), Create(copy_m), run_time=1.0)
        self.at(WT("b4", "same") - 0.2)
        self.play(FadeOut(sh_ar), FadeOut(sh_ar_l), FadeIn(zone), FadeIn(zone_l), run_time=0.8)

        # ---------------- b5: they cross at the zone edge: Bragg
        nuB = 0.5 / nbar
        cross = VGroup(Dot(P(0.5, nuB), 0.08, color=YELLOW), Dot(P(-0.5, nuB), 0.08, color=YELLOW))
        bragg = MathTex("\\lambda", "=", "2", "n", "\\Lambda", font_size=40, color=YELLOW).move_to(P(1.0, nuB) + 0.5 * RIGHT + 0.05 * UP)
        bragg_l = tx("Bragg: reflections add in step", 26, GREY_A).next_to(bragg, DOWN, buff=0.12).align_to(bragg, LEFT)
        dim_out = VGroup(copy_p[0], copy_m[0])
        self.at(S("b5"))
        self.play(FadeIn(cross, scale=1.6), run_time=0.6)
        self.at(WT("b5", "lambda") - 0.2)
        self.play(Write(bragg), run_time=0.9)
        self.at(WT("b5", "bragg") - 0.2)
        self.play(FadeIn(bragg_l), run_time=0.7)

        # ---------------- b6: turn up the contrast -> avoided crossing (computed bands) + mirror spectrum
        band_op = ValueTracker(0.0)

        def bands():
            hv = interp(half_s)
            ok = np.abs(hv) <= 1.0
            x = np.arccos(np.clip(hv, -1, 1)) / TAU
            g = VGroup()
            # split into continuous runs
            idx = np.where(ok)[0]
            if len(idx) == 0:
                return g
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for r_ in runs:
                if len(r_) < 2:
                    continue
                for sgn in (1, -1):
                    g.add(poly_line(OB[0] + sgn * x[r_] * XK, OB[1] + nus_s[r_] * YN, C_KZ, 4.5))
            return g
        bd = faded(bands, band_op)
        # reflectance panel sharing the frequency axis
        RX0, RW = 0.75, 2.45
        rax = VGroup(Line([RX0, OB[1], 0], [RX0 + RW + 0.2, OB[1], 0]).set_stroke(GREY_B, 2),
                     Line([RX0, OB[1], 0], [RX0, OB[1] + NUMAX * YN + 0.1, 0]).set_stroke(GREY_B, 2),
                     Line([RX0 + RW, OB[1] - 0.07, 0], [RX0 + RW, OB[1] + 0.07, 0]).set_stroke(GREY_B, 2))
        rax_l = VGroup(mt("0", 24, GREY_B).next_to([RX0, OB[1], 0], DOWN, buff=0.12),
                       mt("1", 24, GREY_B).next_to([RX0 + RW, OB[1], 0], DOWN, buff=0.12),
                       tx("reflectance", 26, GREY_A).next_to([RX0 + 0.5 * RW, OB[1], 0], DOWN, buff=0.12),
                       tx("10 pairs (computed)", 22, GREY_B).move_to([RX0 + 0.5 * RW, OB[1] + NUMAX * YN + 0.3, 0]))
        r_op = ValueTracker(0.0)

        def refl():
            Rv = interp(R_s)
            return poly_line(RX0 + RW * Rv, OB[1] + nus_s * YN, WHITE, 3)
        rc = faded(refl, r_op)
        # visible colours along the frequency axis (Lambda chosen so the gap centre is blue, as in the hook)
        Lam_nm = float(st["Lam"])
        nrow = 240
        nu_rows = np.linspace(NUMAX, 0, nrow)
        lam_rows = Lam_nm / np.maximum(nu_rows, 1e-6)
        vis = (lam_rows >= LAMV[0]) & (lam_rows <= LAMV[-1])
        bar = np.zeros((nrow, 4, 4), np.uint8)
        ci = np.clip(np.searchsorted(LAMV, lam_rows), 0, len(LAMV) - 1)
        bar[:, :, :3] = (255 * LAMRGB[ci])[:, None, :]
        bar[:, :, 3] = np.where(vis, 255, 0)[:, None]
        bar_m, bar_f = self.static_image(bar, 0.14, [RX0 - 0.14, OB[1] + 0.5 * NUMAX * YN, 0], NUMAX * YN)
        # slider for the index contrast
        SX0, SX1, SY = 4.15, 6.45, 0.75
        sl = VGroup(Line([SX0, SY, 0], [SX1, SY, 0]).set_stroke(GREY_C, 4),
                    tx("index contrast $\\Delta n$", 28, GREY_A).move_to([0.5 * (SX0 + SX1), SY + 0.5, 0]),
                    mt("0", 24, GREY_B).next_to([SX0, SY, 0], DOWN, buff=0.15),
                    mt("0.9", 24, GREY_B).next_to([SX1, SY, 0], DOWN, buff=0.15))
        knob = always_redraw(lambda: VGroup(Line([SX0, SY, 0], [SX0 + (SX1 - SX0) * dn_t.get_value() / DN_MAX, SY, 0])
                                            .set_stroke(YELLOW, 5),
                                            Dot([SX0 + (SX1 - SX0) * dn_t.get_value() / DN_MAX, SY, 0], 0.11, color=YELLOW)))
        sl_l2 = tx("($n_{\\mathrm{high}} - n_{\\mathrm{low}}$)", 22, GREY_B).move_to([0.5 * (SX0 + SX1), SY - 0.55, 0])
        self.at(S("b6"))
        self.add(bd)
        self.play(FadeOut(VGroup(main_v, copy_p, copy_m, zone_l, bragg_l)), band_op.animate.set_value(1.0),
                  FadeIn(sl), FadeIn(sl_l2), run_time=0.8)
        self.add(knob, rc, bar_m)
        self.play(FadeIn(rax), FadeIn(rax_l), r_op.animate.set_value(1.0), bar_f.animate.set_value(1.0), run_time=0.8)
        self.at(WT("b6", "watch") - 0.3)
        self.play(dn_t.animate.set_value(DN_MAX), run_time=6.2, rate_func=rate_functions.ease_in_out_sine)

        # ---------------- b7: the gap
        gap = st["gap"][-1]
        g_band = VGroup(Rectangle(width=XK, height=(gap[1] - gap[0]) * YN).move_to(P(0, 0.5 * (gap[0] + gap[1])))
                        .set_stroke(width=0).set_fill(C_EV, 0.28),
                        Rectangle(width=RW, height=(gap[1] - gap[0]) * YN)
                        .move_to([RX0 + 0.5 * RW, OB[1] + 0.5 * (gap[0] + gap[1]) * YN, 0]).set_stroke(width=0).set_fill(C_EV, 0.16))
        gap_l = tx("band gap", 32, C_EV).move_to([-0.95, OB[1] + gap[1] * YN + 0.3, 0])
        self.at(S("b7"))
        self.play(FadeOut(VGroup(bragg, cross)), FadeIn(g_band), FadeIn(gap_l), run_time=0.9)
        self.bring_to_front(bd, rc)

        # ---------------- b8: light dies off inside -> a mirror (computed field in the finite stack)
        nu_f, Ef, zf = st["nu_f"], st["Ef"], st["zf"]
        jm = int(np.argmin(np.abs(nu_f - 0.5 * (gap[0] + gap[1]))))
        Em = Ef[jm] / np.abs(Ef[jm]).max()
        xf = XZ0 + zf * SC
        sel = (xf >= XW0) & (xf <= XW1)
        f_op = ValueTracker(0.0)

        def gap_field():
            ph = np.exp(-1j * w_b * self.now)
            return VGroup(poly_line(xf[sel], ysc + 0.38 * np.real(Em[sel] * ph), C_U, 4),
                          poly_line(xf[sel], ysc + 0.38 * np.abs(Em[sel]), C_EV, 2.5, 0.9))
        gf = faded(gap_field, f_op)
        cur = DashedLine(P(-0.5, nu_f[jm]), [RX0 + RW + 0.1, OB[1] + nu_f[jm] * YN, 0], dash_length=0.08).set_stroke(YELLOW, 2)
        mirror = tx("nearly perfect mirror", 30, YELLOW).move_to([5.0, OB[1] + nu_f[jm] * YN, 0])
        dies = tx("dies off within a few layers", 26, C_EV).move_to([-1.6, 1.5, 0])
        self.at(S("b8"))
        self.play(fin_t.animate.set_value(1.0), run_time=0.6)
        self.add(gf)
        self.play(f_op.animate.set_value(1.0), Create(cur), run_time=0.8)
        self.play(FadeIn(dies), run_time=0.5)
        self.at(WT("b8", "mirror") - 0.4)
        self.play(FadeIn(mirror, shift=0.1 * LEFT), run_time=0.7)
        self.finish()


# ============================================================ 4  RCWA ===============
class Ch4RCWA(Chapter):
    CH = 4
    TAG = "4\\quad RCWA: one layer at a time"

    def construct(self):
        S = self.S
        self.show_tag()
        rc = load("rcwa.npz")
        Tm, m_all = rc["T"], rc["m"]
        M = (len(m_all) - 1) // 2
        per, fill = float(rc["per"]), float(rc["fill"])
        # ---------------- r1: a thick grating; its orders computed with RCWA
        XC, LP = -3.35, 1.25                        # centre of the cross-section; period in scene units
        hs = 0.9 / per * LP * 1.25                  # ridge height (0.9 lambda; drawn 1.25x taller for legibility)
        y_top = -0.35
        y_mid = y_top - hs
        y_bot = y_mid - 1.0
        xL, xR = XC - 1.5 * LP, XC + 1.5 * LP
        glass_c = "#2a3036"
        sub_r = Rectangle(width=xR - xL, height=y_mid - y_bot).move_to([XC, 0.5 * (y_mid + y_bot), 0])
        sub_r.set_stroke(width=0).set_fill(glass_c, 1.0)
        ridges = VGroup(*[Rectangle(width=fill * LP, height=hs).move_to([XC + j * LP, 0.5 * (y_top + y_mid), 0])
                          .set_stroke(width=0).set_fill(GREY_B, 1.0) for j in (-1, 0, 1)])
        base = Line([xL, y_mid, 0], [xR, y_mid, 0]).set_stroke(GREY_C, 1.5)
        labs = VGroup(tx("air", 24, GREY_B).move_to([xL - 0.5, y_top + 0.5, 0]),
                      tx("grating", 24, GREY_B).move_to([xL - 0.62, 0.5 * (y_top + y_mid), 0]),
                      tx("glass", 24, GREY_B).move_to([xL - 0.55, 0.5 * (y_mid + y_bot), 0]))
        title = tx("rigorous coupled-wave analysis (RCWA)", 30, WHITE).move_to([-3.75, 2.55, 0])
        a_in = Arrow([XC, y_bot + 0.05, 0], [XC, y_mid - 0.04, 0], buff=0, stroke_width=6, color=WHITE,
                     max_tip_length_to_length_ratio=0.22)
        L0 = 1.75
        outs = VGroup()
        for mm in range(-2, 3):
            T = Tm[M + mm]
            if T < 1e-4:
                continue
            sn = mm / per
            d = np.array([sn, np.sqrt(1 - sn ** 2), 0])
            ln = max(L0 * np.sqrt(T / Tm.max()), 0.14)
            outs.add(Arrow([XC, y_top, 0], [XC, y_top, 0] + ln * d, buff=0, stroke_width=5, color=C_U,
                           max_tip_length_to_length_ratio=0.2 if ln > 0.4 else 0.6))
        olab = VGroup()
        for mm, nm in ((0, "0"), (1, "+1"), (-1, "-1")):
            sn = mm / per
            d = np.array([sn, np.sqrt(1 - sn ** 2), 0])
            olab.add(mt(nm, 24, C_U).move_to([XC, y_top, 0] + (L0 * np.sqrt(Tm[M + mm] / Tm.max()) + 0.25) * d))
        comp = tx("(orders computed with RCWA)", 22, GREY_B).next_to(sub_r, DOWN, buff=0.15)
        self.at(S("r1"))
        self.play(FadeIn(sub_r), FadeIn(ridges), FadeIn(base), FadeIn(labs), run_time=0.9)
        self.play(GrowArrow(a_in), run_time=0.5)
        self.at(WT("r1", "rigorous") - 0.2)
        self.play(Write(title), run_time=1.2)
        self.at(WT("r1", "puts") - 0.3)
        self.play(LaggedStart(*[GrowArrow(a) for a in outs], lag_ratio=0.15), FadeIn(olab), FadeIn(comp), run_time=1.1)

        # ---------------- r2: the field in a layer = one amplitude per order
        vec = MathTex("\\begin{pmatrix} a_{-2} \\\\ a_{-1} \\\\ a_0 \\\\ a_1 \\\\ a_2 \\end{pmatrix}", font_size=30, color=C_U)
        vec.move_to([-0.7, -0.85, 0])
        vec_l = tx("one amplitude\\\\per order", 24, GREY_A).next_to(vec, UP, buff=0.18)
        self.at(WT("r2", "list") - 0.4)
        self.play(FadeIn(vec, shift=0.1 * LEFT), FadeIn(vec_l), run_time=0.9)

        # ---------------- r3: uniform layer -> diagonal matrix
        A_uni, A_gr = rc["A_uni"], rc["A_gr"]
        nM = A_uni.shape[0]
        cs = 0.31
        MX, MY = 1.75, -0.95                       # matrix centre
        vmax = np.abs(A_gr).max()

        def cell_style(v, diag):
            a = float(np.clip(abs(v) / vmax, 0, 1)) ** 0.6
            if diag:
                return (C_KZ if v >= 0 else C_EV), 0.15 + 0.85 * a
            return C_K, 0.9 * a
        cells = VGroup()
        for i in range(nM):
            for j in range(nM):
                sq = Square(side_length=cs * 0.92).move_to([MX + (j - (nM - 1) / 2) * cs, MY - (i - (nM - 1) / 2) * cs, 0])
                if i == j:
                    col, op = cell_style(A_uni[i, j], True)
                    sq.set_stroke(width=0).set_fill(col, op)
                else:
                    sq.set_stroke(GREY_D, 1).set_fill(BLACK, 0.0)
                cells.add(sq)
        hw = nM * cs / 2
        mbox = VGroup(Line([MX - hw - 0.08, MY - hw, 0], [MX - hw - 0.08, MY + hw, 0]),
                      Line([MX + hw + 0.08, MY - hw, 0], [MX + hw + 0.08, MY + hw, 0])).set_stroke(GREY_A, 2.5)
        m_bot = tx("orders $m = -4 \\dots 4$", 22, GREY_B).next_to(cells, DOWN, buff=0.18)
        uni_l = tx("uniform layer: diagonal", 30, C_KZ).move_to([MX, MY + hw + 0.35, 0])
        each = MathTex("\\text{each order: }", "e^{\\,i k_{z,m} z}", font_size=32).move_to([MX, MY + hw + 0.95, 0])
        each[1].set_color(C_KZ)
        hl_glass = SurroundingRectangle(sub_r, buff=0.03, color=YELLOW, stroke_width=3)
        self.at(WT("r3", "uniform") - 0.2)
        self.play(Create(hl_glass), FadeIn(cells), FadeIn(mbox), FadeIn(m_bot), FadeOut(vec_l), run_time=1.0)
        self.at(WT("r3", "phase") - 0.2)
        self.play(Write(each), run_time=1.0)
        self.at(WT("r3", "diagonal") - 0.3)
        self.play(FadeIn(uni_l), run_time=0.6)

        # ---------------- r4: grating layer -> eps_{m-p} fills the matrix
        eps_m = rc["eps_m"]
        Me = (len(eps_m) - 1) // 2
        ex0, ey0 = 0.95, 2.1                         # eps(x), one period
        ex, ev = rc["eps_prof_x"], rc["eps_prof"]
        eps_plot = VGroup(Line([ex0 - 0.7, ey0 - 0.3, 0], [ex0 + 0.7, ey0 - 0.3, 0]).set_stroke(GREY_C, 1.5),
                          poly_line(ex0 + 1.4 * ex, ey0 - 0.3 + 0.7 * (ev - 1) / (ev.max() - 1), GREY_A, 3))
        eps_pl = mt("\\varepsilon(x)", 30).next_to(eps_plot, LEFT, buff=0.15)
        sx0 = 3.55
        stems_g = VGroup(Line([sx0 - 1.0, ey0 - 0.3, 0], [sx0 + 1.0, ey0 - 0.3, 0]).set_stroke(GREY_C, 1.5))
        for k in range(-4, 5):
            v = eps_m[Me + k] / eps_m[Me + 1]
            col = C_KZ if k == 0 else C_K
            hgt = 0.55 * min(abs(v), 1.3) * np.sign(v)
            stems_g.add(Line([sx0 + 0.22 * k, ey0 - 0.3, 0], [sx0 + 0.22 * k, ey0 - 0.3 + hgt, 0]).set_stroke(col, 3),
                        Dot([sx0 + 0.22 * k, ey0 - 0.3 + hgt, 0], 0.035, color=col))
        stems_l = mt("\\varepsilon_m", 30, C_K).next_to(stems_g, RIGHT, buff=0.15)
        ft_ar = Arrow([ex0 + 0.8, ey0, 0], [sx0 - 1.05, ey0, 0], buff=0, stroke_width=3, color=GREY_A,
                      max_tip_length_to_length_ratio=0.25)
        gr_l = MathTex("\\text{grating layer: }", "\\varepsilon_{m-p}", "\\text{ couples the orders}", font_size=30)
        gr_l[1].set_color(C_K)
        gr_l.move_to(uni_l)
        hl_gr = Rectangle(width=xR - xL + 0.06, height=hs + 0.1).move_to([XC, 0.5 * (y_top + y_mid), 0])
        hl_gr.set_stroke(YELLOW, 3)
        anims = []
        for i in range(nM):
            for j in range(nM):
                col, op = cell_style(A_gr[i, j], i == j)
                sq = cells[i * nM + j]
                anims.append(sq.animate.set_fill(col, op).set_stroke(width=0) if i != j else sq.animate.set_fill(col, op))
        self.at(S("r4"))
        self.play(FadeOut(each), FadeOut(vec), ReplacementTransform(hl_glass, hl_gr), run_time=0.7)
        self.at(WT("r4", "fourier") - 0.2)
        self.play(FadeIn(eps_plot), FadeIn(eps_pl), run_time=0.6)
        self.play(GrowArrow(ft_ar), FadeIn(stems_g), FadeIn(stems_l), run_time=0.8)
        self.at(WT("r4", "couple") - 0.2)
        self.play(*anims, FadeTransform(uni_l, gr_l), run_time=1.6)

        # ---------------- r5: eig -> Bloch modes with their own k_z
        modes, xs, qv = rc["modes"], rc["xs"], rc["q"]
        PX0, PW, PH = 3.95, 2.6, 0.62
        mode_g = VGroup()
        for k in range(3):
            yc = 0.45 - 1.25 * k
            g = VGroup()
            for j in (-1, 0, 1):
                xa = max(-per, j * per - fill * per / 2)
                xb = min(per, j * per + fill * per / 2)
                if xb > xa:
                    g.add(Rectangle(width=(xb - xa) / (2 * per) * PW, height=PH)
                          .move_to([PX0 + ((xa + xb) / 2 + per) / (2 * per) * PW, yc, 0]).set_stroke(width=0).set_fill(GREY_D, 1.0))
            g.add(Line([PX0, yc, 0], [PX0 + PW, yc, 0]).set_stroke(GREY_C, 1))
            g.add(poly_line(PX0 + (xs + per) / (2 * per) * PW, yc + 0.28 * modes[k], C_U, 3.5))
            lab = MathTex("k_z", "=", f"{qv[k].real:.2f}", "\\,k_0", font_size=26).move_to([PX0 + PW / 2, yc + PH / 2 + 0.2, 0])
            lab[0].set_color(C_KZ)
            g.add(lab)
            mode_g.add(g)
        eig_ar = Arrow([MX + hw + 0.15, MY, 0], [PX0 - 0.12, MY + 0.15, 0], buff=0, stroke_width=4, color=GREY_A,
                       max_tip_length_to_length_ratio=0.25)
        eig_l = tx("eig", 28, YELLOW).next_to(eig_ar, UP, buff=0.05)
        modes_l = tx("Bloch modes (computed)", 24, GREY_B).move_to([PX0 + PW / 2, 1.55, 0])
        self.at(WT("r5", "diagonalize") - 0.2)
        self.play(FadeOut(VGroup(eps_plot, eps_pl, stems_g, stems_l, ft_ar, gr_l)), GrowArrow(eig_ar), FadeIn(eig_l), run_time=0.8)
        self.at(WT("r5", "eigenvectors") - 0.2)
        self.play(LaggedStart(*[FadeIn(g[:-1]) for g in mode_g], lag_ratio=0.25), FadeIn(modes_l), run_time=1.3)
        self.at(WT("r5", "eigenvalues") - 0.2)
        self.play(LaggedStart(*[FadeIn(g[-1]) for g in mode_g], lag_ratio=0.2), run_time=1.0)

        # ---------------- r6: chain the layers with S-matrices (stable)
        chain = MathTex("S", "=", "S_1", "\\star", "S_2", "\\star", "S_3", font_size=46).move_to([2.7, 0.95, 0])
        for i in (2, 4, 6):
            chain[i].set_color(YELLOW)
        tx_ = xR + 0.85
        spots = [(y_mid, y_mid - 0.5), (0.5 * (y_top + y_mid), 0.5 * (y_top + y_mid)), (y_top, y_top + 0.5)]
        tags = VGroup()
        for (yy, yl), nm in zip(spots, ("S_1", "S_2", "S_3")):
            tags.add(Line([xR + 0.05, yy, 0], [tx_ - 0.25, yl, 0]).set_stroke(YELLOW, 1.5),
                     mt(nm, 28, YELLOW).move_to([tx_, yl, 0]))
        stab = VGroup(tx("scattering matrices hold only $e^{\\,i k_z d}$,\\\\ never larger than 1: stable", 28, GREEN),
                      tx("(transfer matrices also need $e^{+|k_z| d}$, which overflows)", 24, C_EV))
        stab.arrange(DOWN, buff=0.25).move_to([3.3, -0.75, 0])
        steps = tx("advance each mode $\\to$ match fields at interfaces $\\to$ chain", 28, GREY_A).move_to([2.75, 1.75, 0])
        self.at(S("r6"))
        self.play(FadeOut(VGroup(cells, mbox, m_bot, eig_ar, eig_l, mode_g, modes_l, hl_gr)), run_time=0.7)
        self.play(FadeIn(steps), run_time=0.6)
        self.at(WT("r6", "chain") - 0.2)
        self.play(FadeIn(tags), Write(chain), run_time=1.2)
        self.at(WT("r6", "never") - 0.2)
        self.play(FadeIn(stab[0]), run_time=0.8)
        self.play(FadeIn(stab[1]), run_time=0.6)
        self.finish()

# ============================================================ 5  on a computer ======
def lens_cells(fills, ph, n_cells=25, f_lam=8.0, per_lam=0.45):
    """Metalens from the computed library: for each cell pick the ridge width whose phase (mod 2 pi) best matches
    the hyperbolic lens phase  -k0 (sqrt(x^2 + f^2) - f)."""
    xs = (np.arange(n_cells) - (n_cells - 1) / 2) * per_lam
    target = (-TAU * (np.sqrt(xs ** 2 + f_lam ** 2) - f_lam)) % TAU
    lib = np.mod(ph, TAU)
    out = []
    for t in target:
        d = np.abs(np.angle(np.exp(1j * (lib - t))))
        out.append(fills[int(np.argmin(d))])
    return xs, np.array(out), target


class Ch5Computer(Chapter):
    CH = 5
    TAG = "5\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        rc = load("rcwa.npz")
        # ---------------- c1: the code (grating equation + RCWA for one layer)
        code = [("kx = kx_in + m*K", "# the orders: grating equation"),
                ("E = toeplitz(fourier(eps))", "# 1  expand eps"),
                ("A = k0**2*E - diag(kx**2)", "# 2  build the matrix"),
                ("kz2, W = eig(A)", "# 3  Bloch modes and kz**2"),
                ("S = star(S, layer(W, kz2, d))", "# 4  chain the layers")]
        cx0, cx1, cy0, dy = -6.05, -0.75, 2.45, 0.42
        rows = VGroup()
        for i, (a, b) in enumerate(code):
            ta = Text(a, font=MONO, font_size=21, color=WHITE if i else GREY_A)
            tb = Text(b, font=MONO, font_size=21, color=GREY_B)
            y = cy0 - i * dy - (0.18 if i else 0.0)
            ta.move_to([cx0, y, 0], aligned_edge=LEFT)
            tb.move_to([cx1, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(ta, tb))
        box = SurroundingRectangle(rows, buff=0.3, corner_radius=0.12, color=GREY_B, stroke_width=1.5).set_fill(GREY_E, 0.6)
        sep = Line([cx0, cy0 - 0.27, 0], [box.get_right()[0] - 0.3, cy0 - 0.27, 0]).set_stroke(GREY_D, 1)
        self.at(S("c1"))
        self.play(FadeIn(box), FadeIn(sep), FadeIn(rows[0]), run_time=0.8)
        for i, w in zip((1, 2, 3, 4), ("expand", "build", "diagonalize", "chain")):
            self.at(WT("c1", w) - 0.25)
            self.play(FadeIn(rows[i], shift=0.1 * RIGHT), run_time=0.45)

        # ---------------- c2: truncation -> convergence (computed)
        Ms, err = rc["Ms"], rc["err"]
        GX0, GY0, GW, GH = 1.75, -2.4, 4.4, 2.05
        lx = np.log10(Ms)
        ly = np.log10(err)
        xl, yl = (0.3, 2.0), (-6.0, -1.0)

        def G(xv, yv):
            return np.array([GX0 + (xv - xl[0]) / (xl[1] - xl[0]) * GW, GY0 + (yv - yl[0]) / (yl[1] - yl[0]) * GH, 0])
        gax = VGroup(Line(G(xl[0], yl[0]), G(xl[1], yl[0])).set_stroke(GREY_B, 2),
                     Line(G(xl[0], yl[0]), G(xl[0], yl[1])).set_stroke(GREY_B, 2))
        for xv, lab in ((np.log10(3), "3"), (1, "10"), (np.log10(81), "81")):
            gax.add(Line(G(xv, yl[0]) + 0.06 * DOWN, G(xv, yl[0]) + 0.06 * UP).set_stroke(GREY_B, 2),
                    mt(lab, 22, GREY_B).next_to(G(xv, yl[0]), DOWN, buff=0.1))
        for yv, lab in ((-2, "10^{-2}"), (-4, "10^{-4}"), (-6, "10^{-6}")):
            gax.add(Line(G(xl[0], yv) + 0.06 * LEFT, G(xl[0], yv) + 0.06 * RIGHT).set_stroke(GREY_B, 2),
                    mt(lab, 22, GREY_B).next_to(G(xl[0], yv), LEFT, buff=0.1))
        gx_l = tx("number of orders kept", 24, GREY_A).next_to(G(0.5 * (xl[0] + xl[1]), yl[0]), DOWN, buff=0.42)
        gy_l = tx("error of the result", 24, GREY_A).next_to(G(xl[0], yl[1]), UP, buff=0.12).align_to(G(xl[0], 0), LEFT)
        conv = poly_line([G(a, b)[0] for a, b in zip(lx, ly)], [G(a, b)[1] for a, b in zip(lx, ly)], GREEN, 3.5)
        conv_l = tx("computed (TE)", 24, GREEN).move_to(G(1.55, -2.05))
        trunc = tx("pitfall: truncation", 34, YELLOW).move_to([-3.4, -0.35, 0])
        keep = tx("keep enough orders, check convergence", 28, GREY_A).move_to([-3.4, -0.95, 0])
        tm = tx("TM polarization: Li's factorization rules", 28, GREY_A).move_to([-3.4, -1.75, 0])
        self.at(S("c2"))
        self.play(FadeIn(trunc), FadeIn(gax), FadeIn(gx_l), FadeIn(gy_l), run_time=0.9)
        self.at(WT("c2", "keep") - 0.2)
        self.play(FadeIn(keep), run_time=0.6)
        self.at(WT("c2", "converges") - 0.6)
        self.play(Create(conv), FadeIn(conv_l), run_time=1.4)
        self.at(WT("c2", "polarization") - 0.3)
        self.play(FadeIn(tm, shift=0.1 * UP), run_time=0.7)

        # ---------------- c3: the unit-cell library (computed with RCWA)
        fills, ph, T0 = rc["fills"], rc["ph"], rc["T0"]
        LX0, LY0, LW, LH = -5.6, -2.4, 4.4, 3.0
        pmax = 7.5

        def L(fv, pv):
            return np.array([LX0 + fv * LW, LY0 + pv / pmax * LH, 0])
        lax = VGroup(Line(L(0, 0), L(1.0, 0)).set_stroke(GREY_B, 2), Line(L(0, 0), L(0, pmax)).set_stroke(GREY_B, 2))
        for fv in (0.2, 0.5, 0.8):
            lax.add(Line(L(fv, 0) + 0.06 * DOWN, L(fv, 0) + 0.06 * UP).set_stroke(GREY_B, 2),
                    mt(f"{fv:.1f}", 22, GREY_B).next_to(L(fv, 0), DOWN, buff=0.1))
        for pv, lab in ((np.pi, "\\pi"), (TAU, "2\\pi")):
            lax.add(Line(L(0, pv) + 0.06 * LEFT, L(0, pv) + 0.06 * RIGHT).set_stroke(GREY_B, 2),
                    mt(lab, 24, GREY_B).next_to(L(0, pv), LEFT, buff=0.1),
                    fast_dashed(L(0, pv), L(1.0, pv), 0.06, GREY_D, 1))
        lx_l = tx("ridge width / period", 24, GREY_A).next_to(L(0.5, 0), DOWN, buff=0.42)
        ly_l = tx("phase of the light passing straight through", 24, GREY_A).next_to(L(0, pmax), UP, buff=0.14)
        ly_l.align_to(L(0, 0) + 0.3 * LEFT, LEFT)
        f_t = ValueTracker(fills[0])

        def lib_curve():
            f = f_t.get_value()
            k = int(np.searchsorted(fills, f)) + 1
            g = VGroup()
            if k >= 2:
                g.add(poly_line([L(a, b)[0] for a, b in zip(fills[:k], ph[:k])],
                                [L(a, b)[1] for a, b in zip(fills[:k], ph[:k])], YELLOW, 4))
                g.add(Dot(L(f, np.interp(f, fills, ph)), 0.07, color=YELLOW))
            return g
        lc = always_redraw(lib_curve)
        UX, UY, UPW = -0.2, -0.15, 1.15            # one unit cell: glass + ridge of the current width

        def cell_icon():
            f = f_t.get_value()
            return VGroup(Rectangle(width=UPW, height=0.3).move_to([UX, UY - 0.5, 0]).set_stroke(width=0).set_fill("#2a3036", 1),
                          Rectangle(width=max(f * UPW, 0.02), height=0.7).move_to([UX, UY, 0]).set_stroke(width=0)
                          .set_fill(GREY_B, 1),
                          DashedVMobject(Rectangle(width=UPW, height=1.0).move_to([UX, UY - 0.15, 0]), num_dashes=22)
                          .set_stroke(GREY_C, 1.2))
        ci = always_redraw(cell_icon)
        ci_l = tx("one cell", 22, GREY_B).move_to([UX, UY + 0.6, 0])
        self.at(S("c3") - 0.1)
        self.add(lc, ci)
        self.play(FadeOut(VGroup(box, sep, rows, gax, gx_l, gy_l, conv, conv_l, trunc, keep, tm)),
                  FadeIn(lax), FadeIn(lx_l), FadeIn(ly_l), FadeIn(ci_l), run_time=0.8)
        self.at(WT("c3", "sweep"))
        self.play(f_t.animate.set_value(fills[-1]), run_time=4.4, rate_func=linear)

        # ---------------- c4: pick a width per phase -> a flat lens (cells looked up in the computed library)
        lib_l = tx("unit-cell library (computed with RCWA;\\\\transmission 77--100\\%)", 26, YELLOW).move_to([-3.4, 2.15, 0])
        xs_c, fw, tgt = lens_cells(fills, ph)
        n_c = len(xs_c)
        CX0, CX1, CY = 1.1, 6.5, -1.85
        cw = (CX1 - CX0) / n_c
        lens = VGroup(Rectangle(width=CX1 - CX0, height=0.25).move_to([0.5 * (CX0 + CX1), CY - 0.12, 0])
                      .set_stroke(width=0).set_fill("#2a3036", 1))
        ridges = VGroup()
        for k in range(n_c):
            ridges.add(Rectangle(width=max(fw[k] * cw * 0.98, 0.012), height=0.62)
                       .move_to([CX0 + (k + 0.5) * cw, CY + 0.31, 0]).set_stroke(width=0).set_fill(GREY_B, 1))
        focus = np.array([0.5 * (CX0 + CX1), 1.9, 0])
        fronts = VGroup()
        for rr in (0.85, 1.55, 2.25, 2.95):
            ang = np.arcsin(min(0.98, (CX1 - CX0) / 2 / rr))
            fronts.add(Arc(radius=rr, start_angle=-PI / 2 - ang, angle=2 * ang, arc_center=focus).set_stroke(C_U, 3))
        fdot = Dot(focus, 0.08, color=WHITE)
        lens_l = tx("a flat lens, cell by cell", 30, WHITE).move_to([0.5 * (CX0 + CX1), CY - 0.62, 0])
        sch = tx("(wavefronts: schematic)", 22, GREY_B).move_to([CX1 - 1.25, 2.75, 0])
        self.at(S("c4"))
        self.play(FadeIn(lib_l), run_time=0.6)
        self.at(WT("c4", "pick") - 0.2)
        self.play(FadeIn(lens), LaggedStart(*[GrowFromEdge(r_, DOWN) for r_ in ridges], lag_ratio=0.06), run_time=1.6)
        self.at(WT("c4", "flat") - 0.4)
        self.play(LaggedStart(*[Create(f_) for f_ in fronts[::-1]], lag_ratio=0.2), FadeIn(fdot), FadeIn(lens_l),
                  FadeIn(sch), run_time=1.4)
        self.finish()


# ============================================================ 6  recap ==============
class Ch6Recap(Chapter):
    CH = 6
    TAG = "6\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        top = tx("symmetry fixes momentum", 44, WHITE).move_to([0, 2.55, 0])
        R = 0.95
        # icon 1: flat interface -> one k_x line through two circles
        c1 = np.array([-4.4, -0.35, 0])
        i1 = VGroup(Line(c1 + 1.45 * LEFT, c1 + 1.45 * RIGHT).set_stroke(GREY_B, 2),
                    half_circle(c1, R, GREY_A, 2.5), half_circle(c1, 1.5 * R, "#8fb3c9", 2.5),
                    fast_dashed(c1 + 0.55 * RIGHT, c1 + 0.55 * RIGHT + 1.65 * UP, 0.06, C_KX, 3))
        l1 = MathTex("\\text{flat: }", "k_x", "\\text{ conserved}", font_size=34).next_to(c1, DOWN, buff=0.35)
        l1[1].set_color(C_KX)
        # icon 2: periodic -> copies spaced by K
        c2 = np.array([0.0, -0.35, 0])
        i2 = VGroup(Line(c2 + 1.6 * LEFT, c2 + 1.6 * RIGHT).set_stroke(GREY_B, 2), half_circle(c2, 1.2 * R, GREY_A, 2.5),
                    fast_dashed(c2, c2 + 1.5 * UP, 0.06, C_KX, 2.5))
        i2c = VGroup(*[fast_dashed(c2 + 0.42 * m * RIGHT, c2 + 0.42 * m * RIGHT + 1.5 * UP, 0.06, C_K, 2.5)
                       for m in (1, -1, 2, -2, 3, -3)])
        l2 = MathTex("\\text{periodic: }", "k_x", "+", "mK", font_size=34).next_to(c2, DOWN, buff=0.35)
        l2[1].set_color(C_KX)
        l2[3].set_color(C_K)
        # icon 3: band gap where copies meet (computed bands, dn = 0.9)
        st = STACK
        c3 = np.array([4.4, -0.35, 0])
        hv = st["half"][-1][::3]
        nus = st["nus"][::3]
        ok = np.abs(hv) <= 1
        xk = np.arccos(np.clip(hv, -1, 1)) / TAU
        i3 = VGroup(Line(c3 + 1.3 * LEFT, c3 + 1.3 * RIGHT).set_stroke(GREY_B, 2),
                    Line(c3, c3 + 1.75 * UP).set_stroke(GREY_B, 2))
        idx = np.where(ok)[0]
        for r_ in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            for sg in (1, -1):
                i3.add(poly_line(c3[0] + sg * xk[r_] * 2.3, c3[1] + nus[r_] * 4.0, C_KZ, 3.5))
        gap = st["gap"][-1]
        gap_r = Rectangle(width=1.15, height=(gap[1] - gap[0]) * 4.0).move_to(c3 + np.array([0, 0.5 * (gap[0] + gap[1]) * 4.0, 0]))
        gap_r.set_stroke(width=0).set_fill(C_EV, 0.35)
        l3 = tx("copies meet: gaps open", 30, WHITE).next_to(c3, DOWN, buff=0.35)
        self.at(S("e1"))
        self.play(FadeIn(top, shift=0.1 * DOWN), run_time=0.8)
        self.at(WT("e1", "flat") - 0.2)
        self.play(FadeIn(i1), FadeIn(l1), run_time=0.8)
        self.at(WT("e1", "periodic") - 0.2)
        self.play(FadeIn(i2), FadeIn(l2), run_time=0.8)
        self.at(WT("e1", "multiples") - 0.2)
        self.play(LaggedStart(*[FadeIn(c_, shift=0.15 * UP) for c_ in i2c], lag_ratio=0.3), run_time=1.8)
        self.at(WT("e1", "where") - 0.2)
        self.play(FadeIn(i3), FadeIn(l3), run_time=0.8)
        self.at(WT("e1", "gaps") - 0.1)
        self.play(GrowFromCenter(gap_r), run_time=0.6)

        # ---------------- e2: where it breaks: periodicity is only borrowed locally
        rc = load("rcwa.npz")
        xs_c, fw, _ = lens_cells(rc["fills"], rc["ph"])
        n_c = len(xs_c)
        CX0, CX1, CY = -5.0, 5.0, -0.6
        cw = (CX1 - CX0) / n_c
        base = Rectangle(width=CX1 - CX0, height=0.3).move_to([0, CY - 0.15, 0]).set_stroke(width=0).set_fill("#2a3036", 1)
        ridges = VGroup(*[Rectangle(width=max(fw[k] * cw * 0.98, 0.015), height=0.75).move_to([CX0 + (k + 0.5) * cw, CY + 0.375, 0])
                          .set_stroke(width=0).set_fill(GREY_B, 1) for k in range(n_c)])
        k0 = 17
        hl = SurroundingRectangle(ridges[k0], buff=0.08, color=YELLOW, stroke_width=3)
        ghost = VGroup()
        for d in (-3, -2, -1, 1, 2, 3):
            xg = CX0 + (k0 + d + 0.5) * cw
            ghost.add(DashedVMobject(Rectangle(width=fw[k0] * cw * 0.98, height=0.75).move_to([xg, CY + 1.35, 0]),
                                     num_dashes=10).set_stroke(YELLOW, 1.5))
        ghost.add(DashedVMobject(Rectangle(width=fw[k0] * cw * 0.98, height=0.75).move_to([CX0 + (k0 + 0.5) * cw, CY + 1.35, 0]),
                                 num_dashes=10).set_stroke(YELLOW, 2.5))
        g_l = tx("RCWA pretends each cell has identical neighbours", 28, YELLOW).move_to([0, CY + 2.35, 0])
        brk = tx("but a lens changes cell by cell: the local periodic approximation", 28, GREY_A).move_to([0, CY - 0.75, 0])
        self.at(S("e2"))
        self.play(FadeOut(VGroup(top, i1, l1, i2, i2c, l2, i3, l3, gap_r)), run_time=0.6)
        self.play(FadeIn(base), FadeIn(ridges), run_time=0.8)
        self.at(WT("e2", "periodicity") - 0.3)
        self.play(Create(hl), FadeIn(ghost), FadeIn(g_l), run_time=1.0)
        self.at(WT("e2", "borrows") - 0.3)
        self.play(FadeIn(brk), run_time=0.8)

        # ---------------- e3: teaser -> trapping light
        tir = VGroup(Rectangle(width=3.2, height=0.7).set_stroke(GREY_B, 2).set_fill("#2a3036", 1))
        zz = np.array([[-1.5, -0.33], [-0.9, 0.33], [-0.3, -0.33], [0.3, 0.33], [0.9, -0.33], [1.5, 0.33]])
        tir.add(VMobject().set_points_as_corners([[a, b, 0] for a, b in zz]).set_stroke(C_U, 3))
        tir.move_to([-3.4, 0.55, 0])
        tir_l = tx("TIR traps light in a slab", 26, GREY_A).next_to(tir, DOWN, buff=0.2)
        mir = VGroup()
        for xm in (-1.3, 1.3):
            for k in range(8):
                mir.add(Rectangle(width=0.07, height=1.1).move_to([xm + (k - 3.5) * 0.09 * np.sign(xm), 0, 0])
                        .set_stroke(width=0).set_fill(GREY_B if k % 2 == 0 else GREY_D, 1))
        mir.add(graph_line(lambda x: 0.32 * np.sin(np.pi * (x + 0.95) / 1.9 * 3), -0.95, 0.95, 0, 1, 1, color=C_U, width=3))
        mir.move_to([3.4, 0.55, 0])
        mir_l = tx("Bragg mirrors trap it between them", 26, GREY_A).next_to(mir, DOWN, buff=0.2)
        nxt = next_card("trapping light: waveguides, modes and resonators", y=-1.95)
        self.at(S("e3"))
        self.play(FadeOut(VGroup(base, ridges, hl, ghost, g_l, brk)), run_time=0.6)
        self.play(FadeIn(tir), FadeIn(tir_l), run_time=0.8)
        self.at(WT("e3", "bragg") - 0.2)
        self.play(FadeIn(mir), FadeIn(mir_l), run_time=0.8)
        self.at(WT("e3", "how") - 0.3)
        self.play(FadeIn(nxt, shift=0.1 * UP), run_time=0.9)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Boundary", "Ch2Grating", "Ch3Bands", "Ch4RCWA", "Ch5Computer", "Ch6Recap"]
