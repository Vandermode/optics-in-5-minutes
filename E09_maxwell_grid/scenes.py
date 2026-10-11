"""Maxwell on a Grid: FDTD and the perfectly matched layer -- chapter scenes (Optics in 5 Minutes, E09).
Render: ./render.sh ; preview: ./preview.sh Ch0Hook

Every field movie is computed by physics.py (2D/1D Yee FDTD with a stretched-coordinate PML) and only loaded here.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403
from scipy.interpolate import PchipInterpolator  # noqa: E402


def _at_whole_frames(self, t):
    """Whole-frame sync (FAST_ANIMATION_PIPELINE.md section 10): wait exactly k frames on every clock, so a target
    lying on the 1/60 s grid cannot shift a play by one frame at a render seam (common/'s at() is the pre-fix one)."""
    x = (t - self.now) * config.frame_rate                # frames still to go
    if x > 0.5 + 1e-6:
        k = int(np.ceil(x - 1e-6))
        self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
    elif x < -0.5 - 1e-6:
        self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


Chapter.at = _at_whole_frames

C_E, C_HF = C_U, GREEN_C             # electric field (series "field" blue), magnetic field (green)
C_GRID = ORANGE                       # the grid's own (lattice) dispersion
DIV_RGBA = np.concatenate([DIVERGE[np.round(np.linspace(0, 1023, 256)).astype(int)],
                           np.full((256, 1), 255, np.uint8)], 1).astype(np.uint8)
_CACHE = {}


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb[:3])


C_POS, C_NEG = hexc(DIVERGE[1000]), hexc(DIVERGE[30])      # colours of +E (out of the screen) and -E


def load(name):
    if name not in _CACHE:
        _CACHE[name] = dict(np.load(data_path(name)))
    return _CACHE[name]


def warp(ts, vs):
    """Smooth monotone map film time -> simulation time through the keyframes (PCHIP): a pure function of t."""
    f = PchipInterpolator(np.asarray(ts, float), np.asarray(vs, float))
    a, b = ts[0], ts[-1]
    return lambda t: float(f(min(max(t, a), b)))


def movie_rgba(F, tf, tau, lut=DIV_RGBA):
    """Frame stack F (uint8, 128 = 0) sampled at times tf -> RGBA at simulation time tau (linear in time)."""
    k = int(np.clip(np.searchsorted(tf, tau) - 1, 0, len(tf) - 2))
    w = float(np.clip((tau - tf[k]) / (tf[k + 1] - tf[k]), 0.0, 1.0))
    if w < 0.02:
        v = F[k]
    elif w > 0.98:
        v = F[k + 1]
    else:
        v = (F[k] * (1.0 - w) + F[k + 1] * w + 0.5).astype(np.uint8)
    return lut[v]


def mtex(s, size=36, color=WHITE, **kw):
    """MathTex with the episode colour code for E and H."""
    m = MathTex(s, font_size=size, color=color, **kw)
    return m


def eh_tex(s, size=36, **kw):
    """MathTex where \\mathbf{E} is blue and \\mathbf{H} green."""
    return MathTex(s, font_size=size, tex_to_color_map={"\\mathbf{E}": C_E, "\\mathbf{H}": C_HF}, **kw)


def label_box(mob, color=GREY_B, buff=0.18, fill=0.75):
    return VGroup(SurroundingRectangle(mob, buff=buff, corner_radius=0.1, color=color, stroke_width=1.5).set_fill(BLACK, fill), mob)


def maxwell_pair(size=40):
    e1 = eh_tex("{\\partial \\mathbf{H} \\over \\partial t}", size=size)
    e1 = VGroup(e1, eh_tex("= -\\,{1 \\over \\mu}\\,\\nabla\\times\\mathbf{E}", size=size)).arrange(RIGHT, buff=0.15)
    e2 = eh_tex("{\\partial \\mathbf{E} \\over \\partial t}", size=size)
    e2 = VGroup(e2, eh_tex("= {1 \\over \\varepsilon}\\,\\nabla\\times\\mathbf{H}", size=size)).arrange(RIGHT, buff=0.15)
    return e1, e2


def code_card_ind(lines, font_size=24, color=GREY_A, title=None):
    """Like o5common.code_card, but keeps the leading indentation of each line (Text drops leading spaces)."""
    cw = Text("M" * 20, font=MONO, font_size=font_size).width / 20
    rows = VGroup(*[Text(l.strip() if l.strip() else " ", font=MONO, font_size=font_size, color=color) for l in lines])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    for t_, l_ in zip(rows, lines):
        t_.shift((len(l_) - len(l_.lstrip(" "))) * cw * RIGHT)
    g = VGroup(rows)
    if title:
        t = Text(title, font="Lato", font_size=font_size - 2, color=YELLOW, weight=BOLD)
        t.next_to(rows, UP, aligned_edge=LEFT, buff=0.25)
        g.add(t)
    box = SurroundingRectangle(g, buff=0.3, corner_radius=0.12, color=GREY_B, stroke_width=1.5).set_fill(GREY_E, 0.6)
    return VGroup(box, g)


def yee_icon(c, s=0.42, nx=3, ny=2, on_e=1.0, on_h=1.0):
    """Mini Yee lattice: E dots at nodes, H arrows on the edge midpoints (for badges and the recap)."""
    g = VGroup()
    o = np.array(c) - np.array([(nx - 1) * s / 2, (ny - 1) * s / 2, 0])
    for i in range(nx):
        for j in range(ny):
            p = o + np.array([i * s, j * s, 0])
            if i < nx - 1:
                q = p + np.array([s / 2, 0, 0])
                g.add(fast_arrow(q + 0.13 * s * DOWN, q + 0.13 * s * UP, C_HF, 3, 0.09, on_h))
            if j < ny - 1:
                q = p + np.array([0, s / 2, 0])
                g.add(fast_arrow(q + 0.13 * s * LEFT, q + 0.13 * s * RIGHT, C_HF, 3, 0.09, on_h))
    for i in range(nx):
        for j in range(ny):
            g.add(Dot(o + np.array([i * s, j * s, 0]), radius=0.055, color=C_E).set_opacity(on_e))
    return g


def ribbon_icon(c, w=1.6, h=0.9):
    """Mini picture of the stretched axis: flat, then bending up into 'Im x', with a decaying wave on it."""
    c = np.array(c)
    xs = np.linspace(-w / 2, w / 2, 160)
    bend = np.clip((xs - 0.1 * w) / (0.4 * w), 0, None) ** 2
    path = [c + np.array([x, -h / 2 + 0.9 * h * b, 0]) for x, b in zip(xs, bend)]
    env = np.exp(-3.2 * bend)
    wave = [p + np.array([0, 0.16 * np.cos(14 * x) * e, 0]) for p, x, e in zip(path, xs, env)]
    return VGroup(VMobject().set_points_as_corners(path).set_stroke(C_IM, 3),
                  VMobject().set_points_as_corners(wave).set_stroke(C_E, 3))


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        d = load("hook.npz")
        F, tf = d["frames"], d["t_frame"]
        X0, X1, Y0, Y1 = [float(v) for v in d["extent"]]
        W = 11.6
        Hh = W * (Y1 - Y0) / (X1 - X0)
        C = np.array([0.0, 0.33, 0.0])
        sc = W / (X1 - X0)
        P = lambda x, y: C + np.array([(x - 0.5 * (X0 + X1)) * sc, (y - 0.5 * (Y0 + Y1)) * sc, 0.0])
        tau = warp([0.0, 0.9, S("h1") + 5.0, S("h2"), S("h2") + 2.0, S("h3"), self.t_end + 0.1],
                   [38.0, 40.0, 52.2, 55.0, 61.0, 80.0, 123.5])
        img, fade = self.live_image(lambda t: movie_rgba(F, tf, tau(t)), W, C, Hh)
        R, w, yb = float(d["R"]), float(d["w"]), float(d["yb"])
        ring = Annulus(inner_radius=(R - w / 2) * sc, outer_radius=(R + w / 2) * sc).move_to(P(0, 0))
        ring.set_fill(GREY_B, 0.0).set_stroke(GREY_B, 1.3, 0.55)
        bus = Rectangle(width=W, height=w * sc).move_to(P(0, yb)).set_fill(GREY_B, 0.0).set_stroke(GREY_B, 1.3, 0.55)
        box = Rectangle(width=W, height=Hh).move_to(C).set_stroke(GREY_D, 1.2)
        geo = VGroup(bus, ring)
        lab_w = Tex("silicon wire", font_size=26, color=GREY_A).move_to(P(-2.75, yb - 0.42))
        lab_r = Tex("ring", font_size=26, color=GREY_A).move_to(P(1.62, 1.08))
        cap = Tex("computed: 2D FDTD", font_size=22, color=GREY_B).move_to(P(2.75, -2.05))
        self.add(img)
        self.play(fade.animate.set_value(1.0), FadeIn(geo), FadeIn(box), run_time=0.8)
        self.at(S("h1") + 0.6)
        self.play(FadeIn(lab_w), FadeIn(cap), run_time=0.6)
        self.at(S("h1") + 4.4)
        self.play(FadeIn(lab_r), run_time=0.5)

        # ---- h3: Maxwell's equations themselves, on a grid
        grid = VGroup()
        gs = 0.25 * sc
        for k in range(int(W / gs) + 1):
            x = C[0] - W / 2 + k * gs
            grid.add(Line([x, C[1] - Hh / 2, 0], [x, C[1] + Hh / 2, 0]))
        for k in range(int(Hh / gs) + 1):
            y = C[1] - Hh / 2 + k * gs
            grid.add(Line([C[0] - W / 2, y, 0], [C[0] + W / 2, y, 0]))
        grid.set_stroke(GREY_C, 0.8, 0.35)
        e1, e2 = maxwell_pair(42)
        eqs = VGroup(e1, e2).arrange(RIGHT, buff=1.0)
        ttl = Tex("Maxwell's equations", font_size=30, color=GREY_A)
        eqg = label_box(VGroup(ttl, eqs).arrange(DOWN, buff=0.25), buff=0.25, fill=0.82).move_to([0, 1.85, 0])
        self.at(S("h3") + 0.2)
        self.play(fade.animate.set_value(0.55), FadeOut(lab_w), FadeOut(lab_r), FadeOut(cap), run_time=0.6)
        self.at(S("h3") + 2.6)
        self.play(FadeIn(eqg, shift=0.1 * DOWN), run_time=0.9)
        self.at(S("h3") + 4.3)
        self.play(Create(grid), run_time=1.4, lag_ratio=0.02)
        q = Tex("?", font_size=72, color=YELLOW).next_to(eqg, RIGHT, buff=0.3)
        self.play(FadeIn(q, scale=1.3), run_time=0.4)

        # ---- h4: title + the two ideas
        card = series_title(9, "Maxwell on a Grid", "FDTD and the perfectly matched layer", y=0.95)
        b1 = VGroup(yee_icon([0, 0, 0], 0.5, 4, 2), Tex("leapfrogging fields", font_size=30, color=GREY_A)).arrange(DOWN, buff=0.22)
        b2 = VGroup(ribbon_icon([0, 0, 0], 1.7, 0.8), Tex("a border of complex numbers", font_size=30, color=GREY_A)).arrange(DOWN, buff=0.22)
        badges = VGroup(*[label_box(b, buff=0.22, fill=0.8) for b in (b1, b2)]).arrange(RIGHT, buff=1.2).move_to([0, -1.75, 0])
        self.at(S("h4"))
        self.play(FadeOut(eqg), FadeOut(q), FadeOut(grid), fade.animate.set_value(0.32), geo.animate.set_opacity(0.05),
                  box.animate.set_stroke(opacity=0.4), run_time=0.6)
        self.play(FadeIn(card[0]), Write(card[1]), run_time=1.4)
        self.at(S("h4") + 4.6)
        self.play(FadeIn(badges[0], shift=0.15 * UP), run_time=0.7)
        self.at(S("h4") + 6.4)
        self.play(FadeIn(badges[1], shift=0.15 * UP), run_time=0.7)
        self.finish()


# ============================================================ 1  Maxwell ===========
class Ch1Maxwell(Chapter):
    CH = 1
    TAG = "1\\quad Maxwell's two rules"

    def construct(self):
        S = self.S
        self.show_tag()
        d = load("bump.npz")
        F, tf, hx, hy, pts = d["frames"], d["t_frame"], d["hx"], d["hy"], d["pts"]
        L = float(d["L"])
        PW = 5.6
        PC = np.array([3.55, -0.12, 0])
        sc = PW / L
        tau = warp([S("m4"), S("m4") + 3.6, S("m5") + 7.8, self.t_end], [0.0, 0.32, 3.35, 4.55])
        img, fade = self.live_image(lambda t: movie_rgba(F, tf, tau(t)), PW, PC, PW)
        frame = Square(side_length=PW).move_to(PC).set_stroke(GREY_B, 1.5)
        xl = MathTex("x", font_size=28, color=GREY_B).next_to(frame, DOWN, buff=0.08).align_to(frame, RIGHT).shift(0.1 * LEFT)
        yl = MathTex("y", font_size=28, color=GREY_B).next_to(frame, LEFT, buff=0.08).align_to(frame, UP).shift(0.1 * DOWN)
        key = VGroup(Square(0.22).set_fill(C_POS, 1).set_stroke(width=0), Tex("$E$ out of the screen", font_size=24),
                     Square(0.22).set_fill(C_NEG, 1).set_stroke(width=0), Tex("into the screen", font_size=24),
                     fast_arrow([0, 0, 0], [0.45, 0, 0], C_HF, 4, 0.14), Tex("$H$", font_size=26, color=C_HF))
        key.arrange(RIGHT, buff=0.12)
        key[2].shift(0.18 * RIGHT)
        key[3].shift(0.18 * RIGHT)
        key[4:].shift(0.36 * RIGHT)
        key.next_to(frame, UP, buff=0.14)
        key[1].set_color(GREY_A)
        key[3].set_color(GREY_A)

        # ---- m1: a 3D sketch: E out of the plane, H in the plane
        o = np.array([-3.7, 0.55, 0])
        u, v = np.array([1.9, 0, 0]), np.array([0.85, 0.7, 0])
        plane = Polygon(o - u - v, o + u - v, o + u + v, o - u + v).set_fill(GREY_E, 0.8).set_stroke(GREY_B, 1.5)
        e_arrow = Arrow(o, o + 1.9 * UP, buff=0, stroke_width=6, color=C_E, max_tip_length_to_length_ratio=0.12)
        e_lab = MathTex("\\mathbf{E}", font_size=40, color=C_E).next_to(e_arrow.get_end(), RIGHT, buff=0.12)
        h_arrows = VGroup(*[Arrow(o + 0.9 * np.array([np.cos(a), 0.45 * np.sin(a), 0]),
                                  o + 0.9 * np.array([np.cos(a + 0.9), 0.45 * np.sin(a + 0.9), 0]), buff=0, stroke_width=4,
                                  color=C_HF, max_tip_length_to_length_ratio=0.25) for a in np.linspace(0, TAU_, 5)[:-1] + 0.3])
        h_lab = MathTex("\\mathbf{H}", font_size=40, color=C_HF).move_to(o + np.array([1.55, -0.62, 0]))
        sk_lab = VGroup(Tex("$E$: straight out of the screen", font_size=28, color=GREY_A),
                        Tex("$H$: flat in the plane", font_size=28, color=GREY_A)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        sk_lab.move_to([-3.7, -1.55, 0])
        sk_lab[0][0][0:1].set_color(C_E)
        sk_lab[1][0][0:1].set_color(C_HF)
        sketch = VGroup(plane, h_arrows, e_arrow, e_lab, h_lab)
        self.at(S("m1"))
        self.add(img)
        self.play(FadeIn(plane), Create(frame), FadeIn(xl), FadeIn(yl), fade.animate.set_value(1.0), run_time=0.9)
        self.at(S("m1") + 1.9)
        self.play(GrowArrow(e_arrow), FadeIn(e_lab), FadeIn(sk_lab[0]), run_time=0.8)
        self.at(S("m1") + 4.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in h_arrows], lag_ratio=0.15), FadeIn(h_lab), FadeIn(sk_lab[1]), run_time=1.0)
        self.play(FadeIn(key), run_time=0.6)

        # ---- m2: the two curl equations
        e1, e2 = maxwell_pair(44)
        g1 = Tex("change of $H$ $\\leftarrow$ swirl of $E$", font_size=26, color=GREY_B)
        g2 = Tex("change of $E$ $\\leftarrow$ swirl of $H$", font_size=26, color=GREY_B)
        col = VGroup(VGroup(e1, g1).arrange(DOWN, buff=0.14), VGroup(e2, g2).arrange(DOWN, buff=0.14)).arrange(DOWN, buff=0.5)
        col.move_to([-3.55, 1.0, 0])
        self.at(S("m2"))
        self.play(FadeOut(sketch), FadeOut(sk_lab), run_time=0.5)
        self.at(S("m2") + 1.6)
        self.play(Write(e1), FadeIn(g1), run_time=1.3)
        self.at(S("m2") + 4.6)
        self.play(Write(e2), FadeIn(g2), run_time=1.3)

        # ---- m3: curl = swirl (a ring of arrows that turns, the curl symbol in its middle)
        vc = np.array([-5.3, -1.85, 0])
        vop = ValueTracker(0.0)

        def vortex():
            g = VGroup()
            o_ = vop.get_value()
            if o_ < 0.01:
                return g
            a0 = -0.6 * self.now
            for k in range(8):
                a = a0 + k * TAU_ / 8
                p = vc + 0.66 * np.array([np.cos(a), np.sin(a), 0])
                tdir = np.array([np.sin(a), -np.cos(a), 0])
                g.add(fast_arrow(p - 0.21 * tdir, p + 0.21 * tdir, C_HF, 3.5, 0.12, o_))
            return g
        vort = always_redraw(vortex)
        nab = MathTex("\\nabla\\times", font_size=40).move_to(vc)
        cl = VGroup(Tex("curl = swirl: how much", font_size=28, color=GREY_A),
                    Tex("a field circulates", font_size=28, color=GREY_A),
                    Tex("around each point", font_size=28, color=GREY_A)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        cl.next_to(vc, RIGHT, buff=1.05)
        self.at(S("m3"))
        self.add(vort)
        self.play(vop.animate.set_value(1.0), FadeIn(nab), FadeIn(cl), run_time=0.8)

        # ---- m4/m5: the bump: H starts to circulate, then E spreads into a ring
        hs = 0.30
        P_ = [PC + np.array([p[0] * sc, p[1] * sc, 0]) for p in pts]

        def quiver():
            g = VGroup()
            o_ = qop.get_value()
            if o_ < 0.01:
                return g
            tt = tau(self.now)
            k = int(np.clip(np.searchsorted(tf, tt) - 1, 0, len(tf) - 2))
            ww = float(np.clip((tt - tf[k]) / (tf[k + 1] - tf[k]), 0, 1))
            HX = hx[k] * (1 - ww) + hx[k + 1] * ww
            HY = hy[k] * (1 - ww) + hy[k + 1] * ww
            M = np.hypot(HX, HY)
            for n_, p in enumerate(P_):
                m = M[n_]
                if m < 0.006:
                    continue
                L_ = 0.44 * min(1.0, (m / hs) ** 0.5)
                dvec = np.array([HX[n_], HY[n_], 0]) / m
                g.add(fast_arrow(p - 0.5 * L_ * dvec, p + 0.5 * L_ * dvec, C_HF, 4, min(0.14, 0.5 * L_), o_))
            return g
        qop = ValueTracker(0.0)
        quiv = always_redraw(quiver)
        hl1 = SurroundingRectangle(col[0], buff=0.12, color=YELLOW, stroke_width=2.5)
        hl2 = SurroundingRectangle(col[1], buff=0.12, color=YELLOW, stroke_width=2.5)
        self.at(S("m4"))
        self.add(quiv)
        self.play(qop.animate.set_value(1.0), Create(hl1), run_time=0.6)
        ring_lab = VGroup(Tex("the bump spreads into a ring:", font_size=30, color=GREY_A),
                          Tex("a wave that pushes itself along", font_size=30, color=YELLOW)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        ring_lab.move_to([-3.55, -1.85, 0])
        self.at(S("m5"))
        self.play(ReplacementTransform(hl1, hl2), FadeOut(cl), FadeOut(nab), vop.animate.set_value(0.0), run_time=0.6)
        self.remove(vort)
        self.at(S("m5") + 4.4)
        self.play(FadeIn(ring_lab), run_time=0.6)

        # ---- m6: only neighbours
        sn = np.array([1.5, 1.0])
        cp = PC + np.array([sn[0] * sc, sn[1] * sc, 0])
        dd = 0.5
        nb = VGroup(*[Line(cp, cp + dd * np.array(v_ + (0,))).set_stroke(WHITE, 2.5) for v_ in ((1, 0), (-1, 0), (0, 1), (0, -1))],
                    *[Dot(cp + dd * np.array(v_ + (0,)), radius=0.075, color=WHITE) for v_ in ((1, 0), (-1, 0), (0, 1), (0, -1))],
                    Dot(cp, radius=0.1, color=YELLOW))
        loc = VGroup(Tex("only derivatives:", font_size=30, color=GREY_A), Tex("each point talks only", font_size=30),
                     Tex("to its neighbours", font_size=30)).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([-3.55, -1.85, 0])
        loc[1:].set_color(YELLOW)
        self.at(S("m6"))
        self.play(FadeOut(hl2), FadeOut(ring_lab), run_time=0.5)
        self.play(FadeIn(nb, scale=0.6), qop.animate.set_value(0.35), run_time=0.8)
        self.play(FadeIn(loc), run_time=0.7)
        self.finish()


# ============================================================ 2  Yee grid ==========
def stems_row(xs, vals, y0, scale, color, width=3.5, dot=0.045, opacity=1.0):
    g = VGroup(Line([xs[0] - 0.2, y0, 0], [xs[-1] + 0.2, y0, 0]).set_stroke(GREY_D, 1.5))
    for x, v in zip(xs, vals):
        if abs(v) * scale < 0.012:
            g.add(Dot([x, y0, 0], radius=0.022, color=color).set_opacity(0.6 * opacity))
            continue
        g.add(Line([x, y0, 0], [x, y0 + scale * v, 0]).set_stroke(color, width, opacity=opacity))
        g.add(Dot([x, y0 + scale * v, 0], radius=dot, color=color).set_opacity(opacity))
    return g


class Ch2Yee(Chapter):
    CH = 2
    TAG = "2\\quad The Yee grid and the leapfrog"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---- y1: sample, differences of neighbours
        xa, xb, yb0 = -6.4, 0.6, 0.35
        f = lambda x: 1.15 * np.sin(1.05 * (x + 3.0)) + 0.1
        df = lambda x: 1.15 * 1.05 * np.cos(1.05 * (x + 3.0))
        axis = Line([xa, yb0 - 1.45, 0], [xb, yb0 - 1.45, 0]).set_stroke(GREY_C, 1.5)
        curve = graph_line(f, xa, xb, yb0, 1, 1, color=C_E, width=4)
        dx = 1.0
        xs = np.arange(xa + 0.3, xb, dx)
        samp = VGroup(*[VGroup(DashedLine([x, yb0 - 1.45, 0], [x, yb0 + f(x), 0], dash_length=0.06).set_stroke(GREY_C, 1.2),
                               Dot([x, yb0 + f(x), 0], radius=0.07, color=C_E)) for x in xs])
        dlab = MathTex("\\Delta x", font_size=30, color=GREY_A)
        br = BraceBetweenPoints([xs[1], yb0 - 1.5, 0], [xs[2], yb0 - 1.5, 0], DOWN, color=GREY_B)
        dlab.next_to(br, DOWN, buff=0.06)
        fd = MathTex("{df \\over dx}", "\\approx", "{f_{i+1} - f_i \\over \\Delta x}", font_size=46).move_to([3.9, 1.7, 0])
        fd_l = Tex("a difference of neighbours", font_size=28, color=GREY_A).next_to(fd, DOWN, buff=0.2)
        self.at(S("y1"))
        self.play(Create(axis), Create(curve), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(sm) for sm in samp], lag_ratio=0.12), GrowFromCenter(br), FadeIn(dlab), run_time=1.3)
        self.at(S("y1") + 3.0)
        self.play(Write(fd), FadeIn(fd_l), run_time=1.3)

        # ---- y2: the secant matches the slope at the midpoint, not at the ends
        i0 = int(np.argmin(np.abs(xs - (-2.1))))
        x0_, x1_ = xs[i0], xs[i0 + 1]
        xm = 0.5 * (x0_ + x1_)
        p0, p1 = np.array([x0_, yb0 + f(x0_), 0]), np.array([x1_, yb0 + f(x1_), 0])
        sl = (p1[1] - p0[1]) / (x1_ - x0_)

        def tline(xc, slope, half, col, w=3.5):
            pc = np.array([xc, yb0 + f(xc), 0])
            dvec = np.array([1, slope, 0]) / np.hypot(1, slope)
            return Line(pc - half * dvec, pc + half * dvec).set_stroke(col, w)
        sec = Line(p0 + 1.2 * (p0 - p1) / np.linalg.norm(p1 - p0), p1 + 1.2 * (p1 - p0) / np.linalg.norm(p1 - p0)).set_stroke(YELLOW, 4)
        big = VGroup(Dot(p0, radius=0.1, color=YELLOW), Dot(p1, radius=0.1, color=YELLOW))
        t_end = tline(x0_, df(x0_), 1.1, C_EV, 4.5)
        t_mid = tline(xm, df(xm), 1.35, C_H, 5)
        mdot = Dot([xm, yb0 + f(xm), 0], radius=0.08, color=C_H)
        mtick = VGroup(DashedLine([xm, yb0 - 1.45, 0], [xm, yb0 + f(xm), 0], dash_length=0.05).set_stroke(C_H, 1.5),
                       MathTex("x_{i+\\frac12}", font_size=30, color=C_H).move_to([xm, yb0 - 1.8, 0]))
        l_sec = Tex("the difference", font_size=28, color=YELLOW)
        l_end = Tex("slope at an end: off", font_size=28, color=C_EV)
        l_mid = Tex("slope at the midpoint: matches", font_size=28, color=C_H)
        legend = VGroup(l_sec, l_end, l_mid).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([3.9, -0.65, 0])
        self.at(S("y2"))
        self.play(FadeIn(big), Create(sec), FadeIn(l_sec), run_time=1.0)
        self.at(S("y2") + 2.5)
        self.play(Create(t_end), FadeIn(l_end), run_time=0.8)
        self.at(S("y2") + 4.0)
        self.play(Create(t_mid), FadeIn(mdot), FadeIn(mtick), FadeIn(l_mid), run_time=0.9)

        # ---- y3: H at the midpoints (1D first), then the 2D Yee lattice
        hmarks = VGroup(*[VGroup(Line([x + dx / 2, yb0 - 1.45 - 0.13, 0], [x + dx / 2, yb0 - 1.45 + 0.13, 0]).set_stroke(C_HF, 4),
                                 MathTex("H", font_size=24, color=C_HF).move_to([x + dx / 2, yb0 - 1.13, 0]))
                          for x in xs[:-1]])
        emarks = VGroup(*[MathTex("E", font_size=24, color=C_E).move_to([x, yb0 - 1.13, 0]) for x in xs])
        self.at(S("y3") + 0.3)
        self.play(FadeOut(VGroup(sec, big, t_end, t_mid, mdot, mtick, legend, fd, fd_l)), curve.animate.set_stroke(opacity=0.35),
                  samp.animate.set_opacity(0.35), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(h, shift=0.1 * UP) for h in hmarks], lag_ratio=0.08), FadeIn(emarks), run_time=1.2)

        nxL, nyL, s_ = 7, 5, 0.95
        LC = np.array([-2.6, 0.05, 0])
        o = LC - np.array([(nxL - 1) * s_ / 2, (nyL - 1) * s_ / 2, 0])
        node = lambda i, j: o + np.array([i * s_, j * s_, 0])
        lines = VGroup(*[Line(node(i, 0), node(i, nyL - 1)) for i in range(nxL)],
                       *[Line(node(0, j), node(nxL - 1, j)) for j in range(nyL)]).set_stroke(GREY_D, 1.2)
        Edots = VGroup(*[Dot(node(i, j), radius=0.08, color=C_E) for i in range(nxL) for j in range(nyL)])
        Hy_ar = VGroup(*[Arrow(node(i, j) + np.array([s_ / 2, -0.17, 0]), node(i, j) + np.array([s_ / 2, 0.17, 0]), buff=0,
                               stroke_width=4, color=C_HF, max_tip_length_to_length_ratio=0.35)
                         for i in range(nxL - 1) for j in range(nyL)])
        Hx_ar = VGroup(*[Arrow(node(i, j) + np.array([-0.17, s_ / 2, 0]), node(i, j) + np.array([0.17, s_ / 2, 0]), buff=0,
                               stroke_width=4, color=C_HF, max_tip_length_to_length_ratio=0.35)
                         for i in range(nxL) for j in range(nyL - 1)])
        Hall = VGroup(Hy_ar, Hx_ar)
        cell = Square(side_length=s_).move_to(node(3, 2) + np.array([s_ / 2, s_ / 2, 0])).set_stroke(YELLOW, 3)
        yee_t = VGroup(Tex("Yee, 1966", font_size=36),
                       Tex("$E$ on the corners,", font_size=30, color=GREY_A),
                       Tex("$H$ on the edges:", font_size=30, color=GREY_A),
                       Tex("half a cell apart", font_size=30, color=YELLOW)).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        yee_t[1][0][0].set_color(C_E)
        yee_t[2][0][0].set_color(C_HF)
        yee_t.move_to([4.1, 1.2, 0])
        self.at(S("y3") + 2.6)
        self.play(FadeOut(VGroup(axis, curve, samp, br, dlab, hmarks, emarks)), run_time=0.5)
        self.play(Create(lines), FadeIn(Edots), run_time=0.8)
        self.play(LaggedStart(*[GrowArrow(a) for a in [*Hy_ar, *Hx_ar]], lag_ratio=0.01), FadeIn(yee_t), Create(cell), run_time=1.2)

        # ---- y4: time is staggered too
        ty = -2.55
        tx0, tdt = 1.15, 1.25
        tax = Arrow([tx0 - 0.35, ty, 0], [tx0 + 4.55 * tdt / 1.0, ty, 0], buff=0, stroke_width=2, color=GREY_B,
                    max_tip_length_to_length_ratio=0.03)
        tl = MathTex("t", font_size=30, color=GREY_B).next_to(tax, RIGHT, buff=0.08)
        tE = VGroup(*[VGroup(Dot([tx0 + k * tdt, ty, 0], radius=0.08, color=C_E),
                             MathTex(["n", "n{+}1", "n{+}2", "n{+}3"][k], font_size=26, color=C_E).move_to([tx0 + k * tdt, ty - 0.36, 0]))
                      for k in range(4)])
        tH = VGroup(*[VGroup(Square(0.13).rotate(PI / 4).move_to([tx0 + (k + 0.5) * tdt, ty, 0]).set_fill(C_HF, 1).set_stroke(width=0),
                             MathTex(["n{+}\\frac12", "n{+}\\frac32", "n{+}\\frac52"][k], font_size=26, color=C_HF)
                             .move_to([tx0 + (k + 0.5) * tdt, ty + 0.38, 0]))
                      for k in range(3)])
        tlab = VGroup(Tex("$E$: whole steps", font_size=26, color=C_E), Tex("$H$: half steps", font_size=26, color=C_HF))
        tlab.arrange(DOWN, buff=0.1, aligned_edge=LEFT).next_to(tax, LEFT, buff=0.3).shift(0.05 * UP)
        self.at(S("y4"))
        self.play(GrowArrow(tax), FadeIn(tl), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(e) for e in tE], lag_ratio=0.2), FadeIn(tlab[0]), run_time=1.0)
        self.at(S("y4") + 2.9)
        self.play(LaggedStart(*[FadeIn(h) for h in tH], lag_ratio=0.2), FadeIn(tlab[1]), run_time=1.0)

        # ---- y5: the leapfrog
        upH = MathTex("\\mathbf{H}", "\\leftarrow", "\\mathbf{H}", "-", "{\\Delta t \\over \\mu}", "\\nabla\\times", "\\mathbf{E}",
                      font_size=38)
        upE = MathTex("\\mathbf{E}", "\\leftarrow", "\\mathbf{E}", "+", "{\\Delta t \\over \\varepsilon}", "\\nabla\\times",
                      "\\mathbf{H}", font_size=38)
        for u_, a_, b_ in ((upH, C_HF, C_E), (upE, C_E, C_HF)):
            u_[0].set_color(a_)
            u_[2].set_color(a_)
            u_[6].set_color(b_)
        ups = VGroup(upH, upE).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([4.15, -0.75, 0])
        hop = Dot([tx0, ty, 0], radius=0.12, color=WHITE)
        ci, cj = 3, 2
        stH = VGroup(Line(node(ci, cj), node(ci + 1, cj)).set_stroke(YELLOW, 5), Dot(node(ci, cj), radius=0.12, color=YELLOW),
                     Dot(node(ci + 1, cj), radius=0.12, color=YELLOW))
        stE = VGroup(*[Circle(radius=0.2).move_to(node(ci + 1, cj) + v_).set_stroke(YELLOW, 4)
                       for v_ in (np.array([s_ / 2, 0, 0]), np.array([-s_ / 2, 0, 0]), np.array([0, s_ / 2, 0]), np.array([0, -s_ / 2, 0]))],
                     Dot(node(ci + 1, cj), radius=0.12, color=YELLOW))

        def leap(target_x, run_time=0.55):
            arc = ArcBetweenPoints(hop.get_center(), np.array([target_x, ty, 0]), angle=-PI * 0.75)
            return MoveAlongPath(hop, arc, run_time=run_time)
        self.at(S("y5"))
        self.play(FadeOut(cell), FadeIn(hop), FadeOut(yee_t), run_time=0.5)
        self.play(FadeIn(upH), Hall.animate.set_color(YELLOW), FadeIn(stH), leap(tx0 + 0.5 * tdt), run_time=0.9)
        self.play(Hall.animate.set_color(C_HF), run_time=0.3)
        self.at(S("y5") + 3.0)
        self.play(FadeIn(upE), Edots.animate.set_color(YELLOW), FadeOut(stH), FadeIn(stE), leap(tx0 + 1.0 * tdt), run_time=0.9)
        self.play(Edots.animate.set_color(C_E), run_time=0.3)
        self.at(S("y5") + 5.0)
        self.play(Hall.animate.set_color(YELLOW), FadeOut(stE), leap(tx0 + 1.5 * tdt, 0.45), run_time=0.45)
        self.play(Hall.animate.set_color(C_HF), Edots.animate.set_color(YELLOW), leap(tx0 + 2.0 * tdt, 0.45), run_time=0.45)
        self.play(Edots.animate.set_color(C_E), Hall.animate.set_color(YELLOW), leap(tx0 + 2.5 * tdt, 0.45), run_time=0.45)
        self.play(Hall.animate.set_color(C_HF), Edots.animate.set_color(YELLOW), leap(tx0 + 3.0 * tdt, 0.45), run_time=0.45)
        self.play(Edots.animate.set_color(C_E), run_time=0.25)

        # ---- y6: one dimension, real numbers
        d = load("leap.npz")
        E, H, S_ = d["E"], d["H"], float(d["S"])
        nj = E.shape[1]
        xj = -6.3 + np.arange(nj) * 0.29
        yE, yH, scl = 0.75, -1.15, 1.05
        stp = ValueTracker(0.0)

        def row_state(s):
            k = int(np.floor(s + 1e-9))
            fr = s - k
            Ek = E[k]
            Hk = H[k + 1] if fr >= 0.5 else H[k]
            return Ek, Hk

        def rows():
            Ek, Hk = row_state(stp.get_value())
            o_ = rop.get_value()
            g = VGroup(stems_row(xj, Ek, yE, scl, C_E, opacity=o_), stems_row(xj + 0.145, Hk, yH, scl, C_HF, opacity=o_))
            return g
        rop = ValueTracker(0.0)
        rw = always_redraw(rows)
        rl = VGroup(MathTex("E_j", font_size=32, color=C_E).move_to([-6.75, yE + 0.35, 0]),
                    MathTex("H_{j+\\frac12}", font_size=30, color=C_HF).move_to([-6.6, yH - 0.45, 0]))
        # worked example: pick a leading-edge cell where the rounded numbers add up exactly
        pick = None
        for k_ in range(3, 9):
            for j_ in range(9, 20):
                r2 = lambda v: round(float(v), 2)
                a_, hr, hl, new = r2(E[k_][j_]), r2(H[k_ + 1][j_]), r2(H[k_ + 1][j_ - 1]), r2(E[k_ + 1][j_])
                diff = round(hr - hl, 2)
                if 0.15 < a_ < 0.7 and diff > 0.1 and abs(round(a_ + S_ * diff, 3) - new) < 1e-9 and round(diff * 100) % 2 == 0:
                    pick = (k_, j_, a_, hr, hl, new)
                    break
            if pick:
                break
        k_, j_, a_, hr, hl, new = pick
        fmt = lambda v: f"{v:.2f}".replace("-", "{-}")
        ex1 = MathTex("E_j", "\\;\\leftarrow\\;", "E_j", "+", "\\tfrac12", "(", "H_{j+\\frac12}", "-", "H_{j-\\frac12}", ")", font_size=38)
        ex2 = MathTex(fmt(new), "\\;\\leftarrow\\;", fmt(a_), "+", "\\tfrac12", "(", fmt(hr), "-", "(" + fmt(hl) + ")", ")", font_size=38)
        for e_ in (ex1, ex2):
            e_[0].set_color(C_E)
            e_[2].set_color(C_E)
            e_[6].set_color(C_HF)
            e_[8].set_color(C_HF)
        ex1.move_to([0.4, 2.35, 0])
        ex2.next_to(ex1, DOWN, buff=0.22).align_to(ex1, LEFT)
        S_l = Tex(f"($c\\,\\Delta t/\\Delta x = {S_:g}$)", font_size=26, color=GREY_B).next_to(ex1, RIGHT, buff=0.35)
        boxE = Square(0.26).move_to([xj[j_], yE, 0]).set_stroke(YELLOW, 3)
        boxH = VGroup(Square(0.26).move_to([xj[j_] + 0.145, yH, 0]), Square(0.26).move_to([xj[j_ - 1] + 0.145, yH, 0])).set_stroke(YELLOW, 3)
        self.at(S("y6"))
        self.play(FadeOut(VGroup(lines, Edots, Hall, ups, tax, tl, tE, tH, tlab, hop, stE)), run_time=0.5)
        stp.set_value(float(k_))
        self.add(rw)
        self.play(rop.animate.set_value(1.0), FadeIn(rl), run_time=0.7)
        self.play(Write(ex1), FadeIn(S_l), Create(boxE), Create(boxH), run_time=1.0)
        self.play(stp.animate.set_value(k_ + 0.5), run_time=0.4, rate_func=lambda t: float(t >= 0.5))
        self.play(FadeIn(ex2, shift=0.1 * DOWN), run_time=0.6)
        self.play(stp.animate.set_value(k_ + 1.0), run_time=0.4, rate_func=lambda t: float(t >= 0.5))
        self.play(FadeOut(boxE), FadeOut(boxH), run_time=0.3)
        self.play(stp.animate.set_value(k_ + 40.0), run_time=3.2, rate_func=linear)

        # ---- y7: matter = one number per cell
        nc_x, nc_y, cs = 34, 20, 0.29
        ccen = np.array([-1.35, -0.05, 0])
        gx = (np.arange(nc_x) - (nc_x - 1) / 2) * 0.1
        gy = (np.arange(nc_y) - (nc_y - 1) / 2) * 0.1
        GX, GY = np.meshgrid(gx, gy)
        rc_y, rr_, hw = 0.2, 0.6, 0.12
        yb_ = rc_y - (rr_ + hw + 0.07 + 0.1)
        inside = (np.abs(np.hypot(GX, GY - rc_y) - rr_) <= hw) | (np.abs(GY - yb_) <= 0.1)
        epsimg = np.zeros((nc_y, nc_x, 4), np.uint8)
        epsimg[..., :3] = np.where(inside[..., None], np.array([150, 150, 150]), np.array([12, 12, 12]))
        epsimg[..., 3] = 255
        epsimg = epsimg[::-1]
        up = 12
        epsimg = np.repeat(np.repeat(epsimg, up, 0), up, 1)
        gw, gh = nc_x * cs, nc_y * cs
        eim, efd, esw = self.sweep_image(epsimg, gw, ccen, gh, axis="right")
        eim.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        glines = VGroup(*[Line(ccen + np.array([-gw / 2 + k * cs, -gh / 2, 0]), ccen + np.array([-gw / 2 + k * cs, gh / 2, 0]))
                          for k in range(nc_x + 1)],
                        *[Line(ccen + np.array([-gw / 2, -gh / 2 + k * cs, 0]), ccen + np.array([gw / 2, -gh / 2 + k * cs, 0]))
                          for k in range(nc_y + 1)]).set_stroke(GREY_D, 0.8)
        e_air = VGroup(MathTex("\\varepsilon = 1", font_size=34), Tex("air", font_size=28, color=GREY_A)).arrange(DOWN, buff=0.08)
        e_si = VGroup(MathTex("\\varepsilon \\approx 12", font_size=34), Tex("silicon", font_size=28, color=GREY_A)).arrange(DOWN, buff=0.08)
        e_air.move_to([5.0, 1.45, 0])
        e_si.move_to([5.0, -0.55, 0])
        a_air = Arrow(e_air.get_left() + 0.1 * LEFT, ccen + np.array([gw / 2 - 0.6, 1.8, 0]), buff=0.1, stroke_width=3, color=GREY_A,
                      max_tip_length_to_length_ratio=0.12)
        a_si = Arrow(e_si.get_left() + 0.1 * LEFT, ccen + np.array([rr_ / 0.1 * cs * 0.97, rc_y / 0.1 * cs, 0]), buff=0.12,
                     stroke_width=3, color=GREY_A, max_tip_length_to_length_ratio=0.12)
        epsq = eh_tex("{\\partial \\mathbf{E} \\over \\partial t} = {1 \\over \\varepsilon}\\,\\nabla\\times\\mathbf{H}", size=36)
        epsq.move_to([5.0, -2.25, 0])
        self.at(S("y7"))
        self.play(rop.animate.set_value(0.0), FadeOut(VGroup(rl, ex1, ex2, S_l)), run_time=0.5)
        self.remove(rw)
        self.add(eim)
        self.play(efd.animate.set_value(1.0), Create(glines), run_time=0.8)
        self.play(esw.animate.set_value(1.03), run_time=1.6, rate_func=linear)
        self.play(FadeIn(e_air), GrowArrow(a_air), FadeIn(e_si), GrowArrow(a_si), run_time=0.8)
        self.at(S("y7") + 4.0)
        self.play(FadeIn(epsq), run_time=0.6)
        self.play(Indicate(epsq, color=YELLOW, scale_factor=1.06), run_time=0.9)
        self.finish()


# ============================================================ 3  grid rules ========
class Ch3GridRules(Chapter):
    CH = 3
    TAG = "3\\quad The grid's own rules"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---- c1: news spreads one cell per step: a diamond
        n_c, cs = 15, 0.34
        GC = np.array([-3.3, -0.25, 0])
        h = n_c // 2
        cellp = lambda i, j: GC + np.array([i * cs, j * cs, 0])
        cells = {}
        allc = VGroup()
        for i in range(-h, h + 1):
            for j in range(-h, h + 1):
                sq = Square(cs).move_to(cellp(i, j)).set_stroke(GREY_D, 1).set_fill(BLACK, 0)
                cells[(i, j)] = sq
                allc.add(sq)
        title1 = Tex("1. a speed limit", font_size=36).move_to([3.6, 2.6, 0])
        self.at(S("c1"))
        self.play(FadeIn(allc), FadeIn(title1), run_time=0.8)
        cells[(0, 0)].set_fill(GOLD, 1)
        src = Dot(cellp(0, 0), radius=0.07, color=WHITE)
        self.play(FadeIn(src), cells[(0, 0)].animate.set_fill(GOLD, 1), run_time=0.3)
        steps_lab = [Tex(f"after {n} step" + ("s" if n > 1 else ""), font_size=30, color=GREY_A).move_to([3.6, 1.9, 0]) for n in range(1, 7)]
        nmax = 6
        self.at(S("c1") + 3.4)
        prev = None
        for n in range(1, nmax + 1):
            ring = [cells[(i, j)] for (i, j) in cells if abs(i) + abs(j) == n]
            anims = [c.animate.set_fill(interpolate_color(ManimColor(C_E), ManimColor(BLACK), 0.15 + 0.6 * n / nmax), 1) for c in ring]
            anims.append(FadeIn(steps_lab[n - 1]) if prev is None else ReplacementTransform(prev, steps_lab[n - 1]))
            self.play(*anims, run_time=0.45)
            prev = steps_lab[n - 1]
        dia = Polygon(cellp(nmax + 0.5, 0), cellp(0, nmax + 0.5), cellp(-nmax - 0.5, 0), cellp(0, -nmax - 0.5)).set_stroke(YELLOW, 3)
        self.play(Create(dia), run_time=0.6)

        # ---- c2: light's circle must fit inside the diamond
        ct = ValueTracker(0.0)              # c dt / dx
        cop = ValueTracker(0.0)

        def light_circle():
            o_ = cop.get_value()
            if o_ < 0.01:
                return VGroup()
            r = nmax * ct.get_value()        # radius in cells after nmax steps
            th = np.linspace(0, TAU_, 361)
            out = r * (np.abs(np.cos(th)) + np.abs(np.sin(th))) > nmax + 1e-6
            X = GC[0] + r * cs * np.cos(th)
            Y = GC[1] + r * cs * np.sin(th)
            g = VGroup(poly_line(X, Y, WHITE, 4, o_))
            if out.any():
                seg = np.split(np.arange(361), np.where(np.diff(out.astype(int)) != 0)[0] + 1)
                for s_ in seg:
                    if out[s_[0]] and len(s_) > 1:
                        g.add(poly_line(X[s_], Y[s_], C_EV, 7, o_))
            return g
        lc = always_redraw(light_circle)
        cour = MathTex("c\\,\\Delta t", "\\;\\le\\;", "{\\Delta x \\over \\sqrt{2}}", font_size=52).move_to([3.6, 0.25, 0])
        cbox = SurroundingRectangle(cour, buff=0.18, color=YELLOW, stroke_width=2.5)
        cl_ = Tex("the Courant limit (2D)", font_size=32, color=YELLOW).next_to(cbox, DOWN, buff=0.2)
        light_l = Tex("light: a circle of radius $c\\,t$", font_size=28, color=GREY_A).move_to([3.6, 1.25, 0])
        self.at(S("c2"))
        self.add(lc)
        ct.set_value(0.35)
        self.play(cop.animate.set_value(1.0), FadeIn(light_l), run_time=0.6)
        self.play(ct.animate.set_value(1 / np.sqrt(2)), run_time=1.6)
        self.play(ct.animate.set_value(0.92), run_time=1.2)
        self.play(ct.animate.set_value(1 / np.sqrt(2)), Write(cour), run_time=1.2)
        self.play(Create(cbox), FadeIn(cl_), run_time=0.6)

        # ---- c3: break it by 6 %: the computed field explodes
        d = load("cfl.npz")
        ok, bad, m_ok, m_bad = d["ok"], d["bad"], d["max_ok"], d["max_bad"]
        nst = len(m_ok) - 1
        t_a, t_b = S("c3") + 0.5, S("c4") - 0.25
        step_at = lambda t: float(np.clip((t - t_a) / (t_b - t_a), 0, 1) * nst)
        tfr = np.arange(nst + 1, dtype=float)
        PW = 3.3
        c_ok, c_bad = np.array([-3.85, 0.75, 0]), np.array([0.15, 0.75, 0])
        im_ok, f_ok = self.live_image(lambda t: movie_rgba(ok, tfr, step_at(t)), PW, c_ok, PW)
        im_bad, f_bad = self.live_image(lambda t: movie_rgba(bad, tfr, step_at(t)), PW, c_bad, PW)
        fr_ok = Square(PW).move_to(c_ok).set_stroke(GREY_B, 1.5)
        fr_bad = Square(PW).move_to(c_bad).set_stroke(GREY_B, 1.5)
        l_ok = MathTex("c\\,\\Delta t = 0.70\\,\\Delta x", font_size=30, color=C_H).next_to(fr_ok, UP, buff=0.12)
        l_bad = MathTex("c\\,\\Delta t = 0.75\\,\\Delta x", font_size=30, color=C_EV).next_to(fr_bad, UP, buff=0.12)
        s_ok = Tex("stable", font_size=26, color=C_H).next_to(fr_ok, DOWN, buff=0.1)
        s_bad = Tex("6\\% over the limit", font_size=26, color=C_EV).next_to(fr_bad, DOWN, buff=0.1)
        # log10 max|E| vs step (the counter)
        pc, pw_, ph_ = np.array([4.55, 0.75, 0]), 2.6, 2.6
        pax = VGroup(Line(pc + np.array([-pw_ / 2, -ph_ / 2, 0]), pc + np.array([pw_ / 2, -ph_ / 2, 0])),
                     Line(pc + np.array([-pw_ / 2, -ph_ / 2, 0]), pc + np.array([-pw_ / 2, ph_ / 2, 0]))).set_stroke(GREY_B, 1.5)
        plab = VGroup(Tex("largest $|E|$", font_size=24, color=GREY_A).next_to(pax, UP, buff=0.1),
                      MathTex("10^{0}", font_size=22, color=GREY_B), MathTex("10^{12}", font_size=22, color=GREY_B),
                      Tex("steps", font_size=22, color=GREY_B).next_to(pax, DOWN, buff=0.08))
        ylo, yhi = -1.0, 12.0
        yv = lambda v: pc[1] - ph_ / 2 + (np.log10(np.maximum(v, 1e-30)) - ylo) / (yhi - ylo) * ph_
        plab[1].next_to([pc[0] - pw_ / 2, yv(1.0), 0], LEFT, buff=0.06)
        plab[2].next_to([pc[0] - pw_ / 2, yv(1e12), 0], LEFT, buff=0.06)

        def counter():
            k = int(step_at(self.now))
            g = VGroup()
            if k < 1:
                return g
            xx = pc[0] - pw_ / 2 + np.arange(k + 1) / nst * pw_
            g.add(poly_line(xx, np.clip(yv(m_ok[:k + 1]), None, pc[1] + ph_ / 2), C_H, 3))
            g.add(poly_line(xx, np.clip(yv(m_bad[:k + 1]), None, pc[1] + ph_ / 2), C_EV, 3))
            return g
        cnt = always_redraw(counter)
        self.at(S("c3"))
        self.play(FadeOut(VGroup(allc, dia, src, prev, light_l, title1)), cop.animate.set_value(0.0),
                  VGroup(cour, cbox, cl_).animate.scale(0.7).move_to([4.4, -2.3, 0]), run_time=0.5)
        self.remove(lc)
        self.add(im_ok, im_bad, cnt)
        self.play(f_ok.animate.set_value(1.0), f_bad.animate.set_value(1.0), FadeIn(fr_ok), FadeIn(fr_bad), FadeIn(l_ok),
                  FadeIn(l_bad), FadeIn(pax), FadeIn(plab), run_time=0.4)
        self.at(S("c3") + 2.2)
        self.play(FadeIn(s_ok), FadeIn(s_bad), run_time=0.5)

        # ---- c4: the exact k-circle
        KC, R = np.array([-3.4, -0.2, 0]), 1.45
        kax = VGroup(Arrow(KC + 2.5 * LEFT, KC + 2.6 * RIGHT, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03),
                     Arrow(KC + 2.5 * DOWN, KC + 2.6 * UP, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.03))
        kxl = MathTex("k_x", font_size=32, color=C_KX).next_to(kax[0], RIGHT, buff=0.06)
        kyl = MathTex("k_y", font_size=32, color=C_KZ).next_to(kax[1], UP, buff=0.06)
        circ = Circle(radius=R).move_to(KC).set_stroke(WHITE, 3)
        ang = ValueTracker(0.6)
        kvec = always_redraw(lambda: fast_arrow(KC, KC + R * np.array([np.cos(ang.get_value()), np.sin(ang.get_value()), 0]), WHITE, 5, 0.2))
        title2 = Tex("2. the grid bends waves", font_size=36).move_to([3.4, 2.6, 0])
        free = VGroup(Tex("free space:", font_size=30, color=GREY_A), MathTex("|k| = 2\\pi/\\lambda", font_size=34),
                      Tex("in every direction", font_size=30, color=GREY_A)).arrange(DOWN, buff=0.12).move_to([3.4, 1.35, 0])
        self.at(S("c4"))
        self.play(*[f_.animate.set_value(0.0) for f_ in (f_ok, f_bad)],
                  FadeOut(VGroup(fr_ok, fr_bad, l_ok, l_bad, s_ok, s_bad, pax, plab, cour, cbox, cl_)), FadeOut(cnt), run_time=0.5)
        for m_ in (im_ok, im_bad, cnt):
            m_.clear_updaters()
            self.remove(m_)
        self.play(GrowArrow(kax[0]), GrowArrow(kax[1]), FadeIn(kxl), FadeIn(kyl), Create(circ), FadeIn(title2), run_time=1.0)
        self.add(kvec)
        self.play(FadeIn(free), ang.animate.set_value(0.6 + TAU_), run_time=3.6, rate_func=linear)

        # ---- c5: the grid's contour bulges along the axes
        dd = load("disp.npz")
        Ns, th, cont = dd["Ns"], dd["th"], dd["cont"]
        Nt = ValueTracker(float(Ns[1]))

        def grid_contour():
            nv = Nt.get_value()
            k = int(np.clip(np.searchsorted(-Ns, -nv) - 1, 0, len(Ns) - 2))
            w_ = float(np.clip((Ns[k] - nv) / (Ns[k] - Ns[k + 1]), 0, 1))
            rr = cont[k] * (1 - w_) + cont[k + 1] * w_
            return poly_line(KC[0] + R * rr * np.cos(th), KC[1] + R * rr * np.sin(th), C_GRID, 5)
        gcont = always_redraw(grid_contour)
        # slider: cells per wavelength
        sx0, sx1, sy = 1.4, 5.6, -0.35
        nmap = lambda nv: sx0 + (np.log(20.0) - np.log(nv)) / (np.log(20.0) - np.log(3.1)) * (sx1 - sx0)
        sl_line = Line([sx0, sy, 0], [sx1, sy, 0]).set_stroke(GREY_B, 3)
        ticks = VGroup(*[VGroup(Line([nmap(v), sy - 0.1, 0], [nmap(v), sy + 0.1, 0]).set_stroke(GREY_B, 2),
                                MathTex(str(v), font_size=26, color=GREY_A).move_to([nmap(v), sy - 0.38, 0])) for v in (20, 10, 5, 3.1)])
        s_t = Tex("cells per wavelength", font_size=28, color=GREY_A).move_to([3.5, sy + 0.5, 0])
        knob = always_redraw(lambda: Dot([nmap(Nt.get_value()), sy, 0], radius=0.11, color=C_GRID))
        legend = VGroup(VGroup(Line(ORIGIN, 0.5 * RIGHT).set_stroke(WHITE, 3), Tex("exact", font_size=26)).arrange(RIGHT, buff=0.15),
                        VGroup(Line(ORIGIN, 0.5 * RIGHT).set_stroke(C_GRID, 5), Tex("Yee grid", font_size=26, color=C_GRID)).arrange(RIGHT, buff=0.15))
        legend.arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([3.5, -1.55, 0])
        slow = Tex("slowest along the axes", font_size=28, color=C_GRID).move_to(KC + np.array([0.0, -2.75, 0]))
        self.at(S("c5"))
        self.play(FadeOut(free), FadeOut(kvec), run_time=0.4)
        self.remove(kvec)
        self.add(gcont, knob)
        self.play(FadeIn(sl_line), FadeIn(ticks), FadeIn(s_t), FadeIn(legend), run_time=0.6)
        self.play(Nt.animate.set_value(3.1), run_time=4.2, rate_func=smooth)
        self.play(FadeIn(slow), run_time=0.5)

        # ---- c6: a pulse on a coarse grid vs a fine grid (computed)
        tc, tfi = dd["t_coarse"], dd["t_fine"]
        Fc, Ff = dd["coarse"], dd["fine"]
        tau = warp([S("c6"), S("c6") + 5.4, self.t_end], [1.0, 4.9, 5.6])
        PW2 = 3.25
        cc, cf = np.array([1.75, 0.55, 0]), np.array([5.25, 0.55, 0])
        imc, fc = self.live_image(lambda t: movie_rgba(Fc, tc, tau(t)), PW2, cc, PW2)
        imf, ff = self.live_image(lambda t: movie_rgba(Ff, tfi, tau(t)), PW2, cf, PW2)
        frc, frf = Square(PW2).move_to(cc).set_stroke(GREY_B, 1.5), Square(PW2).move_to(cf).set_stroke(GREY_B, 1.5)
        lc_ = Tex("5 cells per $\\lambda$", font_size=28, color=C_GRID).next_to(frc, UP, buff=0.12)
        lf_ = Tex("24 cells per $\\lambda$", font_size=28, color=GREY_A).next_to(frf, UP, buff=0.12)
        cap = Tex("computed", font_size=22, color=GREY_B).next_to(frf, DOWN, buff=0.08).align_to(frf, RIGHT)
        rule = Tex("rule of thumb: 10--20 cells per wavelength, or more", font_size=30, color=YELLOW).move_to([3.5, -1.85, 0])
        rule2 = Tex("(count them in the densest material)", font_size=24, color=GREY_B).next_to(rule, DOWN, buff=0.1)
        self.at(S("c6"))
        self.play(FadeOut(VGroup(sl_line, ticks, s_t, legend, slow, title2)), FadeOut(knob), Nt.animate.set_value(5.0), run_time=0.6)
        self.remove(knob)
        self.add(imc, imf)
        self.play(fc.animate.set_value(1.0), ff.animate.set_value(1.0), FadeIn(frc), FadeIn(frf), FadeIn(lc_), FadeIn(lf_),
                  FadeIn(cap), run_time=0.6)
        self.at(S("c6") + 3.6)
        self.play(FadeIn(rule), FadeIn(rule2), run_time=0.7)
        self.finish()


# ============================================================ 4  boundary ==========
def layer_frame(center, inner, outer, color, opacity):
    """Four strips between an inner and an outer square (the absorbing border), filled with a tint."""
    c = np.asarray(center, float)
    t = (outer - inner) / 2
    g = VGroup()
    for dx_, dy_, w_, h_ in ((0, (inner + t) / 2, outer, t), (0, -(inner + t) / 2, outer, t),
                             ((inner + t) / 2, 0, t, inner), (-(inner + t) / 2, 0, t, inner)):
        g.add(Rectangle(width=w_, height=h_).move_to(c + np.array([dx_, dy_, 0])).set_stroke(width=0).set_fill(color, opacity))
    return g


class Ch4Boundary(Chapter):
    CH = 4
    TAG = "4\\quad The edge of the world"

    def construct(self):
        S = self.S
        self.show_tag()
        d = load("boundary.npz")
        tf, L, dl = d["t_frame"], float(d["L"]), float(d["d"])
        PW = 5.8
        PC = np.array([-2.55, -0.18, 0])
        inner = PW * L / (L + 2 * dl)
        tau_w = warp([S("b1") - 0.3, S("b2"), S("b2") + 4.6, S("b3")], [0.0, 5.0, 10.6, 11.4])
        tau_l = warp([S("b3"), S("b3") + 4.4, S("b4"), self.t_end], [2.5, 8.6, 11.0, 14.6])
        imw, fw = self.live_image(lambda t: movie_rgba(d["wall"], tf, tau_w(t)), PW, PC, PW)
        iml, fl = self.live_image(lambda t: movie_rgba(d["lossy"], tf, tau_l(t)), PW, PC, PW)
        wall = Square(inner).move_to(PC).set_stroke(WHITE, 5)
        outer = Square(PW).move_to(PC).set_stroke(GREY_C, 1.2)
        lossy = layer_frame(PC, inner, PW, C_EV, 0.22)
        t1 = Tex("the grid must end somewhere", font_size=34).move_to([3.75, 2.2, 0])
        t2 = VGroup(Tex("hard wall:", font_size=32, color=GREY_A), Tex("a mirror", font_size=32, color=C_EV)).arrange(RIGHT, buff=0.15)
        t2b = Tex("the echo ruins the answer", font_size=30, color=GREY_A)
        g2 = VGroup(t2, t2b).arrange(DOWN, buff=0.14).move_to([3.75, 0.95, 0])
        t3 = VGroup(Tex("lossy layer:", font_size=32, color=GREY_A), Tex("echo from its edge", font_size=32, color=C_EV)).arrange(RIGHT, buff=0.15)
        t3b = Tex("(29\\% of the wall's echo)", font_size=26, color=GREY_B)
        g3 = VGroup(t3, t3b).arrange(DOWN, buff=0.12).move_to([3.75, -0.35, 0])
        t4 = Tex("gentler $\\Rightarrow$ thicker $\\Rightarrow$ more cells", font_size=30, color=GREY_A).move_to([3.75, -1.45, 0])
        t5 = label_box(Tex("an absorber that light cannot see", font_size=34, color=YELLOW), color=YELLOW, buff=0.2, fill=0.0)
        t5.move_to([3.75, -2.45, 0])
        cap = Tex("computed", font_size=22, color=GREY_B).next_to(outer, RIGHT, buff=0.1).align_to(outer, DOWN)
        self.add(imw)
        self.at(S("b1"))
        self.play(fw.animate.set_value(1.0), Create(wall), FadeIn(outer), FadeIn(t1), FadeIn(cap), run_time=0.8)
        self.at(S("b2") + 0.3)
        self.play(FadeIn(g2), run_time=0.7)
        # ---- b3: lossy layer
        self.at(S("b3"))
        self.add(iml)
        self.play(fw.animate.set_value(0.0), fl.animate.set_value(1.0), FadeIn(lossy), wall.animate.set_stroke(GREY_B, 2.5),
                  g2.animate.set_opacity(0.45), run_time=0.7)
        imw.clear_updaters()
        self.remove(imw)
        self.at(S("b3") + 2.6)
        self.play(FadeIn(g3), run_time=0.7)
        # ---- b4: gentler -> thicker; the wish
        self.at(S("b4"))
        self.play(FadeIn(t4), g3.animate.set_opacity(0.6), run_time=0.6)
        self.at(S("b4") + 2.6)
        self.play(FadeIn(t5, scale=1.05), run_time=0.8)
        self.finish()


# ============================================================ 5  PML ===============
class Ch5PML(Chapter):
    CH = 5
    TAG = "5\\quad The perfectly matched layer"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- signature shot: the x axis as a ribbon over the complex x-plane, bending into Im x
        O = np.array([-1.15, -0.5, 0])            # screen position of x = 0, Im x = 0, field = 0
        ex, ez = np.array([1.08, 0, 0]), np.array([0, 0.8, 0])
        xa_, xd, xf_ = -4.9, 2.3, 2.75            # interior start, border depth, floor end (wavelengths)
        B = 1.9                                   # Im x reached at the far end of the border (wavelengths)
        kk, omega = TAU_, TAU_ * 0.55
        xs_ = np.linspace(xa_, xd, 560)
        imx = np.where(xs_ > 0, B * (np.clip(xs_, 0, None) / xd) ** 4, 0.0)       # Im x~ = (1/w) int sigma, sigma ~ depth^3
        t_a, t_b = S("p1"), S("p4")
        tw = ValueTracker(0.0)                    # 0 = real axis, 1 = bent into the complex plane
        sop = ValueTracker(0.0)                   # scene opacity
        env_op = ValueTracker(0.0)

        def ev_at(t):                             # direction of Im x on screen: slowly turning (parallax)
            a = np.radians(np.interp(t, [t_a, t_b], [37.0, 49.0]))
            return 1.0 * np.array([np.cos(a), np.sin(a), 0])

        def P3(x, v, z, ev):
            x, v, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float), np.asarray(z, float))
            return O + x[..., None] * ex + v[..., None] * ev + z[..., None] * ez

        def floor():
            o_ = sop.get_value()
            if o_ < 0.01:
                return VGroup()
            ev = ev_at(self.now)
            g = VGroup(Polygon(*P3([xa_, xf_, xf_, xa_], [-0.25, -0.25, 2.0, 2.0], 0, ev)).set_stroke(GREY_C, 1.5, opacity=0.8 * o_)
                       .set_fill(GREY_E, 0.45 * o_))
            for k in range(8):                     # border tint: sigma grows with depth
                x0, x1 = xd * k / 8, xd * (k + 1) / 8
                op = (0.06 + 0.42 * ((k + 0.5) / 8) ** 2) * o_
                g.add(Polygon(*P3([x0, x1, x1, x0], [-0.25, -0.25, 2.0, 2.0], 0, ev)).set_stroke(width=0).set_fill(C_IM, op))
            for x in np.arange(-4.5, xf_ + 0.01, 0.5):
                g.add(Line(*P3([x, x], [-0.25, 2.0], 0, ev)).set_stroke(GREY_C, 1, opacity=0.55 * o_))
            for v in np.arange(0.0, 2.01, 0.5):
                g.add(Line(*P3([xa_, xf_], [v, v], 0, ev)).set_stroke(GREY_C, 1, opacity=0.55 * o_))
            # the real axis and the Im x direction
            g.add(fast_arrow(P3(xf_, 0, 0, ev), P3(xf_ + 0.75, 0, 0, ev), GREY_B, 2.5, 0.16, o_))
            g.add(fast_arrow(P3(0, 0, 0, ev), P3(0, 2.25, 0, ev), C_IM, 3, 0.18, o_ * min(1.0, 3 * tw.get_value())))
            return g

        def ribbon():
            o_ = sop.get_value()
            if o_ < 0.01:
                return VGroup()
            ev = ev_at(self.now)
            s_ = tw.get_value()
            P = P3(xs_, s_ * imx, 0, ev)
            top, bot = P3(xs_, s_ * imx, 1.12, ev), P3(xs_, s_ * imx, -1.12, ev)
            g = VGroup(Polygon(*top[::8], *bot[::-8]).set_stroke(width=0).set_fill(GREY_B, 0.12 * o_),
                       poly_line(top[:, 0], top[:, 1], GREY_C, 1, 0.55 * o_), poly_line(bot[:, 0], bot[:, 1], GREY_C, 1, 0.55 * o_))
            inn = xs_ <= 0
            g.add(poly_line(P[inn, 0], P[inn, 1], WHITE, 4, o_))
            colb = interpolate_color(ManimColor(WHITE), ManimColor(C_IM), min(1.0, 2 * s_))
            g.add(poly_line(P[~inn, 0], P[~inn, 1], colb, 5, o_))
            if s_ > 0.02:                           # where the real axis would have gone
                R_ = P3(np.array([0, xf_]), 0, 0, ev)
                g.add(fast_dashed(R_[0], R_[1], 0.07, GREY_B, 1.5, o_ * min(1.0, 3 * s_)))
            return g

        def wave():
            o_ = sop.get_value()
            if o_ < 0.01:
                return VGroup()
            ev = ev_at(self.now)
            s_ = tw.get_value()
            amp = np.exp(-kk * imx * s_)
            z = np.cos(kk * xs_ - omega * self.now) * amp
            Q = P3(xs_, s_ * imx, z, ev)
            g = VGroup(poly_line(Q[:, 0], Q[:, 1], C_E, 4.5, o_))
            eo = env_op.get_value() * o_
            if eo > 0.01:
                bd = xs_ >= 0
                for sg in (1, -1):
                    E_ = P3(xs_[bd], s_ * imx[bd], sg * amp[bd], ev)
                    g.add(poly_line(E_[:, 0], E_[:, 1], C_EV, 2.5, eo))
            return g
        fl, rb, wv = always_redraw(floor), always_redraw(ribbon), always_redraw(wave)
        rex_l = MathTex("\\mathrm{Re}\\,x", font_size=32, color=GREY_B)
        rex_l.add_updater(lambda m: m.set_opacity(sop.get_value()).next_to(P3(xf_ + 0.75, 0, 0, ev_at(self.now)), RIGHT, buff=0.08))
        imx_l = MathTex("\\mathrm{Im}\\,x", font_size=32, color=C_IM)
        imx_l.add_updater(lambda m: m.set_opacity(sop.get_value() * min(1.0, 3 * tw.get_value()))
                          .next_to(P3(0, 2.25, 0, ev_at(self.now)), UP, buff=0.06))
        inter_l = Tex("inside the grid", font_size=30, color=GREY_A).move_to([O[0] - 2.6 * 1.08, -1.75, 0])
        border_l = Tex("border", font_size=30, color=C_IM).move_to([O[0] + 1.15 * 1.08 + 0.3, -1.75, 0])
        names = VGroup(Tex("B\\'erenger, 1994", font_size=28, color=GREY_A), Tex("Chew \\& Weedon, 1994", font_size=28, color=GREY_A))
        names.arrange(DOWN, buff=0.1, aligned_edge=RIGHT).move_to([5.25, 3.0, 0])
        idea = Tex("change the coordinates, not the material", font_size=34, color=YELLOW).move_to([0.3, 2.45, 0])
        self.at(S("p1"))
        self.add(fl, rb, wv, rex_l, imx_l)
        self.play(sop.animate.set_value(1.0), FadeIn(inter_l), FadeIn(border_l), run_time=1.0)
        self.play(FadeIn(names), run_time=0.6)
        self.at(S("p1") + 4.6)
        self.play(FadeIn(idea, shift=0.1 * DOWN), run_time=0.8)

        # ---- p2: x picks up an imaginary part that grows with depth -> the ribbon bends into Im x
        xt = MathTex("\\tilde x", "=", "x", "+", "{i", "\\over", "\\omega}", "\\int_0^x", "\\sigma", "\\,dx'", font_size=46)
        xt[4].set_color(C_IM)
        xt[8].set_color(C_EV)
        xt.move_to([-3.75, -2.3, 0])
        self.at(S("p2") + 0.4)
        self.play(tw.animate.set_value(1.0), run_time=4.4, rate_func=smooth)
        self.at(S("p2") + 5.0)
        self.play(FadeOut(inter_l), FadeOut(border_l), run_time=0.4)
        self.play(Write(xt), run_time=1.4)

        # ---- p3: e^{ikx} keeps its wavelength and gains a decaying factor
        dec = MathTex("e^{ik\\tilde x}", "=", "e^{ikx}", "\\cdot", "e^{-\\frac{k}{\\omega}\\int\\sigma\\,dx}", font_size=46)
        dec[2].set_color(C_E)
        dec[4].set_color(C_EV)
        dec.move_to([2.55, -2.2, 0])
        b_same = Brace(dec[2], DOWN, buff=0.06, color=C_E)
        l_same = Tex("same wavelength", font_size=26, color=C_E).next_to(b_same, DOWN, buff=0.04)
        b_dec = Brace(dec[4], DOWN, buff=0.06, color=C_EV)
        l_dec = Tex("dies away", font_size=26, color=C_EV).next_to(b_dec, DOWN, buff=0.04)
        self.at(S("p3"))
        self.play(FadeOut(idea), Write(dec), run_time=1.6)
        self.at(S("p3") + 3.6)
        self.play(GrowFromCenter(b_same), FadeIn(l_same), run_time=0.7)
        self.at(S("p3") + 6.0)
        self.play(GrowFromCenter(b_dec), FadeIn(l_dec), env_op.animate.set_value(1.0), run_time=0.8)

        # ---- p4: same equations in new coordinates: no echo, any angle, any frequency (exact continuum PML)
        ang = ValueTracker(0.0)
        lam_t = ValueTracker(1.0)
        nxp, nyp = 300, 150
        ax_ = np.linspace(-4.2, 2.3, nxp)
        ay_ = np.linspace(-1.6, 1.6, nyp)
        AX, AY = np.meshgrid(ax_, ay_[::-1])
        imA = np.where(AX > 0, B * (np.clip(AX, 0, None) / xd) ** 4, 0.0)
        aop = ValueTracker(0.0)

        def pml2d(t):
            th, lam = ang.get_value(), lam_t.get_value()
            k_ = TAU_ / lam
            kx, ky = k_ * np.cos(th), k_ * np.sin(th)
            # Im x~ = (1/omega) int sigma  ~ lam: so exp(-kx Im x~) = exp(-cos(th) int sigma / c) for every frequency
            v = np.cos(kx * AX + ky * AY - omega / lam * t) * np.exp(-kx * imA * lam)
            img = DIV_RGBA[np.clip((v * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)].copy()
            img[..., 3] = int(255 * aop.get_value())
            return img
        PCW, PCH = 8.8, 4.33
        P2 = np.array([-1.55, 0.25, 0])
        pim, pfd = self.live_image(pml2d, PCW, P2, PCH)
        pfr = Rectangle(width=PCW, height=PCH).move_to(P2).set_stroke(GREY_B, 1.5)
        x0s = P2[0] - PCW / 2 + (0 - ax_[0]) / (ax_[-1] - ax_[0]) * PCW
        pb = Rectangle(width=P2[0] + PCW / 2 - x0s, height=PCH).move_to([0.5 * (x0s + P2[0] + PCW / 2), P2[1], 0])
        pb.set_stroke(C_IM, 2).set_fill(C_IM, 0.10)
        pl = Tex("PML", font_size=30, color=C_IM).next_to(pb, UP, buff=0.08)
        noecho = VGroup(Tex("no echo", font_size=38, color=YELLOW), Tex("at any angle", font_size=32, color=GREY_A),
                        Tex("or frequency", font_size=32, color=GREY_A)).arrange(DOWN, buff=0.14).move_to([5.0, 0.3, 0])
        exact_l = Tex("exact solution of the stretched equations", font_size=24, color=GREY_B).next_to(pfr, DOWN, buff=0.1)
        self.at(S("p4"))
        self.play(sop.animate.set_value(0.0), FadeOut(VGroup(xt, dec, b_same, l_same, b_dec, l_dec, names)),
                  env_op.animate.set_value(0.0), run_time=0.6)
        for m_ in (fl, rb, wv, rex_l, imx_l):
            m_.clear_updaters()
        self.remove(fl, rb, wv, rex_l, imx_l)
        self.add(pim)
        self.play(aop.animate.set_value(1.0), pfd.animate.set_value(1.0), FadeIn(pfr), FadeIn(pb), FadeIn(pl), FadeIn(exact_l), run_time=0.6)
        self.play(ang.animate.set_value(np.radians(55)), FadeIn(noecho[0]), FadeIn(noecho[1]), run_time=2.7)
        self.play(ang.animate.set_value(np.radians(25)), lam_t.animate.set_value(0.6), FadeIn(noecho[2]), run_time=2.5)
        self.play(lam_t.animate.set_value(1.0), run_time=1.2)

        # ---- p5: the perfectly matched layer, side by side with the wall and the lossy layer (computed)
        d = load("boundary.npz")
        tf, Lb, dl = d["t_frame"], float(d["L"]), float(d["d"])
        tau3 = warp([S("p5") + 0.6, S("p5") + 8.0, S("p6")], [3.0, 11.5, 12.5])
        PWb = 3.55
        cs3 = [np.array([-4.45, 0.05, 0]), np.array([0.0, 0.05, 0]), np.array([4.45, 0.05, 0])]
        ims = [self.live_image(lambda t, k=k: movie_rgba(d[k], tf, tau3(t)), PWb, c_, PWb) for k, c_ in zip(("wall", "lossy", "pml"), cs3)]
        inner = PWb * Lb / (Lb + 2 * dl)
        decos = VGroup(Square(inner).move_to(cs3[0]).set_stroke(WHITE, 4),
                       layer_frame(cs3[1], inner, PWb, C_EV, 0.22), layer_frame(cs3[2], inner, PWb, C_IM, 0.25),
                       *[Square(PWb).move_to(c_).set_stroke(GREY_C, 1.2) for c_ in cs3])
        heads = VGroup(*[Tex(t_, font_size=32, color=c_).move_to(cc + np.array([0, PWb / 2 + 0.3, 0]))
                         for t_, c_, cc in (("hard wall", WHITE, cs3[0]), ("lossy layer", C_EV, cs3[1]), ("PML", C_IM, cs3[2]))])
        er = d["echo_rel"]
        echos = VGroup(*[Tex(t_, font_size=28, color=c_).move_to(cc + np.array([0, -PWb / 2 - 0.3, 0]))
                         for t_, c_, cc in (("echo: 100\\%", C_EV, cs3[0]), (f"echo: {100 * er[1]:.0f}\\%", C_EV, cs3[1]),
                                            (f"echo: {100 * er[2]:.4f}\\%", C_H, cs3[2]))])
        ttl = Tex("the perfectly matched layer (PML)", font_size=36, color=YELLOW).move_to([0.9, 3.0, 0])
        self.at(S("p5"))
        self.play(FadeOut(VGroup(pfr, pb, pl, noecho, exact_l)), aop.animate.set_value(0.0), run_time=0.6)
        pim.clear_updaters()
        self.remove(pim)
        for im_, _ in ims:
            self.add(im_)
        self.play(*[f_.animate.set_value(1.0) for _, f_ in ims], FadeIn(decos), FadeIn(heads), FadeIn(ttl), run_time=0.7)
        self.at(S("p5") + 3.8)
        self.play(LaggedStart(*[FadeIn(e_) for e_ in echos], lag_ratio=0.5), run_time=1.8)

        # ---- p6: on a grid a tiny echo remains; sigma ramps up smoothly; grazing and evanescent waves are hardest
        tau4 = warp([S("p6") + 0.6, S("p6") + 6.6, self.t_end], [3.0, 11.5, 12.0])
        c_e = np.array([-3.3, 0.05, 0])
        ime, fe_ = self.live_image(lambda t: movie_rgba(d["pml_echo_x1e5"], tf, tau4(t)), PWb, c_e, PWb)
        efr = VGroup(Square(PWb).move_to(c_e).set_stroke(GREY_C, 1.2), Square(inner).move_to(c_e).set_stroke(C_IM, 1.5))
        eh = Tex("PML echo, magnified $10^5\\times$", font_size=30, color=C_IM).move_to(c_e + np.array([0, PWb / 2 + 0.3, 0]))
        sg_c = np.array([2.2, 0.3, 0])
        sgx = np.linspace(0, 1, 60)
        sg_ax = VGroup(Line(sg_c + np.array([-1.7, -1.2, 0]), sg_c + np.array([1.7, -1.2, 0])),
                       Line(sg_c + np.array([-1.7, -1.2, 0]), sg_c + np.array([-1.7, 1.4, 0]))).set_stroke(GREY_B, 1.5)
        sg_cv = poly_line(np.concatenate([np.linspace(sg_c[0] - 1.7, sg_c[0] - 0.2, 10), sg_c[0] - 0.2 + 1.9 * sgx]),
                          np.concatenate([np.full(10, sg_c[1] - 1.2), sg_c[1] - 1.2 + 2.4 * sgx ** 3]), C_EV, 4)
        sg_band = Rectangle(width=1.9, height=2.6).move_to([sg_c[0] + 0.75, sg_c[1] + 0.1, 0]).set_stroke(width=0).set_fill(C_IM, 0.12)
        sg_l = VGroup(MathTex("\\sigma", font_size=34, color=C_EV).next_to(sg_ax[1], UP, buff=0.06),
                      Tex("depth into the border", font_size=24, color=GREY_B).next_to(sg_ax[0], DOWN, buff=0.08),
                      Tex("ramped up smoothly", font_size=30, color=C_EV).move_to(sg_c + np.array([0, 1.95, 0])))
        hard = VGroup(Tex("hardest to absorb:", font_size=30, color=GREY_A), Tex("grazing waves", font_size=30, color=YELLOW),
                      Tex("evanescent waves", font_size=30, color=YELLOW)).arrange(DOWN, buff=0.12).move_to([5.55, 0.3, 0])
        self.at(S("p6"))
        self.play(*[f_.animate.set_value(0.0) for _, f_ in ims], FadeOut(VGroup(decos, heads, echos, ttl)), run_time=0.6)
        for im_, _ in ims:
            im_.clear_updaters()
            self.remove(im_)
        self.add(ime)
        self.play(fe_.animate.set_value(1.0), FadeIn(efr), FadeIn(eh), FadeIn(sg_ax), FadeIn(sg_band), Create(sg_cv), FadeIn(sg_l),
                  run_time=0.9)
        self.at(S("p6") + 4.2)
        self.play(FadeIn(hard), run_time=0.8)
        self.finish()


# ============================================================ 6  on a computer =====
class Ch6Computer(Chapter):
    CH = 6
    TAG = "6\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        lines = ["for n in range(steps):",
                 "    H -= dt / mu  * curl(E)      # E updates H",
                 "    E += dt / eps * curl(H)      # H updates E",
                 "    absorb_in_pml(E, H)          # stretched border",
                 "    E[source] += pulse(n * dt)   # add the light"]
        card = code_card_ind(lines, font_size=26, title="FDTD")
        card.move_to([0, 0.55, 0])
        rows = card[1][0]
        for r_, col in zip(rows[1:], (C_HF, C_E, C_IM, GOLD)):
            pass
        hl = SurroundingRectangle(rows[1], buff=0.07, color=YELLOW, stroke_width=2.5, corner_radius=0.05)
        self.at(S("a1"))
        self.play(FadeIn(card, shift=0.1 * UP), run_time=0.9)
        self.at(S("a2"))
        self.play(Create(hl), rows[1].animate.set_color(C_HF), run_time=0.5)
        for k, (dt_, col) in enumerate(((1.6, C_E), (3.3, C_IM), (4.8, GOLD))):
            self.at(S("a2") + dt_)
            self.play(Transform(hl, SurroundingRectangle(rows[k + 2], buff=0.07, color=YELLOW, stroke_width=2.5, corner_radius=0.05)),
                      rows[k + 2].animate.set_color(col), run_time=0.4)

        # ---- a3/a4: one pulse -> every wavelength (1D Bragg mirror, computed)
        d = load("bragg.npz")
        Fr, tfr, xf = d["frames"], d["t_frame"], d["xf"]
        xs0, xs1 = float(d["x_stack"][0]), float(d["x_stack"][1])
        sx_ = 1.27
        y0s = 2.05
        X = lambda x: x * sx_
        tau = warp([S("a3"), S("a3") + 6.2, S("a4") + 3.0, self.t_end], [2.6, 21.0, 27.5, 29.5])
        # the layer stack (TiO2 / SiO2 quarter-wave pairs) and the glass behind it
        nprof = d["nprof"]
        stack = VGroup()
        edges = np.where(np.diff(nprof) != 0)[0]
        bnds = [xf[0]] + [0.5 * (xf[e] + xf[e + 1]) for e in edges] + [xf[-1]]
        for a_, b_ in zip(bnds[:-1], bnds[1:]):
            nv = nprof[np.searchsorted(xf, 0.5 * (a_ + b_))]
            if nv > 1.01:
                col = GREY_A if nv > 2 else (GREY_C if nv < 1.49 else GREY_D)
                op = 0.55 if nv > 2 else (0.35 if nv < 1.49 else 0.18)
                stack.add(Rectangle(width=X(b_) - X(a_), height=1.6).move_to([0.5 * (X(a_) + X(b_)), y0s, 0])
                          .set_stroke(width=0).set_fill(col, op))
        sop = ValueTracker(0.0)

        def field1d():
            o_ = sop.get_value()
            if o_ < 0.01:
                return VGroup()
            tt = tau(self.now)
            k = int(np.clip(np.searchsorted(tfr, tt) - 1, 0, len(tfr) - 2))
            w_ = (tt - tfr[k]) / (tfr[k + 1] - tfr[k])
            v = Fr[k] * (1 - w_) + Fr[k + 1] * w_
            return poly_line(X(xf), y0s + 0.62 * v, C_E, 3.5, o_)
        f1 = always_redraw(field1d)
        base = Line([X(xf[0]), y0s, 0], [X(xf[-1]), y0s, 0]).set_stroke(GREY_D, 1)
        xp = float(d["x_probe"])
        probe = VGroup(Line([X(xp), y0s - 0.8, 0], [X(xp), y0s + 0.8, 0]).set_stroke(GOLD, 2),
                       Tex("detector", font_size=24, color=GOLD).move_to([X(xp), y0s + 1.0, 0]))
        mir = Tex("Bragg mirror", font_size=26, color=GREY_A).move_to([X(0.5 * (xs0 + xs1)), y0s + 1.0, 0])
        cap1 = Tex("1D FDTD (computed)", font_size=22, color=GREY_B).move_to([X(4.0), y0s + 1.0, 0])
        # trace at the detector
        tt_, ti, trf = d["t"], d["trace_inc"], d["trace_refl"]
        TC, TW, TH = np.array([-3.55, -1.45, 0]), 5.6, 2.2
        tx = lambda t: TC[0] - TW / 2 + t / 30.0 * TW
        tax = Arrow([TC[0] - TW / 2, TC[1], 0], [TC[0] + TW / 2 + 0.2, TC[1], 0], buff=0, stroke_width=2, color=GREY_B,
                    max_tip_length_to_length_ratio=0.03)
        tlab = VGroup(MathTex("t", font_size=28, color=GREY_B).next_to(tax, RIGHT, buff=0.06),
                      Tex("recorded at the detector", font_size=26, color=GOLD).move_to(TC + np.array([0, TH / 2 + 0.25, 0])),
                      Tex("pulse in", font_size=24, color=C_E).move_to([tx(6.5), TC[1] - TH / 2 + 0.05, 0]),
                      Tex("echo", font_size=24, color=YELLOW).move_to([tx(15.5), TC[1] - TH / 2 + 0.05, 0]))
        tsc = 0.95 / max(np.abs(ti).max(), 1e-9)

        def trace():
            o_ = sop.get_value()
            if o_ < 0.01:
                return VGroup()
            tt = tau(self.now) - 0.0
            n_ = int(np.searchsorted(tt_, tt))
            if n_ < 2:
                return VGroup()
            g = VGroup(poly_line(tx(tt_[:n_]), TC[1] + tsc * ti[:n_], C_E, 2.5, o_),
                       poly_line(tx(tt_[:n_]), TC[1] + tsc * trf[:n_], YELLOW, 2.5, o_))
            return g
        trc = always_redraw(trace)
        # spectrum
        SC, SW, SH = np.array([3.85, -1.45, 0]), 4.6, 2.2
        lam_lo, lam_hi = 0.6, 2.0
        lx = lambda l: SC[0] - SW / 2 + (l - lam_lo) / (lam_hi - lam_lo) * SW
        ly = lambda r: SC[1] - SH / 2 + r * SH
        sax = VGroup(Line([SC[0] - SW / 2, SC[1] - SH / 2, 0], [SC[0] + SW / 2, SC[1] - SH / 2, 0]),
                     Line([SC[0] - SW / 2, SC[1] - SH / 2, 0], [SC[0] - SW / 2, SC[1] + SH / 2, 0])).set_stroke(GREY_B, 1.5)
        slab = VGroup(Tex("reflectance", font_size=24, color=GREY_A).next_to(sax[1], UP, buff=0.08).align_to(sax[1], LEFT).shift(0.05 * RIGHT),
                      MathTex("\\lambda/\\lambda_0", font_size=26, color=GREY_B).next_to(sax[0], DOWN, buff=0.06).align_to(sax[0], RIGHT),
                      MathTex("1", font_size=22, color=GREY_B).next_to([SC[0] - SW / 2, ly(1), 0], LEFT, buff=0.06),
                      MathTex("0", font_size=22, color=GREY_B).next_to([SC[0] - SW / 2, ly(0), 0], LEFT, buff=0.06))
        le, Re_ = d["lam_exact"], d["R_exact"]
        exact = poly_line(lx(le), ly(Re_), WHITE, 2.5)
        lamf, Rf = d["lam"], d["R"]
        sel = (lamf >= lam_lo) & (lamf <= lam_hi)
        dots = VGroup(*[Dot([lx(l_), ly(r_), 0], radius=0.035, color=YELLOW) for l_, r_ in zip(lamf[sel], Rf[sel])])
        sleg = VGroup(VGroup(Line(ORIGIN, 0.4 * RIGHT).set_stroke(WHITE, 2.5), Tex("exact", font_size=22)).arrange(RIGHT, buff=0.1),
                      VGroup(Dot(radius=0.05, color=YELLOW), Tex("FDTD, one pulse", font_size=22, color=YELLOW)).arrange(RIGHT, buff=0.1))
        sleg.arrange(DOWN, buff=0.06, aligned_edge=LEFT).move_to([lx(1.72), ly(0.86), 0])
        ft = VGroup(Arrow([-0.12, -1.3, 0], [1.2, -1.3, 0], buff=0, stroke_width=4, color=GREY_A),
                    Tex("Fourier", font_size=24, color=GREY_A).move_to([0.54, -0.98, 0]),
                    Tex("transform", font_size=24, color=GREY_A).move_to([0.54, -1.62, 0]))
        self.at(S("a3"))
        self.play(FadeOut(card), FadeOut(hl), run_time=0.5)
        self.add(f1, trc)
        self.play(FadeIn(base), FadeIn(stack), FadeIn(probe), FadeIn(mir), FadeIn(cap1), sop.animate.set_value(1.0),
                  FadeIn(tax), FadeIn(tlab[0]), FadeIn(tlab[1]), run_time=0.7)
        self.at(S("a3") + 2.6)
        self.play(FadeIn(tlab[2]), run_time=0.5)
        self.at(S("a3") + 5.0)
        self.play(FadeIn(tlab[3]), GrowArrow(ft[0]), FadeIn(ft[1:]), run_time=0.6)
        self.at(S("a4"))
        self.play(FadeIn(sax), FadeIn(slab), Create(exact), FadeIn(sleg[0]), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(dd) for dd in dots], lag_ratio=0.02), FadeIn(sleg[1]), run_time=1.6)

        # ---- a5: the price: cells x steps; the model ladder
        cost = MathTex("\\text{cost}", "\\;\\propto\\;", "\\text{cells}", "\\times", "\\text{steps}", font_size=46).move_to([0, 2.45, 0])
        names = [("thin element", "TEA", "ms"), ("beam propagation", "BPM", "seconds"), ("periodic modes", "RCWA", "seconds--minutes"),
                 ("full Maxwell", "FDTD", "2D: seconds\\\\3D: minutes--hours")]
        rungs = VGroup()
        for i, (a_, b_, c_) in enumerate(names):
            box = RoundedRectangle(corner_radius=0.12, width=2.85, height=1.25).set_stroke(GREY_B if i < 3 else YELLOW, 2 if i < 3 else 3.5)
            t_ = VGroup(Tex(a_, font_size=28, color=GREY_A if i < 3 else WHITE), Tex(b_, font_size=26, color=GREY_B if i < 3 else YELLOW))
            t_.arrange(DOWN, buff=0.1)
            rungs.add(VGroup(box, t_.move_to(box)))
        rungs.arrange(RIGHT, buff=0.38).move_to([0, 0.55, 0])
        times = VGroup(*[Tex(c_, font_size=24, color=GREY_B if i < 3 else YELLOW).next_to(rungs[i], DOWN, buff=0.14)
                         for i, (_, _, c_) in enumerate(names)])
        arrow_l = Arrow([-5.6, -1.25, 0], [5.6, -1.25, 0], buff=0, stroke_width=3, color=GREY_B, max_tip_length_to_length_ratio=0.02)
        al = VGroup(Tex("faster, more approximate", font_size=24, color=GREY_B).next_to(arrow_l, DOWN, buff=0.08).align_to(arrow_l, LEFT),
                    Tex("slower, exact Maxwell", font_size=24, color=GREY_B).next_to(arrow_l, DOWN, buff=0.08).align_to(arrow_l, RIGHT))
        own = Tex("this episode's ring: 4.8 billion cell updates, 3 seconds on one GPU", font_size=24, color=GREY_B).move_to([0, -2.45, 0])
        self.at(S("a5"))
        self.play(FadeOut(VGroup(base, stack, probe, mir, cap1, tax, tlab, ft, sax, slab, exact, dots, sleg)), sop.animate.set_value(0.0),
                  run_time=0.6)
        self.remove(f1, trc)
        self.play(Write(cost), run_time=1.0)
        self.at(S("a5") + 2.6)
        self.play(LaggedStart(*[FadeIn(r_, shift=0.1 * UP) for r_ in rungs], lag_ratio=0.25), FadeIn(arrow_l), FadeIn(al), run_time=1.4)
        self.play(LaggedStart(*[FadeIn(t_) for t_ in times], lag_ratio=0.25), run_time=1.2)
        self.at(S("a5") + 6.6)
        self.play(FadeIn(own), run_time=0.6)

        # ---- a6: TEA vs FDTD on a fine glass relief (computed)
        r = load("relief.npz")
        Uf, Ut = r["U_fdtd"], r["U_tea"]
        X0r, X1r, Y0r, Y1r = [float(v) for v in r["extent"]]
        Lam, hh = float(r["Lam"]), float(r["h"])
        RH = 5.0
        RW = RH * (X1r - X0r) / (Y1r - Y0r)
        cT, cF = np.array([-3.65, -0.32, 0]), np.array([0.85, -0.32, 0])
        rop = ValueTracker(0.0)
        sc_ = 1.7

        def relief_img(U):
            def fn(t):
                v = np.real(U * np.exp(-1j * TAU_ * 0.55 * t)) / sc_
                img = DIV_RGBA[np.clip((v * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)].copy()
                img[..., 3] = int(255 * rop.get_value())
                return img
            return fn
        imT, fT = self.live_image(relief_img(Ut), RW, cT, RH)
        imF, fF = self.live_image(relief_img(Uf), RW, cF, RH)

        def outline(c_):
            px = lambda x: c_[0] - RW / 2 + (x - X0r) / (X1r - X0r) * RW
            py = lambda y: c_[1] - RH / 2 + (y - Y0r) / (Y1r - Y0r) * RH
            pts = [[px(X0r), py(0), 0]]
            for k in range(int(round((X1r - X0r) / Lam))):
                pts += [[px(k * Lam), py(0), 0], [px((k + 1) * Lam), py(hh), 0], [px((k + 1) * Lam), py(0), 0]]
            g = VGroup(VMobject().set_points_as_corners(pts).set_stroke(WHITE, 2.5),
                       Rectangle(width=RW, height=RH).move_to(c_).set_stroke(GREY_B, 1.5))
            return g
        oT, oF = outline(cT), outline(cF)
        hT = Tex("thin element (TEA)", font_size=30, color=GREY_A).next_to(oT[1], UP, buff=0.12)
        hF = Tex("FDTD (full Maxwell)", font_size=30, color=YELLOW).next_to(oF[1], UP, buff=0.12)
        glass_l = Tex("glass", font_size=24, color=GREY_A).move_to(cF + np.array([RW / 2 - 0.45, -RH / 2 + 0.25, 0]))
        # +1 order efficiency bars
        eff = dict(zip([int(m_) for m_ in r["orders_m"]], r["orders_eff"]))
        bx0, by0, bh = 4.75, -2.55, 3.4
        bars = VGroup()
        for i, (v_, col, lab) in enumerate(((float(r["T_flat"]), GREY_A, "TEA"), (eff[1], YELLOW, "FDTD"))):
            x_ = bx0 + i * 1.15
            bars.add(VGroup(Rectangle(width=0.62, height=bh * v_).move_to([x_, by0 + bh * v_ / 2, 0]).set_stroke(width=0).set_fill(col, 0.85),
                            Tex(f"{100 * v_:.0f}\\%", font_size=30, color=col).move_to([x_, by0 + bh * v_ + 0.25, 0]),
                            Tex(lab, font_size=24, color=col).move_to([x_, by0 - 0.22, 0])))
        bt = VGroup(Tex("light sent", font_size=24, color=GREY_A), Tex("the design way", font_size=24, color=GREY_A)).arrange(DOWN, buff=0.06)
        bt.move_to([bx0 + 0.57, 1.9, 0])
        self.at(S("a6"))
        self.play(FadeOut(VGroup(cost, rungs, times, arrow_l, al, own)), run_time=0.5)
        self.add(imT, imF)
        self.play(rop.animate.set_value(1.0), fT.animate.set_value(1.0), fF.animate.set_value(1.0), FadeIn(oT), FadeIn(oF), FadeIn(hT),
                  FadeIn(hF), FadeIn(glass_l), run_time=0.8)
        self.at(S("a6") + 4.3)
        self.play(FadeIn(bt), FadeIn(bars[0]), run_time=0.7)
        self.at(S("a6") + 10.2)
        self.play(FadeIn(bars[1]), run_time=0.7)
        self.finish()


# ============================================================ 7  recap ============
class Ch7Recap(Chapter):
    CH = 7
    TAG = "7\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---- r1: honest limits
        c1, c2, c3 = np.array([-4.4, 0.45, 0]), np.array([0.0, 0.45, 0]), np.array([4.4, 0.45, 0])
        coarse = VGroup(*[Square(0.5).move_to(c1 + np.array([-1.1 + 0.5 * i - 0.25, -0.5 + 0.5 * j, 0])) for i in range(3) for j in range(3)])
        fine = VGroup(*[Square(0.25).move_to(c1 + np.array([0.6 + 0.25 * i, -0.75 + 0.25 * j, 0])) for i in range(6) for j in range(6)])
        for g_ in (coarse, fine):
            g_.set_stroke(GREY_B, 1.2)
        i1 = VGroup(coarse, fine, Arrow(c1 + np.array([-0.3, 0, 0]), c1 + np.array([0.45, 0, 0]), buff=0, stroke_width=3, color=GREY_A,
                                        max_tip_length_to_length_ratio=0.3))
        circ = Circle(radius=0.95).move_to(c2).set_stroke(WHITE, 2.5)
        cells2 = VGroup()
        cs2 = 0.19
        for i in range(-6, 6):
            for j in range(-6, 6):
                cx, cy = (i + 0.5) * cs2, (j + 0.5) * cs2
                if abs(np.hypot(cx, cy) - 0.95) < 0.11:
                    cells2.add(Square(cs2).move_to(c2 + np.array([cx, cy, 0])).set_stroke(width=0).set_fill(C_GRID, 0.75))
        i2 = VGroup(cells2, circ)
        i3 = VGroup(ribbon_icon(c3 + 0.1 * UP, 2.0, 1.1))
        l1 = Tex("converges only as", font_size=28, color=GREY_A), Tex("the cells shrink", font_size=28, color=GREY_A)
        l2 = Tex("curves become", font_size=28, color=GREY_A), Tex("staircases", font_size=28, color=GREY_A)
        l3 = Tex("the PML adds", font_size=28, color=GREY_A), Tex("a small error", font_size=28, color=GREY_A)
        labs = VGroup(*[VGroup(*l_).arrange(DOWN, buff=0.08).move_to(c_ + np.array([0, -1.5, 0])) for l_, c_ in ((l1, c1), (l2, c2), (l3, c3))])
        top = Tex("FDTD isn't exact either", font_size=36).move_to([0.6, 2.75, 0])
        self.at(S("r1"))
        self.play(FadeIn(top), run_time=0.6)
        self.at(S("r1") + 1.6)
        self.play(FadeIn(i1), FadeIn(labs[0]), run_time=0.7)
        self.at(S("r1") + 3.9)
        self.play(Create(circ), FadeIn(cells2), FadeIn(labs[1]), run_time=0.8)
        self.at(S("r1") + 5.9)
        self.play(FadeIn(i3), FadeIn(labs[2]), run_time=0.7)

        # ---- r2/r3: the idea in two lines
        phase = lambda: (self.now * 1.6) % 2.0
        yop = ValueTracker(0.0)

        def yee_live():
            o_ = yop.get_value()
            if o_ < 0.01:
                return VGroup()
            ph = phase()
            on_h = 1.0 if ph < 1.0 else 0.35
            on_e = 0.35 if ph < 1.0 else 1.0
            g = yee_icon([-4.8, 1.75, 0], 0.55, 4, 3, on_e * o_, on_h * o_)
            return g
        yl = always_redraw(yee_live)
        line1 = Tex("$E$", " updates ", "$H$", ", and ", "$H$", " updates ", "$E$", ",", font_size=44)
        for i_, c_ in ((0, C_E), (2, C_HF), (4, C_HF), (6, C_E)):
            line1[i_].set_color(c_)
        line1b = Tex("one neighbour at a time.", font_size=44)
        L1 = VGroup(line1, line1b).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([0.9, 1.75, 0])
        line2 = Tex("A layer of complex-stretched space", font_size=44)
        line2b = Tex("absorbs whatever leaves.", font_size=44)
        L2 = VGroup(line2, line2b).arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(L1, DOWN, buff=0.55).align_to(L1, LEFT)
        rib2 = ribbon_icon([-4.8, L2.get_center()[1], 0], 2.0, 1.1)
        self.at(S("r2"))
        self.play(FadeOut(VGroup(top, i1, i2, i3, labs)), run_time=0.6)
        self.add(yl)
        self.play(yop.animate.set_value(1.0), Write(line1), run_time=1.6)
        self.play(FadeIn(line1b), run_time=0.7)
        self.at(S("r3"))
        self.play(FadeIn(rib2), Write(line2), run_time=1.4)
        self.play(FadeIn(line2b), run_time=0.6)

        # ---- r4: teaser
        nc = next_card("designing light with gradients: the adjoint method", y=-2.45)
        r1_ = Tex("today: structure $\\;\\longrightarrow\\;$ light", font_size=32, color=GREY_A)
        r2_ = Tex("wanted: light $\\;\\longrightarrow\\;$ structure", font_size=32, color=YELLOW)
        tz = VGroup(r1_, r2_).arrange(RIGHT, buff=1.2).move_to([0, -1.35, 0])
        self.at(S("r4") + 0.6)
        self.play(FadeIn(r1_), run_time=0.7)
        self.at(S("r4") + 2.4)
        self.play(FadeIn(r2_), run_time=0.7)
        self.at(S("r4") + 4.0)
        self.play(FadeIn(nc, shift=0.1 * UP), run_time=0.9)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Maxwell", "Ch2Yee", "Ch3GridRules", "Ch4Boundary", "Ch5PML", "Ch6Computer", "Ch7Recap"]
