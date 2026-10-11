"""Flat Optics: a Lens One Micron Thick -- chapter scenes (Optics in 5 Minutes, E06).
Render: ./render.sh ; preview: ./preview.sh Ch0Hook

Every physical field is precomputed by physics.py (TEA + angular-spectrum propagation, a 2D FDFD Maxwell solver)
and only loaded here. Scene rules: FAST_ANIMATION_PIPELINE.md section 5 (state = function of self.now and plays).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403
from o5common import DIVERGE, HEAT  # noqa: E402


def _at_whole_frames(self, t):
    """Chapter.at() that waits a whole number of frames (FAST_ANIMATION_PIPELINE.md section 10, seam fix)."""
    x = (t - self.now) * config.frame_rate                # frames still to go
    if x > 0.5 + 1e-6:
        k = int(np.ceil(x - 1e-6))
        self.wait((k - 0.5) / config.frame_rate)          # exactly k frames on every clock
    elif x < -0.5 - 1e-6:
        self.log.append(f"LATE by {-x / config.frame_rate:.2f}s at planned {t:.2f}")


Chapter.at = _at_whole_frames

from PIL import Image  # noqa: E402

GLASS_FILL, GLASS_OP, GLASS_LINE = "#7f93a3", 0.55, "#d5dde3"
SUB_FILL = "#5d6d78"
C_R, C_G, C_B = "#ff5a4a", "#7ddc4a", "#4a7bff"          # red / green / blue light (wavelength colours)
MU = "\\mu\\mathrm{m}"


def PPU():
    return config.pixel_height / config.frame_height


def fit_rgba(rgba, w, h):
    """Resample an RGBA image once to its on-screen pixel size (cheap per-frame drawing, crisp result)."""
    W, H = max(1, int(round(w * PPU()))), max(1, int(round(h * PPU())))
    if rgba.shape[1] == W and rgba.shape[0] == H:
        return np.ascontiguousarray(rgba)
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgba), "RGBA").resize((W, H), Image.LANCZOS))


def rgb2rgba(rgb, flip=True):
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = rgb[::-1] if flip else rgb
    out[..., 3] = 255
    return out


def heat_img(A, vmax, gamma=0.5):
    return heat_rgba(A, vmax, gamma=gamma, flip=True)


def glassify(m, op=GLASS_OP, width=2.0):
    return m.set_fill(GLASS_FILL, op).set_stroke(GLASS_LINE, width)


def resample_complex(U, W, H):
    """Bilinear resample of a complex field (rows: top -> bottom) to W x H pixels."""
    from scipy.ndimage import map_coordinates
    ny, nx = U.shape
    yy = (np.arange(H) + 0.5) / H * ny - 0.5
    xx = (np.arange(W) + 0.5) / W * nx - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    re = map_coordinates(U.real, [Y, X], order=1, mode="nearest")
    im = map_coordinates(U.imag, [Y, X], order=1, mode="nearest")
    return re + 1j * im


class LiveWave:
    """Re{U e^(-i omega t)} of a precomputed complex field, drawn with the series' diverging map.
    U: complex array, row 0 = top of the picture. Amplitude is compressed above 1 (focus) like the metasurface film."""

    def __init__(self, U, w, h, omega, amp_ref=None, comp=0.55, intensity_vmax=None):
        W, H = max(1, int(round(w * PPU()))), max(1, int(round(h * PPU())))
        Uc = resample_complex(U, W, H)
        A = np.abs(Uc)
        g = np.where(A <= 1.0, A, 1.0 + comp * np.log(np.maximum(A, 1.0)))
        ref = amp_ref if amp_ref is not None else g.max()
        self.amp = (g / ref).astype(np.float32)
        self.ph = np.angle(Uc).astype(np.float32)
        self.omega = omega
        self.rgba = np.empty((H, W, 4), np.uint8)
        self.rgba[..., 3] = 255
        self.heat = None
        if intensity_vmax is not None:
            v = np.clip(A ** 2 / intensity_vmax, 0, 1) ** 0.6
            self.heat = HEAT[(v * 1023).astype(np.int32)].astype(np.float32)
        self.rows = np.arange(H)[::-1] / max(H - 1, 1)                    # 0 at the bottom row, 1 at the top

    def frame(self, t, sweep=1.0, heat=0.0, soft=0.015):
        v = self.amp * np.cos(self.ph - np.float32(self.omega * t))
        idx = np.clip((v * 0.5 + 0.5) * 1023, 0, 1023).astype(np.int32)
        rgb = DIVERGE[idx]
        if heat > 0.0 and self.heat is not None:
            rgb = ((1.0 - heat) * rgb + heat * self.heat).astype(np.uint8)
        self.rgba[..., :3] = rgb
        if sweep < 1.0:
            self.rgba[..., 3] = (255 * np.clip((sweep - self.rows) / soft, 0, 1))[:, None].astype(np.uint8)
        else:
            self.rgba[..., 3] = 255
        return self.rgba


def lens_outline(r, h, cx, base, sx, sy):
    """Closed polygon of a rotationally symmetric profile h(r) (cross-section), r >= 0 samples ascending."""
    xs = np.concatenate([-r[::-1], r])
    ys = np.concatenate([h[::-1], h])
    pts = [[cx + x * sx, base + y * sy, 0] for x, y in zip(xs, ys)]
    pts = [[cx - r[-1] * sx, base, 0]] + pts + [[cx + r[-1] * sx, base, 0]]
    return Polygon(*pts)


def model_ladder(center, lit=0, width=4.6, size=26):
    """The series' model ladder (E06-E10): rungs from fast/approximate (bottom) to exact/slow (top)."""
    names = ["thin element (TEA)", "multi-slice", "periodic (gratings)", "full Maxwell"]
    eps = ["E06", "E08", "E07", "E09"]
    c = np.asarray(center, float)
    rungs = VGroup()
    for i, (nm, ep) in enumerate(zip(names, eps)):
        y = c[1] - 0.84 + 0.56 * i
        box = RoundedRectangle(corner_radius=0.08, width=width, height=0.44).move_to([c[0], y, 0])
        box.set_stroke(GREY_B, 1.5).set_fill(BLACK, 0.85)
        t = Tex(nm, font_size=size, color=GREY_A).move_to(box).align_to(box, LEFT).shift(0.18 * RIGHT)
        e = Tex(ep, font_size=22, color=GREY_B).move_to(box).align_to(box, RIGHT).shift(0.15 * LEFT)
        rungs.add(VGroup(box, t, e))
    rails = VGroup(Line(rungs[0][0].get_corner(DL) + 0.12 * LEFT, rungs[-1][0].get_corner(UL) + 0.12 * LEFT),
                   ).set_stroke(GREY_C, 2)
    arrow = Arrow(rungs[0][0].get_corner(DR) + 0.25 * RIGHT + 0.05 * UP, rungs[-1][0].get_corner(UR) + 0.25 * RIGHT,
                  buff=0, stroke_width=3, color=GREY_B, max_tip_length_to_length_ratio=0.12)
    lab = Tex("slower,\\\\more exact", font_size=22, color=GREY_B).next_to(arrow, RIGHT, buff=0.1)
    g = VGroup(rungs, rails, arrow, lab)
    return g


def light_rung(ladder, i, color=YELLOW):
    rungs = ladder[0]
    anims = []
    for j, r in enumerate(rungs):
        on = j == i
        anims += [r[0].animate.set_stroke(color if on else GREY_B, 3 if on else 1.5),
                  r[1].animate.set_color(color if on else GREY_A)]
    return anims


# ============================================================ 0  hook ==============
class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        S = self.S
        L = np.load(data_path("lens.npz"))
        xs, zs = L["xs"], L["zs"]
        a, sag, h2 = float(L["a"]), float(L["sag"]), float(L["h2pi"])
        sc = 0.0722                                   # scene units per micron (true scale)
        W, Hf = (xs[-1] - xs[0]) * sc, (zs[-1] - zs[0]) * sc
        ye = -2.0                                     # exit plane of both elements
        cx = (-2.55, 2.55)
        r = np.linspace(0, a, 400)
        rk = np.sort(np.concatenate([np.linspace(0, a, 1600), L["edges"][1:-1] - 1e-6, L["edges"][1:-1] + 1e-6]))
        ht = np.clip((sag - (np.sqrt(r ** 2 + float(L["f0"]) ** 2) - float(L["f0"])) / 0.5), 0, None)
        hk = np.mod(np.clip((sag - (np.sqrt(rk ** 2 + float(L["f0"]) ** 2) - float(L["f0"])) / 0.5), 0, None), h2)
        lensL = glassify(lens_outline(r, ht, cx[0], ye - sag * sc, sc, sc))
        lensR = glassify(lens_outline(rk, hk, cx[1], ye - h2 * sc, sc, sc), width=1.2)
        # true-scale panels: green light (550 nm), then white light (computed, TEA + 2D ASM)
        g_img = fit_rgba(rgb2rgba(L["rgb_g"]), W, Hf)
        imgs = {}
        for k, key in enumerate(("t", "k")):
            c = np.array([cx[k], ye + Hf / 2, 0])
            imgs["g" + key] = self.sweep_image(g_img, W, c, Hf, "up")
            w_img = fit_rgba(rgb2rgba(L["rgb_" + key]), W, Hf)
            imgs["w" + key] = self.static_image(w_img, W, c, Hf)
        frames = VGroup(*[Rectangle(width=W, height=Hf).move_to([c_, ye + Hf / 2, 0]).set_stroke(GREY_D, 1) for c_ in cx])
        tL = Tex("glass lens", font_size=34, color=GREY_A).move_to([cx[0], 3.1, 0])
        tR = Tex("patterned plate", font_size=34, color=GREY_A).move_to([cx[1], 3.1, 0])
        # thickness labels
        bL = Brace(Line([0, ye - sag * sc, 0], [0, ye, 0]), RIGHT, buff=0.05).next_to(lensL, RIGHT, buff=0.08)
        lL = MathTex("\\approx 9\\," + MU, font_size=30).next_to(bL, RIGHT, buff=0.1)
        lR = MathTex("\\approx 1\\," + MU, font_size=30).move_to([cx[1] + a * sc + 0.75, ye - 0.55, 0])
        tick = Arrow(lR.get_corner(UL) + 0.05 * RIGHT, [cx[1] + a * sc + 0.02, ye - 0.5 * h2 * sc, 0], buff=0.05,
                     stroke_width=2.5, color=WHITE, max_tip_length_to_length_ratio=0.3)
        # incoming light below each element: crest lines moving up (pure function of time)
        col = ValueTracker(0.0)                       # 0 = green, 1 = white
        lop = ValueTracker(0.0)
        lam_d, v_d = 0.16, 0.22
        y_lo = -3.15

        def crests():
            g = VGroup()
            o = lop.get_value()
            if o <= 0.002:
                return g
            c_ = interpolate_color(ManimColor("#9be36b"), ManimColor(WHITE), col.get_value())
            ph = (v_d * self.now) % lam_d
            for k, top in enumerate((ye - sag * sc, ye - h2 * sc)):
                y = y_lo + ph
                while y < top - 0.02:
                    f = np.clip((y - y_lo) / 0.25, 0, 1)
                    g.add(Line([cx[k] - a * sc, y, 0], [cx[k] + a * sc, y, 0]).set_stroke(c_, 2.5, opacity=0.75 * o * f))
                    y += lam_d
            return g
        cr = always_redraw(crests)
        # ---- h1: two elements, their thicknesses
        self.add(cr)
        self.play(FadeIn(lensL), FadeIn(lensR), FadeIn(tL), FadeIn(tR), run_time=0.8)
        self.at(S("h1") + 1.0)
        self.play(GrowFromCenter(bL), FadeIn(lL), Indicate(lensL, color=WHITE, scale_factor=1.03), run_time=0.9)
        self.at(S("h1") + 3.4)
        self.play(GrowArrow(tick), FadeIn(lR), Indicate(lensR, color=WHITE, scale_factor=1.05), run_time=0.9)
        # ---- h2: green light, same focus
        mgl, fgl, sgl = imgs["gt"]
        mgr, fgr, sgr = imgs["gk"]
        self.at(S("h2"))
        self.add(mgl, mgr)
        self.play(lop.animate.set_value(1.0), FadeIn(frames), fgl.animate.set_value(1.0), fgr.animate.set_value(1.0),
                  run_time=0.6)
        self.play(sgl.animate.set_value(1.0), sgr.animate.set_value(1.0), run_time=2.4, rate_func=linear)
        zf = ye + float(zs[np.argmax(L["I_k"][:, L["I_k"].shape[1] // 2])]) * sc
        fl = DashedLine([cx[0] - W / 2 - 0.1, zf, 0], [cx[1] + W / 2 + 0.1, zf, 0], dash_length=0.1).set_stroke(WHITE, 2, opacity=0.8)
        same = Tex("same\\\\focus", font_size=28).move_to([0, zf + 0.45, 0])
        self.play(Create(fl), FadeIn(same), run_time=0.7)
        # ---- h3: white light
        mwl, fwl = imgs["wt"]
        mwr, fwr = imgs["wk"]
        self.at(S("h3") + 0.1)
        self.add(mwl, mwr)
        self.play(fwl.animate.set_value(1.0), fwr.animate.set_value(1.0), fgl.animate.set_value(0.0), fgr.animate.set_value(0.0),
                  col.animate.set_value(1.0), FadeOut(fl), FadeOut(same), run_time=1.4)
        for m_ in (mgl, mgr):
            m_.clear_updaters()
            self.remove(m_)
        zf2 = ye + float(zs[np.argmax(L["ax_t_G"])]) * sc
        sharp = Tex("one sharp\\\\focus", font_size=30).move_to([cx[0] - W / 2 - 0.95, zf2, 0])
        smear = Tex("colours\\\\smeared", font_size=30).move_to([cx[1] + W / 2 + 0.95, zf2, 0])
        zr = ye + float(zs[np.argmax(L["ax_k_R"])]) * sc
        zb_ = ye + float(zs[np.argmax(L["ax_k_B"])]) * sc
        span = DoubleArrow([cx[1] + 0.55, zr, 0], [cx[1] + 0.55, zb_, 0], buff=0, stroke_width=3, tip_length=0.14, color=WHITE)
        self.at(S("h3") + 2.2)
        self.play(FadeIn(sharp, shift=0.1 * RIGHT), run_time=0.6)
        self.at(S("h3") + 4.3)
        self.play(GrowFromCenter(span), FadeIn(smear, shift=0.1 * LEFT), run_time=0.8)
        # ---- h4: the question, then the title card
        title = series_title(6, "Flat Optics", "a lens one micron thick", y=0.6)
        self.at(S("h4") - 0.1)
        self.play(fwl.animate.set_value(0.35), fwr.animate.set_value(0.35), FadeOut(VGroup(sharp, smear, span, bL, lL, tick, lR)),
                  lop.animate.set_value(0.3), FadeIn(title[0]), FadeIn(title[1][0]), run_time=0.6)
        self.play(Write(title[1][1]), run_time=1.1)
        self.play(FadeIn(title[1][2], shift=0.1 * UP), run_time=0.5)
        self.finish()

# ============================================================ 1  glass as a delay ==
class Ch1Delay(Chapter):
    CH = 1
    TAG = "1\\quad Glass as a delay"

    def construct(self):
        S = self.S
        self.show_tag()
        # ---------------- d1-d3: two paths, crests in glass bunch up and fall behind (analytic phases)
        n = 1.5
        lam_d, v = 0.6, 0.55
        k = TAU_ / lam_d
        om = k * v
        ybot, ytop = -2.75, 2.25
        yb, hb = -1.15, 1.6
        xB, xA, cw = -4.3, -2.5, 1.2                     # air column (left), glass column (right)
        T_ENTER = S("d2") + 3.0                           # highlighted crest reaches the block
        T1, DEC = S("d3") + 0.4, 0.8                       # then the wave clock slows to a stop ...
        T2, ACC = S("d3") + 5.3, 1.0                       # ... and starts again once the lag has been read

        def tau(t):
            if t <= T1:
                return t
            u = min(t - T1, DEC)
            ts = T1 + u - u * u / (2 * DEC)
            if t <= T2:
                return ts
            w = t - T2
            return ts + (w * w / (2 * ACC) if w <= ACC else ACC / 2 + (w - ACC))
        Pb = k * (yb - ybot)
        P2 = Pb + n * k * hb
        phi0 = Pb - om * T_ENTER

        def y_air(P):
            return ybot + P / k

        def y_glass(P):
            if P < Pb:
                return ybot + P / k
            if P < P2:
                return yb + (P - Pb) / (n * k)
            return yb + hb + (P - P2) / k
        Ptop = {0: k * (ytop - ybot), 1: P2 + k * (ytop - yb - hb)}
        w_op, hl = ValueTracker(0.0), ValueTracker(0.0)

        def crests():
            g = VGroup()
            o = w_op.get_value()
            if o < 0.002:
                return g
            base = om * tau(self.now) + phi0
            for ci, (xc, yf) in enumerate(((xB, y_air), (xA, y_glass))):
                for m in range(int(np.ceil((base - Ptop[ci]) / TAU_)), int(np.floor(base / TAU_)) + 1):
                    y = yf(base - TAU_ * m)
                    f = o * np.clip((y - ybot) / 0.3, 0, 1) * np.clip((ytop - y) / 0.3, 0, 1)
                    if m == 0:
                        c_ = interpolate_color(ManimColor(C_U), ManimColor(YELLOW), hl.get_value())
                        g.add(Line([xc - cw / 2 + 0.04, y, 0], [xc + cw / 2 - 0.04, y, 0]).set_stroke(c_, 3 + 3 * hl.get_value(), opacity=f))
                    else:
                        g.add(Line([xc - cw / 2 + 0.04, y, 0], [xc + cw / 2 - 0.04, y, 0]).set_stroke(C_U, 3, opacity=f))
            return g
        cr = always_redraw(crests)
        cols = VGroup(*[Rectangle(width=cw, height=ytop - ybot).move_to([x_, 0.5 * (ytop + ybot), 0]).set_stroke(GREY_D, 1.5)
                        for x_ in (xB, xA)])
        block = glassify(Rectangle(width=cw, height=hb).move_to([xA, yb + hb / 2, 0]), op=0.45)
        up = VGroup(*[Arrow([x_, ybot - 0.02, 0], [x_, ybot + 0.45, 0], buff=0, stroke_width=3, color=GREY_B,
                            max_tip_length_to_length_ratio=0.35) for x_ in (xB, xA)])
        lab_air = Tex("air", font_size=30, color=GREY_A).next_to(cols[0], UP, buff=0.15)
        lab_gl = Tex("glass", font_size=30, color=GREY_A).next_to(cols[1], UP, buff=0.15)
        sp_air = MathTex("\\text{speed } c", font_size=30, color=GREY_A).next_to(lab_air, UP, buff=0.12)
        sp_gl = MathTex("\\text{speed } c/n", font_size=30, color=GREY_A).next_to(lab_gl, UP, buff=0.12)
        nlab = MathTex("n \\approx 1.5", font_size=34).next_to(block, RIGHT, buff=0.3)
        self.at(S("d1"))
        self.add(cr)
        self.play(FadeIn(cols), FadeIn(block), FadeIn(lab_air), FadeIn(lab_gl), w_op.animate.set_value(1.0), run_time=0.9)
        self.at(S("d1") + 1.8)
        self.play(FadeIn(sp_air), FadeIn(sp_gl), run_time=0.7)
        self.at(S("d1") + 3.2)
        self.play(Write(nlab), run_time=0.8)
        # d2: follow two paths (highlighted crest pair), height h
        hbr = Brace(block, LEFT, buff=0.06)
        hlab = MathTex("h", font_size=36).next_to(hbr, LEFT, buff=0.06)
        self.at(S("d2"))
        self.play(hl.animate.set_value(1.0), Indicate(cols, color=WHITE, scale_factor=1.0), run_time=0.8)
        self.at(S("d2") + 2.5)
        self.play(GrowFromCenter(hbr), FadeIn(hlab), run_time=0.6)
        # d3: the lag (n-1)h, as a phase k0 (n-1) h
        self.at(S("d3") + 1.3)
        base = om * tau(T1 + DEC) + phi0
        ya, yg = y_air(base), y_glass(base)
        gl = VGroup(fast_dashed([xB + cw / 2, ya, 0], [-0.8, ya, 0], 0.07, YELLOW, 2),
                    fast_dashed([xA + cw / 2, yg, 0], [-0.8, yg, 0], 0.07, YELLOW, 2))
        lag = BraceBetweenPoints([-0.85, yg, 0], [-0.85, ya, 0], RIGHT)
        lag_l = MathTex("(n-1)\\,h", font_size=36, color=YELLOW).next_to(lag, RIGHT, buff=0.1)
        self.play(Create(gl), GrowFromCenter(lag), FadeIn(lag_l), run_time=0.9)
        eq = MathTex("\\varphi", "=", "k_0", "(n-1)\\,h", font_size=58).move_to([3.6, 0.95, 0])
        eq[3].set_color(YELLOW)
        k0 = MathTex("k_0 = 2\\pi/\\lambda", font_size=40, color=GREY_A).next_to(eq, DOWN, buff=0.45)
        ph_l = Tex("phase delay", font_size=30, color=GREY_B).next_to(eq, UP, buff=0.35)
        self.at(S("d3") + 3.6)
        self.play(TransformFromCopy(lag_l, eq[3]), FadeIn(eq[:3]), FadeIn(ph_l), run_time=1.2)
        self.at(S("d3") + 4.95)
        self.play(FadeOut(VGroup(gl, lag)), lag_l.animate.set_opacity(0.0), run_time=0.4)
        self.at(S("d3") + 8.4)
        self.play(FadeIn(k0, shift=0.1 * UP), run_time=0.7)

        # ---------------- d4-d5: a lens, column by column (TEA), computed field (TEA columns + ASM)
        T = np.load(data_path("tea_demo.npz"))
        U = T["U"][::-1]                                  # rows: top -> bottom
        lamd = 0.2
        xs, zs = T["xs"], T["zs"]
        Wd, Hd = (xs[-1] - xs[0] + xs[1] - xs[0]) * lamd, (zs[-1] - zs[0] + zs[1] - zs[0]) * lamd
        P0 = np.array([-3.9, -0.45, 0])
        y0 = P0[1] - Hd / 2 - zs[0] * lamd + 0.5 * (zs[1] - zs[0]) * lamd * 0   # z = 0 (lens base)
        y0 = P0[1] - Hd / 2 + (0 - (zs[0] - 0.5 * (zs[1] - zs[0]))) * lamd
        lw = LiveWave(U, Wd, Hd, omega=TAU_ * 1.2, amp_ref=1.9)
        sweep, f_op = ValueTracker(0.0), ValueTracker(0.0)
        wave, wfade = self.live_image(lambda t: lw.frame(t, sweep.get_value()), Wd, P0, Hd)
        box = Rectangle(width=Wd, height=Hd).move_to(P0).set_stroke(GREY_D, 1.5)
        a, hmax = float(T["a"]), float(T["hmax"])
        rr = np.linspace(0, a, 300)
        hh = np.clip(hmax - (np.sqrt(rr ** 2 + float(T["f"]) ** 2) - float(T["f"])) / (n - 1), 0, None)
        lens = lens_outline(rr, hh, P0[0], y0, lamd, lamd).set_fill(GLASS_FILL, 0.22).set_stroke(GLASS_LINE, 2.5)
        cl_x = np.linspace(-a + 1.0, a - 1.0, 9)
        clines = VGroup(*[fast_dashed([P0[0] + x_ * lamd, y0 - 0.25, 0], [P0[0] + x_ * lamd, y0 + hmax * lamd + 0.3, 0], 0.06, WHITE, 1.5, 0.7)
                          for x_ in cl_x])
        self.at(S("d4"))
        self.play(FadeOut(VGroup(cols, block, lab_air, lab_gl, sp_air, sp_gl, nlab, hbr, hlab, gl, lag, lag_l, k0, ph_l)),
                  w_op.animate.set_value(0.0), eq.animate.scale(40 / 58).move_to([3.4, 1.3, 0]), run_time=0.6)
        self.remove(cr)
        self.add(wave)
        self.play(wfade.animate.set_value(1.0), FadeIn(box), FadeIn(lens), run_time=0.4)
        self.play(sweep.animate.set_value((hmax + 3.0 + 1.2) / 25.0), LaggedStart(*[Create(c_) for c_ in clines], lag_ratio=0.08),
                  run_time=1.3, rate_func=linear)
        uo = MathTex("U_{\\rm out}(x)", "=", "U_{\\rm in}(x)", "\\,e^{\\,i\\varphi(x)}", font_size=46).move_to([3.4, 2.4, 0])
        uo[0].set_color(C_U)
        uo[2].set_color(C_U)
        uo[3].set_color(YELLOW)
        eqx = MathTex("\\varphi(x)", "=", "k_0", "(n-1)\\,h(x)", font_size=40).move_to([3.4, 1.3, 0])
        eqx[3].set_color(YELLOW)
        self.at(S("d4") + 4.0)
        self.play(Write(uo), TransformMatchingTex(eq, eqx), run_time=1.6)
        # d5: TEA; thickest in the middle -> curved wave -> focus
        tea = Tex("thin element approximation (TEA)", font_size=32, color=YELLOW).move_to([3.4, 0.35, 0])
        self.at(S("d5") + 0.6)
        self.play(FadeIn(tea, shift=0.1 * UP), run_time=0.7)
        self.at(S("d5") + 3.4)
        self.play(Indicate(lens, color=WHITE, scale_factor=1.04), run_time=0.9)
        self.at(S("d5") + 5.0)
        self.play(sweep.animate.set_value(1.0), run_time=2.8, rate_func=linear)
        zf = float(zs[np.argmax(np.abs(T["U"][:, len(xs) // 2]))])
        fdot = Dot([P0[0], y0 + zf * lamd, 0], radius=0.07, color=WHITE)
        flab = Tex("focus", font_size=30).move_to([P0[0] + Wd / 2 + 0.6, y0 + zf * lamd, 0])
        far = Arrow(flab.get_left() + 0.05 * LEFT, fdot.get_right() + 0.06 * RIGHT, buff=0, stroke_width=3, color=WHITE,
                    max_tip_length_to_length_ratio=0.12)
        self.play(FadeIn(fdot, scale=0.5), FadeIn(flab), GrowArrow(far), run_time=0.6)
        # d6: the ladder of models; remember: every point on its own
        lad = model_ladder([2.9, -1.85, 0], width=3.8)
        self.at(S("d6") + 0.3)
        self.play(FadeIn(lad, shift=0.15 * UP), run_time=0.9)
        self.play(*light_rung(lad, 0), run_time=0.6)
        self.at(S("d6") + 4.2)
        self.play(clines.animate.set_stroke(YELLOW, 2.5, opacity=1.0), run_time=0.6)
        self.play(clines.animate.set_stroke(WHITE, 1.5, opacity=0.7), run_time=0.8)
        self.finish()

# ============================================================ 2  wrap the phase ====
def slab_pieces(L, cx, base, sx, sy):
    """Thick lens cut into slabs of height lam0/(n-1): full slabs (layer m: |r| <= R(m+1)) and curved remainders."""
    a, f0, h2, nz = float(L["a"]), float(L["f0"]), float(L["h2pi"]), int(L["nzone"])
    sag = nz * h2
    R = lambda m: float(L["edges"][nz - m])            # radius where the lens height equals m slabs
    hfun = lambda r: (sag - (np.sqrt(r ** 2 + f0 ** 2) - f0) / 0.5)
    full, rem = [], []
    for m in range(nz):
        r1, r0 = R(m + 1), R(m)
        y_lo, y_hi = base + m * h2 * sy, base + (m + 1) * h2 * sy
        if r1 > 1e-9:
            full.append(Polygon([cx - r1 * sx, y_lo, 0], [cx + r1 * sx, y_lo, 0], [cx + r1 * sx, y_hi, 0], [cx - r1 * sx, y_hi, 0]))
        rr = np.linspace(r1, r0, 40)
        hh = np.clip(hfun(rr), m * h2, (m + 1) * h2)
        if r1 <= 1e-9:                                  # the cap: one piece
            xs_ = np.concatenate([-rr[::-1], rr])
            ys_ = np.concatenate([hh[::-1], hh])
            rem.append((m, Polygon(*[[cx + x_ * sx, base + y_ * sy, 0] for x_, y_ in zip(xs_, ys_)])))
        else:
            for sgn in (1, -1):
                pts = [[cx + sgn * r1 * sx, y_lo, 0]] + [[cx + sgn * r_ * sx, base + h_ * sy, 0] for r_, h_ in zip(rr, hh)]
                rem.append((m, Polygon(*pts)))
    return full, rem


class Ch2Wrap(Chapter):
    CH = 2
    TAG = "2\\quad Wrap the phase"

    def construct(self):
        S = self.S
        self.show_tag()
        L = np.load(data_path("lens.npz"))
        a, h2, nz, f0 = float(L["a"]), float(L["h2pi"]), int(L["nzone"]), float(L["f0"])
        # ---------------- w1: phase is an angle (dial + wave shifted by one wavelength)
        u = ValueTracker(0.0)
        dc, dr = np.array([-4.6, 1.2, 0]), 0.95
        lam_w, yw, x0w, x1w = 1.8, 1.2, -3.0, 6.0
        dial_c = Circle(radius=dr).move_to(dc).set_stroke(GREY_B, 2)

        def dial():
            ang = TAU_ * u.get_value()
            g = VGroup(fast_arrow(dc, dc + dr * np.array([np.cos(ang), np.sin(ang), 0]), YELLOW, 5, 0.2))
            if ang > 0.05:
                g.add(Arc(radius=0.4, start_angle=0, angle=min(ang, TAU_ - 1e-3), arc_center=dc).set_stroke(YELLOW, 3))
            return g
        dl = always_redraw(dial)
        ghost = graph_line(lambda x: 0.6 * np.cos(TAU_ / lam_w * (x - 0.3)), x0w, x1w, yw, 1, 1, color=GREY_B, width=2)
        ghost = DashedVMobject(ghost, num_dashes=90).set_stroke(GREY_B, 2, opacity=0.8)

        def wave():
            sh = u.get_value() * lam_w
            g = VGroup(graph_line(lambda x: 0.6 * np.cos(TAU_ / lam_w * (x - 0.3 - sh)), x0w, x1w, yw, 1, 1, color=C_U, width=4))
            xc = 0.3 + 1 * lam_w + sh
            g.add(Dot([xc, yw + 0.6, 0], radius=0.09, color=YELLOW))
            return g
        wv = always_redraw(wave)
        ang_l = Tex("phase is an angle", font_size=32, color=GREY_A).next_to(dial_c, DOWN, buff=0.3)
        turn_l = MathTex("\\text{one full turn} = 2\\pi", font_size=36, color=YELLOW).move_to([1.6, -0.35, 0])
        same_l = Tex("$\\Rightarrow$ nothing changes", font_size=36).next_to(turn_l, DOWN, buff=0.3)
        self.at(S("w1") + 0.3)
        self.add(dl, wv)
        self.play(Create(dial_c), FadeIn(dl), FadeIn(wv), run_time=0.8)
        self.at(S("w1") + 1.7)
        self.play(FadeIn(ang_l), FadeIn(ghost), run_time=0.6)
        self.at(S("w1") + 2.9)
        self.play(u.animate.set_value(1.0), run_time=2.2)
        self.play(FadeIn(turn_l), run_time=0.4)
        self.at(S("w1") + 5.5)
        self.play(FadeIn(same_l, shift=0.1 * UP), run_time=0.6)

        # ---------------- w2-w5: the signature shot -- slice the lens, drop the remainders
        cx, base, sx, sy = -1.2, -2.25, 0.18, 0.36       # heights drawn x2
        sh_ = h2 * sy                                    # drawn slab height
        sub = Rectangle(width=2 * a * sx + 0.7, height=0.24).move_to([cx, base - 0.12, 0]).set_fill(SUB_FILL, 1).set_stroke(GREY_B, 1.5)
        full, rem = slab_pieces(L, cx, base, sx, sy)
        BODY = "#46515a"
        for p in full + [q for _, q in rem]:
            p.set_fill(BODY, 1).set_stroke(BODY, 1.2)
        rr = np.linspace(0, a, 300)
        hh = np.clip((nz * h2 - (np.sqrt(rr ** 2 + f0 ** 2) - f0) / 0.5), 0, None)
        outline = lens_outline(rr, hh, cx, base, sx, sy).set_fill(opacity=0).set_stroke(GLASS_LINE, 2.5)
        ghost_l = DashedVMobject(outline.copy(), num_dashes=110).set_stroke(GREY_B, 1.5, opacity=0.6)
        exag = Tex("heights $\\times 2$", font_size=24, color=GREY_B).move_to([cx - a * sx - 0.35, base - 0.5, 0])
        lens_all = VGroup(*full, *[q for _, q in rem])
        self.at(S("w2"))
        self.play(FadeOut(VGroup(dial_c, ang_l, turn_l, same_l, ghost)), FadeOut(dl), FadeOut(wv), run_time=0.5)
        self.remove(dl, wv)
        self.play(FadeIn(sub), FadeIn(lens_all), Create(outline), FadeIn(exag), run_time=1.0)
        R = lambda m: float(L["edges"][nz - m])
        cut1 = Line([cx - R(1) * sx, base + sh_, 0], [cx + R(1) * sx, base + sh_, 0]).set_stroke(YELLOW, 2.5)
        br = BraceBetweenPoints([cx + a * sx + 0.12, base + sh_, 0], [cx + a * sx + 0.12, base, 0], RIGHT)
        hl = MathTex("\\lambda/(n-1)", font_size=38, color=YELLOW).next_to(br, RIGHT, buff=0.12)
        hum = MathTex("\\approx 1.1\\," + MU, font_size=34).next_to(hl, DOWN, buff=0.2).align_to(hl, LEFT)
        self.at(S("w2") + 2.2)
        self.play(GrowFromCenter(br), Write(hl), Create(cut1), run_time=1.0)
        self.at(S("w2") + 5.9)
        self.play(FadeIn(hum, shift=0.1 * UP), run_time=0.6)
        # w3: slice
        cuts = VGroup(*[DashedLine([cx - R(m) * sx, base + m * sh_, 0], [cx + R(m) * sx, base + m * sh_, 0], dash_length=0.09)
                        .set_stroke(WHITE, 2) for m in range(2, nz)])
        SHADE = "#5e6c78"
        layer_anims = [p.animate.set_fill(SHADE if (i % 2) else BODY, 1).set_stroke(SHADE if (i % 2) else BODY, 1.2) for i, p in enumerate(full)]
        layer_anims += [q.animate.set_fill(SHADE if (m % 2) else BODY, 1).set_stroke(SHADE if (m % 2) else BODY, 1.2) for m, q in rem]
        self.at(S("w3") + 0.1)
        self.play(cut1.animate.set_stroke(WHITE, 2), LaggedStart(*[Create(c_) for c_ in cuts], lag_ratio=0.15), run_time=1.6)
        self.play(*layer_anims, run_time=0.8)
        # w4: full slabs only add whole turns -> throw them away
        self.at(S("w4") + 0.2)
        self.play(*[p.animate.set_fill(C_EV, 0.55).set_stroke(C_EV, 1.5) for p in full], run_time=0.8)
        self.at(S("w4") + 2.7)
        for _, q in rem:
            q.set_fill(BODY, 1).set_stroke(GLASS_LINE, 2)
        self.add(ghost_l)
        self.play(*[FadeOut(p, shift=0.25 * UP, scale=0.9) for p in full], FadeOut(outline), FadeOut(cut1), FadeOut(cuts),
                  FadeOut(VGroup(br, hl, hum)), FadeIn(ghost_l), run_time=1.0)
        # w5: the remainders drop onto the base (gravity: time ~ sqrt(height))
        self.at(S("w5") + 1.5)
        drops = [q.animate(rate_func=rate_functions.ease_in_quad, run_time=0.45 + 1.0 * np.sqrt(m / (nz - 1))).shift(m * sh_ * DOWN)
                 for m, q in rem if m > 0]
        self.play(AnimationGroup(*drops))
        # w6: a kinoform; rings (computed top view); about one micron
        kin = VGroup(*[q for _, q in rem])
        kname = Tex("kinoform", font_size=44).move_to([4.7, -1.45, 0])
        kbr = BraceBetweenPoints([cx + a * sx + 0.12, base + sh_, 0], [cx + a * sx + 0.12, base, 0], RIGHT)
        kum = MathTex("\\approx 1.1\\," + MU, font_size=32).next_to(kbr, RIGHT, buff=0.1)
        top = fit_rgba(L["top"], 2.6, 2.6)
        tv, tvf = self.static_image(top, 2.6, np.array([4.6, 1.05, 0]), 2.6)
        tvl = Tex("seen from above", font_size=26, color=GREY_B).move_to([4.6, -0.5, 0])
        self.at(S("w6") + 0.4)
        self.play(Write(kname), run_time=0.8)
        self.at(S("w6") + 1.9)
        self.add(tv)
        self.play(tvf.animate.set_value(1.0), FadeIn(tvl), FadeOut(ghost_l), run_time=0.9)
        self.at(S("w6") + 3.5)
        self.play(GrowFromCenter(kbr), FadeIn(kum), run_time=0.7)
        # w7: zones narrow toward the edge (light bends most there); a few wavelengths at the rim
        lam0 = float(L["lam0"])
        yb_ = base - 0.3                                   # braces under the substrate
        bc = BraceBetweenPoints([cx + R(7) * sx, yb_, 0], [cx - R(7) * sx, yb_, 0], DOWN)
        bcl = MathTex("\\approx %d\\,\\lambda" % round(2 * R(7) / lam0), font_size=30).next_to(bc, DOWN, buff=0.06)
        brm = BraceBetweenPoints([cx + R(0) * sx, yb_, 0], [cx + R(1) * sx, yb_, 0], DOWN)
        brl = MathTex("\\approx %.1f\\,\\lambda" % ((R(0) - R(1)) / lam0), font_size=30, color=YELLOW).next_to(brm, DOWN, buff=0.06)
        th = np.arcsin(a / np.hypot(a, f0))
        y_r = base + 0.5 * sh_
        ray_c = Arrow([cx, base + sh_ + 0.05, 0], [cx, base + sh_ + 1.6, 0], buff=0, stroke_width=4, color=C_U,
                      max_tip_length_to_length_ratio=0.15)
        p_r = np.array([cx + 0.5 * (R(0) + R(1)) * sx, y_r, 0])
        ray_r = Arrow(p_r, p_r + 1.75 * np.array([-np.sin(th), np.cos(th), 0]), buff=0, stroke_width=4, color=C_U,
                      max_tip_length_to_length_ratio=0.15)
        p_l = np.array([cx - 0.5 * (R(0) + R(1)) * sx, y_r, 0])
        ray_l = Arrow(p_l, p_l + 1.75 * np.array([np.sin(th), np.cos(th), 0]), buff=0, stroke_width=4, color=C_U,
                      max_tip_length_to_length_ratio=0.15)
        self.at(S("w7") + 0.3)
        self.play(GrowFromCenter(bc), FadeIn(bcl), GrowFromCenter(brm), FadeIn(brl), run_time=0.9)
        self.at(S("w7") + 2.0)
        self.play(GrowArrow(ray_c), GrowArrow(ray_r), GrowArrow(ray_l), run_time=0.9)
        self.at(S("w7") + 4.3)
        self.play(Indicate(brl, scale_factor=1.3), run_time=0.9)
        # w8: the same factor e^{i phi} -> the same focus (computed)
        eqk = MathTex("e^{\\,i\\varphi_{\\rm kinoform}}", "=", "e^{\\,i\\varphi_{\\rm lens}}", "\\,e^{-i\\,2\\pi m}", "=",
                      "e^{\\,i\\varphi_{\\rm lens}}", font_size=44).move_to([0, 2.75, 0])
        eqk[3].set_color(YELLOW)
        one = MathTex("=1", font_size=32, color=YELLOW).next_to(eqk[3], DOWN, buff=0.12)
        mnote = Tex("$m$ = number of slabs removed", font_size=26, color=GREY_B).next_to(one, RIGHT, buff=0.5)
        self.at(S("w8") + 0.2)
        self.play(FadeOut(VGroup(bc, bcl, brm, brl, ray_c, ray_r, ray_l, tvl, kbr, kum)), tvf.animate.set_value(0.0), run_time=0.5)
        tv.clear_updaters()
        self.remove(tv)
        self.play(Write(eqk), run_time=1.6)
        self.play(FadeIn(one, shift=0.1 * UP), FadeIn(mnote), run_time=0.5)
        # two small true-scale panels: lens vs kinoform at 550 nm, plus the focal-plane profiles
        sc = 0.055
        xs, zs = L["xs"], L["zs"]
        W, Hf = (xs[-1] - xs[0]) * sc, (zs[-1] - zs[0]) * sc
        ye = -2.15
        pcx = (-4.6, -1.5)
        g_img = fit_rgba(rgb2rgba(L["rgb_g"]), W, Hf)
        pans = [self.sweep_image(g_img, W, np.array([c_, ye + Hf / 2, 0]), Hf, "up") for c_ in pcx]
        r2 = np.linspace(0, a, 300)
        rk = np.sort(np.concatenate([np.linspace(0, a, 1200), L["edges"][1:-1] - 1e-6, L["edges"][1:-1] + 1e-6]))
        ht = np.clip(nz * h2 - (np.sqrt(r2 ** 2 + f0 ** 2) - f0) / 0.5, 0, None)
        hk = np.mod(np.clip(nz * h2 - (np.sqrt(rk ** 2 + f0 ** 2) - f0) / 0.5, 0, None), h2)
        e1 = glassify(lens_outline(r2, ht, pcx[0], ye - nz * h2 * sc, sc, sc), width=1.5)
        e2 = glassify(lens_outline(rk, hk, pcx[1], ye - h2 * sc, sc, sc), width=1.0)
        fr = VGroup(*[Rectangle(width=W, height=Hf).move_to([c_, ye + Hf / 2, 0]).set_stroke(GREY_D, 1) for c_ in pcx])
        tl = VGroup(Tex("lens", font_size=28, color=GREY_A).next_to(fr[0], UP, buff=0.12),
                    Tex("kinoform", font_size=28, color=GREY_A).next_to(fr[1], UP, buff=0.12))
        # focal-plane profiles (computed): identical
        xp = xs
        sel = np.abs(xp) <= 4.0
        pk = L["prof_k"][sel] / L["prof_k"].max()
        pt = L["prof_t"][sel] / L["prof_k"].max()
        bxc, bw, bh = np.array([3.75, -0.55, 0]), 4.0, 2.7
        axp = axis_pair(bxc, bw, bh, xlab="x", ylab="I", size=28)
        c_t = plot_xy(xp[sel], pt, bxc, bw, bh, (-4, 4), (0, 1.08), color=WHITE, width=7)
        c_k = DashedVMobject(plot_xy(xp[sel], pk, bxc, bw, bh, (-4, 4), (0, 1.08), color=C_G, width=4), num_dashes=40)
        lg = VGroup(VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(WHITE, 6), Tex("lens", font_size=26)).arrange(RIGHT, buff=0.15),
                    VGroup(DashedLine(ORIGIN, 0.45 * RIGHT, dash_length=0.08).set_stroke(C_G, 4), Tex("kinoform", font_size=26)).arrange(RIGHT, buff=0.15)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(bxc + np.array([1.25, 0.85, 0]))
        ptl = Tex("focal plane (computed)", font_size=28, color=GREY_A).next_to(axp, UP, buff=0.15)
        self.at(S("w8") + 4.9)
        self.play(FadeOut(VGroup(kin, sub, kname, exag)), run_time=0.4)
        for m_, f_, s_ in pans:
            self.add(m_)
        self.play(*[f_.animate.set_value(1.0) for _, f_, _ in pans], FadeIn(e1), FadeIn(e2), FadeIn(fr), FadeIn(tl), run_time=0.4)
        self.play(*[s_.animate.set_value(1.0) for _, _, s_ in pans], run_time=1.7, rate_func=linear)
        self.play(FadeIn(axp), FadeIn(ptl), Create(c_t), Create(c_k), FadeIn(lg), run_time=0.9)
        self.finish()

# ============================================================ 3  steps cost light ==
class Ch3Steps(Chapter):
    CH = 3
    TAG = "3\\quad Steps cost light"

    def construct(self):
        S = self.S
        self.show_tag()
        D = np.load(data_path("steps.npz"))
        mm = D["m"]
        sel = (mm >= -7) & (mm <= 7)
        ms = mm[sel]
        # ---------------- profile: two teeth of a blazed grating, N-level staircase (states share vertex counts)
        x0, tw, yb, th = -6.2, 2.0, 1.0, 1.4
        ns = 200

        def prof_poly(N):
            u = (np.arange(2 * ns) + 0.5) / ns            # two periods
            fr = u % 1.0
            h = fr if N == 0 else np.floor(fr * N) / N
            pts = [[x0, yb, 0]] + [[x0 + uu * tw, yb + hh * th, 0] for uu, hh in zip(u, h)] + [[x0 + 2 * tw, yb, 0]]
            return glassify(Polygon(*pts), width=2)
        profs = {N: prof_poly(N) for N in (0, 2, 4, 8, 16)}
        ramp = DashedVMobject(VMobject().set_points_as_corners(
            [[x0, yb, 0], [x0 + tw, yb + th, 0], [x0 + tw, yb, 0], [x0 + 2 * tw, yb + th, 0], [x0 + 2 * tw, yb, 0]]),
            num_dashes=60).set_stroke(YELLOW, 2.5, opacity=0.9)
        prof = profs[0].copy()
        nlab = {N: Tex("$N = %d$ levels" % N, font_size=30).move_to([x0 + tw, yb - 0.4, 0]) for N in (2, 4, 8, 16)}
        nl_cur = nlab[4].copy()
        # ---------------- far-field orders (computed by FFT of one period, physics.py)
        bx0, bdx, bw_, by0, bsc = 0.9, 0.385, 0.22, 0.55, 2.45

        def bars(N):
            et = D[f"eta_{N}"][sel]
            g = VGroup()
            for m_, e_ in zip(ms, et):
                hgt = max(bsc * e_, 0.004)
                g.add(Rectangle(width=bw_, height=hgt).move_to([bx0 + (m_ + 7) * bdx, by0 + hgt / 2, 0])
                      .set_stroke(width=0).set_fill(C_U if m_ == 1 else C_EV, 1.0))
            return g
        bset = {N: bars(N) for N in (0, 2, 4, 8, 16)}
        bcur = bset[0].copy()
        axis = Line([bx0 - 0.3, by0, 0], [bx0 + 14 * bdx + 0.3, by0, 0]).set_stroke(GREY_B, 2)
        mlab = VGroup(MathTex("0", font_size=26, color=GREY_B).move_to([bx0 + 7 * bdx, by0 - 0.25, 0]),
                      MathTex("+1", font_size=26, color=C_U).move_to([bx0 + 8 * bdx, by0 - 0.25, 0]),
                      MathTex("-3", font_size=26, color=GREY_B).move_to([bx0 + 4 * bdx, by0 - 0.25, 0]),
                      MathTex("+5", font_size=26, color=GREY_B).move_to([bx0 + 12 * bdx, by0 - 0.25, 0]))
        btitle = Tex("diffraction orders (computed)", font_size=28, color=GREY_A).move_to([bx0 + 7 * bdx, by0 - 0.65, 0])
        focus_l = Tex("focus", font_size=28, color=C_U).move_to([bx0 + 8 * bdx + 0.75, by0 + bsc + 0.05, 0])
        pct = {N: Tex(t, font_size=32, color=WHITE) for N, t in ((0, "100\\%"), (2, "41\\%"), (4, "81\\%"), (8, "95\\%"), (16, "99\\%"))}
        for N, t_ in pct.items():
            t_.next_to([bx0 + 8 * bdx, by0 + bsc * float(D[f"eta_{N}"][mm == 1][0]), 0], UP, buff=0.1).shift(0.35 * LEFT)
        pcur = pct[0].copy()
        blabel = {k: Tex(t, font_size=28, color=GREY_A).move_to([bx0 + 2.2 * bdx, by0 + 2.3, 0])
                  for k, t in (("ramp", "perfect ramp"), ("N", "with steps"))}
        bl_cur = blabel["ramp"].copy()
        # ---------------- phasor chain: arrows for the phase gap to the ramp, head to tail
        NA = 32
        c0, Lc = np.array([-6.0, -2.45, 0]), 6.0

        def chain_pts(N):
            uu = (np.arange(NA) + 0.5) / NA
            gap = np.zeros(NA) if N == 0 else TAU_ * (np.floor(uu * N) / N - uu) + np.pi / N
            z = np.concatenate([[0], np.cumsum(np.exp(1j * gap))]) * Lc / NA
            return [c0 + np.array([zz.real, zz.imag, 0]) for zz in z]

        def chain(N, k=NA):
            p = chain_pts(N)
            g = VGroup()
            for i in range(NA):
                g.add(fast_arrow(p[i], p[i + 1] if i < k else p[i] + 1e-3 * RIGHT, C_U, 3, 0.085, 1.0 if i < k else 0.0))
            return g
        csets = {N: chain(N) for N in (0, 2, 4, 8, 16)}
        res = {N: fast_arrow(c0, chain_pts(N)[-1], YELLOW, 6, 0.22) for N in (0, 2, 4, 8, 16)}
        ref = VGroup(*[fast_arrow(c0 + (Lc * i / NA) * RIGHT + 0.0 * DOWN, c0 + (Lc * (i + 1) / NA) * RIGHT, GREY_B, 2.5, 0.07, 0.75)
                       for i in range(NA)]).shift(0.22 * DOWN)
        ref_l = Tex("perfect ramp: all aligned", font_size=26, color=GREY_B).next_to(ref, DOWN, buff=0.12)
        nshow, mk_op = ValueTracker(0.0), ValueTracker(0.0)

        def building():
            k = int(np.clip(np.floor(nshow.get_value() + 1e-6), 0, NA))
            p = chain_pts(4)
            g = VGroup(*[fast_arrow(p[i], p[i + 1], C_U, 3, 0.085) for i in range(k)])
            o = mk_op.get_value()
            if o > 0.002:
                uu = np.clip(nshow.get_value() / NA, 0, 1)
                xm = x0 + uu * tw
                hs = np.floor(min(uu, 0.9999) * 4) / 4
                g.add(Line([xm, yb + hs * th, 0], [xm, yb + uu * th, 0]).set_stroke(YELLOW, 5, opacity=o),
                      Dot([xm, yb + uu * th, 0], radius=0.06, color=YELLOW).set_opacity(o))
            return g
        bld = always_redraw(building)
        eta_eq = MathTex("\\eta", "=", "\\mathrm{sinc}^2(1/N)", font_size=48).move_to([3.7, -1.35, 0])
        eta_eq[0].set_color(C_U)
        eta_l = Tex("share of light in the focus", font_size=28, color=GREY_A).next_to(eta_eq, DOWN, buff=0.25)

        # s1: a ramp becomes a staircase of N heights
        self.at(S("s1") + 0.2)
        self.play(FadeIn(prof), Create(ramp), run_time=0.9)
        self.at(S("s1") + 2.0)
        self.play(Transform(prof, profs[4]), FadeIn(nl_cur), run_time=1.2)
        # s2: a tooth of a grating; a perfect ramp -> all light in one order (the focus)
        self.at(S("s2") + 0.4)
        self.play(Indicate(prof, color=WHITE, scale_factor=1.04), run_time=0.9)
        self.at(S("s2") + 2.8)
        self.play(Create(axis), LaggedStart(*[GrowFromEdge(b_, DOWN) for b_ in bcur], lag_ratio=0.03), FadeIn(mlab), FadeIn(btitle),
                  FadeIn(bl_cur), Indicate(ramp, color=YELLOW, scale_factor=1.0), run_time=1.0)
        self.play(FadeIn(pcur), run_time=0.4)
        self.at(S("s2") + 4.6)
        self.play(Indicate(bcur[8], color=WHITE, scale_factor=1.12), run_time=0.9)
        self.at(S("s2") + 6.4)
        self.play(FadeIn(focus_l, shift=0.1 * LEFT), run_time=0.5)
        # s3: steps leak light into other orders
        self.at(S("s3") + 0.2)
        self.play(Transform(bcur, bset[4]), Transform(pcur, pct[4]), Transform(bl_cur, blabel["N"]), run_time=0.9)
        self.at(S("s3") + 2.1)
        self.play(*[Indicate(bcur[i], color=C_EV, scale_factor=1.15) for i, m_ in enumerate(ms) if m_ in (-3, 5, -7)], run_time=0.9)
        # s4: arrows for the phase gap at each point, head to tail
        self.at(S("s4") + 0.8)
        self.add(bld)
        self.play(mk_op.animate.set_value(1.0), run_time=0.3)
        self.play(nshow.animate.set_value(NA), run_time=5.0, rate_func=linear)
        self.play(mk_op.animate.set_value(0.0), run_time=0.3)
        bld.clear_updaters()
        self.remove(bld)
        ch_cur = csets[4].copy()
        self.add(ch_cur)
        # s5: for the ramp the arrows line up; steps curl the chain; the end-to-end arrow is shorter
        self.at(S("s5") + 0.1)
        self.play(Transform(ch_cur, csets[0]), FadeIn(ref_l), run_time=1.0)
        self.at(S("s5") + 2.0)
        self.play(FadeIn(ref), run_time=0.6)
        self.at(S("s5") + 4.6)
        self.play(Transform(ch_cur, csets[4]), run_time=1.3)
        r_cur = res[4].copy()
        self.at(S("s5") + 6.2)
        self.play(GrowFromPoint(r_cur, c0), run_time=0.8)
        # s6: eta = sinc^2(1/N); step through N = 2, 4, 8, 16
        self.at(S("s6") + 0.9)
        self.play(Write(eta_eq), FadeIn(eta_l), run_time=1.3)
        marks = [(2, 6.3), (4, 8.6), (8, 9.9), (16, 11.3)]
        for N, dt in marks:
            self.at(S("s6") + dt)
            self.play(Transform(prof, profs[N]), Transform(bcur, bset[N]), Transform(pcur, pct[N]), Transform(nl_cur, nlab[N]),
                      Transform(ch_cur, csets[N]), Transform(r_cur, res[N]), run_time=0.6)
        self.finish()

# ============================================================ 4  the rainbow problem
class Ch4Rainbow(Chapter):
    CH = 4
    TAG = "4\\quad The rainbow problem"

    def construct(self):
        S = self.S
        self.show_tag()
        L = np.load(data_path("lens.npz"))
        # ---------------- c1: a zone diffracts; longer wavelengths bend more (sin theta = lambda / zone width)
        gx0, gw, gy, gh = -4.6, 1.3, -1.2, 0.62
        teeth = VGroup()
        for i in range(4):
            xl = gx0 + i * gw
            teeth.add(Polygon([xl, gy, 0], [xl, gy + gh, 0], [xl + gw, gy, 0]))
        sub = Rectangle(width=4 * gw, height=0.22).move_to([gx0 + 2 * gw, gy - 0.11, 0])
        grat = VGroup(glassify(sub.set_fill(SUB_FILL, 1), width=1.5), *[glassify(t_, width=1.8) for t_ in teeth])
        hit = np.array([gx0 + 2.0 * gw, gy + 0.35 * gh, 0])
        inc = Arrow(hit + 1.55 * DOWN, hit + 0.02 * DOWN, buff=0, stroke_width=5, color=WHITE, max_tip_length_to_length_ratio=0.12)
        inc_l = Tex("white light", font_size=28, color=GREY_A).next_to(inc, LEFT, buff=0.15).shift(0.45 * DOWN)
        Lz = 1.35                                            # um: an outer zone of the lens
        rays, rlabs = VGroup(), VGroup()
        for lam, c_, nm in ((0.45, C_B, "blue"), (0.55, C_G, "green"), (0.65, C_R, "red")):
            th_ = np.arcsin(lam / Lz)
            d = np.array([-np.sin(th_), np.cos(th_), 0])
            rays.add(Line(hit, hit + 3.3 * d).set_stroke(c_, 5))
            side = {"blue": RIGHT, "green": UP, "red": LEFT}[nm]
            rlabs.add(Tex(nm, font_size=28, color=c_).next_to(hit + 3.3 * d, side, buff=0.12).shift(0.3 * LEFT if nm == "green" else 0 * LEFT))
        bz = BraceBetweenPoints([gx0 + 4 * gw, gy - 0.3, 0], [gx0 + 3 * gw, gy - 0.3, 0], DOWN)
        bzl = Tex("zone: fixed width", font_size=28, color=GREY_A).next_to(bz, DOWN, buff=0.1)
        rule = MathTex("\\sin\\theta", "=", "\\lambda", "/", "\\text{zone width}", font_size=40).move_to([2.6, 0.9, 0])
        rule2 = Tex("longer $\\lambda$ bends more", font_size=32, color=GREY_A).next_to(rule, DOWN, buff=0.3)
        self.at(S("c1") + 0.2)
        self.play(FadeIn(grat), run_time=0.7)
        self.at(S("c1") + 1.2)
        self.play(GrowArrow(inc), FadeIn(inc_l), run_time=0.7)
        self.at(S("c1") + 3.4)
        self.play(GrowFromCenter(bz), FadeIn(bzl), run_time=0.7)
        self.at(S("c1") + 4.8)
        self.play(LaggedStart(*[Create(r_) for r_ in rays], lag_ratio=0.25), LaggedStart(*[FadeIn(l_) for l_ in rlabs], lag_ratio=0.25),
                  Write(rule), run_time=1.4)
        self.play(FadeIn(rule2), run_time=0.5)
        # ---------------- c2/c3: computed foci along the axis (white light: TEA + 2D ASM), kinoform vs glass
        sc = 0.054
        xs, zs = L["xs"], L["zs"]
        W, Hf = (xs[-1] - xs[0]) * sc, (zs[-1] - zs[0]) * sc
        ye = -2.25
        pcx = (-4.7, 1.25)
        a, h2, nz, f0 = float(L["a"]), float(L["h2pi"]), int(L["nzone"]), float(L["f0"])
        r2 = np.linspace(0, a, 300)
        rk = np.sort(np.concatenate([np.linspace(0, a, 1200), L["edges"][1:-1] - 1e-6, L["edges"][1:-1] + 1e-6]))
        ht = np.clip(nz * h2 - (np.sqrt(r2 ** 2 + f0 ** 2) - f0) / 0.5, 0, None)
        hk = np.mod(np.clip(nz * h2 - (np.sqrt(rk ** 2 + f0 ** 2) - f0) / 0.5, 0, None), h2)
        el = [glassify(lens_outline(rk, hk, pcx[0], ye - h2 * sc, sc, sc), width=1.0),
              glassify(lens_outline(r2, ht, pcx[1], ye - nz * h2 * sc, sc, sc), width=1.5)]
        ims = [self.static_image(fit_rgba(rgb2rgba(L[k_]), W, Hf), W, np.array([c_, ye + Hf / 2, 0]), Hf)
               for k_, c_ in (("rgb_k", pcx[0]), ("rgb_t", pcx[1]))]
        frs = VGroup(*[Rectangle(width=W, height=Hf).move_to([c_, ye + Hf / 2, 0]).set_stroke(GREY_D, 1) for c_ in pcx])
        ttl = VGroup(Tex("kinoform", font_size=32).move_to([pcx[0] + 0.9, 2.05, 0]),
                     Tex("glass lens", font_size=32).move_to([pcx[1] + 0.9, 2.05, 0]))
        # on-axis intensity curves, z aligned with the panels
        def axial(key, cxp, col, z_scale=sc):
            I = L[key]
            I = I / I.max()
            x_ = cxp + W / 2 + 0.25 + 1.55 * I
            y_ = ye + zs * z_scale
            return poly_line(x_, y_, col, 3.5)
        cur = {}
        for kind, cxp in (("k", pcx[0]), ("t", pcx[1])):
            for nm, col in (("R", C_R), ("G", C_G), ("B", C_B)):
                cur[kind + nm] = axial(f"ax_{kind}_{nm}", cxp, col)
        axl = VGroup(*[Line([c_ + W / 2 + 0.25, ye, 0], [c_ + W / 2 + 0.25, ye + Hf, 0]).set_stroke(GREY_C, 1.5) for c_ in pcx])
        zf = {k_: ye + float(zs[np.argmax(L[k_])]) * sc for k_ in ("ax_k_R", "ax_k_G", "ax_k_B", "ax_t_R", "ax_t_G", "ax_t_B")}
        fl = MathTex("f", "=", "f_0", "\\,\\lambda_0", "/", "\\lambda", font_size=44).move_to([-4.0, 2.9, 0])
        dk = (float(zs[np.argmax(L["ax_k_B"])]) - float(zs[np.argmax(L["ax_k_R"])]))
        dt = (float(zs[np.argmax(L["ax_t_R"])]) - float(zs[np.argmax(L["ax_t_B"])]))
        cap_k = MathTex("\\text{blue } %.0f\\," % dk + MU + "\\text{ farther than red}", font_size=28, color=GREY_A).next_to(ttl[0], DOWN, buff=0.15)
        cap_t = MathTex("\\text{blue } %.1f\\," % dt + MU + "\\text{ closer than red}", font_size=28, color=GREY_A).next_to(ttl[1], DOWN, buff=0.15)
        lab_k = VGroup(*[Tex(t_, font_size=26, color=c_).move_to([pcx[0] + W / 2 + 2.15, zf["ax_k_" + k_], 0])
                         for k_, t_, c_ in (("R", "red", C_R), ("G", "green", C_G), ("B", "blue", C_B))])
        ion = Tex("on-axis intensity (computed)", font_size=24, color=GREY_B).move_to([pcx[0] + W / 2 + 1.05, ye - 0.72, 0])
        ion2 = ion.copy().move_to([pcx[1] + W / 2 + 1.25, ye - 0.72, 0])
        self.at(S("c2"))
        self.play(FadeOut(VGroup(grat, inc, inc_l, rays, rlabs, bz, bzl, rule, rule2)), run_time=0.5)
        mk, fk = ims[0]
        self.add(mk)
        self.play(fk.animate.set_value(1.0), FadeIn(frs[0]), FadeIn(el[0]), FadeIn(ttl[0]), Create(axl[0]), FadeIn(ion), run_time=0.8)
        self.at(S("c2") + 2.1)
        self.play(Write(fl), Create(cur["kG"]), run_time=1.6)
        self.at(S("c2") + 5.1)
        self.play(Create(cur["kR"]), FadeIn(lab_k[0]), FadeIn(lab_k[1]), run_time=1.0)
        self.at(S("c2") + 7.5)
        self.play(Create(cur["kB"]), FadeIn(lab_k[2]), FadeIn(cap_k), run_time=1.0)
        mt, ft = ims[1]
        self.at(S("c3") + 0.2)
        self.add(mt)
        self.play(ft.animate.set_value(1.0), FadeIn(frs[1]), FadeIn(el[1]), FadeIn(ttl[1]), Create(axl[1]), FadeIn(ion2), run_time=0.8)
        self.play(Create(cur["tG"]), Create(cur["tR"]), Create(cur["tB"]), run_time=1.2)
        self.at(S("c3") + 3.5)
        self.play(FadeIn(cap_t, shift=0.1 * DOWN), run_time=0.7)
        pk = Rectangle(width=1.95, height=0.5).move_to([pcx[1] + W / 2 + 1.15, zf["ax_t_G"], 0]).set_stroke(YELLOW, 3)
        self.at(S("c3") + 5.5)
        self.play(ShowPassingFlash(pk, time_width=0.6), run_time=1.3)
        # ---------------- c4: Abbe numbers
        vh = Tex("Abbe number $V$", font_size=34, color=YELLOW).move_to([3.6, 2.9, 0])
        vk = MathTex("V = -3.45", font_size=36, color=YELLOW).move_to(cap_k)
        vt = MathTex("V \\approx +60", font_size=36, color=YELLOW).move_to(cap_t)
        self.at(S("c4") + 1.2)
        self.play(FadeIn(vh, shift=0.1 * DOWN), run_time=0.6)
        self.at(S("c4") + 2.7)
        self.play(FadeOut(cap_t, shift=0.1 * UP), FadeIn(vt, shift=0.1 * UP), run_time=0.6)
        self.at(S("c4") + 4.9)
        self.play(FadeOut(cap_k, shift=0.1 * UP), FadeIn(vk, shift=0.1 * UP), run_time=0.6)
        # ---------------- c5: opposite signs cancel (thin-lens powers with this glass), or fix it in software
        lam = np.linspace(0.44, 0.66, 120)
        nl = lambda l_: 1.5 - 0.00409 / 0.55 ** 2 + 0.00409 / l_ ** 2
        nd_ = 2 * 0.00409 / 0.55 ** 3                                      # -dn/dlambda at 550 nm
        pr0 = 1 / (1 + 0.55 * nd_ / 0.5)                                    # refractive share of the power
        pd0 = 1 - pr0
        dPr = 100 * pr0 * ((nl(lam) - 1) / 0.5 - 1)
        dPd = 100 * pd0 * (lam / 0.55 - 1)
        bc, bw2, bh2 = np.array([-0.4, -0.45, 0]), 6.4, 3.4
        ylim = (-1.6, 1.6)
        box = axis_pair(bc + np.array([0, 0, 0]), bw2, bh2, size=28)
        zero = DashedLine(bc + np.array([-bw2 / 2, bh2 / 2 * (0 - ylim[0]) / (ylim[1] - ylim[0]) * 2 - bh2 / 2, 0]),
                          bc + np.array([bw2 / 2, bh2 / 2 * (0 - ylim[0]) / (ylim[1] - ylim[0]) * 2 - bh2 / 2, 0]),
                          dash_length=0.08).set_stroke(GREY_C, 1.5)
        cg = plot_xy(lam, dPr, bc, bw2, bh2, (0.44, 0.66), ylim, color=GREY_A, width=4)
        cd = plot_xy(lam, dPd, bc, bw2, bh2, (0.44, 0.66), ylim, color=C_U, width=4)
        cs = plot_xy(lam, dPr + dPd, bc, bw2, bh2, (0.44, 0.66), ylim, color=YELLOW, width=7)
        xl_ = VGroup(*[MathTex("%d" % v_, font_size=24, color=GREY_B).move_to(bc + np.array([-bw2 / 2 + (v_ / 1000 - 0.44) / 0.22 * bw2, -bh2 / 2 - 0.25, 0]))
                       for v_ in (450, 550, 650)])
        xlab = Tex("wavelength (nm)", font_size=26, color=GREY_B).next_to(xl_, DOWN, buff=0.1)
        ylab = Tex("change of focusing power", font_size=26, color=GREY_B).rotate(PI / 2).next_to(box, LEFT, buff=0.15)
        lg = VGroup(Tex("glass lens", font_size=28, color=GREY_A), Tex("weak diffractive surface", font_size=28, color=C_U),
                    Tex("together", font_size=30, color=YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.75, 0.2, 0])
        for i_, c_ in enumerate((cg, cd, cs)):
            pass
        sw = Tex("\\dots or correct the\\\\colours in software", font_size=30, color=GREY_A).move_to([4.75, -1.55, 0])
        z0l = MathTex("0", font_size=24, color=GREY_B).next_to(zero, LEFT, buff=0.1)
        self.at(S("c5"))
        self.play(FadeOut(VGroup(frs, *el, ttl, fl, lab_k, ion, ion2, vk, vt, axl, *cur.values())), fk.animate.set_value(0.0),
                  ft.animate.set_value(0.0), vh.animate.move_to([-3.2, 2.95, 0]).set_opacity(0.0), run_time=0.5)
        for m_ in (mk, mt):
            m_.clear_updaters()
            self.remove(m_)
        self.play(FadeIn(box), FadeIn(zero), FadeIn(z0l), FadeIn(xl_), FadeIn(xlab), FadeIn(ylab), Create(cg), FadeIn(lg[0]), run_time=0.9)
        self.at(S("c5") + 1.7)
        self.play(Create(cd), FadeIn(lg[1]), run_time=0.9)
        self.at(S("c5") + 3.3)
        self.play(Create(cs), FadeIn(lg[2]), run_time=1.0)
        self.at(S("c5") + 6.0)
        self.play(FadeIn(sw, shift=0.1 * UP), run_time=0.7)
        self.finish()

# ============================================================ 5  where thin breaks
class Ch5Thin(Chapter):
    CH = 5
    TAG = "5\\quad Where ``thin'' breaks"

    def construct(self):
        S = self.S
        self.show_tag()
        F = np.load(data_path("fullwave.npz"))
        P_, Hrel = float(F["period"]), float(F["H"])
        fz, fx = F["fz"], F["fx"]
        nx = len(fx)
        sh = nx // 2                                       # roll by half a period: walls at 1/4 and 3/4 of the panel
        Efw = np.roll(np.tile(F["E_fw"], (1, 2)), -sh, axis=1)[::-1]
        Etea = np.roll(np.tile(F["E_tea"], (1, 2)), -sh, axis=1)[::-1]
        dz = fz[1] - fz[0]
        u = 0.52                                           # scene units per wavelength
        Wp, Hp = 2 * P_ * u, (fz[-1] - fz[0] + dz) * u
        cyp = 0.42
        cxs = (0.0, 2.55)                                  # TEA panel starts centred; slides to -2.55 at t3
        lw = [LiveWave(E_, Wp, Hp, omega=TAU_ * 1.0, amp_ref=2.2, intensity_vmax=3.0) for E_ in (Etea, Efw)]
        heat = ValueTracker(0.0)
        sweeps = [ValueTracker(0.0), ValueTracker(0.0)]
        panels = []
        for k in range(2):
            m_, f_ = self.live_image(lambda t, k=k: lw[k].frame(t, sweeps[k].get_value(), heat.get_value()), Wp,
                                     np.array([cxs[k], cyp, 0]), Hp)
            panels.append((m_, f_))
        y_bot = cyp - Hp / 2
        y0 = y_bot + (0 - (fz[0] - dz / 2)) * u            # z = 0: base of the teeth

        def relief(cx):
            xl = cx - Wp / 2
            pts = [[xl, y0 + 0.5 * Hrel * u, 0], [xl + 0.5 * P_ * u, y0 + Hrel * u, 0], [xl + 0.5 * P_ * u, y0, 0],
                   [xl + 1.5 * P_ * u, y0 + Hrel * u, 0], [xl + 1.5 * P_ * u, y0, 0], [xl + 2 * P_ * u, y0 + 0.5 * Hrel * u, 0]]
            return VMobject().set_points_as_corners(pts).set_stroke(WHITE, 2.5)
        rel = [relief(c_) for c_ in cxs]
        frames = [Rectangle(width=Wp, height=Hp).move_to([c_, cyp, 0]).set_stroke(GREY_B, 1.5) for c_ in cxs]
        tl = Tex("thin element (TEA)", font_size=32).next_to(frames[0], UP, buff=0.2)
        tr = Tex("full wave (Maxwell, computed)", font_size=32).next_to(frames[1], UP, buff=0.2)
        # t1: zoom from the kinoform (the Ch2 lens, heights x2) into two zones about 4 wavelengths wide
        L = np.load(data_path("lens.npz"))
        a, h2, nz, f0 = float(L["a"]), float(L["h2pi"]), int(L["nzone"]), float(L["f0"])
        rk = np.sort(np.concatenate([np.linspace(0, a, 1200), L["edges"][1:-1] - 1e-6, L["edges"][1:-1] + 1e-6]))
        hk = np.mod(np.clip(nz * h2 - (np.sqrt(rk ** 2 + f0 ** 2) - f0) / 0.5, 0, None), h2)
        ksx = 0.16
        kin = glassify(lens_outline(rk, hk, 0.0, 1.2, ksx, 0.36), width=1.5)
        kin_base = Rectangle(width=2 * a * ksx + 0.5, height=0.18).move_to([0, 1.11, 0]).set_fill(SUB_FILL, 1).set_stroke(GREY_B, 1)
        e_ = L["edges"]
        z_in, z_out = float(e_[nz - 2]), float(e_[nz - 4])     # zones 3 and 4 from the centre
        zbox = Rectangle(width=(z_out - z_in) * ksx + 0.1, height=0.75).move_to([0.5 * (z_in + z_out) * ksx, 1.38, 0]).set_stroke(YELLOW, 3)
        self.at(S("t1"))
        self.play(FadeIn(kin), FadeIn(kin_base), run_time=0.6)
        self.play(Create(zbox), run_time=0.5)
        mt, ft = panels[0]
        px = ValueTracker(0.0)
        mt.add_updater(lambda m_: m_.move_to([px.get_value(), cyp, 0]))
        self.add(mt)
        self.play(ReplacementTransform(zbox, frames[0]), FadeOut(VGroup(kin, kin_base)), run_time=1.0)
        self.play(ft.animate.set_value(1.0), sweeps[0].animate.set_value(1.0), Create(rel[0]), FadeIn(tl), run_time=1.0)
        cols = VGroup(*[fast_dashed([cxs[0] - Wp / 2 + (i + 0.5) * Wp / 12, y_bot + 0.05, 0],
                                    [cxs[0] - Wp / 2 + (i + 0.5) * Wp / 12, y0 + Hrel * u + 0.15, 0], 0.06, WHITE, 1.5, 0.75)
                        for i in range(12)])
        self.at(S("t1") + 3.6)
        self.play(LaggedStart(*[Create(c_) for c_ in cols], lag_ratio=0.06), run_time=1.0)
        # t2: four wavelengths wide, walls two wavelengths tall
        xw = cxs[0] - Wp / 2 + 0.5 * P_ * u
        bw_ = BraceBetweenPoints([xw + P_ * u, y_bot - 0.08, 0], [xw, y_bot - 0.08, 0], DOWN)
        bwl = MathTex("4\\lambda", font_size=32).next_to(bw_, DOWN, buff=0.08)
        bh_ = BraceBetweenPoints([xw - 0.1, y0, 0], [xw - 0.1, y0 + Hrel * u, 0], LEFT).set_color(YELLOW)
        bhl = MathTex("2\\lambda", font_size=32, color=YELLOW).next_to(bh_, LEFT, buff=0.06)
        bhl_bg = BackgroundRectangle(bhl, fill_opacity=0.75, buff=0.04)
        self.at(S("t2") + 0.5)
        self.play(GrowFromCenter(bw_), FadeIn(bwl), FadeOut(cols), run_time=0.8)
        self.at(S("t2") + 2.5)
        self.play(GrowFromCenter(bh_), FadeIn(bhl_bg), FadeIn(bhl), run_time=0.7)
        # t3: the full-wave solution (2D FDFD, physics.py): leaks, jets, shadows
        mf, ff = panels[1]
        self.at(S("t3") + 0.1)
        self.add(mf)
        slide = VGroup(frames[0], rel[0], tl, bw_, bwl)
        self.play(px.animate.set_value(-2.55), slide.animate.shift(2.55 * LEFT), FadeOut(VGroup(bh_, bhl, bhl_bg)), run_time=0.7)
        self.play(ff.animate.set_value(1.0), FadeIn(frames[1]), FadeIn(tr), Create(rel[1]), run_time=0.6)
        self.play(sweeps[1].animate.set_value(1.0), run_time=1.5, rate_func=linear)
        tav = Tex("time-averaged intensity", font_size=26, color=GREY_B).move_to([0, 3.2, 0])
        self.at(S("t3") + 4.2)
        self.play(heat.animate.set_value(1.0), FadeIn(tav), run_time=1.1)
        wall = cxs[1] - Wp / 2 + 1.5 * P_ * u               # right-hand wall of the full-wave panel
        crowd = Ellipse(width=0.42, height=0.75).move_to([wall - 0.22, y0 + 1.45 * u, 0]).set_stroke(YELLOW, 3)
        xr = cxs[1] + Wp / 2 + 0.3                          # labels sit right of the panel
        jet_p = np.array([wall - 0.3, y0 + 4.6 * u, 0])
        jet = Arrow([xr, jet_p[1] + 0.35, 0], jet_p, buff=0.05, stroke_width=4, color=YELLOW, max_tip_length_to_length_ratio=0.12)
        jet_l = Tex("jet", font_size=30, color=YELLOW).next_to(jet.get_start(), RIGHT, buff=0.08)
        sha_p = np.array([wall + 0.33, y0 + 1.0 * u, 0])
        sha = Arrow([xr, sha_p[1] - 0.35, 0], sha_p, buff=0.05, stroke_width=4, color=YELLOW, max_tip_length_to_length_ratio=0.15)
        sha_l = Tex("shadow", font_size=30, color=YELLOW).next_to(sha.get_start(), RIGHT, buff=0.08)
        self.play(Create(crowd), run_time=0.6)
        self.at(S("t3") + 6.0)
        self.play(GrowArrow(jet), FadeIn(jet_l), run_time=0.6)
        self.at(S("t3") + 7.4)
        self.play(GrowArrow(sha), FadeIn(sha_l), run_time=0.6)
        # t4: light sent to the focus order: TEA 100 %, full wave 71 % (computed, E along the grooves)
        eta = float(F["eta_fw_TE"])
        bx0, bL, byT, byF = -1.9, 4.6, -2.25, -2.8
        lab_t = Tex("thin element", font_size=28).move_to([bx0 - 1.25, byT, 0])
        lab_f = Tex("full wave", font_size=28).move_to([bx0 - 1.25, byF, 0])
        barT = Rectangle(width=bL, height=0.32).move_to([bx0 + bL / 2, byT, 0]).set_stroke(width=0).set_fill(YELLOW, 0.9)
        barF = Rectangle(width=bL * eta, height=0.32).move_to([bx0 + bL * eta / 2, byF, 0]).set_stroke(width=0).set_fill(C_U, 0.95)
        vT = Tex("100\\%", font_size=30).next_to(barT, RIGHT, buff=0.15)
        vF = Tex("%d\\%%" % round(100 * eta), font_size=30).next_to(barF, RIGHT, buff=0.15)
        btl = Tex("light sent\\\\to the focus", font_size=28, color=GREY_B).move_to([5.3, (byT + byF) / 2, 0])
        self.at(S("t4") + 1.3)
        self.play(FadeOut(VGroup(bw_, bwl)), FadeIn(lab_t), GrowFromEdge(barT, LEFT), FadeIn(vT), FadeIn(btl), run_time=0.8)
        self.at(S("t4") + 4.7)
        self.play(FadeIn(lab_f), GrowFromEdge(barF, LEFT), FadeIn(vF), run_time=0.9)
        # t5: efficiency vs zone width (computed sweep)
        ps, eT, eM = F["sweep_p"], F["sweep_TE"], F["sweep_TM"]
        pc, pw, ph = np.array([0.35, -0.25, 0]), 7.0, 3.9
        lx = lambda p: np.log(p / 2.0) / np.log(10.0)
        axes = axis_pair(pc, pw, ph, size=28)
        ylo = 0.0
        def to_xy(p, e):
            return [pc[0] - pw / 2 + lx(p) * pw, pc[1] - ph / 2 + (e - ylo) / (1 - ylo) * ph, 0]
        tea_line = DashedLine(to_xy(2.0, 1.0), to_xy(20.0, 1.0), dash_length=0.1).set_stroke(YELLOW, 3)
        tea_lab = Tex("thin element: 100\\%", font_size=28, color=YELLOW).next_to(tea_line, UP, buff=0.1).align_to(tea_line, RIGHT)
        cTE = VMobject().set_points_smoothly([to_xy(p, e) for p, e in zip(ps, eT)]).set_stroke(C_U, 5)
        cTM = DashedVMobject(VMobject().set_points_smoothly([to_xy(p, e) for p, e in zip(ps, eM)]), num_dashes=40).set_stroke(C_U, 2.5, opacity=0.7)
        dots = VGroup(*[Dot(to_xy(p, e), radius=0.05, color=C_U) for p, e in zip(ps, eT)])
        fw_lab = Tex("full wave (computed)", font_size=28, color=C_U).move_to(to_xy(7.0, 0.62))
        tm_lab = Tex("other polarisation", font_size=24, color=C_U).set_opacity(0.8).move_to(to_xy(2.9, 0.83))
        xt = VGroup(*[MathTex(t_, font_size=26, color=GREY_B).move_to(to_xy(v_, 0) + 0.28 * DOWN) for v_, t_ in
                      ((2, "2\\lambda"), (5, "5\\lambda"), (10, "10\\lambda"), (20, "20\\lambda"))])
        xtl = Tex("zone width", font_size=28, color=GREY_B).next_to(xt, DOWN, buff=0.12)
        yt = VGroup(*[MathTex(t_, font_size=26, color=GREY_B).move_to(to_xy(2.0, v_) + 0.45 * LEFT) for v_, t_ in
                      ((0.0, "0"), (0.5, "50\\%"), (1.0, "100\\%"))])
        safe = Tex("wide zones: safe", font_size=28).move_to(to_xy(12.0, 0.78))
        self.at(S("t5"))
        self.play(FadeOut(VGroup(*frames, *rel, tl, tr, tav, crowd, jet, jet_l, sha, sha_l, lab_t, lab_f, barT, barF, vT, vF, btl)),
                  ft.animate.set_value(0.0), ff.animate.set_value(0.0), run_time=0.5)
        for m_, f_ in panels:
            m_.clear_updaters()
            self.remove(m_)
        self.play(FadeIn(axes), FadeIn(xt), FadeIn(xtl), FadeIn(yt), Create(tea_line), FadeIn(tea_lab), run_time=0.7)
        self.play(Create(cTE), FadeIn(dots), FadeIn(fw_lab), Create(cTM), FadeIn(tm_lab), run_time=1.2)
        qd = ValueTracker(0.0)

        def runner():
            q = qd.get_value()
            if q <= 1e-3:
                return VGroup()
            p_ = cTE.point_from_proportion(min(q, 1.0))
            return VGroup(Dot(p_, radius=0.11, color=YELLOW), Dot(p_, radius=0.2, color=YELLOW).set_opacity(0.3))
        run_dot = always_redraw(runner)
        self.add(run_dot)
        self.at(S("t5") + 2.6)
        self.play(qd.animate.set_value(1.0), FadeIn(safe, shift=0.1 * UP), run_time=2.3)
        # t6: way out 1 -- metasurfaces: sub-wavelength pillars, each delay from a full-wave unit-cell library (computed)
        mw, mph = F["ms_w"], F["ms_phase"] / TAU_
        ok = mw <= 0.361
        mw, mph = mw[ok], mph[ok]
        lib_c, lib_w, lib_h = np.array([3.6, -0.35, 0]), 3.6, 2.9
        lib_ax = axis_pair(lib_c, lib_w, lib_h, size=26)
        wl = lambda w_: lib_c[0] - lib_w / 2 + (w_ * 550 - 30) / (200 - 30) * lib_w
        pl = lambda p_: lib_c[1] - lib_h / 2 + p_ / 1.15 * lib_h
        lib = VMobject().set_points_smoothly([[wl(w_), pl(p_), 0] for w_, p_ in zip(mw, mph)]).set_stroke(GREEN, 5)
        lib_x = VGroup(*[MathTex("%d" % v_, font_size=24, color=GREY_B).move_to([wl(v_ / 550), lib_c[1] - lib_h / 2 - 0.25, 0]) for v_ in (50, 100, 150, 200)])
        lib_xl = Tex("pillar width (nm)", font_size=26, color=GREY_B).next_to(lib_x, DOWN, buff=0.1)
        lib_y = VGroup(MathTex("0", font_size=24, color=GREY_B).move_to([lib_c[0] - lib_w / 2 - 0.3, pl(0), 0]),
                       MathTex("2\\pi", font_size=26, color=GREY_B).move_to([lib_c[0] - lib_w / 2 - 0.3, pl(1.0), 0]))
        lib_yl = VGroup()
        lib_t = Tex("each pillar's delay\\\\(full wave, computed)", font_size=28, color=GREEN).move_to([lib_c[0] + 0.3, lib_c[1] + lib_h / 2 + 0.7, 0])
        # a row of pillars: a sawtooth of delays 0 .. 2pi, widths read from the library
        npl, pp = 12, 0.46
        targ = (np.arange(npl) % 6) / 6.0 + 1 / 12
        wid = np.interp(targ, mph, mw)
        sx0, sy0, ph_ = -6.0, -1.55, 1.2
        sub = Rectangle(width=npl * pp + 0.3, height=0.25).move_to([sx0 + npl * pp / 2, sy0 - 0.125, 0]).set_fill(SUB_FILL, 1).set_stroke(GREY_B, 1.5)
        pills = VGroup(*[Rectangle(width=w_ / 0.6 * pp * 0.95, height=ph_).move_to([sx0 + (i + 0.5) * pp, sy0 + ph_ / 2, 0])
                         .set_fill("#c9d3da", 0.85).set_stroke(GLASS_LINE, 1) for i, w_ in enumerate(wid)])
        dials = VGroup()
        for i, t_ in enumerate(targ):
            c_ = np.array([sx0 + (i + 0.5) * pp, sy0 + ph_ + 0.4, 0])
            dials.add(VGroup(Circle(radius=0.17).move_to(c_).set_stroke(GREY_B, 1.5),
                             fast_arrow(c_, c_ + 0.17 * np.array([np.cos(TAU_ * t_), np.sin(TAU_ * t_), 0]), YELLOW, 3, 0.07)))
        pl_l = Tex("pillars narrower than $\\lambda$", font_size=28, color=GREY_A).move_to([sx0 + npl * pp / 2, sy0 - 0.65, 0])
        ms_t = Tex("metasurface", font_size=36).move_to([sx0 + npl * pp / 2, 0.75, 0])
        self.at(S("t6") + 0.9)
        run_dot.clear_updaters()
        self.play(FadeOut(VGroup(axes, xt, xtl, yt, tea_line, tea_lab, cTE, cTM, dots, fw_lab, tm_lab, safe, run_dot)), run_time=0.5)
        self.play(FadeIn(sub), LaggedStart(*[GrowFromEdge(p_, DOWN) for p_ in pills], lag_ratio=0.08), FadeIn(ms_t), FadeIn(pl_l),
                  run_time=1.3)
        self.at(S("t6") + 5.3)
        self.play(LaggedStart(*[FadeIn(d_, scale=0.6) for d_ in dials], lag_ratio=0.06), FadeIn(lib_ax), FadeIn(lib_x), FadeIn(lib_xl),
                  FadeIn(lib_y), FadeIn(lib_yl), FadeIn(lib_t), Create(lib), run_time=1.6)
        # t7: way out 2 -- climb the ladder: a full-wave solver (Episode 9)
        lad = model_ladder([-0.2, 0.2, 0], width=4.6, size=30).scale(1.25)
        e9 = Tex("Episode 9", font_size=34, color=YELLOW).next_to(lad[0][3], LEFT, buff=0.45)
        self.at(S("t7"))
        self.play(FadeOut(VGroup(sub, pills, dials, pl_l, ms_t, lib_ax, lib_x, lib_xl, lib_y, lib_yl, lib_t, lib)), run_time=0.5)
        self.play(FadeIn(lad, shift=0.1 * UP), *light_rung(lad, 0), run_time=0.8)
        self.at(S("t7") + 1.6)
        self.play(*light_rung(lad, 3), run_time=0.9)
        self.at(S("t7") + 4.4)
        self.play(FadeIn(e9, shift=0.1 * RIGHT), run_time=0.6)
        self.finish()

# ============================================================ 6  on a computer =====
class Ch6Computer(Chapter):
    CH = 6
    TAG = "6\\quad On a computer"

    def construct(self):
        S = self.S
        self.show_tag()
        L = np.load(data_path("lens.npz"))
        card = code_card(["h = mod(h_lens, lam/(n-1))        # wrap the heights",
                          "U = U_in * exp(1j*k0*(n-1)*h)     # thin element",
                          "U = ifft2(fft2(U) * exp(1j*kz*z)) # angular spectrum",
                          "I = abs(U)**2                     # the focus"], font_size=22)
        card.move_to([0.45, 2.2, 0])
        rows = card[1][0]

        def hl(i):
            return SurroundingRectangle(rows[i], buff=0.07, corner_radius=0.05, color=YELLOW, stroke_width=2.5)
        h1, h2_ = hl(1), hl(2)
        # pipeline (computed): wrapped heights -> x e^{i phi} -> angular spectrum -> focus at 550 nm
        a, h2, nz, f0 = float(L["a"]), float(L["h2pi"]), int(L["nzone"]), float(L["f0"])
        rk = np.sort(np.concatenate([np.linspace(0, a, 1200), L["edges"][1:-1] - 1e-6, L["edges"][1:-1] + 1e-6]))
        hk = np.mod(np.clip(nz * h2 - (np.sqrt(rk ** 2 + f0 ** 2) - f0) / 0.5, 0, None), h2)
        kin = glassify(lens_outline(rk, hk, -4.0, -1.6, 0.1, 0.45), width=1.5)
        kin_l = MathTex("\\times\\, e^{\\,i k_0 (n-1) h}", font_size=34).next_to(kin, UP, buff=0.35)
        xs, zs = L["xs"], L["zs"]
        sc = 0.045
        Wg, Hg = (xs[-1] - xs[0]) * sc, (zs[-1] - zs[0]) * sc
        gimg = fit_rgba(rgb2rgba(L["rgb_g"]), Wg, Hg)
        gm, gf, gs = self.sweep_image(gimg, Wg, np.array([1.2, -1.75 + Hg / 2, 0]), Hg, "up")
        gfr = Rectangle(width=Wg, height=Hg).move_to([1.2, -1.75 + Hg / 2, 0]).set_stroke(GREY_D, 1)
        kin2 = glassify(lens_outline(rk, hk, 1.2, -1.75 - h2 * sc, sc, sc), width=1.0)
        par = Arrow([-1.55, -1.35, 0], [-0.25, -1.35, 0], buff=0, stroke_width=4, color=GREY_B)
        par_l = Tex("propagate", font_size=28, color=GREY_A).next_to(par, UP, buff=0.12)
        foc_l = Tex("the focus", font_size=28, color=GREY_A).next_to(gfr, RIGHT, buff=0.25)
        self.at(S("a1") + 0.2)
        self.play(FadeIn(card, shift=0.1 * DOWN), run_time=0.9)
        self.at(S("a1") + 3.0)
        self.play(Create(h1), FadeIn(kin), run_time=0.7)
        self.play(Write(kin_l), run_time=0.9)
        self.at(S("a1") + 7.6)
        self.add(gm)
        self.play(ReplacementTransform(h1, h2_), GrowArrow(par), FadeIn(par_l), gf.animate.set_value(1.0), FadeIn(gfr), FadeIn(kin2),
                  run_time=0.7)
        self.play(gs.animate.set_value(1.0), run_time=1.6, rate_func=linear)
        self.play(FadeIn(foc_l), run_time=0.4)
        # a2: three pitfalls
        items = VGroup(Tex("1\\quad sample the finest zone: $\\Delta x \\ll$ zone width", font_size=30),
                       Tex("2\\quad wrap the heights: $h \\bmod \\lambda/(n-1)$", font_size=30),
                       Tex("3\\quad simulate the profile you can make", font_size=30)).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        items.move_to([-3.35, -0.95, 0]).align_to([-6.6, 0, 0], LEFT)
        head = Tex("pitfalls", font_size=32, color=YELLOW).next_to(items, UP, buff=0.35).align_to(items, LEFT)
        self.at(S("a2"))
        self.play(FadeOut(VGroup(h2_, kin, kin_l, par, par_l, gfr, kin2, foc_l)), gf.animate.set_value(0.0), FadeIn(head), run_time=0.5)
        gm.clear_updaters()
        self.remove(gm)
        self.at(S("a2") + 1.4)
        self.play(FadeIn(items[0], shift=0.1 * RIGHT), run_time=0.6)
        self.at(S("a2") + 3.9)
        self.play(FadeIn(items[1], shift=0.1 * RIGHT), run_time=0.6)
        # computed: drawn vs fabricated (blurred, sigma 0.1 um) outer zones, and what it does to the focus
        rr, hk, hr = L["rr"], L["hk_line"], L["hr_line"]
        sel = rr >= 13.0
        pc, pw, ph = np.array([3.75, -0.2, 0]), 5.0, 1.1
        drawn = plot_xy(rr[sel], hk[sel], pc, pw, ph, (13.0, 20.0), (0, 1.15), color=WHITE, width=3)
        fab = plot_xy(rr[sel], hr[sel], pc, pw, ph, (13.0, 20.0), (0, 1.15), color=YELLOW, width=4)
        base_l = Line(pc + np.array([-pw / 2, -ph / 2, 0]), pc + np.array([pw / 2, -ph / 2, 0])).set_stroke(GREY_C, 1.5)
        lg = VGroup(Tex("drawn", font_size=26), Tex("fabricated (blurred)", font_size=26, color=YELLOW)).arrange(RIGHT, buff=0.5)
        lg.next_to(base_l, UP, buff=ph + 0.18)
        xs, zs = L["xs"], L["zs"]
        jf = int(np.argmax(L["I_k"][:, len(xs) // 2]))
        selx = np.abs(xs) <= 3.0
        Ik, Ir = L["I_k"][jf][selx], L["I_r"][jf][selx]
        fc, fw_, fh = np.array([3.75, -1.95, 0]), 5.0, 1.3
        fk = plot_xy(xs[selx], Ik / Ik.max(), fc, fw_, fh, (-3, 3), (0, 1.05), color=WHITE, width=3)
        fr = plot_xy(xs[selx], Ir / Ik.max(), fc, fw_, fh, (-3, 3), (0, 1.05), color=YELLOW, width=4)
        fbase = Line(fc + np.array([-fw_ / 2, -fh / 2, 0]), fc + np.array([fw_ / 2, -fh / 2, 0])).set_stroke(GREY_C, 1.5)
        ftl = Tex("focus (computed)", font_size=26, color=GREY_A).next_to(fbase, DOWN, buff=0.1)
        lost = 1 - float(L["pow_r"]) / float(L["pow_k"])
        loss_l = Tex("$-%d\\%%$ in the focus" % round(100 * lost), font_size=32, color=YELLOW).move_to(fc + np.array([-1.55, 0.45, 0]))
        self.at(S("a2") + 4.8)
        self.play(FadeIn(items[2], shift=0.1 * RIGHT), Create(base_l), Create(drawn), FadeIn(lg[0]), run_time=0.8)
        self.at(S("a3") + 0.2)
        self.play(Create(fab), FadeIn(lg[1]), Indicate(items[2], color=YELLOW, scale_factor=1.04), run_time=1.0)
        self.at(S("a3") + 2.4)
        self.play(Create(fbase), Create(fk), Create(fr), FadeIn(ftl), run_time=0.9)
        self.play(FadeIn(loss_l, shift=0.1 * UP), run_time=0.5)
        self.finish()


# ============================================================ 7  recap =============
class Ch7Recap(Chapter):
    CH = 7
    TAG = "7\\quad Recap"

    def construct(self):
        S = self.S
        self.show_tag()
        L = np.load(data_path("lens.npz"))
        a, h2, nz, f0 = float(L["a"]), float(L["h2pi"]), int(L["nzone"]), float(L["f0"])
        # icon 1: a lens delays light -> curved wavefront
        r2 = np.linspace(0, a, 200)
        ht = np.clip(nz * h2 - (np.sqrt(r2 ** 2 + f0 ** 2) - f0) / 0.5, 0, None)
        c1 = np.array([-4.4, 0.85, 0])
        lens = glassify(lens_outline(r2, ht, c1[0], c1[1] - 0.75, 0.075, 0.12), width=1.5)
        Fp = c1 + np.array([0, 2.35, 0])                   # focus: wavefronts are cups centred on it (converging)
        arcs = VGroup(*[Arc(radius=1.55 - 0.32 * i, start_angle=-PI / 2 - 0.5, angle=1.0, arc_center=Fp).set_stroke(C_U, 3)
                        for i in range(3)])
        fdot = Dot(Fp, radius=0.06, color=WHITE)
        t1 = MathTex("\\times\\, e^{\\,i\\varphi(x)}", font_size=36).move_to(c1 + np.array([0, -1.3, 0]))
        l1 = Tex("only a delay", font_size=30, color=GREY_A).next_to(t1, DOWN, buff=0.2)
        # icon 2: a full turn is free
        c2 = np.array([0, 0.85, 0])
        dial = VGroup(Circle(radius=0.75).move_to(c2).set_stroke(GREY_B, 2),
                      Arc(radius=0.4, start_angle=0, angle=TAU_ - 0.05, arc_center=c2).set_stroke(YELLOW, 3),
                      fast_arrow(c2, c2 + 0.75 * RIGHT, YELLOW, 5, 0.18))
        t2 = MathTex("2\\pi \\equiv 0", font_size=36).move_to(c2 + np.array([0, -1.3, 0]))
        l2 = Tex("phase wraps", font_size=30, color=GREY_A).next_to(t2, DOWN, buff=0.2)
        # icon 3: a micron-thick kinoform
        c3 = np.array([4.4, 0.85, 0])
        rk = np.sort(np.concatenate([np.linspace(0, a, 1200), L["edges"][1:-1] - 1e-6, L["edges"][1:-1] + 1e-6]))
        hk = np.mod(np.clip(nz * h2 - (np.sqrt(rk ** 2 + f0 ** 2) - f0) / 0.5, 0, None), h2)
        kin = glassify(lens_outline(rk, hk, c3[0], c3[1] - 0.2, 0.075, 0.3), width=1.2)
        t3 = MathTex("\\approx 1\\," + MU, font_size=36).move_to(c3 + np.array([0, -1.3, 0]))
        l3 = Tex("a flat lens", font_size=30, color=GREY_A).next_to(t3, DOWN, buff=0.2)
        ar = VGroup(Arrow(c1 + np.array([1.45, 0, 0]), c2 + np.array([-1.0, 0, 0]), buff=0, stroke_width=4, color=GREY_B),
                    Arrow(c2 + np.array([1.0, 0, 0]), c3 + np.array([-1.65, 0, 0]), buff=0, stroke_width=4, color=GREY_B))
        until = Tex("\\dots until ``thin'' stops being true", font_size=34, color=C_EV).move_to([0, -1.55, 0])
        self.at(S("r1"))
        self.play(FadeIn(lens), FadeIn(arcs), FadeIn(fdot), FadeIn(t1), FadeIn(l1), run_time=0.8)
        self.at(S("r1") + 1.8)
        self.play(GrowArrow(ar[0]), FadeIn(dial), FadeIn(t2), FadeIn(l2), run_time=0.8)
        self.at(S("r1") + 4.1)
        self.play(GrowArrow(ar[1]), FadeIn(kin), FadeIn(t3), FadeIn(l3), run_time=0.8)
        self.at(S("r1") + 5.9)
        self.play(FadeIn(until, shift=0.1 * UP), run_time=0.7)
        # r2: the price
        prices = VGroup(Tex("steps: $\\eta = \\mathrm{sinc}^2(1/N)$", font_size=30),
                        Tex("colour: $f \\propto 1/\\lambda$", font_size=30),
                        Tex("fine zones: full wave", font_size=30)).arrange(RIGHT, buff=0.9).move_to([0, -2.45, 0])
        self.at(S("r2") + 0.7)
        self.play(FadeIn(prices[0], shift=0.1 * UP), run_time=0.5)
        self.at(S("r2") + 2.0)
        self.play(FadeIn(prices[1], shift=0.1 * UP), run_time=0.5)
        self.at(S("r2") + 3.2)
        self.play(FadeIn(prices[2], shift=0.1 * UP), run_time=0.5)
        # r3: outer zones look like a grating -> next time
        zx = np.linspace(-6.0, 6.0, 1600)
        chirp = lambda x_: 0.55 * np.mod(1.2 * x_ + 0.35 * x_ ** 2 / 6, 1.0)
        per = lambda x_: 0.55 * np.mod(1.6 * x_, 1.0)
        g1 = VMobject().set_points_as_corners([[x_, 0.6 + chirp(x_), 0] for x_ in zx]).set_stroke(GLASS_LINE, 3)
        g2 = VMobject().set_points_as_corners([[x_, 0.6 + per(x_), 0] for x_ in zx]).set_stroke(GLASS_LINE, 3)
        gl1 = Tex("outer zones of a lens", font_size=30, color=GREY_A).move_to([0, 1.75, 0])
        gl2 = Tex("a grating", font_size=30, color=C_K).move_to([0, 1.75, 0])
        nxt = next_card("periodic structures: gratings, Bloch waves and band gaps", y=-1.2)
        self.at(S("r3"))
        self.play(FadeOut(VGroup(lens, arcs, fdot, t1, l1, dial, t2, l2, kin, t3, l3, ar, until, prices)), run_time=0.5)
        self.play(Create(g1), FadeIn(gl1), run_time=1.0)
        self.at(S("r3") + 1.9)
        self.play(Transform(g1, g2), FadeTransform(gl1, gl2), run_time=1.0)
        self.at(S("r3") + 3.3)
        self.play(FadeIn(nxt, shift=0.15 * UP), run_time=0.9)
        self.finish(fade=0.9)


CHAPTERS = ["Ch0Hook", "Ch1Delay", "Ch2Wrap", "Ch3Steps", "Ch4Rainbow", "Ch5Thin", "Ch6Computer", "Ch7Recap"]
