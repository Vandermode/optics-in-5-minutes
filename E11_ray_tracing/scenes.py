"""E11 · geometric ray tracing through a real curved singlet.

All rays and spot samples come from physics.py and the canonical course tracer.
Moving dots indicate the order of a ray trace, not a time-of-flight calculation.
The shared Chapter clock is already frame exact; no common helpers are patched.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common
o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403

_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = dict(np.load(data_path("rays.npz")))
    return _DATA


def txt(text, x=0., y=0., size=28, color=GREY_A):
    return Tex(text, font_size=size, color=color).move_to([x, y, 0])


def math(text, x=0., y=0., size=34, color=WHITE):
    return MathTex(text, font_size=size, color=color).move_to([x, y, 0])


def index(values, value):
    return int(np.clip(round((value-values[0])/(values[-1]-values[0])*(len(values)-1)), 0, len(values)-1))


def physical(p, scale=.155, offset=(-4.65, .3)):
    """True equal scales: 3-D (x,y,z) to the displayed meridional (z,x) plane."""
    return np.array([offset[0] + scale*p[2], offset[1] + scale*p[0], 0.])


def path_mobject(points, mapper=physical, color=C_U, width=2.4, opacity=1.):
    p = np.array([mapper(v) for v in points])
    return poly_line(p[:, 0], p[:, 1], color, width, opacity)


def ray_color(i, count):
    frac = abs((i-(count-1)/2)/max(1., (count-1)/2))
    return interpolate_color(C_U, C_REF, frac**1.4)


def fan(paths, mapper=physical, opacity=1., width=2.2):
    return VGroup(*[path_mobject(p, mapper, ray_color(i, len(paths)), width, opacity)
                    for i, p in enumerate(paths)])


def stage_points(points, stage):
    stage = float(np.clip(stage, 0., len(points)-1))
    k = min(int(stage), len(points)-2)
    return np.vstack([points[:k+1], points[k] + (stage-k)*(points[k+1]-points[k])])


def point_along(points, fraction):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.r_[0., np.cumsum(lengths)]
    at = np.clip(fraction, 0., 1.) * cum[-1]
    k = min(int(np.searchsorted(cum, at, side="right")-1), len(lengths)-1)
    k = max(0, k)
    return points[k] + (at-cum[k])/max(lengths[k], 1e-12)*(points[k+1]-points[k])


def riders(scene, paths_fn, mapper=physical, opacity=None, period=5., subset=2):
    """Pure-time markers on already computed paths, with tracker-based fades."""
    def draw():
        paths = paths_fn()
        alpha = 1. if opacity is None else opacity.get_value()
        if alpha < .001:
            return VGroup()
        return VGroup(*[Dot(mapper(point_along(p, ((scene.now-scene.t0)/period + .047*i) % 1.)),
                           radius=.035, color=ray_color(i, len(paths))).set_opacity(alpha)
                       for i, p in enumerate(paths) if i % subset == 0])
    return always_redraw(draw)


def lens_geometry(mapper=physical, axis=True, stop=None):
    d = data()
    p = np.array([mapper([x, 0, z]) for x, z in zip(d["heights"], d["front"])])
    q = np.array([mapper([x, 0, z]) for x, z in zip(d["heights"][::-1], d["back"][::-1])])
    outline = VMobject().set_points_as_corners(np.vstack([p, q, p[:1]]))
    outline.set_fill(C_MAT, .24).set_stroke(C_MAT, 2.2)
    group = VGroup(outline)
    if axis:
        group.add(fast_dashed(mapper([0, 0, -11]), mapper([0, 0, 70]), .12, GREY_D, 1.4))
    if stop is not None:
        for sign in [-1., 1.]:
            group.add(Line(mapper([sign*stop, 0, -5]), mapper([sign*12.7, 0, -5]),
                           color=GREY_B, stroke_width=6))
    return group


def at_plane(paths, z):
    """Clip each existing straight exit segment at a drawn detector plane."""
    result = np.array(paths, copy=True)
    for p in result:
        a, b = p[-2], p[-1]
        p[-1] = a + (z-a[2])/(b[2]-a[2])*(b-a)
    return result


class PlotBox:
    def __init__(self, center, width, height, xlim, ylim):
        self.center = np.array([*center[:2], 0.])
        self.width, self.height, self.xlim, self.ylim = width, height, xlim, ylim

    def xy(self, x, y):
        return self.center + np.array([(x-self.xlim[0])/(self.xlim[1]-self.xlim[0])*self.width-self.width/2,
                                      (y-self.ylim[0])/(self.ylim[1]-self.ylim[0])*self.height-self.height/2, 0.])

    def ray(self, p):
        return self.xy(p[2], p[0])

    def axes(self, xlabel, ylabel=None, xticks=(), yticks=(), color=GREY_B):
        left, right = self.center[0]-self.width/2, self.center[0]+self.width/2
        bottom, top = self.center[1]-self.height/2, self.center[1]+self.height/2
        g = VGroup(Line([left, bottom, 0], [right+.12, bottom, 0], color=color, stroke_width=1.4),
                   Line([left, bottom, 0], [left, top+.12, 0], color=color, stroke_width=1.4))
        g.add(txt(xlabel, self.center[0], bottom-.55, 25, color))
        if ylabel:
            g.add(txt(ylabel, left+.1, top+.4, 24, color).align_to([left, top+.4, 0], LEFT))
        for value in xticks:
            p = self.xy(value, self.ylim[0])
            g.add(Line(p+.055*UP, p-.055*UP, color=color, stroke_width=1.3))
            g.add(math(f"{value:g}", p[0], p[1]-.22, 22, color))
        for value in yticks:
            p = self.xy(self.xlim[0], value)
            g.add(Line(p+.055*LEFT, p-.055*LEFT, color=color, stroke_width=1.3))
            g.add(math(f"{value:g}", p[0]-.31, p[1], 22, color))
        return g

    def clipped_fan(self, paths, opacity=1., width=2.2):
        group = VGroup()
        for i, path in enumerate(paths):
            for a, b in zip(path[:-1], path[1:]):
                clipped = clip_to_box(self.ray(a), self.ray(b), self.center[0], self.center[1],
                                      self.width/2, self.height/2)
                if clipped is not None:
                    group.add(Line(*clipped, color=ray_color(i, len(paths)), stroke_width=width,
                                   stroke_opacity=opacity))
        return group


class SpotCloud(VGroup):
    """Persistent dots; updates place them absolutely and never accumulate state."""
    def __init__(self, box, samples_fn, opacity=None, color=C_U, centred=True):
        self.box, self.samples_fn, self.opacity_tracker, self.centred = box, samples_fn, opacity, centred
        sample = np.asarray(samples_fn())
        super().__init__(*[Dot(radius=.023, color=color) for _ in sample])
        self.add_updater(lambda mob: mob.update_points())
        self.update_points()

    def update_points(self):
        points = np.asarray(self.samples_fn())
        if self.centred:
            points = points-points.mean(axis=0)
        fade = 1. if self.opacity_tracker is None else self.opacity_tracker.get_value()
        for dot, p in zip(self.submobjects, points):
            # The image plane uses horizontal x and vertical y.
            inside = self.box.xlim[0] <= p[0] <= self.box.xlim[1] and self.box.ylim[0] <= p[1] <= self.box.ylim[1]
            dot.move_to(self.box.xy(*p)).set_opacity(fade if inside else 0.)
        return self


def focus_marker(z, mapper=physical, height=1.9, color=WHITE):
    p = mapper([0., 0., z])
    return VGroup(fast_dashed(p-height/2*UP, p+height/2*UP, .09, color, 1.8),
                  Dot(p, radius=.045, color=color))


class Ch0Hook(Chapter):
    CH = 0

    def construct(self):
        d, S = data(), self.S
        paths = d["fan_paths"][-1]
        geometry, rays = lens_geometry(stop=10), fan(paths)
        flow_alpha = ValueTracker(0.)
        moving = riders(self, lambda: paths, opacity=flow_alpha, period=5.5)
        computed = txt(r"Spherical glass lens \quad (computed rays)", 0, 2.85, 28)
        self.add(moving)
        self.play(FadeIn(geometry), FadeIn(computed), run_time=.7)
        self.at(S("h1"))
        self.play(Create(rays), run_time=2.4, rate_func=linear)
        self.play(flow_alpha.animate.set_value(1), run_time=.4)
        self.at(S("h2"))
        self.play(flow_alpha.animate.set_value(0), run_time=.3)
        moving.clear_updaters(); self.remove(moving)
        upper = VGroup(geometry, rays)
        self.play(upper.animate.scale(.60).move_to([0, 1.55, 0]), FadeOut(computed), run_time=.8)
        zoom = PlotBox((0, -1.15), 10.4, 1.65, (49., 54.2), (-.22, .22))
        enlarged = zoom.clipped_fan(paths)
        axis = Line(zoom.xy(49., 0), zoom.xy(54.2, 0), color=GREY_D, stroke_width=1)
        near = focus_marker(float(d["image_z"]), zoom.ray, 1.75)
        note = txt("Enlarged focal region: not a single point", 0, -2.43, 29)
        self.play(Create(enlarged), Create(axis), FadeIn(near), FadeIn(note), run_time=1.3)
        self.at(S("h3"))
        question = txt("Where does each ray actually go?", 0, 3.15, 37, WHITE)
        self.play(FadeIn(question, shift=.1*UP), run_time=.6)
        self.at(S("h4"))
        title = series_title(11, "Tracing Light Through a Lens", "intersect, refract, repeat", y=.35, size=49)
        self.play(FadeOut(VGroup(upper, enlarged, axis, near, note, question)), FadeIn(title), run_time=.8)
        self.finish()


class Ch1RayState(Chapter):
    CH = 1
    TAG = r"1\quad A ray is a position and a direction"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        points = d["selected_path"]
        r0, p1 = physical(points[0]), physical(points[1])
        ray = Line(r0, p1, color=C_U, stroke_width=3)
        start = Dot(r0, radius=.065, color=C_REF)
        alpha = ValueTracker(0.)
        moving = riders(self, lambda: points[None, :2], opacity=alpha, period=4.5, subset=1)
        self.add(moving)
        self.at(S("r1"))
        self.play(Create(ray), FadeIn(start), alpha.animate.set_value(1), run_time=1.)
        label = txt("a path, not the full field", -.1, -.8, 31)
        self.play(FadeIn(label), run_time=.5)
        self.at(S("r2"))
        origin_label = math(r"\mathbf r_0", r0[0]+.1, r0[1]+.43, 32, C_REF)
        direction = fast_arrow(r0+.45*RIGHT, r0+2.0*RIGHT, C_U, 4, .16)
        direction_label = math(r"\hat{\mathbf d}", r0[0]+1.3, r0[1]+.48, 35, C_U)
        eq = math(r"\mathbf r(t)=\mathbf r_0+t\hat{\mathbf d},\qquad \|\hat{\mathbf d}\|=1", 0, -1.8, 37)
        self.play(FadeOut(label), FadeIn(origin_label), Create(direction), FadeIn(direction_label), Write(eq), run_time=1.2)
        self.at(S("r3"))
        geometry = lens_geometry()
        self.play(FadeIn(geometry), run_time=1.)
        self.at(S("r4"))
        numbers = math(r"R_1=+50\,\mathrm{mm}\quad R_2=-50\,\mathrm{mm}\quad d=4\,\mathrm{mm}\quad n=1.5", 0, -2.65, 27)
        self.play(FadeIn(numbers), Indicate(geometry[0], color=WHITE, scale_factor=1.03), run_time=1.)
        self.at(S("r5"))
        slice_label = txt("Meridional slice of a 3D trace", .8, 2.6, 28)
        axis_label = math("z", 6.28, .02, 30, GREY_B)
        self.play(FadeIn(slice_label), FadeIn(axis_label), run_time=.7)
        self.at(S("r6"))
        stop = lens_geometry(axis=False, stop=10)[1:]
        stop_label = txt("entrance stop", -4.6, -1.65, 25, GREY_B)
        # Move the position formula up enough to leave the stop annotation clear.
        self.play(FadeIn(stop), FadeIn(stop_label), FadeOut(origin_label), FadeOut(direction_label), FadeOut(eq), run_time=.7)
        self.finish()


class Ch2Intersection(Chapter):
    CH = 2
    TAG = r"2\quad Find the physical surface"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        mapper = lambda p: physical(p, .048, (-1.7, .2))
        center = mapper([0, 0, 50])
        sphere = Circle(radius=2.4, color=GREY_D, stroke_width=1.8).move_to(center)
        cap = path_mobject(np.column_stack([d["heights"], np.zeros(401), d["front"]]), mapper, C_H, 5)
        roots = d["sphere_roots"]
        incoming = Line(mapper([8, 0, -10]), mapper([8, 0, 110]), color=C_U, stroke_width=3)
        near, far = [Dot(mapper([8, 0, z]), radius=.065, color=c) for z, c in zip(roots, [C_H, C_EV])]
        center_dot = Dot(center, radius=.045, color=GREY_B)
        source = Dot(mapper([8, 0, -10]), radius=.06, color=C_REF)
        line_note = txt("test the incoming line before refraction", -1.2, 2.95, 28)
        self.at(S("i1"))
        self.play(FadeIn(sphere), Create(cap), FadeIn(source), Create(incoming), FadeIn(line_note), run_time=1.5)
        self.at(S("i2"))
        eq = math(r"\|\mathbf r_0+t\hat{\mathbf d}-\mathbf c\|^2=R^2", -1.0, -2.65, 35)
        self.play(Write(eq), FadeIn(near), FadeIn(far), FadeIn(center_dot), run_time=2.5)
        near_label = txt("physical cap", -2.7, 1.18, 26, C_H)
        far_label = txt("other root", 3.5, 1.18, 26, C_EV)
        self.at(S("i3"))
        reject = Cross(far, stroke_color=C_EV, stroke_width=4).scale(2.1)
        self.play(FadeIn(near_label), FadeIn(far_label), Create(reject), run_time=2.8)
        self.at(S("i4"))
        criteria = txt(r"forward \quad + \quad correct cap \quad + \quad clear aperture", 0, -1.95, 28, WHITE)
        self.play(FadeIn(criteria), Indicate(cap, color=C_H, scale_factor=1.02), run_time=2.6)
        self.at(S("i5"))
        old = VGroup(sphere, cap, source, incoming, near, far, center_dot, line_note, near_label, far_label, reject, eq, criteria)
        local = lambda p: physical(p, .19, (-1.7, -.25))
        front = path_mobject(np.column_stack([d["heights"], np.zeros(401), d["front"]]), local, C_H, 4)
        p = local(d["selected_path"][1])
        ray = Line(local(d["selected_path"][0]), p, color=C_U, stroke_width=3)
        normal_vec = np.array([d["selected_normals"][0, 2], d["selected_normals"][0, 0], 0.])
        normal = fast_arrow(p, p+3*normal_vec, C_KZ, 4, .18)
        normal_line = Line(p-1.0*normal_vec, p+3.5*normal_vec, color=GREY_B, stroke_width=1.4)
        dot = Dot(p, radius=.07, color=WHITE)
        normal_label = math(r"\hat{\mathbf n}", p[0]+2.7, p[1]-.1, 38, C_KZ)
        centre_note = txt("toward the sphere centre", 2.55, -.1, 27, GREY_B)
        self.play(FadeOut(old), FadeIn(front), Create(ray), FadeIn(dot), run_time=.7)
        self.play(Create(normal_line), Create(normal), FadeIn(normal_label), FadeIn(centre_note), run_time=1.4)
        self.at(S("i6"))
        media = VGroup(txt("air", -3.5, -.8, 32), txt("glass", .5, -.8, 32))
        note = txt("The normal points into the next medium.", 0, -2.65, 30, WHITE)
        self.play(FadeIn(media), FadeIn(note), run_time=.8)
        angle = float(np.arctan2(normal_vec[1], normal_vec[0]))
        arc = Arc(radius=.8, start_angle=angle, angle=-angle, color=C_REF, stroke_width=3).move_arc_center_to(p)
        theta = math(r"\theta_i", p[0]+1.0, p[1]+.22, 29, C_REF)
        self.at(S("i6")+2.4)
        self.play(Create(arc), FadeIn(theta), run_time=1.2)
        self.finish()


class Ch3Refraction(Chapter):
    CH = 3
    TAG = r"3\quad Change direction with vector Snell"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        origin = np.array([-2.8, -.15, 0.])
        length = 2.25
        angle, eta, shown, out_alpha = ValueTracker(40.), ValueTracker(1.), ValueTracker(0.), ValueTracker(0.)
        normal = RIGHT
        tangent = UP
        def incidence():
            k = index(d["theta"], angle.get_value())
            q = d["incident"][k]
            return np.array([q[2], q[0], 0.])
        def output():
            k = index(d["theta"], angle.get_value())
            q = d["air_glass"][k] if eta.get_value() < 1 else d["glass_air"][k]
            return np.array([q[2], q[0], 0.])
        circle = Circle(radius=length, color=GREY_D, stroke_width=1.5).move_to(origin)
        interface = Line(origin+2.5*UP, origin-2.4*UP, color=GREY_B, stroke_width=2)
        n_arrow = fast_arrow(origin, origin+2.7*RIGHT, C_KZ, 3, .15)
        incoming = always_redraw(lambda: fast_arrow(origin, origin+length*incidence(), C_U, 4, .17))
        tangential = always_redraw(lambda: fast_arrow(origin, origin+length*incidence()[1]*UP, C_KX, 4, .17))
        normal_part = always_redraw(lambda: Line(origin+length*incidence()[1]*UP, origin+length*incidence(),
                                                color=C_KZ, stroke_width=2.4, stroke_opacity=shown.get_value()))
        def transmitted():
            q = output()
            if not np.all(np.isfinite(q)) or out_alpha.get_value() < .001:
                return VGroup()
            return fast_arrow(origin, origin+length*q, C_H, 4, .17, out_alpha.get_value())
        out = always_redraw(transmitted)
        tangent_label = txt("tangent", -4.0, 1.75, 27, C_KX)
        normal_label = txt("normal", -1.1, -.58, 27, C_KZ)
        self.add(incoming, tangential, normal_part, out)
        self.at(S("s1"))
        self.play(FadeIn(circle), Create(interface), Create(n_arrow), shown.animate.set_value(1),
                  FadeIn(tangent_label), FadeIn(normal_label), run_time=1.2)
        premise = txt("match tangential phase", 2.4, 1.65, 33, WHITE)
        self.at(S("s2"))
        self.play(FadeIn(premise), Indicate(tangential, color=C_KX, scale_factor=1.02), run_time=2.4)
        self.at(S("s3"))
        ratio = math(r"\eta=\frac{n_1}{n_2}\qquad \mathbf d_t'=\eta\,\mathbf d_t", 2.35, .45, 37)
        media = txt(r"air $\longrightarrow$ glass", 2.3, -1.1, 32)
        self.play(Write(ratio), FadeIn(media), eta.animate.set_value(1/1.5), out_alpha.animate.set_value(1), run_time=2.4)
        self.at(S("s4"))
        toward = txt("toward the normal", 2.3, -1.85, 29, C_H)
        self.play(FadeIn(toward), run_time=.5)
        self.play(angle.animate.set_value(60.), run_time=2.2, rate_func=smooth)
        self.at(S("s5"))
        full = math(r"\hat{\mathbf d}'=\eta\mathbf d_t+\sqrt{1-\eta^2\|\mathbf d_t\|^2}\,\hat{\mathbf n}", 0, -2.75, 34)
        self.play(Write(full), run_time=2.5)
        self.at(S("s6"))
        equivalent = txt("same law, expressed as vectors", 2.3, 1.65, 30, WHITE)
        self.play(FadeOut(premise), FadeIn(equivalent), run_time=.4)
        premise = equivalent
        self.play(angle.animate.set_value(30.), run_time=1.8)
        self.at(S("s7"))
        reverse = txt(r"glass $\longrightarrow$ air", 2.3, -1.1, 32)
        away = txt("away from the normal", 2.3, -1.85, 29, C_REF)
        self.play(FadeOut(media), FadeIn(reverse), FadeOut(toward), FadeIn(away), eta.animate.set_value(1.5), run_time=.65)
        media, toward = reverse, away
        self.play(angle.animate.set_value(40.), run_time=2.8)
        self.at(S("s8"))
        self.play(angle.animate.set_value(48.), run_time=1.4)
        tir = txt("total internal reflection", 2.3, -1.85, 30, C_EV)
        critical = math(r"\theta_c\approx41.8^\circ", 2.3, .45, 35, C_EV)
        self.play(FadeOut(toward), FadeIn(tir), FadeOut(ratio), FadeIn(critical), FadeOut(full), run_time=.7)
        stop = txt("No transmitted ray for this interface.", 0, -2.75, 30, C_EV)
        self.play(FadeIn(stop), run_time=.6)
        self.at(S("s8")+4.)
        self.play(Indicate(critical, color=C_EV, scale_factor=1.05), run_time=1.8)
        self.finish()


class Ch4ThroughLens(Chapter):
    CH = 4
    TAG = r"4\quad Repeat at the next surface"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        geometry = lens_geometry(stop=10)
        path = at_plane(d["selected_path"][None], float(d["image_z"]))[0]
        stage = ValueTracker(1.)
        ray = always_redraw(lambda: path_mobject(stage_points(path, stage.get_value()), physical, C_REF, 3.5))
        first = Dot(physical(path[1]), radius=.06, color=C_REF)
        second = Dot(physical(path[2]), radius=.065, color=C_H)
        view_label = txt(r"air \quad / \quad glass \quad / \quad air", 0, 2.65, 29)
        self.add(geometry, ray, first)
        self.at(S("l1"))
        self.play(stage.animate.set_value(2), FadeIn(second), FadeIn(view_label), run_time=2.4, rate_func=linear)
        self.at(S("l2"))
        hit = physical(path[2]); n = d["selected_normals"][1]
        normal = fast_arrow(hit, hit+1.5*np.array([n[2], n[0], 0.]), C_KZ, 3, .15)
        rule = math(r"n_1=1.5\quad\longrightarrow\quad n_2=1", 0, -1.75, 35)
        self.play(Create(normal), Write(rule), run_time=2.2)
        self.at(S("l3"))
        image_z = float(d["image_z"])
        detector = focus_marker(image_z, height=4.0, color=GREY_B)
        detector_label = txt("detector", physical([0, 0, image_z])[0], 2.6, 28)
        landing = Dot(physical(path[-1]), radius=.07, color=C_REF)
        self.play(stage.animate.set_value(3), FadeIn(detector), FadeIn(detector_label), run_time=2.5, rate_func=linear)
        self.play(FadeIn(landing), run_time=.4)
        self.at(S("l4"))
        bundle_paths = at_plane(d["fan_paths"][-1], image_z)
        bundle = fan(bundle_paths)
        alpha = ValueTracker(0.)
        flow = riders(self, lambda: bundle_paths, opacity=alpha, period=5.4)
        self.add(flow)
        self.play(FadeOut(rule), FadeOut(normal), FadeIn(bundle), alpha.animate.set_value(.9), run_time=1.)
        self.at(S("l5"))
        near = fan(at_plane(d["fan_paths"][0], image_z), opacity=1., width=3.0)
        position = txt(f"Paraxial image plane: $z={image_z:.2f}$ mm", 0, -2.2, 31, WHITE)
        self.play(bundle.animate.set_stroke(opacity=.2), FadeIn(near), FadeIn(position), run_time=.8)
        self.at(S("l6"))
        detail = txt("Small-angle reference; finite-aperture test next.", 0, -2.85, 27)
        self.play(FadeOut(near), bundle.animate.set_stroke(opacity=1.), FadeIn(detail), run_time=.8)
        self.finish()


class Ch5ImageError(Chapter):
    CH = 5
    TAG = r"5\quad Let the ray bundle test the focus"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        aperture, image_z, field = ValueTracker(1.), ValueTracker(float(d["image_z"])), ValueTracker(0.)
        geometry = always_redraw(lambda: lens_geometry(stop=aperture.get_value()))
        mesh = always_redraw(lambda: fan(d["fan_paths"][index(d["apertures"], aperture.get_value())]))
        opening = txt(r"Open the pupil: radius 1 mm $\longrightarrow$ 10 mm", 0, -2.6, 31)
        self.add(geometry, mesh)
        self.at(S("a1"))
        self.play(FadeIn(opening), aperture.animate.set_value(10.), run_time=4.4, rate_func=smooth)
        self.at(S("a2"))
        geometry.clear_updaters(); mesh.clear_updaters()
        zoom = PlotBox((-2.6, .45), 7.2, 2.8, (49., 54.2), (-.55, .55))
        spot = PlotBox((4.35, .45), 3.25, 2.8, (-.75, .75), (-.65, .65))
        axes = zoom.axes("axial position $z$ (mm)", "ray height (mm)", [49, 51, 53], [-.5, 0, .5])
        spot_axes = spot.axes("centred spot $x$ (mm)", "detector plane", [-.5, 0, .5], [])
        para = focus_marker(float(d["image_z"]), zoom.ray, zoom.height, GREY_B)
        para_label = txt("paraxial", zoom.ray([0, 0, float(d["image_z"])] )[0], 2.3, 24, GREY_B)
        zoom_rays = zoom.clipped_fan(d["fan_paths"][-1])
        line = always_redraw(lambda: Line(zoom.xy(image_z.get_value(), -.55), zoom.xy(image_z.get_value(), .55),
                                           color=C_H, stroke_width=2.8))
        cloud = SpotCloud(spot, lambda: d["focus_xy"][index(d["focus_zs"], image_z.get_value())])
        self.play(FadeOut(VGroup(geometry, mesh, opening)), FadeIn(axes), FadeIn(spot_axes),
                  Create(zoom_rays), FadeIn(para), FadeIn(para_label), run_time=2.8)
        self.add(line, cloud)
        note = txt("Exact rays, different axis crossings", 0, -2.65, 30, WHITE)
        self.play(FadeIn(note), run_time=.6)
        self.at(S("a3"))
        aberration = txt("Spherical aberration", 0, -2.65, 35, C_REF)
        self.play(FadeOut(note), FadeIn(aberration), run_time=.4)
        note = aberration
        self.play(Indicate(zoom_rays, color=C_REF, scale_factor=1.015), run_time=1.8)
        self.at(S("a4"))
        self.play(image_z.animate.set_value(float(d["best_z"])-1.1), run_time=2., rate_func=smooth)
        self.play(image_z.animate.set_value(float(d["best_z"])+.5), run_time=1.9, rate_func=smooth)
        self.at(S("a5"))
        best = float(d["best_z"])
        rms = float(np.sqrt(np.mean(np.sum((d["best_xy"]-d["best_xy"].mean(axis=0))**2, axis=1))))
        metric = txt(f"Centroid RMS at best focus: ${rms:.3f}$ mm", 0, -2.65, 30, C_H)
        self.play(image_z.animate.set_value(best), FadeOut(note), FadeIn(metric), run_time=.8)
        note = metric
        best_label = txt(f"best: {best:.2f} mm", -3.4, 2.75, 25, C_H)
        self.play(FadeIn(best_label), run_time=.5)
        centroid = Dot(spot.xy(0, 0), radius=.045, color=C_H)
        distances = VGroup(*[Line(spot.xy(0, 0), spot.xy(*d["best_xy"][k]), color=C_H,
                                 stroke_width=1.5, stroke_opacity=.65) for k in [12, 40, 84, 180]])
        self.at(S("a5")+2.2)
        self.play(FadeIn(centroid), Create(distances), run_time=2.)
        self.at(S("a6"))
        cloud.clear_updaters(); self.remove(cloud)
        field_cloud = SpotCloud(spot, lambda: d["field_xy"][index(d["fields"], field.get_value())])
        self.add(field_cloud)
        field_note = txt(r"Field 0$^\circ$ $\longrightarrow$ 8$^\circ$; same detector plane", 0, -2.65, 29, WHITE)
        # The ray plot remains the on-axis reference; the field comparison is a spot plot.
        on_axis = txt("on-axis reference", -2.6, 2.75, 25, GREY_B)
        self.play(FadeOut(note), FadeIn(field_note), FadeOut(best_label), FadeIn(on_axis),
                  FadeOut(centroid), FadeOut(distances), run_time=.4)
        note, best_label = field_note, on_axis
        self.play(field.animate.set_value(8.), run_time=3.3)
        self.at(S("a7"))
        field_cloud.clear_updaters(); self.remove(field_cloud)
        aperture_cloud = SpotCloud(spot, lambda: d["aperture_best_xy"][index(d["apertures"], aperture.get_value())])
        self.add(aperture_cloud)
        smaller = txt("Back on axis: smaller pupil, less collected light", 0, -2.65, 29, WHITE)
        self.play(FadeOut(note), FadeIn(smaller), run_time=.4)
        self.play(aperture.animate.set_value(3.), run_time=2.4)
        self.finish()


class Ch6Limits(Chapter):
    CH = 6
    TAG = r"6\quad Know what a ray model leaves out"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        pupil = ValueTracker(10.)
        spot = PlotBox((-3.6, .4), 3.5, 2.8, (-.35, .35), (-.28, .28))
        wave = PlotBox((3.15, .4), 5.0, 2.8, (-14., 14.), (0., 1.12))
        dots = SpotCloud(spot, lambda: d["best_xy"])
        spot_axes = spot.axes("ray landing $x$ (mm)", "geometric spot", [-.2, 0, .2], [])
        wave_axes = wave.axes(r"image coordinate ($\mu$m)", "ideal-pupil wave calculation", [-10, 0, 10], [])
        curve = always_redraw(lambda: plot_xy(d["psf_x_um"], d["ideal_psf"][index(d["apertures"], pupil.get_value())],
                                               wave.center, wave.width, wave.height, wave.xlim, wave.ylim, C_H, 3))
        self.at(S("m1"))
        self.add(dots)
        self.play(FadeIn(spot_axes), run_time=.7)
        not_psf = txt("Geometric samples $\neq$ diffraction PSF", 0, -2.45, 33, WHITE)
        self.play(FadeIn(not_psf), run_time=.6)
        self.at(S("m2"))
        self.add(curve)
        self.play(FadeIn(wave_axes), run_time=.6)
        reference = txt("Separate ideal circular pupil, 550 nm; peak-normalized", 0, -2.45, 26, GREY_A)
        self.play(FadeOut(not_psf), FadeIn(reference), run_time=.4)
        not_psf = reference
        self.play(pupil.animate.set_value(3.), run_time=3.6)
        self.at(S("m3"))
        dots.clear_updaters(); curve.clear_updaters()
        self.play(FadeOut(VGroup(dots, curve, spot_axes, wave_axes, not_psf)), run_time=.6)
        assumptions = VGroup(txt("one refractive index", -3.1, 1.4, 34, WHITE),
                             txt("ideal surfaces", -3.1, .2, 34, WHITE),
                             txt("transmitted paths", -3.1, -1., 34, WHITE))
        omissions = VGroup(txt("dispersion", 3.0, 1.4, 32, GREY_B),
                          txt("manufacturing errors", 3.0, .2, 32, GREY_B),
                          txt("reflected power / absorption", 3.0, -1., 30, GREY_B))
        crosses = VGroup(*[Line(t.get_left()-.12*RIGHT, t.get_right()+.12*RIGHT, color=C_EV, stroke_width=2.2)
                           for t in omissions])
        self.play(LaggedStart(*[AnimationGroup(FadeIn(a), FadeIn(o), Create(c))
                               for a, o, c in zip(assumptions, omissions, crosses)], lag_ratio=.7), run_time=4.3)
        self.at(S("m4"))
        self.play(FadeOut(VGroup(assumptions, omissions, crosses)), run_time=.6)
        statuses = VGroup(txt("transmitted", -4.5, 1.6, 31, C_H),
                          txt("missed / blocked", 0, 1.6, 31, C_EV),
                          txt("total internal reflection", 4.2, 1.6, 28, C_EV))
        note = txt("Count ray outcomes before comparing image errors.", 0, -2.55, 30, WHITE)
        arrows = VGroup(fast_arrow([-5.6, .1, 0], [-3.4, -.25, 0], C_H, 4, .15),
                         Line([-1.3, .1, 0], [.4, .1, 0], color=C_U, stroke_width=3),
                         Line([.4, -.7, 0], [.4, .8, 0], color=GREY_B, stroke_width=5),
                         Line([3.1, -.9, 0], [4.4, .4, 0], color=C_U, stroke_width=3),
                         Line([3.2, .4, 0], [5.7, .4, 0], color=GREY_B, stroke_width=2))
        self.play(FadeIn(statuses), FadeIn(note), Create(arrows), run_time=3.)
        self.at(S("m5"))
        self.play(FadeOut(VGroup(statuses, note, arrows)), run_time=.6)
        geometry = lens_geometry(stop=10)
        few = fan(d["fan_paths"][-1][::3], width=2.7)
        many = fan(d["fan_paths"][-1], width=1.9)
        density = txt("More rays sample the same model more densely.", 0, -2.65, 30, WHITE)
        self.play(FadeIn(geometry), FadeIn(few), FadeIn(density), run_time=.7)
        self.play(ReplacementTransform(few, many), run_time=3.)
        self.finish()


class Ch7Recap(Chapter):
    CH = 7
    TAG = r"7\quad Intersect, refract, repeat"

    def construct(self):
        d, S = data(), self.S
        self.show_tag()
        geometry = lens_geometry(stop=10)
        paths = d["fan_paths"][-1]
        rays = fan(paths, opacity=.65, width=2.)
        alpha = ValueTracker(0.)
        moving = riders(self, lambda: paths, opacity=alpha, period=5.7)
        self.add(geometry, rays, moving)
        words = ["intersect", "normal", "refract", "repeat"]
        recipe = VGroup(*[txt(word, -4.65+3.1*i, 2.7, 33, WHITE) for i, word in enumerate(words)])
        arrows = VGroup(*[fast_arrow([-3.45+3.1*i, 2.7, 0], [-2.75+3.1*i, 2.7, 0], GREY_B, 2.4, .12)
                          for i in range(3)])
        self.at(S("c1"))
        self.play(LaggedStart(*[FadeIn(t) for t in recipe], lag_ratio=.25), Create(arrows), alpha.animate.set_value(1), run_time=2.)
        self.at(S("c2"))
        conditions = txt(r"aperture \quad $\cdot$ \quad field angle \quad $\cdot$ \quad detector plane", 0, -2.45, 30)
        self.play(FadeIn(conditions), run_time=.6)
        self.at(S("c3"))
        marker = focus_marker(float(d["image_z"]), height=2.1, color=GREY_B)
        text = txt("A paraxial reference; an exact geometric test.", 0, -2.45, 30, WHITE)
        self.play(FadeIn(marker), FadeOut(conditions), FadeIn(text), run_time=.7)
        conditions = text
        self.at(S("c4"))
        next_episode = next_card("Design a lens by measuring its error", y=-2.25)
        self.play(FadeOut(conditions), FadeIn(next_episode), alpha.animate.set_value(.45), run_time=.9)
        self.finish()


CHAPTERS = ["Ch0Hook", "Ch1RayState", "Ch2Intersection", "Ch3Refraction", "Ch4ThroughLens",
            "Ch5ImageError", "Ch6Limits", "Ch7Recap"]
