"""E13 closing chapters: schematic extensions and a visual design-loop recap.

Every ray, pixel, and control in this module is a labelled conceptual diagram,
not a second optical calculation. Numerical evidence is shown in chapters 0--5.
"""
from __future__ import annotations

import os
import sys

_TAIL_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TAIL_HERE, "..", "common"))
import o5common

if o5common.EP_DIR != _TAIL_HERE:
    o5common.load_timing(_TAIL_HERE)
from o5common import *


def e13_tail_text(s, xy, size=28, color=WHITE):
    return Tex(s, font_size=size, color=color).move_to([*xy, 0])


def e13_tail_node(s, xy, color, width=2.25, height=1.05, size=28):
    box = RoundedRectangle(width=width, height=height, corner_radius=.12,
                           color=color, stroke_width=1.8,
                           fill_color=color, fill_opacity=.07)
    box.move_to([*xy, 0])
    return VGroup(box, e13_tail_text(s, xy, size, color))


def e13_tail_graph(labels, y=.55, colors=(GREEN, BLUE, YELLOW, RED), size=28):
    xs = (-4.95, -1.65, 1.65, 4.95)
    nodes = VGroup(*[e13_tail_node(s, (x, y), c, size=size)
                     for s, x, c in zip(labels, xs, colors)])
    links = VGroup(*[fast_arrow([a+1.2, y, 0], [b-1.2, y, 0],
                                GREY_B, width=2.3, tip=.12)
                     for a, b in zip(xs[:-1], xs[1:])])
    return nodes, links


def e13_tail_travel(scene, starts, ends, color, duration=1.6):
    """Show a schematic information flow using prebuilt dots and linear paths."""
    dots = VGroup(*[Dot(np.array(a), radius=.065, color=color) for a in starts])
    scene.add(dots)
    scene.play(*[dot.animate.move_to(np.array(b)) for dot, b in zip(dots, ends)],
               run_time=duration, rate_func=linear)
    scene.remove(dots)


def e13_tail_reverse(y=-.5):
    return VGroup(*[fast_arrow([b-1.2, y, 0], [a+1.2, y, 0],
                                PINK, width=2.7, tip=.13)
                     for a, b in zip((-4.95, -1.65, 1.65), (-1.65, 1.65, 4.95))])


def e13_tail_pixel_icon(center, pitch=.18, color=BLUE):
    return VGroup(*[Square(side_length=pitch*.75, color=color, stroke_width=1.1,
                          fill_color=color, fill_opacity=.12 + .12*((r+2*c) % 4))
                    .move_to([center[0]+(c-2)*pitch, center[1]+(r-2)*pitch, 0])
                    for r in range(5) for c in range(5)])


def e13_tail_controls(center, color=GREEN):
    lines, dots = VGroup(), VGroup()
    for j, fraction in enumerate((.3, .7)):
        y = center[1]+(.22 if j == 0 else -.22)
        lines.add(Line([center[0]-.55, y, 0], [center[0]+.55, y, 0],
                       color=color, stroke_width=2))
        dots.add(Dot([center[0]-.55+1.1*fraction, y, 0], color=color, radius=.08))
    return VGroup(lines, dots)


class Ch6Beyond(Chapter):
    CH = 6
    TAG = r"6\quad Extend the same chain"

    def construct(self):
        self.show_tag(.25)
        nodes, links = e13_tail_graph(("Design", "Optical model", "Measurement", "Loss"), size=26)
        heading = e13_tail_text("Differentiable optics", (0, 2.35), 40)
        detail = e13_tail_text("Usable derivatives with respect to design", (0, -1.55), 31, GREY_A)
        schematic = e13_tail_text("Schematic computation graph", (0, -2.7), 24, GREY_B)
        icons = VGroup(e13_tail_controls((-4.95, 1.6)),
                       e13_tail_pixel_icon((1.65, 1.6), pitch=.13, color=YELLOW))
        self.at(self.S("x1"))
        self.play(FadeIn(heading), FadeIn(schematic),
                  LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=.2),
                  Create(links), FadeIn(icons), FadeIn(detail), run_time=1.5)
        e13_tail_travel(self, [[-6.05, .55, 0]], [[6.05, .55, 0]], BLUE, 2.2)

        self.at(self.S("x2"))
        aut = e13_tail_text("Automatic differentiation", (0, 2.35), 40)
        rules = e13_tail_text("Apply local derivative rules along the graph", (0, -1.55), 30, PINK)
        rev = e13_tail_reverse(-.45)
        self.play(ReplacementTransform(heading, aut), ReplacementTransform(detail, rules),
                  Create(rev), run_time=.8)
        e13_tail_travel(self, [[5.9, -.45, 0]], [[-5.9, -.45, 0]], PINK, 2.2)
        explicit = e13_tail_text("This film writes the reverse calculation explicitly", (0, -2.15), 25, GREY_A)
        self.play(FadeIn(explicit), run_time=.5)

        self.at(self.S("x3"))
        adj = e13_tail_text("Adjoint methods", (0, 2.35), 40, PINK)
        adj_rule = e13_tail_text("A solver can supply its own reverse derivative", (0, -1.55), 30, PINK)
        solver = e13_tail_node("Optical solver", (-1.65, .55), GREEN, size=26)
        self.play(ReplacementTransform(aut, adj), ReplacementTransform(rules, adj_rule),
                  ReplacementTransform(nodes[1], solver), FadeOut(explicit), run_time=.8)
        nodes.remove(nodes[1]); nodes.add(solver)
        bracket = RoundedRectangle(width=2.55, height=2.05, corner_radius=.14,
                                   color=PINK, stroke_width=2).move_to([-1.65, .15, 0])
        self.play(Create(bracket), run_time=.65)
        e13_tail_travel(self, [[6, -.45, 0]], [[-6, -.45, 0]], PINK, 2.5)

        self.at(self.S("x4"))
        self.play(FadeOut(VGroup(nodes, links, icons, rev, adj, adj_rule, bracket, schematic)), run_time=.4)
        ray_title = e13_tail_text("Curved lenses", (0, 2.45), 40)
        ray_note = e13_tail_text("Schematic ray path; no lens optimization is run here", (0, -2.7), 24, GREY_B)
        lens = Ellipse(width=1.12, height=3.0, color=GREY_B, stroke_width=2,
                       fill_color=GREY_C, fill_opacity=.13).move_to([-1, .2, 0])
        axis = Line([-5.8, .2, 0], [5.7, .2, 0], color=GREY_D, stroke_width=1)
        p0 = np.array([-5.8, 1.1, 0]); p1 = np.array([-1.45, 1.1, 0])
        p2 = np.array([-.529, 1.012, 0]); p3 = np.array([5.2, -.6, 0])
        segments = VGroup(Line(p0, p1, color=BLUE, stroke_width=3),
                          Line(p1, p2, color=BLUE, stroke_width=3),
                          Line(p2, p3, color=BLUE, stroke_width=3))
        hits = VGroup(Dot(p1, radius=.08, color=GOLD), Dot(p2, radius=.08, color=GOLD))
        same = e13_tail_text("Follow one intersection branch", (-2.5, -1.65), 27, GOLD)
        snell = e13_tail_text("Differentiate refraction", (2.65, -1.65), 27, BLUE)
        self.play(FadeIn(ray_title), FadeIn(ray_note), FadeIn(axis), FadeIn(lens),
                  LaggedStart(*[Create(s) for s in segments], lag_ratio=.3), run_time=1.6)
        self.play(FadeIn(hits), FadeIn(same), FadeIn(snell), run_time=.7)
        self.play(Indicate(hits, color=PINK, scale_factor=1.5), run_time=1.5)
        ray_dot = Dot(p0, radius=.065, color=BLUE)
        self.add(ray_dot)
        self.play(ray_dot.animate.move_to(p1), run_time=.85, rate_func=linear)
        self.play(ray_dot.animate.move_to(p2), run_time=.35, rate_func=linear)
        self.play(ray_dot.animate.move_to(p3), run_time=1., rate_func=linear)
        self.remove(ray_dot)

        self.at(self.S("x5"))
        stopx = 2.7
        stop_y = p2[1] + (p3[1]-p2[1])*(stopx-p2[0])/(p3[0]-p2[0])
        stop = Line([stopx, stop_y+.45, 0], [stopx, 1.75, 0], color=GREY_A, stroke_width=7)
        cutoff = VGroup(Line([stopx-.1, stop_y-.1, 0], [stopx+.1, stop_y+.1, 0], color=RED, stroke_width=3),
                        Line([stopx-.1, stop_y+.1, 0], [stopx+.1, stop_y-.1, 0], color=RED, stroke_width=3))
        warning = e13_tail_text("Clipping and path switches can break smoothness", (0, -1.65), 30, RED)
        self.play(FadeIn(stop), FadeOut(same), FadeOut(snell), run_time=.35)
        self.play(stop.animate.shift(.6*DOWN), run_time=.65)
        self.play(FadeIn(cutoff), FadeIn(warning), segments[2].animate.set_opacity(.28), run_time=.5)

        self.at(self.S("x6"))
        self.play(FadeOut(VGroup(ray_title, ray_note, axis, lens, segments, hits,
                                stop, cutoff, warning)), run_time=.4)
        joint, joint_links = e13_tail_graph(("Scene", r"Optics +\\sensor", "Reconstruct", "Task loss"),
                                             y=.65, colors=(GOLD, GREEN, BLUE, RED), size=26)
        joint_title = e13_tail_text("Joint optical and digital design", (0, 2.55), 38)
        optical = e13_tail_text("Optical parameters", (-1.65, -1.3), 27, GREEN)
        digital = e13_tail_text("Digital parameters", (1.65, -1.3), 27, BLUE)
        back = VGroup(fast_arrow([4.9, -.15, 0], [1.65, -.15, 0], PINK, 2.5, .13),
                      fast_arrow([1.65, -.15, 0], [-1.65, -.15, 0], PINK, 2.5, .13),
                      fast_arrow([-1.65, -.35, 0], [-1.65, -.9, 0], PINK, 2.5, .13),
                      fast_arrow([1.65, -.35, 0], [1.65, -.9, 0], PINK, 2.5, .13))
        reference = VGroup(Line([-4.95, 1.25, 0], [-4.95, 1.9, 0], color=GREY_B, stroke_width=1.3),
                           Line([-4.95, 1.9, 0], [4.95, 1.9, 0], color=GREY_B, stroke_width=1.3),
                           fast_arrow([4.95, 1.9, 0], [4.95, 1.25, 0], GREY_B, 1.3, .10),
                           e13_tail_text("reference scene", (0, 2.05), 23, GREY_B))
        extension = e13_tail_text("Schematic extension", (0, -2.7), 24, GREY_B)
        self.play(FadeIn(joint_title), FadeIn(extension),
                  LaggedStart(*[FadeIn(n) for n in joint], lag_ratio=.15),
                  Create(joint_links), Create(reference), run_time=1.35)
        self.play(Create(back), FadeIn(optical), FadeIn(digital), run_time=.8)
        e13_tail_travel(self, [[4.85, -.15, 0]], [[-1.65, -.15, 0]], PINK, 1.7)
        self.play(Indicate(joint[3], color=RED, scale_factor=1.04), run_time=1.2)

        self.at(self.S("x7"))
        computed = e13_tail_text(r"Computed here: phase $\to$ intensity $\to$ loss", (0, -2.05), 29, GREEN)
        self.play(FadeIn(computed), run_time=.7)
        self.play(Indicate(computed, color=GREEN, scale_factor=1.035), run_time=2.2)
        self.finish()


class Ch7Recap(Chapter):
    CH = 7
    TAG = r"7\quad Close the design loop"

    def construct(self):
        self.show_tag(.25)
        nodes, links = e13_tail_graph(("Design", "Field", "Intensity", "Loss"), y=.85)
        forward = e13_tail_text("Predict", (0, 2.35), 38, BLUE)
        backward = e13_tail_text("Carry sensitivity back", (0, -.95), 32, PINK)
        rev = e13_tail_reverse(-.25)
        self.at(self.S("r1"))
        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=.17),
                  Create(links), FadeIn(forward), run_time=1.25)
        e13_tail_travel(self, [[-6, .85, 0]], [[6, .85, 0]], BLUE, 1.7)
        self.play(Create(rev), FadeIn(backward), run_time=.55)
        e13_tail_travel(self, [[6, -.25, 0]], [[-6, -.25, 0]], PINK, 1.7)

        self.at(self.S("r2"))
        steps = VGroup(*[e13_tail_node(s, (x, -2.1), c, width=2.5, height=.7, size=28)
                         for s, x, c in zip(("Check", "Update", "Recompute"),
                                            (-3.65, 0, 3.65), (GOLD, GREEN, RED))])
        step_links = VGroup(fast_arrow([-2.3, -2.1, 0], [-1.35, -2.1, 0], GREY_B, 2, .10),
                           fast_arrow([1.35, -2.1, 0], [2.3, -2.1, 0], GREY_B, 2, .10))
        self.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=.25), Create(step_links), run_time=1.15)
        self.play(Indicate(nodes[0], color=GREEN, scale_factor=1.035),
                  Indicate(nodes[3], color=RED, scale_factor=1.035), run_time=1.5)

        self.at(self.S("r3"))
        self.play(FadeOut(VGroup(nodes, links, forward, backward, rev, steps, step_links)), run_time=.4)
        limit = e13_tail_text("One wavelength; polarization omitted", (0, 2.35), 34, GREY_A)
        tests = e13_tail_text("What must be tested next?", (0, 1.55), 32)
        self.play(FadeIn(limit), FadeIn(tests), run_time=.7)
        pixels = e13_tail_pixel_icon((-4, .05), pitch=.3, color=BLUE)
        stairs = VMobject(color=GREEN, stroke_width=3)
        stairs.set_points_as_corners([[-1, -.6, 0], [-.65, -.6, 0], [-.65, -.25, 0],
                                      [-.3, -.25, 0], [-.3, .1, 0], [.1, .1, 0],
                                      [.1, .45, 0], [.5, .45, 0], [.5, .65, 0], [1, .65, 0]])
        samples = VGroup(*[RoundedRectangle(width=1.15, height=.9, corner_radius=.06,
                                            color=GOLD, stroke_width=1.6,
                                            fill_color=BLACK, fill_opacity=1)
                          .move_to([3.7+.22*j, -.2+.19*j, 0]) for j in range(3)])
        names = VGroup(e13_tail_text("Sensor noise", (-4, -1.3), 30, BLUE),
                       e13_tail_text("Fabrication", (0, -1.3), 30, GREEN),
                       e13_tail_text("New samples", (4, -1.3), 30, GOLD))
        scope = e13_tail_text("Schematic validation questions", (0, -2.55), 24, GREY_B)
        self.play(LaggedStart(FadeIn(VGroup(pixels, names[0])),
                              FadeIn(VGroup(stairs, names[1])),
                              FadeIn(VGroup(samples, names[2])), lag_ratio=.35),
                  FadeIn(scope), run_time=2.0)
        self.play(Indicate(samples[-1], color=GOLD, scale_factor=1.05), run_time=1.0)
        self.play(Indicate(pixels, color=BLUE, scale_factor=1.04),
                  Indicate(stairs, color=GREEN, scale_factor=1.04), run_time=1.4)

        self.at(self.S("r4"))
        self.play(FadeOut(VGroup(limit, tests, pixels, stairs, samples, names, scope)), run_time=.45)
        image = e13_tail_pixel_icon((-3.5, 1), pitch=.27, color=BLUE)
        controls = e13_tail_controls((3.5, 1), GREEN).scale(1.5)
        arrow = fast_arrow([-2.45, 1, 0], [2.45, 1, 0], PINK, 4, .2)
        image_label = e13_tail_text("Image error", (-3.5, -.05), 36, BLUE)
        design_label = e13_tail_text("Design change", (3.5, -.05), 36, GREEN)
        takeaway = e13_tail_text("Useful gradients need useful models.", (0, -1.45), 40)
        self.play(FadeIn(image), FadeIn(controls), GrowFromCenter(arrow),
                  FadeIn(image_label), FadeIn(design_label), run_time=1.25)
        self.play(Write(takeaway), run_time=1.65)
        e13_tail_travel(self, [[-2.35, 1, 0]], [[2.35, 1, 0]], PINK, 1.1)
        self.finish()
