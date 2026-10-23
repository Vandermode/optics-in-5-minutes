"""E12: Lens Design -- computed rays, explicit merit, and honest comparisons.

All ray paths, spot positions, scans and the ideal Airy reference come from
physics.py. Moving markers only indicate traversal; they are not time-domain
light simulations. Every redraw is a pure function of the chapter clock and
trackers. No numerical tracing, random sampling or text layout occurs per frame.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403


_DATA = None
AXIS_COLOR = GOLD
ALL_COLOR = BLUE
FIELD_COLORS = (BLUE, GOLD, PINK)


def D():
    global _DATA
    if _DATA is None:
        _DATA = dict(np.load(data_path("design.npz")))
        _DATA["design_centered"] = _DATA["design_spots"] - _DATA["design_centroids"][:, :, None, :]
        _DATA["scan_centered"] = _DATA["scan_spots"] - _DATA["scan_centroids"][:, :, :, None, :]
    return _DATA


def text(s, xy=(0, 0), size=30, color=WHITE):
    return Tex(s, font_size=size, color=color).move_to([*xy, 0])


def math(s, xy=(0, 0), size=34, color=WHITE):
    return MathTex(s, font_size=size, color=color).move_to([*xy, 0])


def till(scene, end, maximum=6.0):
    """Whole-frame animation duration ending before a spoken beat ends."""
    fps = config.frame_rate
    frames = max(1, int(np.floor(min(maximum, max(.15, end - scene.now - .15)) * fps)))
    return (frames - .5) / fps


def nearest(q):
    return int(np.argmin(abs(D()["q_scan"] - float(q))))


def rgba_points(xy, extent, color=BLUE, size=384, radius=2, count=None):
    """Rasterize computed point positions, keeping a transparent background."""
    xy = np.asarray(xy)
    if count is not None:
        xy = xy[:max(0, min(len(xy), int(count)))]
    im = np.zeros((size, size, 4), np.uint8)
    if not len(xy):
        return im
    px = np.rint((xy[:, 0] / extent + 1) * (size - 1) / 2).astype(int)
    py = np.rint((1 - xy[:, 1] / extent) * (size - 1) / 2).astype(int)
    rgb = np.rint(255 * np.asarray(ManimColor(color).to_rgb())).astype(np.uint8)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius + .5:
                continue
            xx, yy = px + dx, py + dy
            good = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
            im[yy[good], xx[good], :3] = rgb
            im[yy[good], xx[good], 3] = 255
    return im


class Spot:
    def __init__(self, scene, xy, center, width=3.0, extent=.16, title=None,
                 rms=None, color=BLUE, count=None, value_size=26):
        self.scene = scene
        self.center, self.width, self.extent = np.array([*center, 0.]), width, extent
        source = xy if callable(xy) else lambda: xy
        if callable(xy) or count is not None:
            self.image, self.fade = scene.live_image(
                lambda t: rgba_points(source(), extent, color,
                                      count=None if count is None else count.get_value()),
                width, self.center, width)
        else:
            self.image, self.fade = scene.static_image(rgba_points(xy, extent, color), width, self.center, width)
        self.frame = VGroup(Square(side_length=width, color=GREY_D, stroke_width=1.2).move_to(self.center),
                            Line(self.center + .09 * LEFT, self.center + .09 * RIGHT, color=GREY_B, stroke_width=1),
                            Line(self.center + .09 * DOWN, self.center + .09 * UP, color=GREY_B, stroke_width=1))
        if title:
            self.frame.add(text(title, (center[0], center[1] + width / 2 + .32), 28))
        self.value = None
        if rms is not None:
            self.value = math(rf"{rms * 1000:.1f}\,\mu\mathrm{{m}}", (center[0], center[1] - width / 2 - .28), value_size, color)
            self.frame.add(self.value)
        scene.add(self.image)

    def show(self, duration=.5):
        self.scene.play(self.fade.animate.set_value(1), FadeIn(self.frame), run_time=duration)

    def hide(self, duration=.4):
        self.scene.play(self.fade.animate.set_value(0), FadeOut(self.frame), run_time=duration)
        self.image.clear_updaters()
        self.scene.remove(self.image)


def ray_shape(profile, paths, image_z, center=(0, .3), width=11.5, height=4.2,
              now=0., color=BLUE, alpha=1., zlim=(-10., 56.), xlim=(-13., 13.), stop=True,
              bad=None, profile_y=None):
    """Draw the computed meridional projection (horizontal z, vertical x)."""
    cx, cy = center
    def point(z, x):
        return np.array([cx - width/2 + (z-zlim[0])/(zlim[1]-zlim[0])*width,
                         cy - height/2 + (x-xlim[0])/(xlim[1]-xlim[0])*height, 0.])
    yy = (D()["y_profile"] if profile_y is None else profile_y)[::4]
    front, back = profile[:, ::4]
    vertices = [point(z, x) for z, x in zip(front, yy)] + [point(z, x) for z, x in zip(back[::-1], yy[::-1])]
    glass = Polygon(*vertices, color=GREY_B, stroke_width=1.8, fill_color=GREY_C, fill_opacity=.22)
    result = VGroup(Line(point(zlim[0], 0), point(zlim[1], 0), color=GREY_D, stroke_width=.9), glass)
    if stop:
        result.add(Line(point(-5, xlim[0]), point(-5, -6), color=GREY_A, stroke_width=4),
                   Line(point(-5, 6), point(-5, xlim[1]), color=GREY_A, stroke_width=4))
    if image_z is not None:
        result.add(Line(point(image_z, xlim[0]+1), point(image_z, xlim[1]-1), color=GREEN, stroke_width=2))
    paths = np.asarray(paths)
    if paths.ndim == 3:
        paths = paths[None]
    for field_index, field_paths in enumerate(paths):
        ray_color = FIELD_COLORS[field_index] if len(paths) > 1 else color
        for j, coordinates in enumerate(field_paths[::2]):
            good = np.all(np.isfinite(coordinates), axis=1)
            coordinates = coordinates[good]
            if len(coordinates) < 2:
                continue
            pts = np.array([point(p[2], p[0]) for p in coordinates])
            failed = bad is not None and not bool(bad[2*j])
            c = RED if failed else ray_color
            result.add(poly_line(pts[:, 0], pts[:, 1], c, 1.6, .75).set_fill(opacity=0))
            if failed:
                end = pts[-1]
                result.add(Line(end+[-.07,-.07,0],end+[.07,.07,0],color=RED,stroke_width=2),
                           Line(end+[-.07,.07,0],end+[.07,-.07,0],color=RED,stroke_width=2))
            elif j % 3 == 0:
                distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                target = ((now / 3.8 + j * .127 + field_index * .2) % 1) * sum(distances)
                index = min(len(distances)-1, int(np.searchsorted(np.cumsum(distances), target)))
                before = sum(distances[:index])
                fraction = (target-before) / max(distances[index],1e-12)
                result.add(Dot(pts[index]*(1-fraction)+pts[index+1]*fraction, radius=.035, color=c))
    return fade_vgroup(result, alpha)


class RayView:
    def __init__(self, scene, profile, paths, image_z, **kwargs):
        self.scene, self.fade = scene, ValueTracker(0)
        pfn = profile if callable(profile) else lambda: profile
        rfn = paths if callable(paths) else lambda: paths
        zfn = image_z if callable(image_z) else lambda: image_z
        self.mob = always_redraw(lambda: ray_shape(pfn(), rfn(), zfn(), now=scene.now,
                                                    alpha=self.fade.get_value(), **kwargs))
        scene.add(self.mob)

    def show(self, duration=.6):
        self.scene.play(self.fade.animate.set_value(1), run_time=duration)

    def hide(self, duration=.4):
        self.scene.play(self.fade.animate.set_value(0), run_time=duration)
        self.mob.clear_updaters(); self.scene.remove(self.mob)


class Plot:
    def __init__(self, center, width, height, xlim, ylim, xlabel, ylabel,
                 xticks=None, yticks=None):
        self.center, self.width, self.height = center, width, height
        self.xlim, self.ylim = xlim, ylim
        self.axes = VGroup(Line(self.p(xlim[0],ylim[0]),self.p(xlim[1],ylim[0]),color=GREY_B,stroke_width=1.4),
                           Line(self.p(xlim[0],ylim[0]),self.p(xlim[0],ylim[1]),color=GREY_B,stroke_width=1.4),
                           text(xlabel,(center[0],center[1]-height/2-.5),26),
                           text(ylabel,(center[0],center[1]+height/2+.35),26))
        for value in ([] if xticks is None else xticks):
            p = self.p(value,ylim[0]); self.axes.add(Line(p+.05*UP,p+.05*DOWN,color=GREY_B),
                math(f"{value:g}",(p[0],p[1]-.22),22,GREY_A))
        for value in ([] if yticks is None else yticks):
            p = self.p(xlim[0],value); self.axes.add(Line(p+.05*LEFT,p+.05*RIGHT,color=GREY_B),
                math(f"{value:g}",(p[0]-.28,p[1]),22,GREY_A))

    def p(self, x, y):
        return np.array([self.center[0]-self.width/2+(x-self.xlim[0])/(self.xlim[1]-self.xlim[0])*self.width,
                         self.center[1]-self.height/2+(y-self.ylim[0])/(self.ylim[1]-self.ylim[0])*self.height,0.])

    def curve(self, x, y, color=BLUE, width=3):
        pts=np.array([self.p(a,b) for a,b in zip(x,y)])
        return poly_line(pts[:,0],pts[:,1],color,width)


def live_marker(scene, plot, tracker, xs, ys, color=GOLD):
    fade=ValueTracker(0)
    dot=Dot(radius=.075,color=color)
    dot.add_updater(lambda m:m.move_to(plot.p(tracker.get_value(),np.interp(tracker.get_value(),xs,ys)))
                    .set_opacity(fade.get_value()))
    scene.add(dot)
    return dot,fade


class Ch0Hook(Chapter):
    CH=0
    def construct(self):
        d=D()
        q=ValueTracker(0.)
        rays=RayView(self,lambda:d["scan_profiles"][nearest(q.get_value())],
                     lambda:d["scan_paths"][nearest(q.get_value()),0],
                     lambda:d["scan_z"][nearest(q.get_value()),0],center=(-2.25,.55),width=8.3,height=3.8)
        spot=Spot(self,lambda:d["scan_centered"][nearest(q.get_value()),0,0],(4.8,.55),2.5,.18,
                  title="On-axis rays",color=AXIS_COLOR)
        title=text("Same effective focal length. Different ray bundles.",(0,2.9),35)
        scale=text(r"Computed intersections $\cdot$ spot square: $0.36$ mm wide",(0,-2.15),25,GREY_A)
        self.at(self.S("h1")); self.play(FadeIn(title),FadeIn(scale),rays.fade.animate.set_value(1),
                                        spot.fade.animate.set_value(1),FadeIn(spot.frame),run_time=.7)
        self.play(q.animate.set_value(float(d["q_design"][1])),run_time=till(self,self.E("h1")))
        self.at(self.S("h2"))
        edge=Spot(self,d["design_centered"][1,2],(4.8,.55),2.5,.18,title=r"Same lens, $6^\circ$",color=PINK)
        self.play(spot.fade.animate.set_value(0),FadeOut(spot.frame),edge.fade.animate.set_value(1),FadeIn(edge.frame),run_time=1.)
        self.play(Indicate(edge.frame[0],color=PINK),run_time=1.2)
        self.at(self.S("h3")); rays.hide(.4)
        self.play(edge.fade.animate.set_value(0),FadeOut(edge.frame),FadeOut(title),FadeOut(scale),run_time=.4)
        question=text("Which lens should we build?",(0,.65),50)
        self.play(Write(question),run_time=1.2)
        self.at(self.S("h4")); self.play(FadeOut(question),run_time=.4)
        card=series_title(12,"Lens Design","choose what better means",y=.7,size=64)
        self.play(FadeIn(card,shift=.15*UP),run_time=.8)
        self.finish()


class Ch1Score(Chapter):
    CH=1; TAG=r"1\quad Define the score"
    def construct(self):
        d=D(); self.show_tag(.25)
        rays=RayView(self,d["design_profiles"][1],d["design_paths"][1,2],d["z_design"][1])
        recall=text(r"Intersect $\longrightarrow$ normal $\longrightarrow$ refract",(0,-2.35),32)
        self.at(self.S("s1")); rays.show(.6); self.play(FadeIn(recall),run_time=.5)
        self.at(self.S("s2")); detector=text("One object direction, many arrival positions",(0,2.85),33)
        self.play(FadeIn(detector),run_time=.6)
        self.at(self.S("s3")); rays.hide(.4)
        self.play(FadeOut(recall),FadeOut(detector),run_time=.35)
        count=ValueTracker(1)
        spot=Spot(self,d["design_centered"][1,2],(-3.2,.4),3.7,.18,title="Computed spot",count=count)
        spot.show(.5); self.play(count.animate.set_value(96),run_time=1.5,rate_func=linear)
        centre=Dot([-3.2,.4,0],radius=.07,color=YELLOW)
        centre_label=text("Centroid",(-.2,1.5),31,YELLOW)
        centre_arrow=fast_arrow([-.8,1.2,0],[-3.03,.5,0],YELLOW,width=2,tip=.12)
        self.play(FadeIn(centre),FadeIn(centre_label),Create(centre_arrow),run_time=.8)
        self.at(self.S("s4")); self.play(FadeOut(centre_label),FadeOut(centre_arrow),run_time=.3)
        equation=math(r"r_{\rm RMS}=\sqrt{\frac{1}{N}\sum_j\left|\mathbf{x}_j-\overline{\mathbf{x}}\right|^2}",
                      (2.9,.85),37)
        value=math(rf"r_{{\rm RMS}}={d['design_rms'][1,2]*1000:.1f}\,\mu\mathrm{{m}}",(2.9,-.25),33,BLUE)
        ring=Circle(radius=float(d["design_rms"][1,2])/.18*3.7/2,color=YELLOW,stroke_width=2).move_to(centre)
        sample_lines=VGroup()
        for xy in d["design_centered"][1,2,::12]:
            end=np.array([-3.2+xy[0]/.18*3.7/2,.4+xy[1]/.18*3.7/2,0])
            sample_lines.add(Line(centre.get_center(),end,color=YELLOW,stroke_width=1.1,stroke_opacity=.6))
        self.play(Create(sample_lines),Write(equation),run_time=2.)
        self.play(Create(ring),FadeIn(value),run_time=1.6)
        self.at(self.S("s5")); note=text("Geometric spread, not a diffraction PSF",(0,-2.5),31,GREY_A)
        self.play(FadeIn(note),Indicate(ring,color=YELLOW),run_time=2.5)
        self.at(self.S("s6")); note2=text("Spot spread and image position are different requirements",(0,-2.5),29,GREY_A)
        self.play(ReplacementTransform(note,note2),FadeOut(sample_lines),run_time=.7)
        centroid_value=math(rf"\overline{{x}}={d['design_centroids'][1,2,0]:.3f}\,\mathrm{{mm}}",(2.9,-1.25),30,GREY_A)
        self.play(FadeIn(centroid_value),run_time=.7)
        self.finish()


class Ch2Bending(Chapter):
    CH=2; TAG=r"2\quad Choose the variables"
    def construct(self):
        d=D(); self.show_tag(.25); q=ValueTracker(0.)
        rays=RayView(self,lambda:d["scan_profiles"][nearest(q.get_value())],
                     lambda:d["scan_paths"][nearest(q.get_value()),1],
                     lambda:d["scan_z"][nearest(q.get_value()),1],center=(-2.9,.5),width=6.6,height=4.,
                     zlim=(-2.,8.),stop=False)
        # In this lens close-up, only show the material, not paths extending beyond it.
        rays.mob.clear_updaters(); self.remove(rays.mob)
        def closeup():
            profile=d["scan_profiles"][nearest(q.get_value())]
            yy=d["y_profile"][::4]; zz=profile[:,::4]
            def p(z,x):return [-3.+(z-2)*.65,.4+x*.16,0]
            verts=[p(z,x) for z,x in zip(zz[0],yy)]+[p(z,x) for z,x in zip(zz[1,::-1],yy[::-1])]
            return fade_vgroup(Polygon(*verts,color=GREY_B,fill_color=GREY_C,fill_opacity=.25,stroke_width=2), rays.fade.get_value())
        rays.mob=always_redraw(closeup); self.add(rays.mob)
        formula=math(r"c_1=1/R_1\qquad c_2=1/R_2",(2.8,1.5),34)
        close_label=text("Lens cross-section: axial scale enlarged",(-2.7,-2.15),24,GREY_A)
        self.at(self.S("b1")); rays.show(.6); self.play(Write(formula),FadeIn(close_label),run_time=1.)
        self.at(self.S("b2")); constants=VGroup(text(r"$n=1.5$, thickness $4$ mm",(2.8,.45),29),
            text(r"Entrance radius $6$ mm",(2.8,-.2),29),text(r"Effective focal length $50$ mm",(2.8,-.85),29))
        self.play(LaggedStart(*[FadeIn(m) for m in constants],lag_ratio=.3),run_time=2.)
        self.play(q.animate.set_value(.35),run_time=3.2)
        self.at(self.S("b3")); qeq=math(r"q=\frac{c_1+c_2}{c_1-c_2}",(2.8,1.5),42,YELLOW)
        self.play(ReplacementTransform(formula,qeq),run_time=.8)
        self.play(q.animate.set_value(-1.2),run_time=3.5)
        self.at(self.S("b4")); self.play(q.animate.set_value(1.2),run_time=till(self,self.E("b4")),rate_func=linear)
        self.at(self.S("b5")); rays.hide(.35)
        self.play(FadeOut(qeq),FadeOut(constants),FadeOut(close_label),run_time=.4)
        full=RayView(self,d["design_profiles"][2],d["design_paths"][2],d["z_design"][2])
        full.show(.6)
        focal=text("Fixed effective focal length; detector position can change",(0,2.85),31)
        detector=text(r"One detector at $z="+f"{d['z_design'][2]:.3f}"+r"$ mm",(0,-2.3),31,GREEN)
        self.play(FadeIn(focal),FadeIn(detector),run_time=.8)
        self.at(self.S("b6")); shared=text("ONE plane shared by all object directions",(0,2.85),35,GREEN)
        self.play(ReplacementTransform(focal,shared),run_time=.8)
        self.finish()


class Ch3Axis(Chapter):
    CH=3; TAG=r"3\quad Optimize the centre"
    def construct(self):
        d=D(); self.show_tag(.25); q=ValueTracker(-1.2)
        plot=Plot((-2.55,.35),6.4,3.65,(-1.2,1.2),(0,90),"Bending $q$",r"On-axis RMS radius $(\mu\mathrm{m})$",
                  xticks=[-1.2,0,1.2],yticks=[0,40,80])
        values=1000*np.sqrt(d["scan_merit"][:,0]); curve=plot.curve(d["q_scan"],values,AXIS_COLOR)
        spot=Spot(self,lambda:d["scan_centered"][nearest(q.get_value()),0,0],(4.65,.5),2.7,.16,
                  title=r"Field $0^\circ$",color=AXIS_COLOR)
        dot,fade=live_marker(self,plot,q,d["q_scan"],values,AXIS_COLOR)
        note=text("Each shape uses its best on-axis detector plane",(0,-2.55),29,GREY_A)
        self.at(self.S("a1")); self.play(FadeIn(plot.axes),spot.fade.animate.set_value(1),FadeIn(spot.frame),FadeIn(note),run_time=.7)
        self.play(Create(curve),fade.animate.set_value(1),q.animate.set_value(1.2),run_time=till(self,self.E("a1")),rate_func=linear)
        self.at(self.S("a2")); samples=VGroup(*[Dot(plot.p(x,y),radius=.026,color=AXIS_COLOR) for x,y in zip(d["q_scan"][::4],values[::4])])
        self.play(LaggedStart(*[FadeIn(m) for m in samples],lag_ratio=.035),run_time=2.5)
        self.play(q.animate.set_value(-.1),run_time=2.4)
        self.at(self.S("a3")); self.play(q.animate.set_value(float(d["q_design"][1])),run_time=3.)
        optimum=math(rf"q={d['q_design'][1]:.3f}",(1.15,2.65),29,AXIS_COLOR)
        self.play(FadeIn(optimum),run_time=.6)
        self.at(self.S("a4")); result=math(rf"r_{{\rm RMS}}={d['design_rms'][1,0]*1000:.1f}\,\mu\mathrm{{m}}",(4.65,-1.55),29,AXIS_COLOR)
        self.play(FadeIn(result),Indicate(dot,color=AXIS_COLOR),run_time=1.2)
        self.at(self.S("a5")); edge=Spot(self,d["design_centered"][1,2],(4.65,.5),2.7,.16,
                                        title=r"Same design, $6^\circ$",color=PINK)
        edge_value=math(rf"{d['design_rms'][1,2]*1000:.1f}\,\mu\mathrm{{m}}",(4.65,-1.55),29,PINK)
        self.play(spot.fade.animate.set_value(0),FadeOut(spot.frame),edge.fade.animate.set_value(1),FadeIn(edge.frame),
                  ReplacementTransform(result,edge_value),run_time=.8)
        self.play(Indicate(edge.frame[0],color=PINK),run_time=1.4)
        self.finish()


class Ch4Fields(Chapter):
    CH=4; TAG=r"4\quad Score the whole task"
    def construct(self):
        d=D(); self.show_tag(.25)
        top=text("The on-axis design, tested at three fields",(0,2.95),34,AXIS_COLOR)
        first=[Spot(self,d["design_centered"][1,k],(x,.4),2.6,.16,title=rf"${int(d['fields'][k])}^\circ$",
                    rms=d["design_rms"][1,k],color=FIELD_COLORS[k]) for k,x in enumerate([-4.3,0,4.3])]
        scale=text(r"Identical square size: $0.32$ mm $\times$ $0.32$ mm",(0,-2.25),26,GREY_A)
        self.at(self.S("f1")); self.play(FadeIn(top),FadeIn(scale),run_time=.5)
        for panel in first:panel.show(.65)
        self.at(self.S("f2")); self.play(Indicate(first[0].frame[0],color=AXIS_COLOR),run_time=2.2)
        self.play(Indicate(first[2].frame[0],color=PINK),run_time=2.2)
        self.at(self.S("f3")); merit=math(r"J=\frac{r_0^2+r_3^2+r_6^2}{3}",(0,-2.55),38,GREEN)
        self.play(FadeOut(scale),Write(merit),run_time=1.6)
        self.play(*[Indicate(p.value,color=GREEN) for p in first],run_time=1.2)
        self.at(self.S("f4")); self.play(FadeOut(top),*[p.fade.animate.set_value(0) for p in first],
                                       *[FadeOut(p.frame) for p in first],run_time=.4)
        for p in first:p.image.clear_updaters();self.remove(p.image)
        grid=[]
        fields=VGroup(*[text(rf"${field}^\circ$",(x,2.93),28) for field,x in zip([0,3,6],[-2.8,.55,3.9])])
        row_labels=VGroup(text(r"On-axis\\objective",(-5.75,1.45),28,AXIS_COLOR),
                          text(r"Three-field\\objective",(-5.75,-.85),28,ALL_COLOR),
                          math(rf"q={d['q_design'][1]:.3f}",(-5.75,.75),23,AXIS_COLOR),
                          math(rf"z={d['z_design'][1]:.3f}\,\mathrm{{mm}}",(-5.75,.3),22,GREY_A),
                          math(rf"q={d['q_design'][2]:.3f}",(-5.75,-1.55),23,ALL_COLOR),
                          math(rf"z={d['z_design'][2]:.3f}\,\mathrm{{mm}}",(-5.75,-2.0),22,GREY_A))
        self.play(FadeIn(fields),FadeIn(row_labels),run_time=.6)
        for row,(index,cy,color) in enumerate([(1,1.45,AXIS_COLOR),(2,-.85,ALL_COLOR)]):
            for k,cx in enumerate([-2.8,.55,3.9]):
                panel=Spot(self,d["design_centered"][index,k],(cx,cy),1.7,.16,
                           rms=d["design_rms"][index,k],color=color,value_size=26)
                grid.append(panel)
            self.play(*[p.fade.animate.set_value(1) for p in grid[row*3:row*3+3]],
                      *[FadeIn(p.frame) for p in grid[row*3:row*3+3]],run_time=.7)
        self.play(merit.animate.scale(.8).move_to([.5,-2.8,0]),run_time=.5)
        self.at(self.S("f5")); compromise=text("The task determines the compromise",(.4,-2.8),30,GREEN)
        self.play(ReplacementTransform(merit,compromise),run_time=.7)
        self.play(Indicate(grid[0].value,color=AXIS_COLOR),Indicate(grid[3].value,color=ALL_COLOR),run_time=2.8)
        self.at(self.S("f6")); self.play(Indicate(grid[2].value,color=PINK),Indicate(grid[5].value,color=ALL_COLOR),run_time=1.5)
        final=text("Always report every tested field",(.4,-2.8),31,WHITE)
        self.play(ReplacementTransform(compromise,final),run_time=.6)
        self.finish()


class Ch5Constraints(Chapter):
    CH=5; TAG=r"5\quad Prevent false improvements"
    def construct(self):
        d=D(); self.show_tag(.25)
        yy=d["y_profile"][::4]; prof=d["collision_profile"][:,::4]
        p=lambda z,x:np.array([-2.8+(z-2)*.45,.4+x*.16,0.])
        front=np.array([p(z,x) for z,x in zip(prof[0],yy)]);back=np.array([p(z,x) for z,x in zip(prof[1],yy)])
        collision=VGroup(poly_line(front[:,0],front[:,1],BLUE,3),poly_line(back[:,0],back[:,1],RED,3))
        warning=text(r"Rejected example:\\surfaces collide",(3.,.8),36,RED)
        edge=text(r"Required edge thickness $\geq 0.5$ mm",(0,-2.4),30)
        self.at(self.S("c1")); self.play(Create(collision),FadeIn(warning),FadeIn(edge),run_time=1.3)
        self.play(Indicate(collision,color=RED),run_time=1.2)
        self.at(self.S("c2")); self.play(FadeOut(collision),FadeOut(warning),FadeOut(edge),run_time=.4)
        paths=RayView(self,d["clipped_profile"],d["clipped_paths"],d["z_design"][2],
                      center=(0,.4),width=11.5,height=3.7,bad=d["clipped_valid"],
                      profile_y=d["clipped_y_profile"])
        paths.show(.6)
        aperture=text("Forbidden change: a smaller clear opening",(0,2.8),32,RED)
        retained=text(rf"Only {int(d['clipped_accepted'])} of {int(d['clipped_requested'])} rays survive",(0,-2.15),34,RED)
        self.play(FadeIn(aperture),FadeIn(retained),run_time=.7)
        self.at(self.S("c3")); reject=text("Dropping rays is not an image-quality improvement",(0,-2.75),29,RED)
        self.play(Write(reject),run_time=1.5)
        self.at(self.S("c4")); paths.hide(.4)
        self.play(FadeOut(aperture),FadeOut(retained),FadeOut(reject),run_time=.35)
        pupil=Circle(radius=1.8*6/6.5,color=GREEN,stroke_width=2).move_to([-3,.3,0])
        points,fade=self.static_image(rgba_points(d["pupil_sparse"],6.5,GREEN),3.6,[-3,.3,0],3.6)
        self.add(points)
        fixed=VGroup(text("Fixed entrance stop",(2.6,1.3),36,GREEN),
                     math(r"a=6\,\mathrm{mm}\qquad z=-5\,\mathrm{mm}",(2.6,.4),33),
                     text("Same light collection",(2.6,-.6),31,GREY_A))
        self.play(Create(pupil),fade.animate.set_value(1),FadeIn(fixed),run_time=1.8)
        self.at(self.S("c5")); self.play(FadeOut(pupil),fade.animate.set_value(0),FadeOut(fixed),run_time=.4)
        boxes=VGroup()
        for y,label in [(1.5,"Fixed aperture and material"),(.35,"Valid surfaces and edge thickness"),(-.8,"Every entrance ray reaches the image")]:
            line=text(label,(.4,y),34)
            mark=math(r"\checkmark",(-5.5,y),37,GREEN)
            boxes.add(VGroup(mark,line))
        self.play(LaggedStart(*[FadeIn(m,shift=.1*RIGHT) for m in boxes],lag_ratio=.35),run_time=2.)
        bounds=math(r"-1.2\leq q\leq1.2",(0,-2.3),32,GREY_A)
        self.play(FadeIn(bounds),run_time=.5)
        self.play(Indicate(boxes[2],color=GREEN),run_time=1.2)
        self.finish()


class Ch6Validate(Chapter):
    CH=6; TAG=r"6\quad Check what the model predicts"
    def construct(self):
        d=D(); self.show_tag(.25)
        sparse,sf=self.static_image(rgba_points(d["pupil_sparse"],6.5,BLUE),3.65,[-3,.3,0],3.65)
        dense,df=self.static_image(rgba_points(d["pupil_dense"],6.5,BLUE,radius=1),3.65,[-3,.3,0],3.65)
        self.add(sparse,dense)
        circle=Circle(radius=1.685,color=GREY_B,stroke_width=1.5).move_to([-3,.3,0])
        counts=text(r"$96\ \longrightarrow\ 1{,}536$\\rays per field",(2.7,.7),40)
        unchanged=text("Hold the selected lens and detector fixed",(0,-2.4),29,GREY_A)
        self.at(self.S("v1")); self.play(sf.animate.set_value(1),FadeIn(circle),FadeIn(counts),FadeIn(unchanged),run_time=.6)
        self.play(sf.animate.set_value(0),df.animate.set_value(1),run_time=2.8)
        self.at(self.S("v2")); self.play(df.animate.set_value(0),FadeOut(circle),FadeOut(counts),FadeOut(unchanged),run_time=.4)
        plot=Plot((0,.35),9.8,3.65,(0,6),(0,80),"Object field angle (degrees)",r"Geometric RMS radius $(\mu\mathrm{m})$",
                  xticks=[0,1.5,3,4.5,6],yticks=[0,40,80])
        ca=plot.curve(d["field_curve"],d["field_rms"][1]*1000,AXIS_COLOR)
        cb=plot.curve(d["field_curve"],d["field_rms"][2]*1000,ALL_COLOR)
        legend=VGroup(text("On-axis objective",(-2.5,3.0),27,AXIS_COLOR),text("Three-field objective",(2.7,3.0),27,ALL_COLOR))
        self.play(FadeIn(plot.axes),FadeIn(legend),Create(ca),Create(cb),run_time=1.7)
        heldout=VGroup(*[Dot(plot.p(d["validation_fields"][j],d["validation_rms"][2,j]*1000),radius=.08,color=GREEN)
                        for j in [1,3]])
        note=text(r"New checks at $1.5^\circ$ and $4.5^\circ$",(0,-2.5),29,GREEN)
        self.play(FadeIn(heldout),FadeIn(note),run_time=.6)
        self.at(self.S("v3")); caveat=text("Constant glass index: no chromatic prediction",(0,-2.5),29,RED)
        self.play(ReplacementTransform(note,caveat),run_time=.7)
        field=ValueTracker(0)
        marker,mf=live_marker(self,plot,field,d["field_curve"],d["field_rms"][2]*1000,ALL_COLOR)
        self.play(mf.animate.set_value(1),field.animate.set_value(6),run_time=5.7,rate_func=linear)
        self.at(self.S("v4")); self.play(FadeOut(plot.axes),FadeOut(legend),FadeOut(ca),FadeOut(cb),FadeOut(heldout),FadeOut(caveat),run_time=.4)
        marker.clear_updaters();self.remove(marker)
        spot=Spot(self,d["design_centered"][1,0],(-3.2,.45),3.15,.08,title="Geometric ray intersections",color=AXIS_COLOR)
        airy,af=self.static_image(heat_rgba(d["airy"],1.,gamma=.35),3.15,[3.2,.45,0],3.15)
        self.add(airy)
        ideal=VGroup(Square(side_length=3.15,color=GREY_D,stroke_width=1.2).move_to([3.2,.45,0]),
                     text("Ideal circular-pupil diffraction",(3.2,2.35),27),
                     math(r"\lambda=550\,\mathrm{nm},\quad f/4.2",(3.2,-1.45),23,GREY_A))
        scales=text(r"Separate scales: $160\,\mu$m square / $20\,\mu$m square",(0,-1.8),25,GREY_A)
        reference=text(r"Ideal wave reference only; not this lens's PSF",(0,-2.45),30,WHITE)
        self.play(spot.fade.animate.set_value(1),FadeIn(spot.frame),af.animate.set_value(1),FadeIn(ideal),FadeIn(scales),FadeIn(reference),run_time=1.)
        airy_ring=Circle(radius=float(d["airy_first_null_um"])/10*3.15/2,color=YELLOW,stroke_width=1.4).move_to([3.2,.45,0])
        self.play(Create(airy_ring),run_time=1.4)
        self.play(Indicate(scales,color=YELLOW),run_time=2.)
        self.at(self.S("v5")); self.play(spot.fade.animate.set_value(0),FadeOut(spot.frame),af.animate.set_value(0),FadeOut(ideal),FadeOut(scales),FadeOut(reference),run_time=.4)
        self.play(FadeOut(airy_ring),run_time=.25)
        links=VGroup(text("E03",(-3.2,1.3),48,BLUE),text("Diffraction sets a wave-optical scale",(-3.2,.25),29),
                     text("E10",(3.2,1.3),48,PINK),text("Gradients follow a chosen objective",(3.2,.25),29))
        self.play(LaggedStart(*[FadeIn(m) for m in links],lag_ratio=.18),run_time=1.8)
        self.play(Indicate(links[0],color=BLUE),run_time=1.5)
        self.play(Indicate(links[2],color=PINK),run_time=1.7)
        self.at(self.S("v6")); method=text("Here: bending search + one shared detector position",(0,-1.25),32,GREEN)
        limitation=text("A bounded design study, not a universal best lens",(0,-2.35),29,GREY_A)
        self.play(FadeIn(method),run_time=1.2)
        self.at(self.S("v6")+4.)
        self.play(FadeIn(limitation),run_time=1.2)
        self.play(Indicate(method,color=GREEN),run_time=1.7)
        self.finish()


class Ch7Recap(Chapter):
    CH=7; TAG=r"7\quad Close the design loop"
    def construct(self):
        self.show_tag(.25)
        positions=[(-4.65,.7),(-1.55,.7),(1.55,.7),(4.65,.7)]
        names=["Specify","Trace","Measure","Change"]
        cards=VGroup(*[VGroup(RoundedRectangle(width=2.45,height=1.1,corner_radius=.12,color=GREEN if j==0 else GREY_B,
                      stroke_width=1.5).move_to([*p,0]),text(name,p,34)) for j,(name,p) in enumerate(zip(names,positions))])
        arrows=VGroup(*[fast_arrow([positions[j][0]+1.3,.7,0],[positions[j+1][0]-1.3,.7,0],BLUE,width=2,tip=.10) for j in range(3)])
        self.at(self.S("r1")); self.play(LaggedStart(*[FadeIn(c) for c in cards],lag_ratio=.35),Create(arrows),run_time=2.5)
        flow=ValueTracker(-5.8)
        dot=Dot(radius=.07,color=YELLOW)
        dot.add_updater(lambda m:m.move_to([flow.get_value(),-.2,0]));self.add(dot)
        self.play(flow.animate.set_value(5.8),run_time=3.,rate_func=linear)
        self.at(self.S("r2")); requirements=text(r"Same aperture $\cdot$ shared detector $\cdot$ all fields $\cdot$ all rays",(0,-1.3),30,GREY_A)
        self.play(FadeIn(requirements),run_time=.8)
        self.play(flow.animate.set_value(-5.8),run_time=4.2,rate_func=linear)
        self.at(self.S("r3")); dot.clear_updaters();self.play(FadeOut(dot),FadeOut(cards),FadeOut(arrows),FadeOut(requirements),run_time=.5)
        questions=VGroup(text("Questions to explore",(0,2.0),35,GREY_A),
                         text("What changes across colour?",(0,.75),42,BLUE),
                         text("What can several lens elements correct?",(0,-.45),38,GREEN))
        self.play(LaggedStart(*[FadeIn(m) for m in questions],lag_ratio=.3),run_time=2.7)
        self.at(self.S("r4")); self.play(FadeOut(questions),run_time=.4)
        takeaway=text(r"Choose what better means.\\Then make the comparison honest.",(0,.55),46)
        self.play(Write(takeaway),run_time=2.7)
        self.finish()


CHAPTERS=["Ch0Hook","Ch1Score","Ch2Bending","Ch3Axis","Ch4Fields","Ch5Constraints","Ch6Validate","Ch7Recap"]
