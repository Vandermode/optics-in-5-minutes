"""Differentiable optics: computed phase, intensity, sensitivities and updates.
All plotted optical quantities are precomputed in physics.py. Diagrams are schematic.
"""
from __future__ import annotations
import os
import sys
from functools import lru_cache
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import o5common
o5common.load_timing(HERE)
from o5common import *


@lru_cache(None)
def data(name):
    with np.load(data_path(name)) as z:
        return {k: z[k] for k in z.files}


def label(text, at, size=28, color=GREY_A):
    return Tex(text, font_size=size, color=color).move_to(at)


def formula(text, at, size=36, color=WHITE):
    return MathTex(text, font_size=size, color=color).move_to(at)


class Plot:
    """Axes and numerical curves with explicit, fixed limits."""
    def __init__(self, center, width, height, xlim, ylim, xlabel, ylabel, xticks=(), yticks=()):
        self.center, self.w, self.h = np.array(center), width, height
        self.xlim, self.ylim = xlim, ylim
        p = self.point(xlim[0], ylim[0])
        self.axes = VGroup(Line(p, p + width * RIGHT, color=GREY_B, stroke_width=1.7),
                           Line(p, p + height * UP, color=GREY_B, stroke_width=1.7))
        self.axes.add(label(xlabel, self.center + [0, -height/2-.52, 0], 24),
                      label(ylabel, self.center + [0, height/2+.35, 0], 26))
        for val, txt in xticks:
            pt = self.point(val, ylim[0])
            self.axes.add(Line(pt+.045*UP, pt+.045*DOWN, color=GREY_B), label(txt, pt+.22*DOWN, 22))
        for val, txt in yticks:
            pt = self.point(xlim[0], val)
            self.axes.add(Line(pt+.045*LEFT, pt+.045*RIGHT, color=GREY_B), label(txt, pt+.29*LEFT, 22))
    def point(self, x, y):
        return self.center + [(x-self.xlim[0])/(self.xlim[1]-self.xlim[0])*self.w-self.w/2,
                              (y-self.ylim[0])/(self.ylim[1]-self.ylim[0])*self.h-self.h/2, 0]
    def curve(self, x, y, color=C_U, width=3):
        x, y = np.asarray(x), np.asarray(y)
        m = (x >= self.xlim[0]) & (x <= self.xlim[1])
        return plot_xy(x[m], y[m], self.center, self.w, self.h, self.xlim, self.ylim, color, width)


def interp(a, index):
    index = float(np.clip(index, 0, len(a)-1))
    lo = int(index)
    hi = min(lo+1, len(a)-1)
    return a[lo]*(1-index+lo) + a[hi]*(index-lo)


def legend(items, center, size=25):
    rows = VGroup(*[VGroup(Line(ORIGIN, .45*RIGHT, color=c, stroke_width=4),
                          Tex(t, color=c, font_size=size)).arrange(RIGHT, buff=.12) for t,c in items])
    return rows.arrange(RIGHT, buff=.45).move_to(center)


def arrow_chain(items, y=0, colors=None):
    colors = colors or [C_H, C_U, C_KX, C_EV]
    xs = np.linspace(-5.1, 5.1, len(items))
    nodes = VGroup()
    for i, (x, name) in enumerate(zip(xs,items)):
        box = RoundedRectangle(width=2.1, height=.9, corner_radius=.1, color=colors[i%len(colors)], stroke_width=2)
        box.set_fill(colors[i%len(colors)], .08).move_to([x,y,0])
        txt = formula(name,[x,y,0],32,colors[i%len(colors)])
        nodes.add(VGroup(box,txt))
    arrows = VGroup(*[fast_arrow([a+1.1,y,0],[b-1.1,y,0],GREY_B,2.5) for a,b in zip(xs[:-1],xs[1:])])
    return nodes, arrows


def note(scene, text, y=-2.55, color=GREY_A):
    m=label(text,[0,y,0],28,color)
    scene.play(FadeIn(m,shift=.1*UP),run_time=.45)
    return m


# Populated after physics data schema is finalized.
class Ch0Hook(Chapter):
    CH=0
    def construct(self):
        d=data('differentiable.npz'); S=self.S
        p=Plot([-3.7,.25,0],4.5,2.65,(-1,1),(-3,4),'Pupil position $x$','Phase (rad)',[(-1,'$-1$'),(0,'$0$'),(1,'$1$')],[(0,'$0$'),(3,'$3$')])
        q=Plot([3.4,.25,0],5.4,2.65,(-2,2),(0,.27),'Spatial frequency (cycles / half-width)','Power fraction per sample',[(-2,'$-2$'),(0,'$0$'),(2,'$2$')],[(0,'$0$'),(.25,'$.25$')])
        t=ValueTracker(0); mode=ValueTracker(0)
        def sweep(key):
            return interp(d[('sweep_a_' if mode.get_value()<.5 else 'sweep_b_')+key],t.get_value())
        phase=always_redraw(lambda:p.curve(d['x'],sweep('phase'),C_H))
        intensity=always_redraw(lambda:q.curve(d['frequency'],sweep('intensity'),C_U))
        self.add(p.axes,q.axes,phase,intensity)
        bridge=fast_arrow([-1.1,.4,0],[.4,.4,0],GREY_B,3)
        self.add(bridge,label('Fourier',[ -.35,.9,0],24))
        self.at(S('h1')); self.play(t.animate.set_value(80),run_time=2,rate_func=smooth)
        self.play(t.animate.set_value(40),run_time=.8)
        mode.set_value(1);t.set_value(0)
        self.play(t.animate.set_value(80),run_time=2.4,rate_func=smooth)
        self.at(S('h2'))
        power=note(self,r'Phase changes; total power $\sum I=1$ stays fixed.',color=C_H)
        self.play(t.animate.set_value(64),run_time=2)
        self.at(S('h3'))
        target=q.curve(d['frequency'],d['target_intensity'],C_REF,2.4)
        self.play(Create(target),FadeOut(power),run_time=.8)
        leg=legend([('Current',C_U),('Target',C_REF)],[3.5,2.4,0]);self.play(FadeIn(leg),run_time=.4)
        self.at(S('h4'))
        self.play(FadeOut(Group(*self.mobjects)),run_time=.45)
        title=series_title(13,'Differentiable Optics','From image error to design changes',y=.6,size=60)
        self.play(FadeIn(title,shift=.12*UP),run_time=.6)
        chain=label('Predict  $\\longrightarrow$  Differentiate  $\\longrightarrow$  Update',[0,-1.5,0],30,C_U)
        self.play(FadeIn(chain),run_time=.6)
        self.finish()


class Ch1Forward(Chapter):
    CH=1; TAG=r'1\quad From phase to intensity'
    def construct(self):
        d=data('differentiable.npz');S=self.S;self.show_tag(.25)
        p=Plot([-3.5,.25,0],4.8,2.5,(-1,1),(-3,4),'Pupil position $x$','Phase (rad)',[(-1,'$-1$'),(0,'$0$'),(1,'$1$')],[(0,'$0$'),(3,'$3$')])
        q=Plot([3.5,.25,0],4.9,2.5,(-2,2),(0,.27),'Spatial frequency (cycles / half-width)','Power fraction per sample',[(-2,'$-2$'),(0,'$0$'),(2,'$2$')],[(0,'$0$'),(.25,'$.25$')])
        t=ValueTracker(40); mode=ValueTracker(0)
        def values(key):
            return interp(d['sweep_a_'+key],t.get_value()) if mode.get_value()<.5 else interp(d['sweep_b_'+key],t.get_value())
        phase=always_redraw(lambda:p.curve(d['x'],values('phase'),C_H))
        inten=always_redraw(lambda:q.curve(d['frequency'],values('intensity'),C_U))
        self.at(S('f1')); self.play(FadeIn(p.axes),FadeIn(q.axes),run_time=.6);self.add(phase,inten)
        fixed=note(self,'Fixed aperture and amplitude',color=C_MAT)
        self.at(S('f2')); self.play(FadeOut(fixed),run_time=.3)
        eq=formula(r'\phi(x)=a x+b x^2',[0,2.65,0],38,C_H)
        self.play(Write(eq),run_time=.9)
        self.at(S('f3')); a=note(self,r'Tilt $a$: a phase ramp moves the pattern',color=C_H)
        self.play(t.animate.set_value(80),run_time=1.1,rate_func=linear)
        self.play(t.animate.set_value(40),run_time=1.1)
        self.at(S('f4'));self.play(FadeOut(a),run_time=.25)
        mode.set_value(1);t.set_value(0)
        b=note(self,r'Defocus $b$: quadratic phase spreads the pattern',color=C_H)
        self.play(t.animate.set_value(80),run_time=4.2,rate_func=linear)
        self.at(S('f5'));self.play(FadeOut(b),FadeOut(eq),run_time=.3)
        equation=formula(r'u=Ae^{i\phi}\quad\longrightarrow\quad v=Fu',[0,2.65,0],37)
        self.play(Write(equation),run_time=.9)
        n=note(self,r'$F$: unitary Fourier transform; $A$: fixed amplitude')
        self.at(S('f5')+3.5)
        self.play(Indicate(phase,color=C_H,scale_factor=1),run_time=1.0)
        travel=fast_arrow([-.75,.1,0],[.75,.1,0],C_U,2.5)
        self.play(Create(travel),run_time=.8)
        self.at(S('f5')+7)
        self.play(Indicate(inten,color=C_U,scale_factor=1),run_time=1.2)
        self.at(S('f6'));self.play(FadeOut(n),run_time=.3)
        out=formula(r'I=|v|^2\qquad\sum_k I_k=1',[0,-2.55,0],38,C_U)
        self.play(Write(out),Indicate(q.axes[1],color=C_U,scale_factor=1),run_time=1.1)
        self.finish()


class Ch2Loss(Chapter):
    CH=2; TAG=r'2\quad Define the error'
    def construct(self):
        d=data('differentiable.npz');S=self.S;self.show_tag(.25)
        q=Plot([-3.35,.15,0],5.25,2.7,(-2,2),(0,.27),'Frequency (cycles / half-width)','Power fraction per sample',[(-2,'$-2$'),(0,'$0$'),(2,'$2$')],[(0,'$0$'),(.25,'$.25$')])
        r=Plot([3.55,.15,0],4.8,2.7,(-2,2),(-.18,.18),'Frequency (cycles / half-width)',r'Residual $I-I_\star$',[(-2,'$-2$'),(0,'$0$'),(2,'$2$')],[(-.1,'$-.1$'),(0,'$0$'),(.1,'$.1$')])
        self.at(S('l1'));self.play(FadeIn(q.axes),run_time=.5)
        actual=q.curve(d['frequency'],d['initial_intensity'],C_U)
        target=q.curve(d['frequency'],d['target_intensity'],C_REF)
        self.play(Create(actual),Create(target),run_time=1)
        self.add(legend([('Start: $(a,b)=(1.5,2)$',C_U),('Target: $(0,0)$',C_REF)],[0,2.7,0]))
        self.at(S('l1')+4)
        self.play(Indicate(target,color=C_REF,scale_factor=1),run_time=1.4)
        self.at(S('l2'));self.play(FadeIn(r.axes),run_time=.5)
        residual=r.curve(d['frequency'],d['initial_residual'],C_EV)
        zero=Line(r.point(-2,0),r.point(2,0),color=GREY_B,stroke_width=1)
        self.play(Create(residual),Create(zero),run_time=.8)
        pick=np.where((abs(d['frequency'])<1.8))[0][::2]
        sticks=VGroup(*[Line(q.point(d['frequency'][i],d['target_intensity'][i]),q.point(d['frequency'][i],d['initial_intensity'][i]),color=C_EV,stroke_width=2) for i in pick])
        self.play(Create(sticks),run_time=1.3)
        marker=ValueTracker(-1.5)
        sample=always_redraw(lambda:Line(q.point(marker.get_value(),0),q.point(marker.get_value(),.25),color=GREY_B,stroke_width=1))
        self.add(sample)
        self.play(marker.animate.set_value(1.5),run_time=2.4,rate_func=linear)
        self.at(S('l3'))
        loss=formula(r'L=\frac{\sum_k(I_k-I_{\star,k})^2}{2S},\qquad S=\sum_k I_{\star,k}^2',[0,-2.55,0],32)
        self.play(Write(loss),run_time=1.1)
        self.at(S('l3')+4)
        self.play(FadeOut(sample),Indicate(residual,color=C_EV,scale_factor=1),run_time=1.2)
        self.at(S('l4'));self.play(Indicate(target,color=C_REF,scale_factor=1),Indicate(loss,color=C_REF,scale_factor=1),run_time=1.3)
        self.finish()


class Ch3Backward(Chapter):
    CH=3; TAG=r'3\quad Carry sensitivity backwards'
    def construct(self):
        d=data('differentiable.npz');S=self.S;self.show_tag(.25)
        nodes,arrows=arrow_chain([r'\theta=(a,b)',r'u=Ae^{i\phi}',r'v=Fu',r'L(I,I_\star)'],y=1.8)
        self.at(S('b1'));self.play(FadeIn(nodes),Create(arrows),run_time=1)
        back_arrows=VGroup(*[fast_arrow([b-1.1,1.05,0],[a+1.1,1.05,0],C_ADJ,3) for a,b in zip(np.linspace(-5.1,5.1,4)[:-1],np.linspace(-5.1,5.1,4)[1:])])
        self.play(LaggedStart(*[Create(m) for m in reversed(back_arrows)],lag_ratio=.3),run_time=1.2)
        r=Plot([0,-.65,0],8,1.9,(-2,2),(-.18,.18),'Frequency (cycles / pupil half-width)',r'Residual $I-I_\star$',[(-2,'$-2$'),(0,'$0$'),(2,'$2$')],[(-.1,'$-.1$'),(0,'$0$'),(.1,'$.1$')])
        self.at(S('b2'))
        rr=r.curve(d['frequency'],d['initial_residual'],C_EV)
        zr=Line(r.point(-2,0),r.point(2,0),color=GREY_B,stroke_width=1)
        self.play(FadeIn(r.axes),Create(rr),Create(zr),run_time=.8)
        self.at(S('b3'));self.play(FadeOut(r.axes),FadeOut(rr),FadeOut(zr),run_time=.4)
        q=formula(r'r_kv_k=\frac{I_k-I_{\star,k}}{S}\,v_k',[0,-.3,0],52,C_ADJ)
        qdesc=label('Sensor sensitivity: the residual weighted by the focal field',[0,-1.45,0],28)
        self.play(Write(q),FadeIn(qdesc),run_time=1)
        self.at(S('b4'));self.play(FadeOut(qdesc),run_time=.3)
        ceq=formula(r'w=F^\dagger(r\odot v)',[0,-.3,0],52,C_ADJ)
        self.play(Transform(q,ceq),Indicate(back_arrows[1],color=C_ADJ,scale_factor=1),run_time=1)
        desc=note(self,r'$F^\dagger$: conjugate transpose; here, the inverse Fourier transform',y=-1.65)
        self.at(S('b5'));self.play(FadeOut(desc),run_time=.3)
        desc=note(self,'A mathematical sensitivity field',y=-1.65,color=C_ADJ)
        self.play(Indicate(q,color=C_ADJ,scale_factor=1.04),run_time=1)
        self.at(S('b6'));self.play(FadeOut(desc),FadeOut(q),run_time=.35)
        maxg=max(abs(d['initial_phase_gradient']))
        g=Plot([-3.1,-.55,0],5.3,1.65,(-1,1),(-1.15*maxg,1.15*maxg),'Pupil position $x$',r'Phase sensitivity $g_\phi$ (1/rad)',[(-1,'$-1$'),(0,'$0$'),(1,'$1$')],[(-.02,'$-.02$'),(0,'$0$'),(.02,'$.02$')])
        # rebuild zero line after selecting a symmetric data scale
        gc=g.curve(d['x'],d['initial_phase_gradient'],C_ADJ)
        geq=formula(r'g_\phi=2\,\mathrm{Im}(u^*w)',[3.6,-.1,0],37,C_ADJ)
        self.play(FadeIn(g.axes),Create(gc),Write(geq),run_time=1)
        self.at(S('b7'))
        sums=formula(r'\frac{\partial L}{\partial a}=\sum_xg_\phi x\qquad\frac{\partial L}{\partial b}=\sum_xg_\phi x^2',[0,-2.6,0],33)
        self.play(Write(sums),Indicate(nodes[0],color=C_H,scale_factor=1),run_time=1.3)
        self.finish()


class Ch4Check(Chapter):
    CH=4; TAG=r'4\quad Check the derivative'
    def construct(self):
        d=data('differentiable.npz');S=self.S;self.show_tag(.25)
        eq=formula(r'g_j\;\approx\;\frac{L(\theta+h e_j)-L(\theta-h e_j)}{2h}',[0,2.45,0],38)
        p=Plot([0,-.2,0],9.8,2.8,(-10,-1),(-12,-2),'Perturbation $h$ (rad, log scale)','Maximum absolute gradient error (1/rad)', [(-10,'$10^{-10}$'),(-7,'$10^{-7}$'),(-4,'$10^{-4}$'),(-1,'$10^{-1}$')],[(-12,'$10^{-12}$'),(-8,'$10^{-8}$'),(-4,'$10^{-4}$')])
        self.at(S('c1'))
        minus=VGroup(formula(r'\theta-h e_j',[-3.1,.5,0],40,C_U),label('recompute loss',[-3.1,-.35,0],28))
        plus=VGroup(formula(r'\theta+h e_j',[3.1,.5,0],40,C_REF),label('recompute loss',[3.1,-.35,0],28))
        self.play(FadeIn(minus),FadeIn(plus),run_time=.8)
        self.at(S('c2'));self.play(Write(eq),run_time=.9)
        self.at(S('c3'));self.play(FadeOut(minus),FadeOut(plus),FadeIn(p.axes),run_time=.5)
        err=p.curve(np.log10(d['fd_h']),np.log10(d['fd_error']),C_ADJ)
        self.play(Create(err),run_time=2.4)
        self.play(FadeIn(label('roundoff',[-3.4,.8,0],25,GREY_A)),FadeIn(label('large perturbation',[2.4,.8,0],25,GREY_A)),run_time=.6)
        self.at(S('c4'))
        values=label(r'Backward: $(0.21890,\,-0.04178)$  $\approx$  finite difference',[0,-2.55,0],29,C_ADJ)
        self.play(FadeIn(values),run_time=.5)
        k=int(np.argmin(abs(np.log10(d['fd_h'])+5)))
        pt=p.point(np.log10(d['fd_h'][k]),np.log10(d['fd_error'][k]))
        self.play(Create(Circle(radius=.15,color=C_REF).move_to(pt)),run_time=.6)
        self.at(S('c5'));self.play(FadeOut(values),run_time=.3)
        note(self,'Checks the implemented derivative; physical accuracy needs its own test.',color=GREY_A)
        self.finish()


class Ch5Update(Chapter):
    CH=5; TAG=r'5\quad Update, then evaluate again'
    def construct(self):
        d=data('differentiable.npz');S=self.S;self.show_tag(.25)
        eq=formula(r'\theta_{\rm trial}=\theta-\eta\nabla_\theta L',[0,2.6,0],38)
        p=Plot([-3.4,.1,0],4.9,2.8,(-2,2),(0,.27),'Frequency (cycles / half-width)','Power fraction per sample',[(-2,'$-2$'),(0,'$0$'),(2,'$2$')],[(0,'$0$'),(.25,'$.25$')])
        q=Plot([3.5,.1,0],4.8,2.8,(0,40),(-5,0),'Accepted update','Loss $L$ (log scale)',[(0,'$0$'),(20,'$20$'),(40,'$40$')],[(-5,'$10^{-5}$'),(-3,'$10^{-3}$'),(-1,'$10^{-1}$')])
        t=ValueTracker(0)
        def idx():return int(np.clip(round(t.get_value()),0,40))
        curve=always_redraw(lambda:p.curve(d['frequency'],d['history_intensity'][idx()],C_U))
        trail=always_redraw(lambda:q.curve(np.r_[0,d['history'][:idx()+1,0]],np.log10(np.r_[d['history'][0,3],d['history'][:idx()+1,3]]),C_U))
        dot=always_redraw(lambda:Dot(q.point(idx(),np.log10(d['history'][idx(),3])),radius=.065,color=C_U))
        self.at(S('u1'));self.play(Write(eq),FadeIn(p.axes),FadeIn(q.axes),run_time=.8)
        target=p.curve(d['frequency'],d['target_intensity'],C_REF,2)
        self.add(target,curve,trail,dot)
        current_legend=legend([('Current',C_U),('Target',C_REF)],[-3.4,2.15,0],24)
        self.add(current_legend)
        self.at(S('u2'))
        step=note(self,r'First step: $b:2\to2.50$ while $L:0.261\to0.196$',color=C_ADJ)
        self.play(t.animate.set_value(1),run_time=1.4)
        self.at(S('u3'));self.play(FadeOut(step),run_time=.3)
        search=note(self,r'Try a step $\longrightarrow$ recompute $L$ $\longrightarrow$ accept a decrease')
        self.play(Indicate(eq,color=C_ADJ,scale_factor=1.02),run_time=.8)
        self.at(S('u4'));self.play(FadeOut(search),run_time=.3)
        self.play(t.animate.set_value(40),run_time=5.4,rate_func=linear)
        result=note(self,r'$40$ updates: $L=0.2606\;\longrightarrow\;0.000046$',color=C_U)
        self.at(S('u5'));self.play(FadeOut(result),run_time=.3)
        self.play(FadeOut(curve),FadeOut(current_legend),run_time=.3)
        alternate=ValueTracker(0)
        def alt_idx():return int(np.clip(round(alternate.get_value()),0,40))
        difficult_image=always_redraw(lambda:p.curve(d['frequency'],d['difficult_history_intensity'][alt_idx()],C_EV))
        difficult=always_redraw(lambda:q.curve(np.r_[0,d['difficult_history'][:alt_idx()+1,0]],
            np.log10(np.r_[d['difficult_history'][0,3],d['difficult_history'][:alt_idx()+1,3]]),C_EV))
        alternate_legend=legend([('Current: start $(3,3)$',C_EV),('Target',C_REF)],[-3.4,2.15,0],23)
        self.add(difficult_image,difficult,alternate_legend)
        leg=legend([('Start $(1.5,2)$',C_U),('Start $(3,3)$',C_EV)],[0,-2.55,0],26)
        self.play(FadeIn(leg),alternate.animate.set_value(40),run_time=2.4,rate_func=linear)
        self.at(S('u6'));self.play(FadeOut(leg),run_time=.3)
        note(self,'The second run stalls at $L=0.299$; a zero-loss design is known.',color=C_EV)
        self.finish()


from scenes_tail import Ch6Beyond as _TailBeyond, Ch7Recap as _TailRecap


class Ch6Beyond(_TailBeyond):
    """Local wrapper so Manim's scene discovery includes this chapter."""


class Ch7Recap(_TailRecap):
    """Local wrapper so Manim's scene discovery includes this chapter."""


CHAPTERS=['Ch0Hook','Ch1Forward','Ch2Loss','Ch3Backward','Ch4Check','Ch5Update','Ch6Beyond','Ch7Recap']
