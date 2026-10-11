"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).

    CUDA_VISIBLE_DEVICES=<free gpu> python physics.py [part ...]

Parts: hook bump leap cfl disp boundary bragg relief (default: all).  Deterministic (no randomness).

Everything is computed with the method the episode teaches: a 2D Yee-grid FDTD for the TMz fields (Ez out of the
screen, Hx and Hy in the plane), leapfrogged in time, with a convolutional PML (the stretched coordinate
x -> x + (i/omega) * integral of sigma, implemented in the time domain, Roden & Gedney 2000).  The 1D demos use the
same scheme in one dimension.

Units: wavelength lambda0 = 1, c = 1, eps0 = mu0 = 1 (so the period is 1 and the vacuum impedance is 1).
Grid: dx = 1 / N (N cells per vacuum wavelength), dt = S dx (S = Courant number; 2D limit 1/sqrt(2)).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


# =============================================================== 2D FDTD core ====
def cpml_coefs(n, npml, dt, m=3, R=1e-7, alpha_max=0.0, half=False, lo=True, hi=True, dx=1.0):
    """CPML recursion coefficients (b, a) along one axis with n nodes; PML occupies npml cells at each chosen end.
    half=True gives them at the half-integer positions (where the H components live).
    sigma(d) = sigma_max (d/L)^m with sigma_max = -(m+1) ln R / (2 L)  (eta0 = 1)."""
    p = np.arange(n) + (0.5 if half else 0.0)
    d = np.zeros(n)
    if lo:
        d = np.maximum(d, npml - p)
    if hi:
        d = np.maximum(d, p - (n - 1 - npml))
    d = np.clip(d, 0, npml) * dx
    L = npml * dx
    sig = (-(m + 1) * np.log(R) / (2 * L)) * (d / L) ** m if npml > 0 else 0 * d
    alpha = alpha_max * (1 - d / L) * (d > 0) if npml > 0 else 0 * d
    b = np.exp(-(sig + alpha) * dt)
    a = np.where(sig > 0, sig / np.maximum(sig + alpha, 1e-30) * (b - 1.0), 0.0)
    return b, a, sig


def sigma_profile(n, npml, dx, m=3, R=1e-7, half=False):
    return cpml_coefs(n, npml, 1.0, m, R, 0.0, half, dx=dx)[2]


class Grid2D:
    """TMz Yee grid: Ez[i, j] at (i dx, j dx); Hx[i, j] at (i, j+1/2); Hy[i, j] at (i+1/2, j).
    Outer edge is a perfect conductor (Ez = 0) unless periodic_x.  PML (CPML) of npml cells on the chosen sides."""

    def __init__(self, nx, ny, N, S=0.5, npml=0, pml_x=True, pml_y=True, eps=None, sig_e=None,
                 periodic_x=False, R=1e-7, m=3, alpha_max=0.0, dtype=np.float32):
        self.nx, self.ny, self.N, self.S = nx, ny, N, S
        self.dx = 1.0 / N
        self.dt = S * self.dx
        self.dtype = dtype
        self.periodic_x = periodic_x
        eps = np.ones((nx, ny)) if eps is None else eps
        sig_e = np.zeros((nx, ny)) if sig_e is None else sig_e
        dt = self.dt
        self.Ca = ((1 - sig_e * dt / (2 * eps)) / (1 + sig_e * dt / (2 * eps))).astype(dtype)
        self.Cb = ((dt / eps) / (1 + sig_e * dt / (2 * eps)) / self.dx).astype(dtype)
        self.Ch = dtype(dt / self.dx)

        def co(n, on, half):
            if not on or npml == 0:
                return np.ones(n), np.zeros(n)
            b, a, _ = cpml_coefs(n, npml, dt, m, R, alpha_max, half, dx=self.dx)
            return b, a
        self.bxe, self.axe = co(nx, pml_x, False)
        self.bxh, self.axh = co(nx, pml_x, True)
        self.bye, self.aye = co(ny, pml_y, False)
        self.byh, self.ayh = co(ny, pml_y, True)

    def run(self, nsteps, every, src_vals=None, src_prof=None, crop=None, sub=1, ez0=None, probes=None,
            dft_freqs=None, dft_start=0, hq=None):
        """Leapfrog nsteps steps.  Returns dict with
        frames: Ez[crop][::sub, ::sub] every `every` steps (frame k = state after k*every steps),
        probes: Ez at probe nodes every step, dft: complex DFT of Ez at dft_freqs over steps >= dft_start,
        hq: (Hx, Hy) sampled at nodes hq=(ii, jj) with each frame.
        Source: Ez += src_vals[n] * src_prof after the E update of step n (time (n+1) dt)."""
        dt, dtype = self.dt, self.dtype
        nx, ny = self.nx, self.ny
        f32 = lambda a: jnp.asarray(np.asarray(a), dtype)
        Ca, Cb, Ch = f32(self.Ca), f32(self.Cb), self.Ch
        bxe, axe, bxh, axh = f32(self.bxe), f32(self.axe), f32(self.bxh), f32(self.axh)
        bye, aye, byh, ayh = f32(self.bye), f32(self.aye), f32(self.byh), f32(self.ayh)
        src_vals = f32(np.zeros(nsteps) if src_vals is None else src_vals)
        src_prof = f32(np.zeros((nx, ny)) if src_prof is None else src_prof)
        per = self.periodic_x
        x0, x1, y0, y1 = crop if crop is not None else (0, nx, 0, ny)
        pr = np.zeros((0, 2), int) if probes is None else np.asarray(probes, int)
        fr = np.zeros(0) if dft_freqs is None else np.asarray(dft_freqs, float)
        hqi = None if hq is None else (jnp.asarray(hq[0]), jnp.asarray(hq[1]))
        z = jnp.zeros((nx, ny), dtype)
        Ez = z if ez0 is None else f32(ez0)
        state = (Ez, z, z, z, z, z, z, jnp.zeros((len(fr), nx, ny), jnp.complex128 if dtype == np.float64 else jnp.complex64))

        def step(n, st):
            Ez, Hx, Hy, pEx, pEy, pHx, pHy, acc = st
            if per:
                dEx = jnp.roll(Ez, -1, axis=0) - Ez
            else:
                dEx = jnp.concatenate([Ez[1:] - Ez[:-1], jnp.zeros_like(Ez[:1])], 0)
            dEy = jnp.concatenate([Ez[:, 1:] - Ez[:, :-1], jnp.zeros_like(Ez[:, :1])], 1)
            pEx = bxh[:, None] * pEx + axh[:, None] * dEx
            pEy = byh[None, :] * pEy + ayh[None, :] * dEy
            Hx = Hx - Ch * (dEy + pEy)
            Hy = Hy + Ch * (dEx + pEx)
            if per:
                dHy = Hy - jnp.roll(Hy, 1, axis=0)
            else:
                dHy = jnp.concatenate([Hy[:1], Hy[1:] - Hy[:-1]], 0)
            dHx = jnp.concatenate([Hx[:, :1], Hx[:, 1:] - Hx[:, :-1]], 1)
            pHx = bxe[:, None] * pHx + axe[:, None] * dHy
            pHy = bye[None, :] * pHy + aye[None, :] * dHx
            Ez = Ca * Ez + Cb * ((dHy + pHx) - (dHx + pHy))
            Ez = Ez + src_vals[n] * src_prof
            if not per:
                Ez = Ez.at[0].set(0).at[-1].set(0)
            Ez = Ez.at[:, 0].set(0).at[:, -1].set(0)
            if len(fr):
                t = (n + 1) * dt
                ph = (jnp.exp(1j * TAU * jnp.asarray(fr) * t) * (n >= dft_start)).astype(acc.dtype)
                acc = acc + ph[:, None, None] * Ez[None].astype(acc.dtype)
            return (Ez, Hx, Hy, pEx, pEy, pHx, pHy, acc)

        nfr = nsteps // every

        @jax.jit
        def chunk(st, k):
            def inner(i, c):
                s, tr = c
                s = step(k * every + i, s)
                tr = tr.at[i].set(s[0][pr[:, 0], pr[:, 1]])
                return (s, tr)
            tr0 = jnp.zeros((every, len(pr)), dtype)
            st, tr = jax.lax.fori_loop(0, every, inner, (st, tr0))
            out = st[0][x0:x1:sub, y0:y1:sub]
            hout = (st[1][hqi[0], hqi[1]], st[2][hqi[0], hqi[1]]) if hqi is not None else (jnp.zeros(0), jnp.zeros(0))
            return st, (out, tr, hout)

        t0 = time.time()
        st = state
        frames, traces, hxs, hys = [], [], [], []
        frames.append(np.asarray(st[0][x0:x1:sub, y0:y1:sub]))
        if hqi is not None:
            hxs.append(np.zeros(len(hq[0])))
            hys.append(np.zeros(len(hq[0])))
        for k in range(nfr):
            st, (o, tr, (hx, hy)) = chunk(st, k)
            frames.append(np.asarray(o))
            traces.append(np.asarray(tr))
            if hqi is not None:
                hxs.append(np.asarray(hx))
                hys.append(np.asarray(hy))
        jax.block_until_ready(st[0])
        res = dict(frames=np.stack(frames), probes=np.concatenate(traces, 0) if traces else None,
                   dft=np.asarray(st[7]), Ez=np.asarray(st[0]), Hx=np.asarray(st[1]), Hy=np.asarray(st[2]),
                   secs=time.time() - t0)
        if hqi is not None:
            res["hx"], res["hy"] = np.stack(hxs), np.stack(hys)
        return res


def pulse(t, t0, tau, f0=1.0):
    """Gaussian-modulated sine (zero mean): exp(-((t-t0)/tau)^2) sin(2 pi f0 (t-t0))."""
    return np.exp(-((t - t0) / tau) ** 2) * np.sin(TAU * f0 * (t - t0))


def to_u8(F, scale, gamma=1.0, floor=0.0):
    """Signed field -> uint8 (128 = 0) with a signed power law (gamma < 1 lifts faint waves).
    floor (in units of scale): values below it are shown as zero (a display noise gate)."""
    v = np.clip(np.asarray(F, np.float64) / scale, -1, 1)
    v = np.sign(v) * (np.maximum(np.abs(v) - floor, 0) / (1 - floor)) ** gamma
    return np.clip(np.round(127.5 + 127.0 * v), 0, 255).astype(np.uint8)


def save(name, **kw):
    p = os.path.join(DATA, name)
    np.savez(p, **kw)
    print(f"  wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)", flush=True)


PARTS = {}


def part(f):
    PARTS[f.__name__.replace("part_", "")] = f
    return f


def supersample(shape_fn, x, y, N, ss=4):
    """Area fraction of each cell inside shape_fn(X, Y) -> bool (ss x ss sub-samples): smooth permittivity edges."""
    o = (np.arange(ss) + 0.5) / ss - 0.5
    frac = np.zeros((len(x), len(y)))
    for a in o:
        for b in o:
            X, Y = np.meshgrid(x + a / N, y + b / N, indexing="ij")
            frac += shape_fn(X, Y)
    return frac / ss ** 2


def slab_mode(y, w, n, f=1.0):
    """Fundamental Ez (TM-slab, E along the slab) mode profile of a slab |y| < w/2 of index n in air."""
    k0 = TAU * f
    V2 = k0 ** 2 * (n ** 2 - 1)
    from scipy.optimize import brentq
    kap = brentq(lambda k: k * np.tan(k * w / 2) - np.sqrt(max(V2 - k * k, 0)), 1e-9, min(np.pi / w, np.sqrt(V2)) - 1e-9)
    gam = np.sqrt(V2 - kap ** 2)
    out = np.where(np.abs(y) <= w / 2, np.cos(kap * y), np.cos(kap * w / 2) * np.exp(-gam * (np.abs(y) - w / 2)))
    return out, np.sqrt(k0 ** 2 * n ** 2 - kap ** 2) / k0     # profile, effective index


# ===================================================================== parts ====
@part
def part_hook():
    """Ring resonator: a few-cycle pulse runs along a silicon bus waveguide, part of it couples into a small ring,
    circulates and leaks back out every lap.  2D FDTD, n = 3.5, N = 56 cells per vacuum wavelength.
    The source sits 14 wavelengths upstream so its stray radiation has left before the guided pulse arrives;
    frames are kept from t = T_KEEP on."""
    N, S, npml = 56, 0.5, 20
    n_si, R, w, gap = 3.5, 1.0, 0.30, 0.07
    X0, X1, Y0, Y1 = -15.0, 4.2, -2.6, 2.2
    nx, ny = int(round((X1 - X0) * N)) + 2 * npml, int(round((Y1 - Y0) * N)) + 2 * npml
    x, y = X0 + (np.arange(nx) - npml) / N, Y0 + (np.arange(ny) - npml) / N
    yb = -(R + w + gap)
    frac = supersample(lambda X, Y: (np.abs(np.hypot(X, Y) - R) <= w / 2) | (np.abs(Y - yb) <= w / 2), x, y, N)
    eps = 1 + (n_si ** 2 - 1) * frac
    g = Grid2D(nx, ny, N, S, npml=npml, eps=eps)
    T, T_KEEP = 124.0, 34.0
    nsteps = int(round(T / g.dt))
    t = (np.arange(nsteps) + 1) * g.dt
    sv = pulse(t, 2.2, 0.8)
    prof = np.zeros((nx, ny))
    x_src = -14.0
    prof[npml + int(round((x_src - X0) * N)), :] = slab_mode(y - yb, w, n_si)[0]
    fpp = 12                                                    # frames per optical period
    every = int(round(1 / (g.dt * fpp)))
    DX0, DX1, DY0, DY1 = -3.6, 3.6, -2.25, 1.8                  # displayed window (wavelengths)
    ia, ib = npml + int(round((DX0 - X0) * N)), npml + int(round((DX1 - X0) * N))
    ja, jb = npml + int(round((DY0 - Y0) * N)), npml + int(round((DY1 - Y0) * N))
    r = g.run(nsteps, every, sv, prof, crop=(ia, ib, ja, jb))
    tf = np.arange(len(r["frames"])) * every * g.dt
    keep = tf >= T_KEEP - 1e-9
    F, tf = r["frames"][keep], tf[keep]
    print(f"  hook: grid {nx}x{ny}, {nsteps} steps in {r['secs']:.1f}s, kept frames {F.shape}")
    peak = np.abs(F).max()
    scale = peak * 0.32                                      # the bus pulse saturates; the ring's light stays bright
    # display gate: |E| below 0.4% of the pulse peak is shown black (hides a ~1e-3 residue of the source's
    # imperfect mode match that runs along the bus; the ring's leaked light is far above it)
    U8 = to_u8(F, scale, 0.55, floor=0.004 * peak / scale).transpose(0, 2, 1)[:, ::-1]   # (frame, row = +y at top, col = x)
    Xc, Yc = np.meshgrid(x[ia:ib], y[ja:jb], indexing="ij")
    ringm = np.abs(np.hypot(Xc, Yc) - R) <= w / 2
    rr = np.array([np.abs(f[ringm]).max() for f in F]) / peak
    print("  hook: max |E| in the ring / bus pulse peak, every 8 periods from t =", tf[0], ":", " ".join(f"{v:.2f}" for v in rr[::96]))
    np.savez_compressed(os.path.join(DATA, "hook.npz"), frames=U8, t_frame=tf, extent=np.array([DX0, DX1, DY0, DY1]),
                        R=R, w=w, gap=gap, yb=yb, n=n_si, N=N, x_src=x_src, fpp=fpp)
    print(f"  wrote hook.npz ({os.path.getsize(os.path.join(DATA, 'hook.npz')) / 1e6:.1f} MB)")


@part
def part_bump():
    """Ch1: an Ez bump with H = 0 spreads into a ring.  H is sampled on rings of points around the bump (it is purely
    azimuthal by symmetry, so rings of arrows show the circulation)."""
    N, S, npml = 32, 0.5, 16
    L = 8.0
    n_in = int(L * N)
    nx = ny = n_in + 2 * npml
    x = (np.arange(nx) - npml - n_in / 2) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    ez0 = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.32 ** 2))
    g = Grid2D(nx, ny, N, S, npml=npml)
    nsteps = 300                                               # 4.7 periods
    pts = []
    for r_ in (0.32, 0.75, 1.25, 1.8, 2.4, 3.0, 3.6):
        n_ = max(6, int(round(TAU * r_ / 0.62)))
        for a in (np.arange(n_) + 0.5 * (len(pts) % 2)) * TAU / n_:
            pts.append((r_ * np.cos(a), r_ * np.sin(a)))
    pts = np.array(pts)
    c = npml + n_in // 2
    ii, jj = c + np.round(pts[:, 0] * N).astype(int), c + np.round(pts[:, 1] * N).astype(int)
    r = g.run(nsteps, 2, None, None, crop=(npml, npml + n_in, npml, npml + n_in), ez0=ez0, hq=(ii, jj))
    F = r["frames"]
    U8 = to_u8(F, 1.0, 0.7).transpose(0, 2, 1)[:, ::-1]
    hx, hy = r["hx"], r["hy"]                                  # (frames, points); Hx at (i, j+1/2), Hy at (i+1/2, j)
    save("bump.npz", frames=U8, t_frame=np.arange(len(F)) * 2 * g.dt, hx=hx.astype(np.float32),
         hy=hy.astype(np.float32), pts=np.stack([(ii - c) / N, (jj - c) / N], 1), L=L)
    print(f"  bump: {nsteps} steps, {len(pts)} arrows, |H|max {np.hypot(hx, hy).max():.3f}")


@part
def part_leap():
    """Ch2: 1D leapfrog with real numbers.  E_j at integer cells, H_{j+1/2} at half cells, S = 0.5.
    H += S (E[j+1] - E[j]);  E += S (H[j+1/2] - H[j-1/2]).  E[k], H[k] = state after step k."""
    n, S = 44, 0.5
    j = np.arange(n, dtype=float)
    f = lambda xx: np.exp(-((xx - 8.0) / 2.4) ** 2)
    E = f(j)
    H = -f(j + 0.5 + 0.5 * S)                                 # right-going pulse (H at time -dt/2)
    Es, Hs = [E.copy()], [H.copy()]
    for _ in range(120):
        H = H.copy()
        H[:-1] += S * (E[1:] - E[:-1])
        H[-1] = 0
        E = E.copy()
        E[1:] += S * (H[1:] - H[:-1])
        E[0] = 0
        Es.append(E.copy())
        Hs.append(H.copy())
    save("leap.npz", E=np.array(Es), H=np.array(Hs), S=S)


@part
def part_cfl():
    """Ch3: the same smooth Ez bump at Courant S = 0.70 (stable) and 0.75 (6% above 1/sqrt 2): round-off grows
    by about 2x per step in the unstable run.  float64."""
    N, npml, n_in = 20, 12, 110
    nx = ny = n_in + 2 * npml
    x = (np.arange(nx) - npml - n_in / 2) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    ez0 = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.35 ** 2))
    out = {}
    for tag, S in (("ok", 0.70), ("bad", 0.75)):
        g = Grid2D(nx, ny, N, S, npml=npml, dtype=np.float64)
        r = g.run(96, 1, None, None, crop=(npml, npml + n_in, npml, npml + n_in), ez0=ez0)
        F = r["frames"]
        out[tag] = F
        out["max_" + tag] = np.abs(F).max(axis=(1, 2))
        print(f"  cfl S={S}: max|E| at steps 0,30,50,60,70,80,96:",
              " ".join(f"{out['max_' + tag][k]:.1e}" for k in (0, 30, 50, 60, 70, 80, 96)))
    save("cfl.npz", ok=to_u8(out["ok"], 0.5, 0.6).transpose(0, 2, 1)[:, ::-1],
         bad=to_u8(out["bad"], 0.5, 0.6).transpose(0, 2, 1)[:, ::-1], max_ok=out["max_ok"], max_bad=out["max_bad"],
         S=np.array([0.70, 0.75]))


def yee_k(theta, Ncell, S):
    """|k|/k0 on the Yee grid's dispersion contour at frequency f0 (Ncell cells per vacuum wavelength), direction theta:
    sin^2(kx d/2) + sin^2(ky d/2) = sin^2(pi S / Ncell) / S^2.  nan where no real solution."""
    from scipy.optimize import brentq
    rhs = np.sin(np.pi * S / Ncell) ** 2 / S ** 2
    out = []
    for th in np.atleast_1d(theta):
        c, s = abs(np.cos(th)), abs(np.sin(th))
        f = lambda k: np.sin(k * c / 2) ** 2 + np.sin(k * s / 2) ** 2 - rhs
        kmax = np.pi / max(c, s)                               # edge of the grid's Brillouin zone along this ray
        out.append(brentq(f, 1e-9, kmax) / (TAU / Ncell) if f(kmax) >= 0 else np.nan)
    return np.array(out)


@part
def part_disp():
    """Ch3: dispersion contours (analytic) and a point-source pulse on a coarse (5 cells / lambda) and a fine
    (24 cells / lambda) grid, same physical size and time."""
    S = 0.5
    th = np.linspace(0, TAU, 721)
    Ns = np.array([30, 20, 14, 10, 8, 6, 5, 4, 3.4, 3.1])
    cont = np.array([yee_k(th, n_, S) for n_ in Ns])
    res = {}
    L = 9.0
    for tag, Nc in (("coarse", 5), ("fine", 24)):
        npml = 3 * Nc
        n_in = int(L * Nc)
        nx = ny = n_in + 2 * npml
        g = Grid2D(nx, ny, Nc, S, npml=npml)
        Tend = 6.6
        nsteps = int(round(Tend / g.dt))
        t = (np.arange(nsteps) + 1) * g.dt
        prof = np.zeros((nx, ny))
        prof[nx // 2, ny // 2] = 1.0
        every = max(1, int(round(0.05 / g.dt)))
        r = g.run(nsteps, every, pulse(t, 1.6, 0.55), prof, crop=(npml, npml + n_in, npml, npml + n_in))
        F = r["frames"]
        ref = np.abs(F[len(F) // 2:]).max()
        res[tag] = to_u8(F, ref * 0.8, 0.75).transpose(0, 2, 1)[:, ::-1]
        res["t_" + tag] = np.arange(len(F)) * every * g.dt
        print(f"  disp {tag}: {F.shape} frames, every {every} steps")
    save("disp.npz", Ns=Ns, th=th, cont=cont, S=S, L=L, **res)


@part
def part_boundary():
    """Ch4-5: one point-source pulse in an 8 x 8 wavelength box, three ways to end the grid:
    wall (perfect conductor right at the edge), lossy layer (step conductivity, 1 wavelength thick, then a wall),
    PML (graded stretched-coordinate layer, 1 wavelength).  Echo = difference from a reference run in a huge box."""
    N, S, L, d = 20, 0.5, 8.0, 1.0
    n_in, nl = int(L * N), int(d * N)
    T = 15.0
    out = {}
    nx = ny = n_in + 2 * nl
    for tag in ("wall", "lossy", "pml", "ref"):
        if tag == "wall":
            g = Grid2D(n_in, n_in, N, S, npml=0)
            off = 0
            nxx = n_in
        elif tag == "lossy":
            sig = np.zeros((nx, ny))
            sig[:nl] = sig[-nl:] = 10.0
            sig[:, :nl] = sig[:, -nl:] = 10.0
            g = Grid2D(nx, ny, N, S, npml=0, sig_e=sig)
            off, nxx = nl, nx
        elif tag == "pml":
            g = Grid2D(nx, ny, N, S, npml=nl)
            off, nxx = nl, nx
        else:
            nxx = n_in + 2 * 400
            off = 400
            g = Grid2D(nxx, nxx, N, S, npml=0)
        nsteps = int(round(T / g.dt))
        t = (np.arange(nsteps) + 1) * g.dt
        prof = np.zeros((nxx, nxx))
        prof[off + n_in // 2, off + n_in // 2] = 1.0
        every = int(round(0.1 / g.dt))
        a0, a1 = off - (nl if tag != "ref" else nl), off + n_in + nl
        crop = (max(a0, 0), min(a1, nxx), max(a0, 0), min(a1, nxx)) if tag != "wall" else (0, n_in, 0, n_in)
        r = g.run(nsteps, every, pulse(t, 1.4, 0.55), prof, crop=crop)
        F = r["frames"]
        if tag == "wall":                                       # pad to the same (n_in + 2 nl)^2 window: dead zone = 0
            F = np.pad(F, ((0, 0), (nl, nl), (nl, nl)))
        out[tag] = F
        print(f"  boundary {tag}: {F.shape}, {r['secs']:.1f}s")
    ref = out["ref"]
    inner = (slice(None), slice(nl, nl + n_in), slice(nl, nl + n_in))
    inc = np.abs(ref[inner][:, :3, :]).max()                     # incident amplitude arriving at the edge
    stats = {}
    for tag in ("wall", "lossy", "pml"):
        echo = np.abs(out[tag][inner] - ref[inner]).max()
        stats[tag] = echo / inc
        print(f"  echo {tag}: {echo / inc:.2e} of the incident amplitude at the edge")
    scale = np.abs(ref[inner][int(3.0 / 0.1):]).max() * 0.9
    # the PML's residual echo (PML run minus the huge-box reference), magnified 10^5 for display
    res_pml = np.zeros_like(out["pml"])
    res_pml[inner] = out["pml"][inner] - ref[inner]
    rel = [stats[k] / stats["wall"] for k in ("wall", "lossy", "pml")]
    print("  echo relative to the wall's echo:", " ".join(f"{r_:.2e}" for r_ in rel))
    save("boundary.npz", **{k: to_u8(out[k], scale, 0.6).transpose(0, 2, 1)[:, ::-1] for k in ("wall", "lossy", "pml")},
         pml_echo_x1e5=to_u8(res_pml * 1e5, scale, 0.6).transpose(0, 2, 1)[:, ::-1],
         t_frame=np.arange(len(ref)) * int(round(0.1 / (S / N))) * (S / N), L=L, d=d,
         echo=np.array([stats["wall"], stats["lossy"], stats["pml"]]), echo_rel=np.array(rel))


def fdtd1d(n_of_x, dx, S, nsteps, src_i, src_vals, probe_i, frame_every=0, frame_slice=None):
    """1D Yee leapfrog (Ez, Hy), c = 1: H += S (E[j+1]-E[j]);  E += S/eps (H[j]-H[j-1]); soft source at src_i."""
    eps = n_of_x ** 2
    n = len(eps)
    E, H = np.zeros(n), np.zeros(n)
    tr = np.zeros(nsteps)
    frames = []
    for k in range(nsteps):
        H[:-1] += S * (E[1:] - E[:-1])
        E[1:] += S / eps[1:] * (H[1:] - H[:-1])
        E[src_i] += src_vals[k]
        tr[k] = E[probe_i]
        if frame_every and k % frame_every == 0:
            frames.append(E[frame_slice].copy())
    return tr, (np.array(frames) if frames else None)


def tmm_R(lam, layers, n_in=1.0, n_out=1.5):
    """Normal-incidence reflectance of a layer stack [(n, d), ...] (characteristic matrices)."""
    R = []
    for l_ in np.atleast_1d(lam):
        M = np.eye(2, dtype=complex)
        for n_, d_ in layers:
            dl = TAU / l_ * n_ * d_
            M = M @ np.array([[np.cos(dl), -1j * np.sin(dl) / n_], [-1j * n_ * np.sin(dl), np.cos(dl)]])
        B, C = M @ np.array([1.0, n_out])
        r = (n_in * B - C) / (n_in * B + C)
        R.append(abs(r) ** 2)
    return np.array(R)


@part
def part_bragg():
    """Ch6: one pulse -> the whole reflection spectrum of an 8-pair quarter-wave Bragg mirror (TiO2/SiO2-like,
    n = 2.3 / 1.45 on glass 1.5), 1D FDTD vs the exact transfer-matrix result for the same (grid-rounded) layers."""
    N, S = 100, 0.5
    dx = 1.0 / N
    nH, nL, ns, pairs = 2.3, 1.45, 1.5, 8
    cH, cL = int(round(N / (4 * nH))), int(round(N / (4 * nL)))   # layer thickness in whole cells
    layers = [(nH, cH * dx), (nL, cL * dx)] * pairs
    xs0, xa = -6.0, 0.0                                         # source position, stack front face
    n0 = 13000
    x = (np.arange(2 * n0) - n0) * dx
    nprof = np.ones_like(x)
    j = n0
    for n_, d_ in layers:
        c_ = int(round(d_ / dx))
        nprof[j:j + c_] = n_
        j += c_
    nprof[j:] = ns
    x_end = x[j]
    T = 125.0
    nsteps = int(round(T / (S * dx)))
    t = (np.arange(nsteps) + 1) * S * dx
    sv = pulse(t, 2.0, 0.42)
    si, pi_ = np.searchsorted(x, xs0), np.searchsorted(x, -2.0)
    fe = int(round(0.05 / (S * dx)))
    fsl = slice(np.searchsorted(x, -5.0), np.searchsorted(x, 5.0), 4)
    tr, frames = fdtd1d(nprof, dx, S, nsteps, si, sv, pi_, fe, fsl)
    tr0, frames0 = fdtd1d(np.ones_like(x), dx, S, nsteps, si, sv, pi_, fe, fsl)
    refl = tr - tr0
    f = np.fft.rfftfreq(nsteps, S * dx)
    Fi, Fr = np.fft.rfft(tr0), np.fft.rfft(refl)
    band = (f > 0.5) & (f < 1.6)
    Rf = np.abs(Fr[band]) ** 2 / np.abs(Fi[band]) ** 2
    lam = 1 / f[band]
    lam_exact = np.linspace(0.6, 2.0, 700)
    R_exact = tmm_R(lam_exact, layers, 1.0, ns)
    err = np.abs(Rf - tmm_R(lam, layers, 1.0, ns))
    print(f"  bragg: |R_fdtd - R_exact| max {err.max():.4f}, mean {err.mean():.4f} over {lam.min():.2f}-{lam.max():.2f} lambda0;"
          f" R(lambda0) exact {tmm_R(1.0, layers, 1.0, ns)[0]:.4f}")
    tt = t[: int(round(30.0 / (S * dx)))]
    save("bragg.npz", lam=lam, R=Rf, lam_exact=lam_exact, R_exact=R_exact, t=tt[::10],
         trace_inc=tr0[: len(tt)][::10], trace_refl=refl[: len(tt)][::10], frames=frames.astype(np.float32),
         frames0=frames0.astype(np.float32), xf=x[fsl], t_frame=np.arange(len(frames)) * fe * S * dx,
         nprof=nprof[fsl], x_stack=np.array([xa, x_end]), x_probe=-2.0, x_src=xs0, err=err.max())


@part
def part_relief():
    """Ch6: CW plane wave (lambda0, E along the grooves) from glass through a blazed relief of period 3 lambda0 and
    height lambda0/(n-1) = 2 lambda0, periodic in x.  FDTD phasor vs the thin-element (TEA) prediction, plus
    the diffraction efficiency of the +1 order."""
    N, S, npml = 30, 0.5, 24
    n, Lam, h = 1.5, 3.0, 2.0
    nxp = int(Lam * N)
    Yb, Yt = -2.5, 7.0
    ny = int(round((Yt - Yb) * N)) + 2 * npml
    x = np.arange(nxp) / N
    y = Yb + (np.arange(ny) - npml) / N
    prof_h = lambda X: h * (X % Lam) / Lam
    frac = supersample(lambda X, Y: (Y < 0) | ((Y >= 0) & (Y < prof_h(X))), x, y, N)
    eps = 1 + (n ** 2 - 1) * frac
    out = {}
    for tag, e in (("relief", eps), ("ref", np.where(np.arange(ny)[None, :] >= 0, n ** 2, 1.0) * np.ones((nxp, 1)))):
        g = Grid2D(nxp, ny, N, S, npml=npml, pml_x=False, eps=e, periodic_x=True)
        T = 60.0
        nsteps = int(round(T / g.dt))
        t = (np.arange(nsteps) + 1) * g.dt
        ramp = np.clip(t / 6.0, 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        prof = np.zeros((nxp, ny))
        prof[:, npml + int(round((-1.8 - Yb) * N))] = 1.0
        r = g.run(nsteps, nsteps, ramp * np.sin(TAU * t), prof, dft_freqs=[1.0], dft_start=nsteps - int(round(10 / g.dt)))
        out[tag] = r["dft"][0]
        print(f"  relief {tag}: {r['secs']:.1f}s")
    U, Uref = out["relief"], out["ref"]
    j0 = npml + int(round((-1.0 - Yb) * N))
    a_inc = np.abs(Uref[:, j0]).mean()                          # incident amplitude in the glass
    U = U / a_inc
    # FDTD diffraction efficiencies from the field 3 wavelengths above the relief top
    jt = npml + int(round((h + 3.0 - Yb) * N))
    am = np.fft.fft(U[:, jt]) / nxp
    m = np.fft.fftfreq(nxp, 1.0 / nxp).astype(int)
    kx = m / Lam
    prop = np.abs(kx) < 1.0
    cos_t = np.sqrt(np.clip(1 - kx ** 2, 0, None))
    eff = np.abs(am) ** 2 * cos_t / n * prop                     # power flux of each order / incident flux (eta0 = 1)
    order = {int(mm): float(e_) for mm, e_ in zip(m, eff) if prop[list(m).index(mm)]}
    T_flat = 1 - ((n - 1) / (n + 1)) ** 2
    print(f"  relief: FDTD orders {', '.join(f'{k:+d}: {v:.3f}' for k, v in sorted(order.items()))}; total {sum(order.values()):.3f};"
          f" TEA: +1 order {T_flat:.3f}")
    # TEA field: glass below 0, column delay inside the relief, angular-spectrum propagation above the top plane
    k0 = TAU
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    hx = prof_h(Xg)
    t_amp = 2 * n / (n + 1)                                     # flat glass->air amplitude transmission (E along grooves)
    col = np.exp(1j * k0 * (n * np.minimum(np.maximum(Yg, 0), hx) + np.maximum(Yg - hx, 0)))
    U_tea = np.where(Yg < 0, np.exp(1j * k0 * n * Yg), t_amp * col)
    top = U_tea[:, np.searchsorted(y, h)]
    kxs = TAU * np.fft.fftfreq(nxp, 1.0 / N)
    kz = np.sqrt((k0 ** 2 - kxs ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    A0 = np.fft.fft(top)
    for jj in np.where(y > h)[0]:
        U_tea[:, jj] = np.fft.ifft(A0 * np.exp(1j * kz * (y[jj] - y[np.searchsorted(y, h)])))
    # display window: y in [-1, h + 4], 2 periods wide (tile the periodic cell)
    ja, jb = npml + int(round((-1.0 - Yb) * N)), npml + int(round((h + 4.0 - Yb) * N))
    sl = lambda A: np.tile(A[:, ja:jb], (2, 1)).T[::-1]          # rows: +y at top
    save("relief.npz", U_fdtd=sl(U).astype(np.complex64), U_tea=sl(U_tea).astype(np.complex64),
         extent=np.array([0.0, 2 * Lam, -1.0, h + 4.0]), Lam=Lam, h=h, n=n,
         orders_m=np.array(sorted(order)), orders_eff=np.array([order[k] for k in sorted(order)]), T_flat=T_flat)


if __name__ == "__main__":
    want = sys.argv[1:] or list(PARTS)
    print("devices:", jax.devices())
    for w in want:
        t = time.time()
        print(f"[{w}]", flush=True)
        PARTS[w]()
        print(f"[{w}] done in {time.time() - t:.1f}s", flush=True)
