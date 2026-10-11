"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).
    OMP_NUM_THREADS=1 python physics.py        (single-threaded, about 2 CPU-minutes)

Units: lengths in vacuum wavelengths (lambda = 1) except the ring, which is in microns.
  slab.npz     Ch1  symmetric slab (n1 = 2.0, n2 = 1.45, d = 0.95): TE modes from the transverse-resonance
                    condition 2 kx d - 4 arctan(gamma/kx) = 2 pi m (root finding), checked against the eigen-solver
  solver.npz   Ch2  finite-difference mode solver: H = D2/dx^2 + diag(k0^2 n^2), numpy eigh (coarse matrix for the
                    picture, fine grid for the profiles)
  mmi.npz      Ch2  multimode guide (width 6): off-centre spot expanded in the eigenmodes, each advanced by
                    e^{i beta_m z}, summed back (the 'angular spectrum method of a waveguide') -> |E(x,z)|^2
  coupler.npz  Ch3  directional coupler: split-step BPM (FFT diffraction step + thin phase screen e^{i k0 dn dz});
                    supermodes and kappa from the eigen-solver; coupling length BPM vs pi/(2 kappa)
  ring.npz     Ch0/4/6  add-drop ring (2D scalar Helmholtz = Ez polarisation, finite differences + stretched-coordinate
                    PML, sparse LU): resonance found by a wavelength scan, fields at the resonance and two
                    off-resonance wavelengths
  window.npz   Ch6  the same slab mode solved in a narrow and a wide window (Dirichlet edges)
"""
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi
K0 = TAU                                   # lambda = 1


def fd_modes(nprof, x, k0=K0):
    """Finite-difference slab mode solver: eigenpairs of D2/dx^2 + diag(k0^2 n^2), sorted by decreasing beta."""
    dx = x[1] - x[0]
    N = len(x)
    H = (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) / dx ** 2 \
        + np.diag(k0 ** 2 * nprof ** 2)
    w, v = np.linalg.eigh(H)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    for j in range(v.shape[1]):                         # sign convention: first big lobe positive (from the left)
        k = np.argmax(np.abs(v[:, j]) > 0.5 * np.abs(v[:, j]).max())
        v[:, j] *= np.sign(v[k, j])
    return np.sqrt(np.clip(w, 0, None)) / k0, v, H


# ----------------------------------------------------------------------- Ch1: slab ------
def slab():
    n1, n2, d = 2.0, 1.45, 0.95
    thmax = np.arccos(n2 / n1)

    def phi_rt(th):                                     # round-trip phase: 2 kx d + 2 phi_r, phi_r = -2 atan(g/kx)
        kx = K0 * n1 * np.sin(th)
        g = K0 * np.sqrt(np.maximum((n1 * np.cos(th)) ** 2 - n2 ** 2, 0.0))
        return 2 * kx * d - 4 * np.arctan(g / kx)
    thetas = []
    for m in range(6):
        try:
            thetas.append(brentq(lambda t: phi_rt(t) - TAU * m, 1e-6, thmax - 1e-12, xtol=1e-14))
        except ValueError:
            break
    thetas = np.array(thetas)
    neff = n1 * np.cos(thetas)
    x = (np.arange(1200) - 599.5) * 0.005               # core edges fall midway between samples
    ne_fd, V, _ = fd_modes(np.where(np.abs(x) <= d / 2, n1, n2), x)
    print(f"[slab] {len(thetas)} TE modes, theta = {np.degrees(thetas).round(2)} deg (theta_max {np.degrees(thmax):.2f}); "
          f"n_eff analytic {neff.round(5)}, eigen-solver {ne_fd[:len(thetas)].round(5)}")
    th_tab = np.linspace(np.radians(0.5), thmax, 600)
    np.savez(os.path.join(DATA, "slab.npz"), n1=n1, n2=n2, d=d, thmax=thmax, thetas=thetas, neff=neff,
             neff_fd=ne_fd[:len(thetas)], th_tab=th_tab, phi_tab=phi_rt(th_tab))


# ----------------------------------------------------------------------- Ch2: solver -----
def solver():
    n1, n2, d = 2.0, 1.45, 0.95
    # coarse matrix (for the picture): 36 samples over 2.4 lambda
    Nc = 36
    dxc = 2.4 / Nc
    xc = (np.arange(Nc) - (Nc - 1) / 2) * dxc
    nc = np.where(np.abs(xc) <= d / 2, n1, n2)
    D2 = (np.diag(-2.0 * np.ones(Nc)) + np.diag(np.ones(Nc - 1), 1) + np.diag(np.ones(Nc - 1), -1)) / dxc ** 2
    Dn = np.diag(K0 ** 2 * nc ** 2)
    # fine grid for the profiles
    x = (np.arange(1000) - 499.5) * 0.005
    nprof = np.where(np.abs(x) <= d / 2, n1, n2)
    ne, V, _ = fd_modes(nprof, x)
    guided = int(np.sum(ne > n2))
    prof = V[:, :guided] / np.abs(V[:, :guided]).max(axis=0)
    ne_c, Vc, _ = fd_modes(nc, xc)
    print(f"[solver] coarse 36x36: n_eff {ne_c[:3].round(4)}; fine 1000: {ne[:guided].round(5)}")
    np.savez(os.path.join(DATA, "solver.npz"), n1=n1, n2=n2, d=d, xc=xc, D2=D2, Dn=Dn, x=x, nprof=nprof,
             neff=ne[:guided], prof=prof, neff_c=ne_c[:3], Vc=Vc[:, :3])


# ----------------------------------------------------------------------- Ch2: MMI --------
def mmi():
    n1, n2, W = 2.0, 1.45, 6.0
    x = np.linspace(-6, 6, 1201)
    nprof = np.where(np.abs(x) <= W / 2, n1, n2)
    ne, V, _ = fd_modes(nprof, x)
    beta = K0 * ne
    Lpi = np.pi / (beta[0] - beta[1])
    x0, w0 = -1.5, 0.75                                  # off-centre input spot
    u0 = np.exp(-((x - x0) / w0) ** 2)
    c = V.T @ u0                                         # modal coefficients (complete basis of the window)
    zmax = 6.35 * Lpi
    zs = np.linspace(0, zmax, 1100)
    sel = np.abs(x) <= 4.0
    U = (V[sel] * c[None, :]) @ np.exp(1j * (beta[:, None] - beta[0]) * zs[None, :])   # (nx_sel, nz)
    I = np.abs(U) ** 2
    guided = int(np.sum(ne > n2))
    pw = np.abs(c[:guided]) ** 2 / np.sum(np.abs(c) ** 2)
    # self-image positions: best re-focus near 3 L_pi (mirrored) and 6 L_pi (direct)
    def best(zc):
        zz = np.linspace(zc - 0.12 * Lpi, zc + 0.12 * Lpi, 241)
        pk = [np.max(np.abs(V @ (c * np.exp(1j * (beta - beta[0]) * q))) ** 2) for q in zz]
        return zz[int(np.argmax(pk))], max(pk)
    z3, p3 = best(3 * Lpi)
    z6, p6 = best(6 * Lpi)
    print(f"[mmi] {guided} guided modes, L_pi = {Lpi:.1f}; mirrored image at z = {z3:.1f} (peak {p3:.2f}), "
          f"direct image at z = {z6:.1f} (peak {p6:.2f}); guided power {pw.sum():.5f}")
    np.savez(os.path.join(DATA, "mmi.npz"), x=x[sel], z=zs, I=I.astype(np.float32), Lpi=Lpi, W=W, x0=x0, w0=w0,
             z3=z3, z6=z6, n1=n1, n2=n2, nmodes=guided, cpow=pw[:12])


# ----------------------------------------------------------------------- Ch3: coupler ----
def coupler():
    n1, n2, w, g = 1.50, 1.45, 1.0, 1.0
    N = 1024
    x = np.linspace(-8, 8, N, endpoint=False)
    dx = x[1] - x[0]
    c1, c2 = -(w + g) / 2, (w + g) / 2                   # guide 1 below (x < 0), guide 2 above
    nprof = np.where((np.abs(x - c1) <= w / 2) | (np.abs(x - c2) <= w / 2), n1, n2)
    ne, V, _ = fd_modes(nprof, x)                       # supermodes: 0 symmetric, 1 antisymmetric
    ns, Vs, _ = fd_modes(np.where(np.abs(x - c1) <= w / 2, n1, n2), x)
    bS, bA = K0 * ne[0], K0 * ne[1]
    kappa = (bS - bA) / 2
    Lc = np.pi / (2 * kappa)
    # split-step BPM: diffraction step (FFT, paraxial transfer function) + refraction step (thin phase screen)
    nref = ns[0]
    kx = TAU * np.fft.fftfreq(N, d=dx)
    dz = 0.05
    P = np.exp(-1j * kx ** 2 / (2 * K0 * nref) * dz)
    Q = np.exp(1j * K0 * (nprof - nref) * dz)
    absorb = np.exp(-3.0 * np.clip(np.abs(x) - 6.5, 0, None) ** 2)
    u = Vs[:, 0].astype(complex)
    u /= np.sqrt(np.sum(np.abs(u) ** 2) * dx)
    zmax = 4.3 * Lc
    nst = int(round(zmax / dz))
    keep = 4
    sel = np.abs(x) <= 3.0
    cols, zs, p2 = [], [], []
    for i in range(nst + 1):
        if i % keep == 0:
            cols.append(np.abs(u[sel]) ** 2)
            zs.append(i * dz)
            p2.append(np.sum(np.abs(u[x > 0]) ** 2) / np.sum(np.abs(u) ** 2))
        u = np.fft.ifft(np.fft.fft(u) * P) * Q * absorb
    Imap = np.array(cols).T                             # rows x, cols z
    zs, p2 = np.array(zs), np.array(p2)
    i1 = int(np.argmax(p2[zs < 1.5 * Lc]))
    print(f"[coupler] n_eff single {ns[0]:.5f}; supermodes S {ne[0]:.5f}, A {ne[1]:.5f}; kappa {kappa:.5f}; "
          f"L = pi/2kappa = {Lc:.1f}; BPM: max transfer {p2[i1]:.3f} at z = {zs[i1]:.1f}; "
          f"power kept {np.sum(np.abs(u) ** 2) * dx:.4f}")
    S = V[:, 0] / np.abs(V[:, 0]).max()
    A = V[:, 1] / np.abs(V[:, 1]).max()
    if A[np.argmin(np.abs(x - c1))] < 0:                # A positive in guide 1 (so S + A sits in guide 1)
        A = -A
    np.savez(os.path.join(DATA, "coupler.npz"), x=x[sel], z=zs, I=Imap.astype(np.float32), p2=p2, Lc=Lc,
             Lc_bpm=zs[i1], kappa=kappa, nS=ne[0], nA=ne[1], n_single=ns[0], S=S[sel], A=A[sel], c1=c1, c2=c2,
             w=w, g=g, n1=n1, n2=n2, dz=dz)


# ----------------------------------------------------------------------- ring (FDFD) ----
RING = dict(nco=2.9, ncl=1.45, w=0.30, R=2.5, gap=0.18, h=0.025, lp=1.0)


class RingSim:
    def __init__(self, p=RING):
        self.p = p
        h, lp, R, w, gap = p["h"], p["lp"], p["R"], p["w"], p["gap"]
        self.XH = R + w / 2 + 1.3 + lp
        self.YH = R + w / 2 + gap + w + 0.9 + lp
        self.x, self.xh = self.axis(self.XH)
        self.y, self.yh = self.axis(self.YH)
        self.nx, self.ny = len(self.x), len(self.y)
        self.yb = -(R + w / 2 + gap + w / 2)                 # bottom bus centre; top bus at -yb
        self.eps = self.eps_grid(True)
        self.eps_ref = self.eps_grid(False)

    def axis(self, half):
        h = self.p["h"]
        n = int(round(2 * half / h))
        return -half + (np.arange(n) + 0.5) * h, -half + (np.arange(n) + 1.0) * h

    def core(self, X, Y, ring=True):
        p = self.p
        m = (np.abs(Y - self.yb) <= p["w"] / 2)
        if ring:
            m |= (np.abs(Y + self.yb) <= p["w"] / 2) | (np.abs(np.hypot(X, Y) - p["R"]) <= p["w"] / 2)
        return m

    def eps_grid(self, ring, ss=5):
        p = self.p
        acc = np.zeros((self.ny, self.nx))
        o = (np.arange(ss) + 0.5) / ss - 0.5
        for a in o:                                         # sub-pixel averaging of eps (smooth staircase)
            for b in o:
                X, Y = np.meshgrid(self.x + a * p["h"], self.y + b * p["h"])
                acc += np.where(self.core(X, Y, ring), p["nco"] ** 2, p["ncl"] ** 2)
        return acc / ss ** 2

    def sfac(self, u, half):
        d = np.clip(np.abs(u) - (half - self.p["lp"]), 0, None) / self.p["lp"]
        return 1 + 1j * 5.0 * d ** 3                         # e^{-i omega t}: Im s > 0 absorbs outgoing waves

    def lap(self, n, s_int, s_half):
        h = self.p["h"]
        sm = np.r_[s_half[0], s_half[:-1]]
        M = sp.diags([1 / s_half[:-1], -(1 / s_half + 1 / sm), 1 / s_half[:-1]], [-1, 0, 1], shape=(n, n), format="csr")
        return (sp.diags(1 / s_int) @ M) / h ** 2

    def bus_mode(self, lam):
        p = self.p
        e1 = np.where(np.abs(self.y - self.yb) <= p["w"] / 2, p["nco"] ** 2, p["ncl"] ** 2)
        ne, V, _ = fd_modes(np.sqrt(e1), self.y, TAU / lam)
        return ne[0], np.abs(V[:, 0])

    def solve(self, lam, ring=True):
        k0 = TAU / lam
        Ax = self.lap(self.nx, self.sfac(self.x, self.XH), self.sfac(self.xh, self.XH))
        Ay = self.lap(self.ny, self.sfac(self.y, self.YH), self.sfac(self.yh, self.YH))
        eps = self.eps if ring else self.eps_ref
        A = sp.kron(sp.identity(self.ny), Ax) + sp.kron(Ay, sp.identity(self.nx)) + sp.diags(k0 ** 2 * eps.ravel())
        neff, psi = self.bus_mode(lam)
        self.i_src = int(np.searchsorted(self.x, -self.XH + self.p["lp"] + 0.3))
        b = np.zeros((self.ny, self.nx), complex)
        b[:, self.i_src] = psi / self.p["h"]
        E = spl.splu(A.tocsc()).solve(-b.ravel()).reshape(self.ny, self.nx)
        return E, psi, neff

    def ports(self, E, psi):
        i_t = int(np.searchsorted(self.x, self.XH - self.p["lp"] - 0.3))
        i_d = int(np.searchsorted(self.x, -self.XH + self.p["lp"] + 0.3))
        return np.sum(E[:, i_t] * psi), np.sum(E[:, i_d] * psi[::-1])


def ring():
    t0 = time.time()
    rs = RingSim()
    _, psi, _ = rs.solve(1.555, ring=False)
    Eref, psi, _ = rs.solve(1.555, ring=False)
    a_ref, _ = rs.ports(Eref, psi)
    cache = {}

    def TD(lam):
        if lam not in cache:
            E, psi, _ = rs.solve(lam)
            at, ad = rs.ports(E, psi)
            cache[lam] = (abs(at / a_ref) ** 2, abs(ad / a_ref) ** 2)
        return cache[lam]
    scan = np.arange(1.540, 1.5751, 0.0025)
    for lam in scan:
        TD(lam)
    best = max(scan, key=lambda l: TD(l)[1])
    r = minimize_scalar(lambda l: TD(l)[0], bracket=(best - 0.0015, best, best + 0.0015), tol=1e-6)
    lam_g = float(r.x)
    lams = {"G": lam_g, "R": lam_g + 0.021, "B": lam_g - 0.021}
    out = {}
    crop_x = (rs.x >= -3.45) & (rs.x <= 3.95 - 0.05)
    crop_y = np.abs(rs.y) <= 3.95
    for key, lam in lams.items():
        E, psi, _ = rs.solve(lam)
        Er, psi, _ = rs.solve(lam, ring=False)
        ar, _ = rs.ports(Er, psi)
        at, ad = rs.ports(E, psi)
        T, D = abs(at / ar) ** 2, abs(ad / ar) ** 2
        # normalise: incident bus-mode amplitude -> peak 1 (reference run, same wavelength)
        i_in = int(np.searchsorted(rs.x, -2.0))
        amp = np.abs(Er[:, i_in]).max()
        out[key] = (E[np.ix_(crop_y, crop_x)] / amp).astype(np.complex64)
        out["TD_" + key] = (T, D)
        print(f"[ring] {key}: lambda = {lam:.5f} um, through T = {T:.4f}, drop D = {D:.4f}, "
              f"max |E| in ring {np.abs(E / amp)[np.abs(np.hypot(*np.meshgrid(rs.x, rs.y)) - RING['R']) < 0.1].max():.2f}")
    xs, ys = rs.x[crop_x], rs.y[crop_y]
    X, Y = np.meshgrid(xs, ys)
    core = rs.core(X, Y, True).astype(np.float32)
    # 'arrival' path length (for the reveal): along the bus, around the ring (counter-clockwise from its bottom),
    # then left along the top bus; plus the distance to the nearest structure
    R, yb = RING["R"], rs.yb
    x_in = xs[0]
    t_bus = (X - x_in) + np.abs(Y - yb)
    ang = (np.arctan2(Y, X) + np.pi / 2) % TAU
    t_ring = (0 - x_in) + R * ang + np.abs(np.hypot(X, Y) - R)
    t_top = (0 - x_in) + np.pi * R + np.abs(X) + np.abs(Y + yb)
    arrival = np.minimum(np.minimum(t_bus, t_ring), t_top).astype(np.float32)
    sc = np.array([[l, T, D] for l, (T, D) in sorted(cache.items())])
    np.savez(os.path.join(DATA, "ring.npz"), ER=out["R"], EG=out["G"], EB=out["B"], lamR=lams["R"], lamG=lams["G"],
             lamB=lams["B"], TD_R=out["TD_R"], TD_G=out["TD_G"], TD_B=out["TD_B"], x=xs, y=ys, core=core,
             arrival=arrival, scan=sc, yb=yb, **{k: v for k, v in RING.items()})
    print(f"[ring] {len(cache) + 7} sparse solves ({rs.nx}x{rs.ny} grid) in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------- Ch6: window -----
def window():
    n1, n2, d = 2.0, 1.45, 0.95
    res = {}
    for name, half in (("narrow", 0.72), ("wide", 3.0)):
        n = int(round(2 * half / 0.005))
        x = (np.arange(n) - (n - 1) / 2) * 0.005
        ne, V, _ = fd_modes(np.where(np.abs(x) <= d / 2, n1, n2), x)
        prof = V[:, 2] / np.abs(V[:, 2]).max()
        res[name] = (x, prof, ne[2])
    print(f"[window] mode m=2: n_eff narrow window {res['narrow'][2]:.4f} vs wide {res['wide'][2]:.4f}")
    np.savez(os.path.join(DATA, "window.npz"), xn=res["narrow"][0], pn=res["narrow"][1], nn=res["narrow"][2],
             xw=res["wide"][0], pw=res["wide"][1], nw=res["wide"][2], d=d)


if __name__ == "__main__":
    t = time.time()
    slab()
    solver()
    mmi()
    coupler()
    window()
    ring()
    print(f"done in {time.time() - t:.0f}s")
