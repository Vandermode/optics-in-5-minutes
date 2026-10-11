"""Precompute every physical field shown in E10 (adjoint inverse design) -> data/*.npz (run once; scenes only load).

    python physics.py [splitter asm edof gap land]      (default: all)

Parts
  splitter  2D frequency-domain (FDFD) scalar Helmholtz solver, Ez polarisation, e^{-i w t}, stretched-coordinate PML,
            symmetrised so that A^T = A (reciprocity).  Adjoint inverse design of a symmetric 1x2 splitter
            (silicon eps=12 in glass eps=2.1, lambda = 1.55 um, dx = 30 nm, 1.5 x 2.4 um design box = 50 x 80 pixels):
            dF/deps = -2 Re{ lambda * k0^2 * x },  A lambda = dF/dx  (one extra solve, same LU factors).
            Checked against central finite differences.  Also: the human Y-branch (sub-cell averaged), a single-pixel
            nudge (its scattered wavelet), two other starting guesses, a 6x6-grid example of A.   CPU, scipy SuperLU.
  asm       thin phase mask + exact angular-spectrum propagation (E01), gradient descent with JAX autodiff toward a
            target picture; check: the backward pass equals propagating the error by -z (H -> H*).
  edof      end-to-end extended depth of field: pupil phase (radial Zernike terms) + one depth-blind Wiener filter
            (E03), trained on the expected restoration error over 9 defocus values (JAX).
  gap       model gap: a glass relief optimised under the thin-element model (E06, TEA + ASM), then simulated with the
            full-wave FDFD solver (E09's Maxwell, frequency domain) -> the focus loses about half its light.
  land      schematic 2-parameter merit landscape + gradient-ascent path (Ch1; not physics).
JAX parts run on a GPU if one is visible (choose it with CUDA_VISIBLE_DEVICES), otherwise on the CPU; FDFD runs single-threaded on the CPU (~3 CPU-min).
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi
F16 = np.float16


def log(*a):
    print(*a, flush=True)


# ===================================================================== 1. FDFD splitter ==========
class Sim:
    """2D scalar Helmholtz (Ez): [d/dx (1/sx) d/dx + d/dy (1/sy) d/dy] / (sx sy) + k0^2 eps, multiplied by sx*sy so the
    sparse matrix is complex-symmetric (A^T = A).  Unknowns ordered i*ny + j (i = x index, j = y index)."""

    def __init__(self, dx=0.03, nx=240, ny=180, npml=20, lam=1.55, eps_core=12.0, eps_clad=2.1, WG=8, SEP=30,
                 I_D0=80, NXD=50, NYD=80, i_src=40, i_mon=200, R=4.0):
        self.dx, self.nx, self.ny, self.npml, self.lam = dx, nx, ny, npml, lam
        self.eps_core, self.eps_clad, self.WG, self.SEP = eps_core, eps_clad, WG, SEP
        self.I_D0, self.NXD, self.NYD, self.i_src, self.i_mon, self.R = I_D0, NXD, NYD, i_src, i_mon, R
        self.k0 = TAU / lam
        self.JC = ny // 2
        self.I_D1 = I_D0 + NXD
        self.J_D0, self.J_D1 = self.JC - NYD // 2, self.JC + NYD // 2
        sxi, sxh = self.s_profile(nx, False), self.s_profile(nx, True)
        syi, syh = self.s_profile(ny, False), self.s_profile(ny, True)
        self.L0 = (sp.kron(self.lap1d(nx, sxh), sp.diags(syi)) + sp.kron(sp.diags(sxi), self.lap1d(ny, syh))).tocsc()
        self.SXSY = np.outer(sxi, syi)
        eline = np.full(ny, eps_clad)
        eline[self.wg_rows(0)] = eps_core
        D2 = sp.diags([np.ones(ny - 1), -2 * np.ones(ny), np.ones(ny - 1)], [-1, 0, 1]) / dx ** 2
        w, v = np.linalg.eigh((D2 + sp.diags(self.k0 ** 2 * eline)).toarray())
        self.n_guided = int(np.sum(w > self.k0 ** 2 * eps_clad))
        m = v[:, -1] * np.sign(v[self.JC, -1])
        self.m_in = m / np.sqrt(np.sum(m ** 2) * dx)                 # fundamental mode of the cross-section (E08)
        self.neff = np.sqrt(w[-1]) / self.k0
        self.m_up, self.m_dn = np.roll(self.m_in, SEP), np.roll(self.m_in, -SEP)
        B = np.zeros((nx, ny), complex)
        B[i_src] = self.m_in / dx                                      # mode source (a current sheet)
        self.b = B.ravel()
        eref = np.full((nx, ny), eps_clad)
        eref[:, self.wg_rows(0)] = eps_core
        self.x_ref = spl.splu(self.operator(eref)).solve(self.b).reshape(nx, ny)
        self.P_ref = abs(np.sum(self.m_in * self.x_ref[i_mon]) * dx) ** 2      # power of a straight guide = 100 %
        self.build_filter()

    def s_profile(self, n, half):
        pos = np.arange(n) + (0.5 if half else 0.0)
        d = np.maximum(np.maximum(self.npml - pos, pos - (n - 1 - self.npml)), 0) / self.npml
        return 1 + 1j * 4.0 * d ** 3                                   # s = 1 + i sigma/omega (e^{-i w t})

    def lap1d(self, n, sh):
        D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n, n)) / self.dx
        return -(D.T @ sp.diags(1 / sh) @ D)

    def operator(self, eps):
        return (self.L0 + sp.diags((self.k0 ** 2 * self.SXSY * eps).ravel())).tocsc()

    def wg_rows(self, off):
        return slice(self.JC - self.WG // 2 + off, self.JC + self.WG // 2 + off)

    def base_eps(self):
        e = np.full((self.nx, self.ny), self.eps_clad)
        e[:self.I_D0, self.wg_rows(0)] = self.eps_core
        e[self.I_D1:, self.wg_rows(self.SEP)] = self.eps_core
        e[self.I_D1:, self.wg_rows(-self.SEP)] = self.eps_core
        return e

    def y_arm_centre(self, t, sgn):
        return self.JC + sgn * self.SEP * (1 - np.cos(np.pi * t)) / 2

    def ybranch_fill(self, ss=8, scale=1):
        """Fill fraction of the human Y-branch (two cosine S-bends) on the design box, supersampled ss x ss,
        on a grid `scale` times finer than the simulation grid -> (NXD*scale, ny*scale)."""
        nxs, nys = self.NXD * scale, self.ny * scale
        out = np.zeros((nxs, nys))
        for i in range(nxs):
            acc = np.zeros(nys)
            for sub in range(ss):
                t = (i + (sub + 0.5) / ss) / nxs
                jj = (np.arange(nys)[:, None] + (np.arange(ss)[None, :] + 0.5) / ss) / scale
                inside = np.zeros_like(jj, bool)
                for sgn in (1, -1):
                    inside |= np.abs(jj - self.y_arm_centre(t, sgn)) <= self.WG / 2
                acc += inside.mean(1) / ss
            out[i] = acc
        return out

    def ybranch_eps(self):
        e = self.base_eps()
        e[self.I_D0:self.I_D1] = self.eps_clad + self.ybranch_fill() * (self.eps_core - self.eps_clad)
        return e

    def forward(self, eps):
        lu = spl.splu(self.operator(eps))
        x = lu.solve(self.b).reshape(self.nx, self.ny)
        au = np.sum(self.m_up * x[self.i_mon]) * self.dx
        ad = np.sum(self.m_dn * x[self.i_mon]) * self.dx
        return x, lu, au, ad

    def transmission(self, eps):
        x, _, au, ad = self.forward(eps)
        return (abs(au) ** 2 + abs(ad) ** 2) / self.P_ref, x

    def merit_grad_eps(self, eps):
        """F = (|<m_up, x>|^2 + |<m_dn, x>|^2) / P_ref and dF/deps by the adjoint method."""
        x, lu, au, ad = self.forward(eps)
        F = (abs(au) ** 2 + abs(ad) ** 2) / self.P_ref
        g = np.zeros((self.nx, self.ny), complex)
        g[self.i_mon] = (np.conj(au) * self.m_up + np.conj(ad) * self.m_dn) * self.dx / self.P_ref   # dF/dx (Wirtinger)
        lam_ = lu.solve(g.ravel()).reshape(self.nx, self.ny)          # A^T lam = dF/dx, and A^T = A: same LU
        dFde = -2 * np.real(lam_ * self.k0 ** 2 * self.SXSY * x)      # dA/deps_j = k0^2 e_j e_j^T (sx sy = 1 there)
        return F, dFde, x, lam_, lu

    def build_filter(self):
        NX, NY, R = self.NXD, self.NYD, self.R
        idx = np.arange(NX * NY).reshape(NX, NY)
        rows, cols, vals = [], [], []
        r = int(np.ceil(R))
        I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                wgt = max(0.0, R - np.hypot(di, dj))
                if wgt <= 0:
                    continue
                I2, J2 = I + di, J + dj
                ok = (I2 >= 0) & (I2 < NX) & (J2 >= 0) & (J2 < NY)
                rows.append(idx[ok])
                cols.append(idx[I2[ok], J2[ok]])
                vals.append(np.full(ok.sum(), wgt))
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(NX * NY, NX * NY))
        self.K = (sp.diags(1 / np.asarray(K.sum(1)).ravel()) @ K).tocsr()       # conic min-feature filter
        self.KT = self.K.T.tocsr()

    @staticmethod
    def project(rt, b, eta=0.5):
        den = np.tanh(b * eta) + np.tanh(b * (1 - eta))
        return (np.tanh(b * eta) + np.tanh(b * (rt - eta))) / den, b * (1 - np.tanh(b * (rt - eta)) ** 2) / den

    def design_eps(self, rho, b):
        rs = 0.5 * (rho + rho[:, ::-1])                                # mirror symmetry -> equal split
        rt = (self.K @ rs.ravel()).reshape(self.NXD, self.NYD)
        rp, drp = self.project(rt, b)
        e = self.base_eps()
        e[self.I_D0:self.I_D1, self.J_D0:self.J_D1] = self.eps_clad + rp * (self.eps_core - self.eps_clad)
        return e, rp, drp

    def F_and_grad(self, rho, b):
        e, rp, drp = self.design_eps(rho, b)
        F, dFe, x, lam_, lu = self.merit_grad_eps(e)
        sl = (slice(self.I_D0, self.I_D1), slice(self.J_D0, self.J_D1))
        g_rp = dFe[sl] * (self.eps_core - self.eps_clad)
        g_rs = (self.KT @ (g_rp * drp).ravel()).reshape(self.NXD, self.NYD)
        return F, 0.5 * (g_rs + g_rs[:, ::-1]), dict(x=x, lam=lam_, dFde=dFe, rp=rp, eps=e, lu=lu)

    def optimize(self, rho0, betas, lr=0.03, on_iter=None, verbose=True):
        rho = rho0.copy()
        m1, m2 = np.zeros_like(rho), np.zeros_like(rho)
        b1, b2 = 0.9, 0.999
        hist = []
        t0 = time.time()
        for it, bb in enumerate(betas):
            F, g, info = self.F_and_grad(rho, bb)
            hist.append(F)
            if on_iter:
                on_iter(it, F, g, info, rho)
            m1 = b1 * m1 + (1 - b1) * g
            m2 = b2 * m2 + (1 - b2) * g ** 2
            mh, vh = m1 / (1 - b1 ** (it + 1)), m2 / (1 - b2 ** (it + 1))
            rho = np.clip(rho + lr * mh / (np.sqrt(vh) + 1e-12), 0, 1)       # Adam, ascent
            if verbose and it % 20 == 0:
                gray = np.mean(4 * info["rp"] * (1 - info["rp"]))
                log(f"    it {it:3d} beta {bb:4.0f} F {F:.4f} grey {gray:.3f}  {time.time() - t0:.0f}s")
        return rho, np.array(hist)


BETAS = [2.0] * 40 + [4.0] * 25 + [8.0] * 25 + [16.0] * 25 + [32.0] * 25 + [64.0] * 20     # 160 iterations
CROP = (slice(44, 204), slice(30, 150))           # displayed window (cells): x 1.32..6.12 um, y 0.9..4.5 um


def up2(a, order=3):
    """2x upsampling of a (real or complex) map for display."""
    if np.iscomplexobj(a):
        return ndimage.zoom(a.real, 2, order=order) + 1j * ndimage.zoom(a.imag, 2, order=order)
    return ndimage.zoom(a, 2, order=order)


def run_splitter():
    t0 = time.time()
    s = Sim()
    log(f"[splitter] grid {s.nx}x{s.ny}, guided modes {s.n_guided}, n_eff {s.neff:.3f}")
    out = dict(dx=s.dx, lam=s.lam, crop=np.array([44, 204, 30, 150]), I_D0=s.I_D0, I_D1=s.I_D1, J_D0=s.J_D0,
               J_D1=s.J_D1, JC=s.JC, WG=s.WG, SEP=s.SEP, i_src=s.i_src, i_mon=s.i_mon, neff=s.neff)
    vref = np.abs(s.x_ref[CROP]).max()
    out["vref"] = vref
    out["x_ref"] = up2(s.x_ref[CROP]).astype(np.complex64)          # straight guide: the incident mode alone

    # ---- the human design: Y-branch (cosine S-bends, sub-cell averaged)
    ey = s.ybranch_eps()
    T_y, x_y = s.transmission(ey)
    log(f"[splitter] Y-branch T = {T_y:.4f}")
    fill_hi = np.zeros((s.nx * 4, s.ny * 4))
    fill_hi[: s.I_D0 * 4, (s.JC - s.WG // 2) * 4:(s.JC + s.WG // 2) * 4] = 1
    for sgn in (1, -1):
        fill_hi[s.I_D1 * 4:, (s.JC - s.WG // 2 + sgn * s.SEP) * 4:(s.JC + s.WG // 2 + sgn * s.SEP) * 4] = 1
    fill_hi[s.I_D0 * 4:s.I_D1 * 4] = s.ybranch_fill(ss=4, scale=4)
    out.update(T_y=T_y, x_y=up2(x_y[CROP]).astype(np.complex64),
               ymask_hi=fill_hi[CROP[0].start * 4:CROP[0].stop * 4, CROP[1].start * 4:CROP[1].stop * 4].astype(F16))

    # ---- iteration 0 (uniform grey box): forward, adjoint, sensitivity; finite-difference check
    rho0 = np.full((s.NXD, s.NYD), 0.5)
    F0, g0, info0 = s.F_and_grad(rho0, BETAS[0])
    fd_pix = [(10, 20), (25, 40), (40, 12), (30, 60), (45, 33)]
    fd_adj, fd_num = [], []
    for (i, j) in fd_pix:
        h = 1e-3
        rp_, rm_ = rho0.copy(), rho0.copy()
        rp_[i, j] += h
        rm_[i, j] -= h
        Fp = s.merit_grad_eps(s.design_eps(rp_, BETAS[0])[0])[0]
        Fm = s.merit_grad_eps(s.design_eps(rm_, BETAS[0])[0])[0]
        fd_adj.append(g0[i, j])
        fd_num.append((Fp - Fm) / (2 * h))
        log(f"[splitter] FD check pixel {(i, j)}: adjoint {g0[i, j]:+.6e}  finite diff {(Fp - Fm) / (2 * h):+.6e}")
    fd_adj, fd_num = np.array(fd_adj), np.array(fd_num)
    out.update(F0=F0, fd_pix=np.array(fd_pix), fd_adj=fd_adj, fd_num=fd_num,
               fd_relerr=float(np.max(np.abs(fd_adj - fd_num) / np.abs(fd_num))),
               x0=up2(info0["x"][CROP]).astype(np.complex64), lam0=up2(info0["lam"][CROP]).astype(np.complex64),
               eps0=info0["eps"][CROP].astype(np.float32),
               dFde0=info0["dFde"][s.I_D0:s.I_D1, s.J_D0:s.J_D1].astype(np.float32), g0=g0.astype(np.float32))
    # one nudged pixel: its scattered wavelet  dx = -A^{-1} (dA/deps_j) x  (exact first-order response)
    pi, pj = 30, 52                                                    # design-box pixel (x, y index)
    gi, gj = s.I_D0 + pi, s.J_D0 + pj
    src = np.zeros((s.nx, s.ny), complex)
    src[gi, gj] = -s.k0 ** 2 * info0["x"][gi, gj]
    dxn = info0["lu"].solve(src.ravel()).reshape(s.nx, s.ny)
    out.update(nudge_pix=np.array([pi, pj]), nudge=up2(dxn[CROP]).astype(np.complex64),
               nudge_x=complex(info0["x"][gi, gj]), nudge_lam=complex(info0["lam"][gi, gj]),
               nudge_slope=float(info0["dFde"][gi, gj]))

    # ---- main optimisation (adjoint gradients + Adam), every iteration stored
    rp_snaps, absx_snaps = [], []

    def keep(it, F, g, info, rho):
        rp_snaps.append(info["rp"].astype(F16))
        absx_snaps.append(np.abs(info["x"][CROP]).astype(F16))
    log("[splitter] optimising (160 iterations) ...")
    rho, hist = s.optimize(rho0, BETAS, on_iter=keep)
    e_opt, rp_opt, _ = s.design_eps(rho, 1000.0)                      # hard binarisation (all silicon / all glass)
    T_opt, x_opt = s.transmission(e_opt)
    log(f"[splitter] optimised, last grey-ish iterate F = {hist[-1]:.4f}; binarised design T = {T_opt:.4f}")
    out.update(F_hist=hist, rp_snaps=np.array(rp_snaps), absx_snaps=np.array(absx_snaps), T_opt=T_opt,
               rp_opt=rp_opt.astype(np.float32), bin_opt=(rp_opt > 0.5).astype(np.uint8),
               x_opt=up2(x_opt[CROP]).astype(np.complex64), eps_opt=e_opt[CROP].astype(np.float32))
    # the optimised design's adjoint field (for the backward-light shot at the end)
    _, _, _, lam_opt, _ = s.merit_grad_eps(e_opt)
    out["lam_opt"] = up2(lam_opt[CROP]).astype(np.complex64)

    # ---- two more starting guesses -> different local optima
    alts, T_alts, starts = [], [], [rho0.astype(F16)]
    for seed in (11, 12):
        rng = np.random.default_rng(seed)
        nz_ = ndimage.gaussian_filter(rng.standard_normal((s.NXD, s.NYD)), 3.0)
        r0 = np.clip(0.5 + 0.35 * nz_ / nz_.std(), 0, 1)
        starts.append(r0.astype(F16))
        log(f"[splitter] local optimum from seed {seed} ...")
        ra, _ = s.optimize(r0, BETAS, verbose=False)
        ea, rpa, _ = s.design_eps(ra, 1000.0)
        Ta, _ = s.transmission(ea)
        log(f"[splitter]   seed {seed}: binarised T = {Ta:.4f}")
        alts.append((rpa > 0.5).astype(np.uint8))
        T_alts.append(Ta)
    out.update(alt_bins=np.array(alts), T_alts=np.array(T_alts), starts=np.array(starts))

    # ---- what A looks like: a 6 x 6 grid -> 36 x 36 matrix (no PML), eps from a little pattern
    n6 = 6
    D = sp.diags([-np.ones(n6), np.ones(n6 - 1)], [0, 1], shape=(n6, n6))
    L6 = -(D.T @ D)
    A6 = (sp.kron(L6, sp.eye(n6)) + sp.kron(sp.eye(n6), L6)).toarray() + np.diag(np.linspace(2.0, 3.5, n6 * n6))
    out["A6"] = A6.astype(np.float32)
    np.savez_compressed(os.path.join(DATA, "splitter.npz"), **out)
    log(f"[splitter] done in {time.time() - t0:.0f}s -> data/splitter.npz")


# ===================================================================== 2. ASM phase mask (JAX) ===
def target_nabla(N, size=200, blur=1.2):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    img = Image.new("L", (N, N), 0)
    dr = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    bb = dr.textbbox((0, 0), "∇", font=font)
    dr.text(((N - (bb[2] - bb[0])) / 2 - bb[0], (N - (bb[3] - bb[1])) / 2 - bb[1]), "∇", fill=255, font=font)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(blur)), float) / 255.0


def run_asm():
    import jax
    import jax.numpy as jnp
    t0 = time.time()
    N, P, dx, z = 256, 512, 2.0, 1200.0                  # mask pixels, padded grid, pixel = 2 wavelengths, distance
    k = TAU
    kx = TAU * np.fft.fftfreq(P, d=dx)
    KX, KY = np.meshgrid(kx, kx)
    kz = np.sqrt((k ** 2 - KX ** 2 - KY ** 2).astype(complex))
    H = np.exp(1j * kz * z) * (np.real(kz) > 0)          # transfer function (E01); evanescent part ~ e^{-kappa z} = 0
    c = slice((P - N) // 2, (P + N) // 2)
    T = np.zeros((P, P))
    T[c, c] = target_nabla(N)
    ap = np.zeros((P, P))
    ap[c, c] = 1.0
    Hj, Tj, Aj = jnp.asarray(H, jnp.complex64), jnp.asarray(T, jnp.float32), jnp.asarray(ap, jnp.float32)

    def forward(phi):
        u1 = jnp.fft.ifft2(jnp.fft.fft2(Aj * jnp.exp(1j * phi)) * Hj)
        return u1, jnp.abs(u1) ** 2

    def loss_I(I):
        s_ = jnp.sum(I * Tj) / jnp.sum(Tj * Tj)
        return jnp.sum((I - s_ * Tj) ** 2) / jnp.sum((s_ * Tj) ** 2)

    def loss(phi):
        return loss_I(forward(phi)[1])

    vg = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.uniform(0, TAU, (P, P)), jnp.float32) * Aj
    # check: reverse-mode autodiff == propagate the error backward by -z (multiply by conj(H))
    L0, g0 = vg(phi)
    u1, I = forward(phi)
    err = 2 * jax.grad(loss_I)(I) * u1                  # error field at the image plane (2 dL/d conj(u1))
    back = jnp.fft.ifft2(jnp.fft.fft2(err) * jnp.conj(Hj))   # ... propagated by -z
    u0 = Aj * jnp.exp(1j * phi)
    g_manual = jnp.real(jnp.conj(1j * u0) * back)
    relerr = float(jnp.abs(g_manual - g0).max() / jnp.abs(g0).max())
    log(f"[asm] backward pass = propagation by -z: max rel. difference {relerr:.1e}")
    snap_its = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 19, 24, 30, 38, 48, 60, 75, 95, 120, 150, 190, 240, 299]
    lr, b1, b2 = 0.08, 0.9, 0.999
    m1, m2 = jnp.zeros_like(phi), jnp.zeros_like(phi)
    hist, ph_s, I_s = [], [], []
    for it in range(300):
        L, g = vg(phi)
        hist.append(float(L))
        if it in snap_its:
            Ii = np.asarray(forward(phi)[1])[c, c]
            ph_s.append((np.mod(np.asarray(phi)[c, c], TAU)).astype(F16))
            I_s.append((Ii / np.percentile(Ii, 99.5)).astype(F16))
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        phi = (phi - lr * (m1 / (1 - b1 ** (it + 1))) / (jnp.sqrt(m2 / (1 - b2 ** (it + 1))) + 1e-8)) * Aj
    log(f"[asm] loss {hist[0]:.3f} -> {hist[-1]:.4f} in {time.time() - t0:.1f}s")
    eb = np.abs(np.asarray(err))[c, c]
    bk = np.abs(np.asarray(back))[c, c]
    np.savez_compressed(os.path.join(DATA, "asm.npz"), hist=np.array(hist), snap_its=np.array(snap_its),
                        phase=np.array(ph_s), inten=np.array(I_s), target=T[c, c].astype(F16), relerr=relerr,
                        err0=(eb / eb.max()).astype(F16), back0=(bk / bk.max()).astype(F16), z=z, dx=dx, N=N)
    log("[asm] -> data/asm.npz")


# ===================================================================== 3. EDOF end-to-end (JAX) ==
def test_scene(M):
    """Synthetic test scene: Siemens star, bar groups and letters (values 0..1)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (M, M), 20)
    dr = ImageDraw.Draw(img)
    cx, cy, R = M * 0.32, M * 0.36, M * 0.25
    for k in range(18):
        a0, a1 = TAU * k / 18, TAU * (k + 0.5) / 18
        pts = [(cx, cy)] + [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in np.linspace(a0, a1, 8)]
        dr.polygon(pts, fill=235)
    for g, (x0, w) in enumerate(((M * 0.66, 6), (M * 0.80, 4), (M * 0.90, 3))):
        for b in range(3):
            dr.rectangle([x0 + 2 * b * w, M * 0.12, x0 + 2 * b * w + w - 1, M * 0.55], fill=235)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(M * 0.2))
    dr.text((M * 0.08, M * 0.66), "light", fill=235, font=font)
    return np.asarray(img, float) / 255.0


def run_edof():
    import jax
    import jax.numpy as jnp
    from math import factorial
    t0 = time.time()
    M, NP, K = 256, 80, 9
    PSI = 3.0 * np.pi                                   # defocus phase at the pupil edge: +-3 pi (+-1.5 waves)
    psis = np.linspace(-PSI, PSI, K)
    SIG2 = 1e-8
    u = (np.arange(M) - M // 2) / (NP / 2)
    UX, UY = np.meshgrid(u, u)
    RR = np.hypot(UX, UY)
    P = (RR <= 1.0).astype(np.float32)

    def zern_radial(n):
        Rn = np.zeros_like(RR)
        for s_ in range(n // 2 + 1):
            Rn += (-1) ** s_ * factorial(n - s_) / (factorial(s_) * factorial(n // 2 - s_) ** 2) * RR ** (n - 2 * s_)
        return Rn * P
    orders = list(range(4, 25, 2))
    Z = np.stack([zern_radial(n) for n in orders]).astype(np.float32)
    f = np.fft.fftfreq(M)
    FX, FY = np.meshgrid(f, f)
    S = 1.0 / (FX ** 2 + FY ** 2 + (1.0 / M) ** 2)
    S /= S.sum()                                        # natural-image-like spectrum ~ 1/f^2
    Pj, R2j, Sj, Zj = jnp.asarray(P), jnp.asarray(RR ** 2, jnp.float32), jnp.asarray(S, jnp.float32), jnp.asarray(Z)

    def phase(cf):
        return jnp.tensordot(cf, Zj, 1)

    def psfs(phi, ps):
        pupil = Pj[None] * jnp.exp(1j * (phi[None] + jnp.asarray(ps, jnp.float32)[:, None, None] * R2j[None]))
        p = jnp.abs(jnp.fft.fft2(jnp.fft.ifftshift(pupil, axes=(1, 2)))) ** 2
        return p / jnp.sum(p, axis=(1, 2), keepdims=True)

    def wiener_mse(phi):
        Hk = jnp.fft.fft2(psfs(phi, psis))
        W = jnp.sum(jnp.conj(Hk), 0) * Sj / (jnp.sum(jnp.abs(Hk) ** 2, 0) * Sj + K * SIG2)   # one depth-blind filter
        E = jnp.sum(Sj[None] * jnp.abs(1 - W[None] * Hk) ** 2 + SIG2 * jnp.abs(W[None]) ** 2, axis=(1, 2))
        return W, E

    def loss(cf):
        return jnp.mean(wiener_mse(phase(cf))[1])

    vg = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(3)
    cf = jnp.asarray(0.05 * rng.standard_normal(len(orders)), jnp.float32)
    lr, b1, b2 = 0.05, 0.9, 0.999
    m1, m2 = jnp.zeros_like(cf), jnp.zeros_like(cf)
    snap_its = sorted(set([int(round(v)) for v in np.geomspace(1, 1500, 40)] + [0]))
    hist, cfs = [], []
    for it in range(1500):
        L, g = vg(cf)
        hist.append(float(L))
        if it in snap_its:
            cfs.append(np.asarray(cf))
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        cf = cf - lr * (m1 / (1 - b1 ** (it + 1))) / (jnp.sqrt(m2 / (1 - b2 ** (it + 1))) + 1e-8)
    if snap_its[-1] != 1499:
        cfs.append(np.asarray(cf))
    zero = jnp.zeros((M, M), jnp.float32)
    W0, E0 = wiener_mse(zero)
    W1, E1 = wiener_mse(phase(cf))
    log(f"[edof] restoration error per depth, lens:    {np.round(np.asarray(E0), 3)}")
    log(f"[edof] restoration error per depth, learned: {np.round(np.asarray(E1), 3)}  ({time.time() - t0:.1f}s)")
    show = np.array([0, 2, 4, 6, 8])                    # 5 displayed depths
    cc = slice(M // 2 - 32, M // 2 + 32)

    def crop_psfs(phi):
        p = np.asarray(psfs(phi, psis[show]))
        return np.array([np.fft.fftshift(q)[cc, cc] for q in p])
    psf_lens = crop_psfs(zero)
    psf_snaps = np.array([crop_psfs(phase(jnp.asarray(c_))) for c_ in cfs])
    pc = slice(M // 2 - NP // 2 - 2, M // 2 + NP // 2 + 2)
    pupil_snaps = np.array([np.where(P[pc, pc] > 0, np.asarray(phase(jnp.asarray(c_)))[pc, pc], np.nan) for c_ in cfs])
    # images through the whole pipeline (scene -> PSF -> noise -> Wiener), at three depths
    scene = test_scene(M)
    sig_n = np.sqrt(M * M * SIG2 * np.mean(scene ** 2))  # noise energy / signal energy = M^2 SIG2, as in the model
    noise = np.random.default_rng(7).standard_normal((3, M, M)) * sig_n
    imgs = {}
    for name, phi, W in (("lens", zero, W0), ("learned", phase(cf), W1)):
        ps = np.asarray(psfs(phi, psis[[0, 4, 8]]))
        raw, rest = [], []
        for q in range(3):
            y = np.real(np.fft.ifft2(np.fft.fft2(scene) * np.fft.fft2(ps[q]))) + noise[q]
            raw.append(y)
            rest.append(np.real(np.fft.ifft2(np.fft.fft2(y) * np.asarray(W))))
        imgs[name + "_raw"] = np.array(raw).astype(F16)
        imgs[name + "_rest"] = np.array(rest).astype(F16)
    np.savez_compressed(os.path.join(DATA, "edof.npz"), psis=psis, show=show, E_lens=np.asarray(E0), E_learn=np.asarray(E1),
                        hist=np.array(hist), snap_its=np.array(snap_its), psf_lens=psf_lens.astype(np.float32),
                        psf_snaps=psf_snaps.astype(np.float32), pupil_snaps=pupil_snaps.astype(np.float32),
                        scene=scene.astype(F16), **imgs)
    log("[edof] -> data/edof.npz")


# ===================================================================== 4. TEA vs full wave ======
def run_gap():
    import jax
    import jax.numpy as jnp
    t0 = time.time()
    lam, n_g = 1.0, 1.5                                 # lengths in wavelengths; glass n = 1.5
    k0 = TAU / lam
    hmax = lam / (n_g - 1)                              # a full 2 pi of relief (E06)
    dx = 1 / 30
    XT, ZT, WSPOT, A_HALF = 6.0, 9.0, 0.75, 8.0        # target focus (x, z), spot half-width, aperture half-width
    nxp = 15
    X0, X1 = -13.0, 13.0
    nxi = int(round((X1 - X0) / dx))
    nx = nxi + 2 * nxp
    xs = X0 + (np.arange(nx) - nxp + 0.5) * dx
    amp = np.exp(-(xs / (A_HALF + 0.5)) ** 12)         # flat-top illumination with soft edges
    NPd = 4 * nx
    kx = TAU * np.fft.fftfreq(NPd, d=dx)
    kz = np.sqrt((k0 ** 2 - kx ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    pad0 = (NPd - nx) // 2
    spot = (np.abs(xs - XT) <= WSPOT).astype(np.float32)
    ampj, spotj = jnp.asarray(amp, jnp.float32), jnp.asarray(spot, jnp.float32)
    Pinc = float(np.sum(amp ** 2) * dx)

    def tea_field(h, Hz):
        u = ampj * jnp.exp(1j * k0 * (n_g - 1) * h)    # thin element: each column only delays the light
        U = jnp.zeros(NPd, jnp.complex64).at[pad0:pad0 + nx].set(u)
        return jnp.fft.ifft(jnp.fft.fft(U) * Hz)[pad0:pad0 + nx]

    Hz_t = jnp.asarray(np.exp(1j * kz * (ZT - hmax)), jnp.complex64)

    def eff(theta):
        return jnp.sum(spotj * jnp.abs(tea_field(hmax * jax.nn.sigmoid(theta), Hz_t)) ** 2) * dx / Pinc

    vg = jax.jit(jax.value_and_grad(lambda t: -eff(t)))
    theta = jnp.zeros(nx, jnp.float32)
    m1, m2 = jnp.zeros_like(theta), jnp.zeros_like(theta)
    for it in range(1500):
        _, g = vg(theta)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        theta = theta - 0.1 * (m1 / (1 - 0.9 ** (it + 1))) / (jnp.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
    h = np.array(hmax * jax.nn.sigmoid(theta))
    h[np.abs(xs) > A_HALF + 1.5] = 0.0
    eta_tea = float(jnp.sum(spotj * jnp.abs(tea_field(jnp.asarray(h), Hz_t)) ** 2) * dx / Pinc)
    zs = np.linspace(0.0, 12.0, 241)
    Umap = np.array([np.asarray(tea_field(jnp.asarray(h), jnp.asarray(np.exp(1j * kz * (z - hmax)), jnp.complex64)))
                     for z in np.maximum(zs, hmax)])
    log(f"[gap] relief optimised under TEA: {eta_tea:.3f} of the light in the focal spot (TEA's prediction)")
    # full-wave check with FDFD (same scalar Helmholtz solver as the splitter, glass relief on a glass substrate)
    Z0, Z1 = -2.5, 12.5
    nzp = 15
    nz = int(round((Z1 - Z0) / dx)) + 2 * nzp
    zc = Z0 + (np.arange(nz) - nzp + 0.5) * dx

    def s_prof(n, npml, half):
        pos = np.arange(n) + (0.5 if half else 0.0)
        d = np.maximum(np.maximum(npml - pos, pos - (n - 1 - npml)), 0) / npml
        return 1 + 1j * 5.0 * d ** 3

    def lap1d(n, sh):
        D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n, n)) / dx
        return -(D.T @ sp.diags(1 / sh) @ D)
    sxi, sxh, szi, szh = s_prof(nx, nxp, False), s_prof(nx, nxp, True), s_prof(nz, nzp, False), s_prof(nz, nzp, True)
    L0 = (sp.kron(lap1d(nx, sxh), sp.diags(szi)) + sp.kron(sp.diags(sxi), lap1d(nz, szh))).tocsc()
    SXSZ = np.outer(sxi, szi)
    B = np.zeros((nx, nz), complex)
    B[:, np.argmin(np.abs(zc + 1.5))] = amp / dx       # line source in the substrate

    def fdfd(hprof):
        Xg, Zg = np.meshgrid(xs, zc, indexing="ij")
        frac = np.clip((hprof[:, None] - (Zg - dx / 2)) / dx, 0, 1)
        eps = np.where(Zg < -dx / 2, n_g ** 2, 1.0 + frac * (n_g ** 2 - 1))
        A = (L0 + sp.diags((k0 ** 2 * SXSZ * eps).ravel())).tocsc()
        return spl.splu(A).solve(B.ravel()).reshape(nx, nz)
    iz = np.argmin(np.abs(zc - ZT))
    x_flat = fdfd(np.zeros(nx))
    P_flat = np.sum(np.abs(x_flat[:, iz]) ** 2) * dx
    x_rel = fdfd(h)
    eta_fdfd = float(np.sum(spot * np.abs(x_rel[:, iz]) ** 2) * dx / P_flat)
    log(f"[gap] same relief, full-wave FDFD: {eta_fdfd:.3f}  (TEA/FDFD = {eta_tea / eta_fdfd:.2f}; {time.time() - t0:.0f}s)")
    # display maps on a common grid: x in [-12, 12], z in [0, 12]; normalised by the flat-window intensity
    xsel = np.abs(xs) <= 12.0
    zsel = (zc >= 0) & (zc <= 12.0)
    I_fd = np.abs(x_rel[np.ix_(xsel, zsel)]) ** 2 / np.mean(np.abs(x_flat[np.abs(xs) < 4, iz]) ** 2)
    I_tea = (np.abs(Umap[:, xsel]) ** 2).T / np.mean(amp[np.abs(xs) < 4] ** 2)
    np.savez_compressed(os.path.join(DATA, "gap.npz"), xs=xs[xsel], zs_tea=zs, zs_fd=zc[zsel], h=h[xsel],
                        I_tea=I_tea[::2, ::1].astype(np.float32), I_fd=I_fd[::2, ::2].astype(np.float32),
                        eta_tea=eta_tea, eta_fdfd=eta_fdfd, XT=XT, ZT=ZT, hmax=hmax)
    log("[gap] -> data/gap.npz")


# ===================================================================== 5. landscape (schematic) ==
def run_land():
    n = 240
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)

    def Ff(x, y):
        return (1.0 * np.exp(-((x - 0.45) ** 2 + (y - 0.38) ** 2) / 0.10) + 0.62 * np.exp(-((x + 0.5) ** 2 + (y - 0.45) ** 2) / 0.07)
                + 0.45 * np.exp(-((x + 0.15) ** 2 + (y + 0.55) ** 2) / 0.09) + 0.08 * np.cos(3.0 * x + 1.0) * np.cos(2.4 * y))
    Fg = Ff(X, Y)
    p = np.array([-0.55, -0.62])
    path = [p.copy()]
    h = 1e-4
    for _ in range(400):                                   # gradient ascent on the schematic landscape
        gx = (Ff(p[0] + h, p[1]) - Ff(p[0] - h, p[1])) / (2 * h)
        gy = (Ff(p[0], p[1] + h) - Ff(p[0], p[1] - h)) / (2 * h)
        p = p + 0.01 * np.array([gx, gy]) / max(np.hypot(gx, gy), 0.25)
        path.append(p.copy())
    np.savez_compressed(os.path.join(DATA, "landscape.npz"), F=Fg.astype(np.float32), path=np.array(path))
    log("[land] -> data/landscape.npz")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["splitter", "asm", "edof", "gap", "land"]
    for p_ in parts:
        {"splitter": run_splitter, "asm": run_asm, "edof": run_edof, "gap": run_gap, "land": run_land}[p_]()
