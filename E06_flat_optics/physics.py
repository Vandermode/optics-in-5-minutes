"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).
    python physics.py            (2D ASM runs in JAX on a GPU if one is visible)

Models
------
* Thin element approximation (TEA): a column of glass of height h delays the light by phi = k0 (n(lam) - 1) h, so
  U_out(x, y) = U_in(x, y) exp(i phi(x, y)).
* Exact scalar angular-spectrum propagation (ASM, Episode 1) behind the element, kz = sqrt(k^2 - kx^2 - ky^2).
* Microlens (hook, Ch2, Ch4, Ch6): circular, radius a = 20 um, f0 = 43.25 um at lam0 = 550 nm (exactly 8 zones),
  glass with a BK7-like Cauchy dispersion n(lam) = A + B/lam^2, n(550 nm) = 1.50, Abbe number ~64.
  Thick lens h(r) = (f0 + 8 lam0 - sqrt(r^2 + f0^2)) / (n0 - 1) (TEA phase = ideal lens), kinoform = h mod lam0/(n0-1).
  Full 2D (x, y) ASM on a 1024^2 grid (80 um, dx = 78 nm); the x-z slice through the axis is stored.
* Ch1 demo lens (wavelength scale, cylindrical): 1D TEA columns inside the lens, then 1D ASM.
* Ch3: N-level staircase of one blazed period; diffraction orders |c_m|^2 by FFT; phasor chains.
* Ch5: 2D finite-difference frequency-domain (FDFD) Maxwell solver, periodic in x, PML in z, both polarisations
  (TE: E along the grooves, TM: H along the grooves); blazed glass grating lit from the substrate side; metasurface
  ridge library (TiO2-like ridges on glass, TE). Energy conservation T + R = 1 to < 1e-5; Fresnel check T = 0.9607.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi

# ------------------------------------------------------------------ the microlens --------------------------------
LAM0 = 0.55                     # um, design wavelength (green)
N0 = 1.50                       # index at LAM0
CB = 0.00409                    # Cauchy B (um^2): BK7-like dispersion, Abbe number ~64
CA = N0 - CB / LAM0 ** 2
A_LENS = 20.0                   # um, lens radius
NZONE = 8
F0 = (A_LENS ** 2 - (NZONE * LAM0) ** 2) / (2 * NZONE * LAM0)     # sqrt(a^2+f^2) - f = 8 lam0  -> 43.25 um
H2PI = LAM0 / (N0 - 1)          # 1.1 um: one full turn of delay
SAG = NZONE * H2PI              # 8.8 um: thick-lens sag


def n_glass(lam):
    return CA + CB / lam ** 2


def h_thick(r):
    return np.clip((F0 + NZONE * LAM0 - np.sqrt(r ** 2 + F0 ** 2)) / (N0 - 1), 0, None)


def h_kino(r):
    h = h_thick(r)
    return np.where(r <= A_LENS, np.mod(h, H2PI), 0.0)


def zone_edges():
    m = np.arange(0, NZONE + 1)
    return np.sqrt((F0 + m * LAM0) ** 2 - F0 ** 2)                  # r where the thick lens crosses slab lines


# ------------------------------------------------------------------ CIE colour (white light composites) ----------
def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz(lam_nm):
    """CIE 1931 2-degree colour matching functions, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013)."""
    l = np.asarray(lam_nm, float)
    x = 1.056 * _g(l, 599.8, 37.9, 31.0) + 0.362 * _g(l, 442.0, 16.0, 26.7) - 0.065 * _g(l, 501.1, 20.4, 26.2)
    y = 0.821 * _g(l, 568.8, 46.9, 40.5) + 0.286 * _g(l, 530.9, 16.3, 31.1)
    z = 1.217 * _g(l, 437.0, 11.8, 36.0) + 0.681 * _g(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


M_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def tone_map(rgb, expo):
    """Linear RGB (white = (1,1,1) at unit intensity) -> uint8. The luminance is compressed (1 - e^-x) while the
    chromaticity is kept; whatever exceeds the gamut is blended towards white, like an overexposed photograph."""
    Y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    Yt = 1 - np.exp(-expo * Y)
    out = rgb * (Yt / np.maximum(Y, 1e-12))[..., None]
    m = out.max(-1, keepdims=True)
    over = np.clip((m - 1) / np.maximum(m, 1e-12), 0, 1)
    out = np.where(m > 1, out / np.maximum(m, 1e-12), out) * (1 - over) + over
    return (255 * np.clip(out, 0, 1) ** (1 / 1.6)).astype(np.uint8)


def white_light_rgb(stack, lams_um, expo):
    """stack: (nl, nz, nx) intensities for an equal-energy spectrum -> uint8 RGB (nz, nx, 3)."""
    cmf = cie_xyz(lams_um * 1000)                                    # (nl, 3)
    XYZ = np.einsum("lzx,lc->zxc", stack, cmf)
    rgb = XYZ @ M_XYZ2RGB.T
    white = (cmf.sum(0) @ M_XYZ2RGB.T)                               # equal-energy white -> (1, 1, 1)
    return tone_map(np.clip(rgb / white, 0, None), expo)


def mono_rgb(I, lam_um, expo):
    """Intensity of one wavelength -> uint8 RGB in that wavelength's own (gamut-clipped) sRGB colour."""
    c = np.clip(cie_xyz(lam_um * 1000) @ M_XYZ2RGB.T, 0, None)
    c = c / (c @ np.array([0.2126, 0.7152, 0.0722]))                 # unit luminance
    return tone_map(I[..., None] * c[None, None, :], expo)


# ------------------------------------------------------------------ 2D ASM (JAX on GPU, numpy fallback) ----------
def make_xz_engine(N, dx):
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", False)
        dev = jax.devices()[0]
        use_jax = dev.platform == "gpu"
    except Exception:                                               # pragma: no cover
        use_jax = False
    kx = TAU * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, kx)
    K2 = (KX ** 2 + KY ** 2).astype(np.float32)
    if use_jax:
        K2j = jnp.asarray(K2)

        @jax.jit
        def rows(F, k, zs):
            kz = jnp.sqrt((k ** 2 - K2j).astype(jnp.complex64))
            kz = jnp.where(jnp.imag(kz) < 0, -kz, kz)

            def one(z):
                G = F * jnp.exp(1j * kz * z)
                return jnp.fft.ifft(jnp.sum(G, axis=0)) / N        # the y = 0 row of ifft2(G)
            return jax.vmap(one)(zs)

        def xz(U0, lam, zs):
            F = jnp.fft.fft2(jnp.asarray(U0, jnp.complex64))
            out = []
            for i in range(0, len(zs), 32):
                out.append(np.asarray(rows(F, np.float32(TAU / lam), jnp.asarray(zs[i:i + 32], jnp.float32))))
            return np.concatenate(out, 0)

        def plane(U0, lam, z):
            F = jnp.fft.fft2(jnp.asarray(U0, jnp.complex64))
            k = TAU / lam
            kz = jnp.sqrt((k ** 2 - K2j).astype(jnp.complex64))
            kz = jnp.where(jnp.imag(kz) < 0, -kz, kz)
            return np.asarray(jnp.fft.ifft2(F * jnp.exp(1j * kz * z)))
        return xz, plane, "jax-gpu"

    def xz_np(U0, lam, zs):
        F = np.fft.fft2(U0)
        k = TAU / lam
        kz = np.sqrt((k ** 2 - K2).astype(complex))
        kz = np.where(kz.imag < 0, -kz, kz)
        return np.array([np.fft.ifft(np.sum(F * np.exp(1j * kz * z), axis=0)) / N for z in zs])

    def plane_np(U0, lam, z):
        k = TAU / lam
        kz = np.sqrt((k ** 2 - K2).astype(complex))
        kz = np.where(kz.imag < 0, -kz, kz)
        return np.fft.ifft2(np.fft.fft2(U0) * np.exp(1j * kz * z))
    return xz_np, plane_np, "numpy"


EXPO = 0.07                     # white-light exposure (incident beam -> dim grey, foci saturate)
GREEN_GAIN = 1.00               # single-wavelength image brightness relative to the white composite
SIG_ROUND = 0.1                 # um, Gaussian blur of the fabricated kinoform (Ch6)


def lens_fields():
    """Hook / Ch2 / Ch4 / Ch6: x-z slices of the circular microlens and its kinoform (TEA + 2D ASM)."""
    t0 = time.time()
    N, L = 1024, 80.0
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    taper = np.clip((A_LENS - R) / 2.0, 0, 1)                         # raised-cosine edge over the last 2 um
    aperture = np.sin(0.5 * np.pi * taper) ** 2                       # (suppresses the hard-edge axial ripples)
    xz, plane, eng = make_xz_engine(N, dx)
    zmax, nz = 66.0, 528
    zs = np.linspace(0, zmax, nz)
    xw = 24.0                                                         # stored half width (um)
    cols = np.where(np.abs(x) <= xw)[0]
    ht, hk = h_thick(R), h_kino(R)

    def run(h, lam, ap=aperture):
        U0 = ap * np.exp(1j * TAU / lam * (n_glass(lam) - 1) * h)
        U = np.fft.fftshift(xz(np.fft.ifftshift(U0), lam, zs), axes=1)  # centre at index N//2
        return np.abs(U[:, cols]) ** 2

    lams = np.round(np.arange(0.42, 0.6801, 0.01), 3)
    st_t = np.array([run(ht, l) for l in lams])
    st_k = np.array([run(hk, l) for l in lams])
    i550 = int(np.argmin(np.abs(lams - 0.55)))
    I_t, I_k = st_t[i550], st_k[i550]
    # downsample x by 2 (box) for display: 615 -> ~307 columns; keep full z
    def ds(A):
        n2 = A.shape[-1] // 2 * 2
        return 0.5 * (A[..., :n2:2] + A[..., 1:n2:2])
    rgb_t = white_light_rgb(st_t, lams, EXPO)
    rgb_k = white_light_rgb(st_k, lams, EXPO)
    rgb_g = mono_rgb(I_k, LAM0, EXPO * GREEN_GAIN)
    ic = len(cols) // 2
    ax = {}
    for nm, l in (("B", 0.45), ("G", 0.55), ("R", 0.65)):
        i = int(np.argmin(np.abs(lams - l)))
        ax["ax_t_" + nm] = st_t[i][:, ic]
        ax["ax_k_" + nm] = st_k[i][:, ic]
    # focal-plane profiles (z = f0) at 550 nm for both elements -> identical under TEA
    jf = int(np.argmin(np.abs(zs - F0)))
    prof_t, prof_k = I_t[jf], I_k[jf]
    # Ch6: fabricated (rounded) kinoform: 2D Gaussian blur of the height map
    sig = SIG_ROUND
    KX = TAU * np.fft.fftfreq(N, d=dx)
    G = np.exp(-0.5 * sig ** 2 * (KX[None, :] ** 2 + KX[:, None] ** 2))
    hk_r = np.real(np.fft.ifft2(np.fft.fft2(hk) * G))
    I_r = run(hk_r, LAM0)
    jpk = int(np.argmax(I_k[:, ic]))
    peak_ratio = float(I_r[:, ic].max() / I_k[:, ic].max())
    # encircled power within 3 um of the axis at the focal plane: 2D field at z = f0
    Uk = plane(np.fft.ifftshift(aperture * np.exp(1j * TAU / LAM0 * (N0 - 1) * hk)), LAM0, F0)
    Ur = plane(np.fft.ifftshift(aperture * np.exp(1j * TAU / LAM0 * (N0 - 1) * hk_r)), LAM0, F0)
    Rs = np.fft.ifftshift(R)
    rin = 1.5
    pow_k = float((np.abs(Uk) ** 2)[Rs < rin].sum() / (aperture ** 2).sum())
    pow_r = float((np.abs(Ur) ** 2)[Rs < rin].sum() / (aperture ** 2).sum())
    # 1D cut of the rounded vs sharp profile (for drawing), r in [0, a]
    rr = np.linspace(0, A_LENS, 4001)
    hk_line = h_kino(rr)
    hr_line = np.interp(rr, x[N // 2:], hk_r[N // 2, N // 2:])
    # top view of the kinoform (shaded relief), 420 x 420 over [-a-1, a+1]
    tv = np.linspace(-A_LENS - 1, A_LENS + 1, 420)
    TX, TY = np.meshgrid(tv, tv)
    TR = np.hypot(TX, TY)
    th = h_kino(TR)
    gy, gx = np.gradient(th, tv[1] - tv[0])
    shade = np.clip(0.55 + 0.45 * (-gx * 0.6 + gy * 0.6) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    base = 0.25 + 0.6 * th / H2PI
    v = np.clip(base * (0.55 + 0.6 * shade), 0, 1)
    top = np.zeros(TR.shape + (4,), np.uint8)
    top[..., :3] = (np.stack([v * 0.93, v * 0.95, v], -1) * 255).astype(np.uint8)
    top[..., 3] = (255 * np.clip((A_LENS + 0.3 - TR) / 0.3, 0, 1)).astype(np.uint8)
    np.savez_compressed(os.path.join(DATA, "lens.npz"), lam0=LAM0, n0=N0, a=A_LENS, f0=F0, h2pi=H2PI, sag=SAG,
                        nzone=NZONE, edges=zone_edges(), xs=x[cols], zs=zs, I_t=I_t.astype(np.float32), rgb_g=rgb_g,
                        I_k=I_k.astype(np.float32), rgb_t=rgb_t, rgb_k=rgb_k, lams=lams, prof_t=prof_t, prof_k=prof_k,
                        I_r=I_r.astype(np.float32), peak_ratio=peak_ratio, pow_k=pow_k, pow_r=pow_r, sig=sig, rr=rr,
                        hk_line=hk_line, hr_line=hr_line, top=top, **ax)
    print(f"[lens] engine {eng}; f0 = {F0:.3f} um, NA = {A_LENS / np.hypot(A_LENS, F0):.3f}, sag = {SAG:.2f} um, "
          f"H2pi = {H2PI:.3f} um; max|I_t - I_k| / max = {np.abs(I_t - I_k).max() / I_t.max():.2e}; "
          f"rounded peak ratio {peak_ratio:.3f}, encircled power {pow_k:.3f} -> {pow_r:.3f} "
          f"({time.time() - t0:.1f}s)")
    for nm, l in (("B", 0.45), ("G", 0.55), ("R", 0.65)):
        zk = zs[np.argmax(ax['ax_k_' + nm])]
        zt = zs[np.argmax(ax['ax_t_' + nm])]
        print(f"   {nm} {l * 1000:.0f} nm: kinoform focus {zk:.2f} um (f0 lam0/lam = {F0 * LAM0 / l:.2f}); "
              f"glass focus {zt:.2f} um (f0 (n0-1)/(n-1) = {F0 * (N0 - 1) / (n_glass(l) - 1):.2f})")
    nF, nC, nd = n_glass(0.4861), n_glass(0.6563), n_glass(0.5876)
    print(f"   Abbe numbers: glass {(nd - 1) / (nF - nC):.1f}, diffractive {0.5876 / (0.4861 - 0.6563):.2f}")


# ------------------------------------------------------------------ Ch1: wavelength-scale demo lens ---------------
def tea_demo():
    """Cylindrical demo lens in units of lambda: incident wave below, TEA columns inside, ASM above (complex field)."""
    lam, n = 1.0, 1.5
    k0 = TAU / lam
    a, f = 9.0, 14.0
    hmax = (np.sqrt(a ** 2 + f ** 2) - f) / (n - 1)
    h = lambda x: np.clip(hmax - (np.sqrt(x ** 2 + f ** 2) - f) / (n - 1), 0, None) * (np.abs(x) <= a)
    beam = lambda x: np.clip((a + 0.6 - np.abs(x)) / 1.2, 0, 1) ** 2 * (3 - 2 * np.clip((a + 0.6 - np.abs(x)) / 1.2, 0, 1))
    xw, z0, z1 = 11.0, -3.0, 22.0
    nxd, nzd = 440, 500                                              # display grid (20 px per lambda)
    xs = -xw + (np.arange(nxd) + 0.5) * 2 * xw / nxd
    zs = z0 + (np.arange(nzd) + 0.5) * (z1 - z0) / nzd
    U = np.zeros((nzd, nxd), complex)
    A = beam(xs)[None, :]
    Z = zs[:, None]
    hx = h(xs)[None, :]
    below = Z < 0
    U = np.where(below, A * np.exp(1j * k0 * Z), U)
    inl = (Z >= 0) & (Z <= hmax)
    U = np.where(inl & (Z < hx), A * np.exp(1j * k0 * n * Z), U)
    U = np.where(inl & (Z >= hx), A * np.exp(1j * (k0 * n * hx + k0 * (Z - hx))), U)
    # ASM above the exit plane z = hmax (fine 1D grid, padded)
    Nf, dxf = 8192, 0.05
    xf = (np.arange(Nf) - Nf // 2) * dxf
    u0 = beam(xf) * np.exp(1j * (k0 * hmax + k0 * (n - 1) * h(xf)))
    kx = TAU * np.fft.fftfreq(Nf, d=dxf)
    kz = np.sqrt((k0 ** 2 - kx ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    F = np.fft.fft(u0)
    above = np.where(zs > hmax)[0]
    for j in above:
        row = np.fft.ifft(F * np.exp(1j * kz * (zs[j] - hmax)))
        U[j] = np.interp(xs, xf, row.real) + 1j * np.interp(xs, xf, row.imag)
    np.savez_compressed(os.path.join(DATA, "tea_demo.npz"), U=U.astype(np.complex64), xs=xs, zs=zs, a=a, f=f, n=n,
                        hmax=hmax, hx=h(xs))
    zf = zs[above][np.argmax(np.abs(U[above, nxd // 2]))]
    print(f"[tea_demo] hmax = {hmax:.2f} lam, focus at z = {zf:.2f} (exit plane + f = {hmax + f:.2f}); "
          f"max|U| = {np.abs(U).max():.2f}")


# ------------------------------------------------------------------ Ch3: quantised blaze ---------------------------
def steps():
    M = 4096
    u = (np.arange(M) + 0.5) / M
    out = {}
    levels = [0, 2, 4, 8, 16]                                        # 0 = ideal ramp
    mm = np.arange(-9, 10)
    for N in levels:
        ph = TAU * (u if N == 0 else np.floor(u * N) / N)
        c = np.fft.fft(np.exp(1j * ph)) / M
        eta = np.abs(c[mm % M]) ** 2
        out[f"eta_{N}"] = eta
        # phasor chain: 64 arrows (difference to the ideal ramp), head to tail, total length 1
        ns = 64
        us = (np.arange(ns) + 0.5) / ns
        phs = TAU * (us if N == 0 else np.floor(us * N) / N)
        arr = np.exp(1j * (phs - TAU * us)) / ns
        out[f"chain_{N}"] = np.concatenate([[0], np.cumsum(arr)])
        out[f"prof_{N}"] = (us if N == 0 else np.floor(us * N) / N)
    out["m"] = mm
    for N in levels[1:]:
        print(f"[steps] N={N:2d}: eta(+1) = {out[f'eta_{N}'][mm == 1][0]:.4f} (sinc^2(1/N) = {np.sinc(1 / N) ** 2:.4f}), "
              f"|chain| = {abs(out[f'chain_{N}'][-1]):.4f}")
    np.savez_compressed(os.path.join(DATA, "steps.npz"), **out)


# ------------------------------------------------------------------ Ch5: 2D FDFD full-wave solver ------------------
def fill_eps(nx, nzr, dx, glass_fn, n_hi, n_lo=1.0, sub=8):
    """Cell-averaged permittivity on the nodes (node (i, j) at x = i dx, z = j dx), by sub x sub supersampling."""
    s = (np.arange(sub) + 0.5) / sub - 0.5
    xs = ((np.arange(nx)[:, None] + s[None, :]) * dx).ravel()
    zs = ((np.arange(nzr)[:, None] + s[None, :]) * dx).ravel()
    g = glass_fn(xs[None, :], zs[:, None])
    fill = g.reshape(nzr, sub, nx, sub).mean(axis=(1, 3))
    return n_lo ** 2 + (n_hi ** 2 - n_lo ** 2) * fill


def fdfd_solve(eps, dx, k0, pml, src_row, pol="TE", sigma_max=6.0, p=3):
    """Solve d/dx(a d/dx u) + (1/s) d/dz(a/s d/dz u) + k0^2 b u = -J  (TE: u = E_y, a = 1, b = eps;
    TM: u = H_y, a = 1/eps, b = 1). Periodic in x, stretched-coordinate PML (s = 1 + i sigma) at both z ends."""
    nzr, nx = eps.shape
    Nn = nzr * nx

    def sig(zc):
        lo, hi = pml, nzr - 1 - pml
        d = np.where(zc < lo, (lo - zc) / pml, np.where(zc > hi, (zc - hi) / pml, 0.0))
        return sigma_max * d ** p
    j = np.arange(nzr, dtype=float)
    s0 = (1 + 1j * sig(j))[:, None]
    sp_ = (1 + 1j * sig(j + 0.5))[:, None]
    sm_ = (1 + 1j * sig(j - 0.5))[:, None]
    if pol == "TE":
        axp = axm = azp = azm = np.ones((nzr, nx))
        kk = k0 ** 2 * eps
    else:
        ie = 1 / eps
        axp, axm = 0.5 * (ie + np.roll(ie, -1, 1)), 0.5 * (ie + np.roll(ie, 1, 1))
        azp = 0.5 * (ie + np.vstack([ie[1:], ie[-1:]]))
        azm = 0.5 * (ie + np.vstack([ie[:1], ie[:-1]]))
        kk = np.full((nzr, nx), k0 ** 2)
    idx = np.arange(Nn).reshape(nzr, nx)
    i2 = 1 / dx ** 2
    cxp, cxm = axp * i2, axm * i2
    czp, czm = azp * i2 / (s0 * sp_), azm * i2 / (s0 * sm_)
    diag = -(cxp + cxm + czp + czm) + kk
    R = [idx.ravel(), idx.ravel(), idx.ravel(), idx[:-1].ravel(), idx[1:].ravel()]
    C = [idx.ravel(), np.roll(idx, -1, 1).ravel(), np.roll(idx, 1, 1).ravel(), idx[1:].ravel(), idx[:-1].ravel()]
    V = [diag.ravel(), cxp.ravel(), cxm.ravel(), czp[:-1].ravel(), czm[1:].ravel()]
    Am = sp.csc_matrix((np.concatenate(V).astype(complex), (np.concatenate(R), np.concatenate(C))), shape=(Nn, Nn))
    b = np.zeros(Nn, complex)
    b[idx[src_row]] = -1.0 / dx
    return spla.spsolve(Am, b).reshape(nzr, nx)


def order_flux(E, j, w=1.0):
    """Exactly conserved discrete flux through the half row j + 1/2, split into Fourier orders."""
    a1 = np.fft.fft(E[j]) / E.shape[1]
    a2 = np.fft.fft(E[j + 1]) / E.shape[1]
    return w * np.imag(np.conj(a1) * a2)


def periodic_cell(period, glass_fn, n_hi, H_rel, pol="TE", ppl=40, lam=1.0, n_in=1.5, n_out=1.0, sub_l=1.0,
                  air_l=3.0, pml_l=1.5, keep=False):
    """Plane wave from the substrate (index n_in, below) through a periodic layer of height H_rel (relief between
    z = zb and zb + H_rel) into the cover (n_out). glass_fn(x, z_rel) -> True where the high index n_hi sits
    (z_rel = z - zb; the substrate z_rel < 0 is filled by the caller's glass_fn)."""
    dx = lam / ppl
    k0 = TAU / lam
    nx = int(round(period * ppl))
    per = nx * dx
    pml = int(round(pml_l * ppl))
    below = 20
    n_sub, n_rel, n_air = int(round(sub_l * ppl)), int(round(H_rel / dx)), int(round(air_l * ppl))
    nzr = pml + below + 1 + n_sub + n_rel + n_air + pml
    js = pml + below
    zb = (js + 1 + n_sub) * dx
    eps = fill_eps(nx, nzr, dx, lambda x, z: glass_fn(np.mod(x, per), z - zb), n_hi, n_out)
    eps[: js + 1 + n_sub] = np.where(eps[: js + 1 + n_sub] > 0, n_in ** 2, eps[: js + 1 + n_sub])
    E = fdfd_solve(eps, dx, k0, pml, js, pol)
    Er = fdfd_solve(np.full_like(eps, n_in ** 2), dx, k0, pml, js, pol)
    jt, jr = nzr - pml - 10, js + 5
    w_in = 1.0 if pol == "TE" else 1 / n_in ** 2
    w_out = 1.0 if pol == "TE" else 1 / n_out ** 2
    Finc = order_flux(Er, jr, w_in).sum()
    Ft = order_flux(E, jt, w_out) / Finc
    Rr = -order_flux(E - Er, jr, w_in).sum() / Finc
    mm = np.fft.fftfreq(nx, 1 / nx).astype(int)
    res = dict(period=per, T=float(Ft.sum()), R=float(Rr), eta={int(m): float(Ft[i]) for i, m in enumerate(mm)})
    jb = int(round(zb / dx))
    # complex 0th-order transmission (phase reference: plane wave e^{i k (z - zb)} continued from the substrate)
    a0 = np.fft.fft(E[jt])[0] / nx
    res["t0"] = a0 / Er[jb, 0] * np.exp(-1j * k0 * n_out * (jt - jb) * dx)
    if keep:
        res.update(E=E / Er[jb, 0], eps=eps, dx=dx, js=js, jb=jb, pml=pml, nzr=nzr, nx=nx)
    return res


def fullwave():
    t0 = time.time()
    n = 1.5
    H = 1 / (n - 1)                                                  # 2 lambda: one full turn of delay
    blaze = lambda per: (lambda x, z: (z < 0) | (z < H * x / per))
    out = {}
    # --- field picture: 4-lambda zones, TE (E along the grooves), 2 periods shown
    P = 4.0
    r = periodic_cell(P, blaze(P), n, H, "TE", air_l=4.6, keep=True)
    lo, hi = r["js"] + 6, r["nzr"] - r["pml"]
    E = r["E"][lo:hi]
    eps = r["eps"][lo:hi]
    dx = r["dx"]
    z = (np.arange(lo, hi) - r["jb"]) * dx
    x = np.arange(r["nx"]) * dx
    # TEA column picture on the same grid
    k0 = TAU
    Z, X = np.meshgrid(z, x, indexing="ij")
    hx = H * np.mod(X, r["period"]) / r["period"]
    U = np.exp(1j * k0 * n * Z)
    U = np.where((Z >= 0) & (Z >= hx) & (Z <= H), np.exp(1j * (k0 * n * hx + k0 * (Z - hx))), U)
    kxx = TAU / r["period"]
    U = np.where(Z > H, np.exp(1j * (k0 * H + kxx * X + np.sqrt(k0 ** 2 - kxx ** 2) * (Z - H))), U)
    out.update(E_fw=E.astype(np.complex64), E_tea=U.astype(np.complex64), eps=eps.astype(np.float32), fx=x, fz=z,
               period=r["period"], H=H, eta_fw_TE=r["eta"].get(1, 0.0), T_TE=r["T"], R_TE=r["R"])
    rm = periodic_cell(P, blaze(P), n, H, "TM")
    out.update(eta_fw_TM=rm["eta"].get(1, 0.0), T_TM=rm["T"], R_TM=rm["R"])
    orders_TE = np.array([r["eta"].get(m, 0.0) for m in range(-3, 4)])
    orders_TM = np.array([rm["eta"].get(m, 0.0) for m in range(-3, 4)])
    out.update(orders_m=np.arange(-3, 4), orders_TE=orders_TE, orders_TM=orders_TM)
    print(f"[fullwave] 4-lambda blaze: TE eta+1 {r['eta'][1]:.4f} (T {r['T']:.4f}, R {r['R']:.4f}, "
          f"T+R {r['T'] + r['R']:.6f});  TM eta+1 {rm['eta'][1]:.4f} (T {rm['T']:.4f}, R {rm['R']:.4f})")
    # --- efficiency vs zone width
    periods = np.array([2.4, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0])
    eT, eM = [], []
    for per in periods:
        eT.append(periodic_cell(per, blaze(per), n, H, "TE", ppl=32 if per > 9 else 40)["eta"][1])
        eM.append(periodic_cell(per, blaze(per), n, H, "TM", ppl=32 if per > 9 else 40)["eta"][1])
    out.update(sweep_p=periods, sweep_TE=np.array(eT), sweep_TM=np.array(eM))
    print("[fullwave] sweep zone width/lambda:", " ".join(f"{p:g}:{a:.3f}/{b:.3f}" for p, a, b in zip(periods, eT, eM)))
    # --- metasurface ridge library (TE): TiO2-like ridges (n = 2.4), period 0.6 lambda, height 1.2 lambda, on glass
    pm, Hm, nm_ = 0.6, 1.2, 2.4
    ws = np.linspace(0.06, 0.54, 25)
    t0s, T0s = [], []
    for w in ws:
        g = (lambda w_: (lambda x, z: (z < 0) | ((z < Hm) & (np.abs(x - pm / 2) < w_ / 2))))(w)
        rr = periodic_cell(pm, g, nm_, Hm, "TE", ppl=60, air_l=2.0)
        t0s.append(rr["t0"])
        T0s.append(rr["eta"][0])                                     # flux transmission into the 0th order
    t0s = np.array(t0s)
    ph = np.unwrap(np.angle(t0s))
    ph -= ph[0]
    out.update(ms_w=ws, ms_phase=ph, ms_T=np.array(T0s), ms_period=pm, ms_H=Hm, ms_n=nm_)
    print(f"[fullwave] metasurface library: phase span {ph.max() - ph.min():.2f} rad ({(ph.max() - ph.min()) / TAU:.2f} turns), "
          f"T0 in [{min(T0s):.2f}, {max(T0s):.2f}]  ({time.time() - t0:.1f}s)")
    np.savez_compressed(os.path.join(DATA, "fullwave.npz"), **out)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["lens", "tea", "steps", "fullwave"]
    if "lens" in which:
        lens_fields()
    if "tea" in which:
        tea_demo()
    if "steps" in which:
        steps()
    if "fullwave" in which:
        fullwave()
