"""Precompute every physical field shown in E02 -> data/*.npz (run once; scenes only load).

    CUDA_VISIBLE_DEVICES=<free gpu> python physics.py [part ...]
    parts: hook zoom fresnel lens fourf zernike chirp   (default: all)

Units: wavelength lambda = 1, k = 2 pi.  Conventions: plane wave e^{i(kx x + kz z - w t)}; a delay adds +phase,
so a thin lens multiplies by e^{-i k x^2 / 2f}; spectrum U^(kx) = int U(x) e^{-i kx x} dx.

What is computed how (README has the summary):
  hook     2D letter-F slide. Far wall: single-FFT Fresnel propagation to Z = 4e5 lambda (Fresnel number ~0.004).
           Lens: thin-lens phase, then 2D band-limited angular-spectrum (ASM) propagation by f.
  zoom     1D slit (D = 12 lambda): exact band-limited ASM (Matsushima & Shimobaba 2009) on a 2^21-point grid,
           evaluated over 64 zoom levels from z <= 150 lambda to z <= 1.5e5 lambda (JAX on the GPU).
  fresnel  1D slit: the Fresnel diffraction integral, summed directly, for 240 distances (pattern vs kx = k x / z).
  lens     1D x-z fields behind a thin lens (paraxial phase, f = 40, aperture 24): plane waves at several tilts and
           a 6-lambda slit, propagated with exact ASM.
  fourf    2D 4f system on a letter F: FFT -> mask -> FFT (the second forward FFT gives the inverted image).
  zernike  2D weak phase object ("cell"); brightfield and Zernike phase contrast (DC phase-shifted by pi/2).
  chirp    sampled chirps (ASM transfer function in kx, Fresnel kernel in x) and their band-limited
           reconstructions from the samples, for a sweep of z through z_c = N dx^2 / lambda.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi
K = TAU                                   # lambda = 1


def save(name, **arrs):
    p = os.path.join(DATA, name)
    np.savez_compressed(p, **arrs)
    print(f"  wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)", flush=True)


def jx():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


# ---------------------------------------------------------------- shapes ------
def rect_cover(x, a, b, dx):
    """Fraction of each pixel [x - dx/2, x + dx/2] covered by [a, b] (anti-aliased edges)."""
    return np.clip((np.minimum(b, x + dx / 2) - np.maximum(a, x - dx / 2)) / dx, 0, 1)


def letter_F(n, dx, H=40.0, W=28.0, s=8.0, mid=21.0):
    """Amplitude transmission of a letter F (bounding box centred on the grid), rows = y (index 0 at the bottom)."""
    x = (np.arange(n) - n // 2) * dx
    x0, y0 = -W / 2, -H / 2
    ym = y0 + 0.5 * H + 0.5                 # middle arm a touch above half height
    parts = [((x0, x0 + s), (y0, y0 + H)),                  # stem
             ((x0 + s, x0 + W), (y0 + H - s, y0 + H)),      # top arm
             ((x0 + s, x0 + mid), (ym - s / 2, ym + s / 2))]  # middle arm
    U = np.zeros((n, n))
    for (xa, xb), (ya, yb) in parts:
        U += rect_cover(x, ya, yb, dx)[:, None] * rect_cover(x, xa, xb, dx)[None, :]
    return np.clip(U, 0, 1)


def box_resample(img, out):
    """Area-average (box filter) a 2D float image down to out x out: keeps fine fringes from aliasing."""
    from PIL import Image
    return np.asarray(Image.fromarray(img.astype(np.float32)).resize((out, out), Image.BOX))


# ---------------------------------------------------------------- hook --------
def part_hook():
    """Letter F (80 lambda tall): far wall (single-FFT Fresnel) and lens focal plane (ASM through a thin lens)."""
    jax, jnp = jx()
    th = 0.30                                  # half-width of the displayed angular range (rad)
    out = 600
    N, dx = 8192, 0.5
    Fk = dict(H=80.0, W=56.0, s=16.0, mid=42.0)
    U = letter_F(N, dx, **Fk)
    x = (np.arange(N) - N // 2) * dx
    X2 = x[None, :] ** 2 + x[:, None] ** 2
    # far wall: single-FFT Fresnel to Z (Fresnel number H^2 / Z = 0.0064), output pitch dx' = lambda Z / (N dx)
    Z = 1.0e6
    Ui = jnp.asarray(U * np.exp(1j * K * X2 / (2 * Z)))
    Uf = np.asarray(jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(Ui)))) ** 2)
    dxp = Z / (N * dx)
    nc = int(round(th * Z / dxp))
    far = box_resample(Uf[N // 2 - nc:N // 2 + nc, N // 2 - nc:N // 2 + nc], out)
    del Ui, Uf
    # lens right behind the slide: thin-lens phase, then band-limited ASM by f
    f = 4000.0
    L = N * dx
    U0 = jnp.asarray(U * np.exp(-1j * K * X2 / (2 * f)))
    kx = TAU * np.fft.fftfreq(N, d=dx)
    KX2 = kx[None, :] ** 2 + kx[:, None] ** 2
    kz = np.sqrt(np.maximum(K ** 2 - KX2, 0.0))
    ok = np.sqrt(KX2) <= kz * L / (2 * f)      # Matsushima band limit (also drops evanescent waves)
    Hf = jnp.asarray(np.exp(1j * kz * f) * ok)
    del KX2, kz, ok
    Ul = np.asarray(jnp.abs(jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.fft2(jnp.fft.ifftshift(U0)) * Hf))) ** 2)
    ncl = int(round(th * f / dx))
    lens = box_resample(Ul[N // 2 - ncl:N // 2 + ncl, N // 2 - ncl:N // 2 + ncl], out)
    no = int(round(56 / dx))                   # the slide itself (display crop +-56 lambda)
    obj = U[N // 2 - no:N // 2 + no, N // 2 - no:N // 2 + no]
    far, lens = far / far.max(), lens / lens.max()
    print(f"  hook: far vs lens max |diff| {np.abs(far - lens).max():.3f}, mean {np.abs(far - lens).mean():.4f}")
    save("hook.npz", far=far.astype(np.float32), lens=lens.astype(np.float32), obj=obj.astype(np.float32),
         th=th, Z=Z, f=f)


# ---------------------------------------------------------------- zoom --------
def part_zoom():
    """Slit D=12: weighted |U|^2 maps for a log zoom of the viewing window, z in [0, Zl], x in [-rho Zl, rho Zl]."""
    jax, jnp = jx()
    D, dx, N = 12.0, 0.25, 2 ** 21
    L = N * dx
    zF = D ** 2                                  # D^2 / lambda
    rho = 0.23                                   # window half-height / length
    Z0, Z1, nlev = 150.0, 1.5e5, 64
    nz, nx = 512, 256
    x = (np.arange(N) - N // 2) * dx
    u0 = rect_cover(x, -D / 2, D / 2, dx)
    F0 = jnp.asarray(np.fft.fft(np.fft.ifftshift(u0)))
    kx = TAU * np.fft.fftfreq(N, d=dx)
    kzr = np.sqrt(np.maximum(K ** 2 - kx ** 2, 0.0))
    kzi = np.sqrt(np.maximum(kx ** 2 - K ** 2, 0.0))
    kxa = jnp.asarray(np.abs(kx))
    kzr_j, kzi_j = jnp.asarray(kzr), jnp.asarray(kzi)
    prop = jnp.asarray(kx ** 2 < K ** 2)
    xg = jnp.asarray(x)

    @jax.jit
    def rows(zs, edges):
        """zs (B,), edges (nx+1,) -> bin-averaged |U|^2 (B, nx) and the full row at the last z."""
        def one(z):
            ok = jnp.where(prop, kxa <= kzr_j * L / (2 * z), True)
            H = jnp.where(prop, jnp.exp(1j * kzr_j * z), jnp.exp(-kzi_j * z)) * ok
            I = jnp.abs(jnp.fft.fftshift(jnp.fft.ifft(F0 * H))) ** 2
            c = jnp.concatenate([jnp.zeros(1), jnp.cumsum(I) * dx])      # cumulative power at x_n - dx/2
            xe = jnp.concatenate([xg - dx / 2, xg[-1:] + dx / 2])
            ce = jnp.interp(edges, xe, c)
            return (ce[1:] - ce[:-1]) / (edges[1:] - edges[:-1])
        return jax.vmap(one)(zs)

    Zl = Z0 * (Z1 / Z0) ** (np.arange(nlev) / (nlev - 1))
    maps = np.zeros((nlev, nx, nz), np.float32)
    t0 = time.time()
    B = 32
    for li, Zt in enumerate(Zl):
        zs = Zt * (np.arange(nz) + 0.5) / nz
        X = rho * Zt
        edges = jnp.asarray(np.linspace(-X, X, nx + 1))
        out = np.concatenate([np.asarray(rows(jnp.asarray(zs[i:i + B]), edges)) for i in range(0, nz, B)])
        w = 1.0 + zs / zF                          # undo the 1/z far-field fall-off (2D: |U|^2 ~ D^2 / (lambda z))
        maps[li] = (out * w[:, None]).T[::-1]      # rows: +x at the top; columns: z
    print(f"  zoom: {nlev} levels in {time.time() - t0:.1f}s")
    # far wall profile at Z1 (fine) and the slit's spectrum at k sin(theta)
    xw = np.linspace(-rho * Z1, rho * Z1, 1201)
    ok = np.where(kx ** 2 < K ** 2, np.abs(kx) <= kzr * L / (2 * Z1), True)
    H = np.where(kx ** 2 < K ** 2, np.exp(1j * kzr * Z1), np.exp(-kzi * Z1)) * ok
    Uw = np.fft.fftshift(np.fft.ifft(np.asarray(F0) * H))
    Aw = np.interp(xw, x, np.abs(Uw))
    sin_t = xw / np.sqrt(xw ** 2 + Z1 ** 2)
    spec = np.abs(D * np.sinc(K * sin_t * D / 2 / np.pi))          # |FT of the slit| at kx = k sin(theta)
    kq = np.linspace(-0.25 * K, 0.25 * K, 1201)                      # spectrum vs kx (for the strip under the map)
    Fs = np.fft.fftshift(np.abs(np.asarray(F0)) * dx)
    kxs = np.fft.fftshift(kx)
    spec_k = np.interp(kq, kxs, Fs)
    print(f"  zoom: wall profile vs spectrum, max |diff| {np.abs(Aw / Aw.max() - spec / spec.max()).max():.4f}")
    save("zoom.npz", maps=maps.astype(np.float16), Zl=Zl, rho=rho, D=D, xw=xw, Aw=Aw / Aw.max(), spec=spec / spec.max(),
         kq=kq, spec_k=spec_k / spec_k.max(), Z1=Z1)


# ---------------------------------------------------------------- fresnel -----
def part_fresnel():
    """Fresnel integral of a slit, I(x) = int_{-D/2}^{D/2} e^{ik(x-x')^2/2z} dx' (= sqrt(lambda z) |U_wall|), vs kx = k x / z.
    Far away |I| -> |U^(kx)|, so the curves are directly comparable with the spectrum."""
    D = 12.0
    zF = D ** 2
    zs = zF * np.logspace(np.log10(0.03), np.log10(40.0), 240)
    kq = np.linspace(-0.25 * K, 0.25 * K, 601)
    xp = np.linspace(-D / 2, D / 2, 4001)
    wq = np.full_like(xp, xp[1] - xp[0])
    wq[[0, -1]] *= 0.5
    pat = np.zeros((len(zs), len(kq)))
    for i, z in enumerate(zs):
        x = kq * z / K
        Ph = np.exp(1j * K * (x[:, None] - xp[None, :]) ** 2 / (2 * z))
        pat[i] = np.abs(Ph @ wq)
    spec = np.abs(D * np.sinc(kq * D / 2 / np.pi))
    print(f"  fresnel: last-z pattern vs spectrum max |diff| {np.abs(pat[-1] / pat[-1].max() - spec / spec.max()).max():.3f}")
    save("fresnel.npz", zs=zs, kq=kq, pat=pat / spec.max(), spec=spec / spec.max(), D=D)


# ---------------------------------------------------------------- lens --------
def part_lens():
    """x-z fields behind a thin lens (paraxial phase) for tilted plane waves and for a slit, exact ASM."""
    f, DL = 40.0, 24.0
    N, dx = 4096, 1 / 8
    x = (np.arange(N) - N // 2) * dx
    kx = TAU * np.fft.fftfreq(N, d=dx)
    kz = np.sqrt((K ** 2 - kx ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    ap = 0.5 * (1 - np.tanh((np.abs(x) - DL / 2) / 0.35))
    lens = np.exp(-1j * K * x ** 2 / (2 * f))
    zpre, zpost, xh = 10.0, 50.0, 15.0           # display window: z in [-zpre, zpost], x in [-xh, xh]
    ppl = 8                                       # display samples per wavelength
    zs = (np.arange(int((zpre + zpost) * ppl)) + 0.5) / ppl - zpre
    xs = xh - (np.arange(int(2 * xh * ppl)) + 0.5) / ppl      # rows: +x at the top
    sel = np.searchsorted(x, xs[::-1])[::-1]
    zpos = zs[zs > 0]

    def field(u_after, u_before):
        F0 = np.fft.fft(np.fft.ifftshift(u_after))
        out = np.zeros((len(xs), len(zs)), np.complex64)
        for j, z in enumerate(zs):
            if z <= 0:
                out[:, j] = u_before(xs, z)
            else:
                out[:, j] = np.fft.fftshift(np.fft.ifft(F0 * np.exp(1j * kz * z)))[sel]
        return out

    def tilted(deg):
        s = np.sin(np.radians(deg))
        c = np.cos(np.radians(deg))
        inc = lambda xx, z: np.exp(1j * K * (s * xx + c * z))
        return field(ap * lens * np.exp(1j * K * s * x), inc)

    t0 = time.time()
    sweep_deg = np.linspace(0.0, 7.0, 29)
    sweep = np.stack([tilted(d) for d in sweep_deg])
    minus = tilted(-7.0)
    Ds = 6.0
    slit = rect_cover(x, -Ds / 2, Ds / 2, dx)
    sl = field(slit * lens, lambda xx, z: np.ones_like(xx) * np.exp(1j * K * z))
    # focal-plane profile of the slit (fine) and its spectrum mapped to x = f tan(theta)
    F0 = np.fft.fft(np.fft.ifftshift(slit * lens))
    Uf = np.abs(np.fft.fftshift(np.fft.ifft(F0 * np.exp(1j * kz * f))))
    xf = np.linspace(-xh, xh, 721)
    prof = np.interp(xf, x, Uf)
    tan_t = xf / f
    sin_t = tan_t / np.sqrt(1 + tan_t ** 2)
    spec = np.abs(Ds * np.sinc(K * sin_t * Ds / 2 / np.pi))
    print(f"  lens: {len(sweep_deg) + 2} fields in {time.time() - t0:.1f}s; focal profile vs spectrum max |diff| "
          f"{np.abs(prof / prof.max() - spec / spec.max()).max():.3f}")
    # focus positions (argmax of |U| in the focal column) for the tilted waves
    jf = np.argmin(np.abs(zs - f))
    for d, U in ((0.0, sweep[0]), (7.0, sweep[-1]), (-7.0, minus)):
        print(f"    tilt {d:+.0f} deg: focus at x = {xs[np.argmax(np.abs(U[:, jf]))]:+.2f} (f tan = {f * np.tan(np.radians(d)):+.2f})")
    save("lens.npz", sweep=sweep, sweep_deg=sweep_deg, minus=minus, slit=sl, zs=zs, xs=xs, f=f, DL=DL, Ds=Ds,
         xf=xf, prof=prof / prof.max(), spec=spec / spec.max())


# ---------------------------------------------------------------- 4f ----------
def part_fourf():
    """4f system on a letter F: spectrum = FFT(U); image = FFT(mask * FFT(U)) / N^2 = (filtered U)(-x, -y).
    Stored as display crops: object/images +-32 lambda, spectra +-64 frequency samples (+-0.5 cycles/lambda)."""
    N, dx = 512, 0.25
    U = letter_F(N, dx, H=40.0, W=28.0, s=8.0, mid=21.0)
    S = np.fft.fft2(np.fft.ifftshift(U))
    fx = np.fft.fftfreq(N)
    R = np.hypot(fx[None, :], fx[:, None]) * N           # radius in frequency samples
    r_pin, r_stop = PIN_R, STOP_R
    masks = {"open": np.ones_like(R), "pinhole": (R <= r_pin).astype(float), "stop": (R > r_stop).astype(float)}
    c, h = N // 2, int(32 / dx)
    out = {}
    for name, M in masks.items():
        img = np.fft.fftshift(np.fft.fft2(M * S)) / N ** 2
        out["img_" + name] = (np.abs(img[c - h:c + h, c - h:c + h]) ** 2).astype(np.float32)
        out["spec_" + name] = (np.fft.fftshift(np.abs(M * S) ** 2)[c - 64:c + 64, c - 64:c + 64]).astype(np.float32)
    ref = np.roll(U[::-1, ::-1], 1, axis=(0, 1))          # U[(N - m) mod N]: the reversal about index N/2
    full = np.abs(np.fft.fftshift(np.fft.fft2(S)) / N ** 2) ** 2
    print(f"  fourf: open 4f image vs |U(-x,-y)|^2 max err {np.abs(full - ref ** 2).max():.2e}")
    out["obj"] = U[c - h:c + h, c - h:c + h].astype(np.float32)
    save("fourf.npz", **out, N=N, r_pin=r_pin, r_stop=r_stop, spec_half=64)


PIN_R, STOP_R = 10.0, 14.0


# ---------------------------------------------------------------- zernike -----
def part_zernike():
    """Weak phase object: brightfield |e^{i phi}|^2 = 1; phase contrast: DC disc multiplied by e^{i pi/2}."""
    N = 512
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:N, 0:N] - N / 2
    def blob(cx, cy, a, b, ang, edge):
        c, s = np.cos(ang), np.sin(ang)
        u = ((x - cx) * c + (y - cy) * s) / a
        v = (-(x - cx) * s + (y - cy) * c) / b
        r = np.sqrt(u ** 2 + v ** 2)
        return 0.5 * (1 - np.tanh((r - 1) / edge))
    th = np.arctan2(y + 10, x - 5)
    wob = 1 + 0.06 * np.sin(3 * th + 0.7) + 0.04 * np.sin(5 * th + 2.1)     # a slightly irregular outline
    cell = blob(5, -10, 150 * wob, 112 * wob, 0.35, 0.05)
    phi = 0.30 * cell
    phi += 0.22 * blob(-35, 5, 44, 38, 0.2, 0.08)                            # nucleus
    phi += 0.10 * blob(-35, 5, 14, 12, 0.0, 0.15)                            # nucleolus
    for _ in range(14):                                                      # organelles inside the cell
        while True:
            cx, cy = rng.uniform(-130, 140), rng.uniform(-110, 90)
            if blob(5, -10, 135, 100, 0.35, 0.05)[int(cy + N / 2), int(cx + N / 2)] > 0.9 and np.hypot(cx + 35, cy - 5) > 60:
                break
        r = rng.uniform(5, 11)
        phi += 0.12 * blob(cx, cy, r * 1.4, r, rng.uniform(0, np.pi), 0.2)
    phi = phi.astype(float)
    U = np.exp(1j * phi)
    S = np.fft.fft2(U)
    fx = np.fft.fftfreq(N)
    R = np.hypot(fx[None, :], fx[:, None]) * N
    plate = np.where(R < 0.5, np.exp(1j * np.pi / 2), 1.0)       # plane-wave illumination: DC is one point
    bright = np.abs(np.fft.ifft2(S)) ** 2
    pc = np.abs(np.fft.ifft2(S * plate)) ** 2
    lin = 1 + 2 * (phi - phi.mean())
    print(f"  zernike: phi in [{phi.min():.2f}, {phi.max():.2f}] rad; brightfield range {bright.min():.4f}-{bright.max():.4f}; "
          f"phase-contrast range {pc.min():.2f}-{pc.max():.2f}; corr with 1+2phi {np.corrcoef(pc.ravel(), lin.ravel())[0, 1]:.4f}")
    save("zernike.npz", phi=phi.astype(np.float32), bright=bright.astype(np.float32), pc=pc.astype(np.float32),
         spec=np.fft.fftshift(np.abs(S) ** 2).astype(np.float32))


# ---------------------------------------------------------------- chirp -------
def part_chirp():
    """Sampled chirps through z_c = N dx^2 / lambda.  ASM transfer function e^{i z sqrt(k^2 - kx^2)} on the kx grid
    and the Fresnel kernel e^{i k x^2 / 2z} on the x grid (N = 32, dx = 2 lambda -> z_c = 128 lambda), plus the
    band-limited (trigonometric) reconstruction of each set of samples, which is what the FFT 'sees'."""
    N, dx = 32, 2.0
    zc = N * dx ** 2
    zs = zc * np.logspace(np.log10(0.35), np.log10(2.8), 300)
    m = np.arange(N) - N // 2
    kx = TAU * m / (N * dx)                     # kx grid (centred)
    xg = m * dx                                 # x grid (centred)
    up = 24
    t = (np.arange(N * up) - N * up // 2) / up  # fine sample index (continuous curve)

    def recon(samples):
        """Trigonometric interpolation of N periodic samples onto the fine index grid t."""
        c = np.fft.fft(np.fft.ifftshift(samples))
        F = np.zeros(N * up, complex)
        F[:N // 2] = c[:N // 2]
        F[-N // 2:] = c[-N // 2:]
        return np.fft.fftshift(np.fft.ifft(F)) * up

    asm_s, asm_c, asm_r, fr_s, fr_c, fr_r, asm_ok, fr_ok = [], [], [], [], [], [], [], []
    for z in zs:
        ka = TAU * t / (N * dx)
        # the constant phase e^{ikz} is dropped for display: it does not change the sampling, but it would spin the
        # whole curve by k dz per frame during the z sweep (strobing)
        Hs = np.exp(1j * z * (np.sqrt(K ** 2 - kx ** 2) - K))
        Hc = np.exp(1j * z * (np.sqrt(np.maximum(K ** 2 - ka ** 2, 0)) - K))
        asm_s.append(Hs.real)
        asm_c.append(Hc.real)
        asm_r.append(recon(Hs).real)
        ph = z * np.sqrt(K ** 2 - kx ** 2)
        asm_ok.append(np.abs(np.diff(ph)).max() <= np.pi)
        hs = np.exp(1j * K * xg ** 2 / (2 * z))
        hc = np.exp(1j * K * (t * dx) ** 2 / (2 * z))
        fr_s.append(hs.real)
        fr_c.append(hc.real)
        fr_r.append(recon(hs).real)
        fr_ok.append(np.abs(np.diff(K * xg ** 2 / (2 * z))).max() <= np.pi)
    asm_ok, fr_ok = np.array(asm_ok), np.array(fr_ok)
    print(f"  chirp: z_c = {zc}; ASM sampled up to z = {zs[asm_ok].max():.1f}; Fresnel sampled from z = {zs[fr_ok].min():.1f}")
    save("chirp.npz", zs=zs, zc=zc, N=N, dx=dx, t=t, m=m, asm_s=np.array(asm_s), asm_c=np.array(asm_c),
         asm_r=np.array(asm_r), fr_s=np.array(fr_s), fr_c=np.array(fr_c), fr_r=np.array(fr_r), asm_ok=asm_ok, fr_ok=fr_ok)


PARTS = {"hook": part_hook, "zoom": part_zoom, "fresnel": part_fresnel, "lens": part_lens, "fourf": part_fourf,
         "zernike": part_zernike, "chirp": part_chirp}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(PARTS)
    for name in todo:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        PARTS[name]()
        print(f"[{name}] {time.time() - t0:.1f}s", flush=True)
