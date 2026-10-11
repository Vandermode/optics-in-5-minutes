"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).
    python physics.py

E04 Coherence. All random quantities use fixed seeds; single-threaded numpy, a few seconds in total.
  hook.npz      laser speckle spot vs LED (incoherent) spot; double-slit fringes: laser / LED / LED + pinhole
  jitter.npz    two-wave fringes averaged over Gaussian phase jitter (Monte Carlo) -> visibility vs jitter
  temporal.npz  two-beam interferograms vs path difference for Gaussian spectra (envelope = |FT of spectrum|)
  spatial.npz   van Cittert-Zernike: visibility vs slit separation for a uniform strip source and a uniform disk
                (Betelgeuse, Michelson & Pease 1920)
  speckle.npz   far-field speckle of 40 random scatterers; histogram; averaging N independent patterns
  modes.npz     mutual intensity J(x1,x2) of a slit lit by an extended incoherent source; eigenmodes
  doe.npz       phase-only diffractive element (Gerchberg-Saxton): far field under a laser vs an LED
                (shifted copies for source points, scaled copies for colours)
"""
import os

import numpy as np
from scipy import ndimage, special

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi
f32 = np.float32


def save(name, **kw):
    np.savez_compressed(os.path.join(DATA, name), **kw)
    print(f"  wrote data/{name}")


def smooth_noise(rng, n, sig, complex_=False):
    """Seeded Gaussian noise, low-passed by a Gaussian of sig samples, unit standard deviation."""
    w = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    w = ndimage.gaussian_filter1d(w.real, sig, mode="wrap") + (
        1j * ndimage.gaussian_filter1d(w.imag, sig, mode="wrap") if complex_ else 0)
    return w / np.sqrt(np.mean(np.abs(w) ** 2))


# =============================================================== hook =========
def hook():
    rng = np.random.default_rng(11)
    N = 512
    yy, xx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2].astype(float)
    r2 = xx ** 2 + yy ** 2
    amp = np.exp(-r2 / 100.0 ** 2)                                 # laser spot on the wall (field amplitude)
    u = amp * np.exp(1j * TAU * rng.random((N, N)))               # rough wall: random phase per pixel
    f = np.fft.fftfreq(N)
    FX, FY = np.meshgrid(f, f)
    pupil = (np.hypot(FX, FY) <= 0.09).astype(float)              # camera aperture -> grain ~ 1.22/0.09 = 14 px
    I_laser = np.abs(np.fft.ifft2(np.fft.fft2(u) * pupil)) ** 2   # coherent imaging: |field * h|^2
    h = np.fft.ifft2(pupil)
    otf = np.fft.fft2(np.abs(h) ** 2)
    I_led = np.real(np.fft.ifft2(np.fft.fft2(amp ** 2) * otf))     # incoherent imaging: |amp|^2 * |h|^2
    I_led *= I_laser.sum() / I_led.sum()
    c = slice(N // 2 - 170, N // 2 + 170)
    spot_l, spot_e = I_laser[c, c], I_led[c, c]
    print(f"  hook spot: laser contrast in centre {spot_l[150:190, 150:190].std() / spot_l[150:190, 150:190].mean():.2f}, "
          f"LED {spot_e[150:190, 150:190].std() / spot_e[150:190, 150:190].mean():.3f}")

    # double slit, Fraunhofer: slit width a, separation d; s = sin(theta) on the screen
    lam0, d, a = 0.530, 250.0, 10.0                                # microns
    P = lam0 / d                                                   # fringe period in s
    s = np.linspace(-12 * P, 12 * P, 1600)

    def pattern(s_, lam, alpha):
        q = s_[None, :] - alpha[:, None]
        return np.sinc(a * q / lam) ** 2 * np.cos(np.pi * d * q / lam) ** 2

    def led(half_width):
        lams = lam0 + 0.040 / 2.3548 * np.linspace(-2.5, 2.5, 41)  # Gaussian spectrum, 40 nm FWHM
        wl = np.exp(-0.5 * ((lams - lam0) / (0.040 / 2.3548)) ** 2)
        al = np.linspace(-half_width, half_width, 241)            # uniform source, angular half-width
        out = np.zeros_like(s)
        for lam, w in zip(lams, wl):
            out += w * pattern(s, lam, al).mean(0)
        return out / wl.sum()
    fr_laser = pattern(s, lam0, np.array([0.0]))[0]
    fr_led = led(0.010)                                            # 1 mm LED die 50 mm away
    fr_pin = led(0.0002)                                           # 20 um pinhole 50 mm away
    for nm, v in (("laser", fr_laser), ("LED", fr_led), ("LED+pinhole", fr_pin)):
        m = np.abs(s) < 1.5 * P
        print(f"  hook fringes {nm}: central visibility {(v[m].max() - v[m].min()) / (v[m].max() + v[m].min()):.3f}")
    save("hook.npz", spot_laser=f32(spot_l), spot_led=f32(spot_e), s=f32(s / P), fr_laser=f32(fr_laser),
         fr_led=f32(fr_led), fr_pin=f32(fr_pin))


# =============================================================== Ch1 ==========
def jitter():
    rng = np.random.default_rng(2)
    g = rng.standard_normal(20000)                                 # Monte Carlo samples of the jitter
    sig = np.linspace(0, 3.5, 71)
    gam = np.array([np.mean(np.exp(1j * s_ * g)) for s_ in sig])   # <e^{i sigma g}> (computed, ~ e^{-sigma^2/2})
    x = np.linspace(0, 1, 801)                                     # screen position in units of the panel width
    dphi = TAU * 4.0 * x                                           # 4 fringes across the panel
    I = 2 + 2 * np.real(gam[:, None] * np.exp(1j * dphi[None, :]))  # |a1 + a2 e^{i(dphi + jitter)}|^2, averaged
    V = (I.max(1) - I.min(1)) / (I.max(1) + I.min(1))
    print(f"  jitter: V(0)={V[0]:.3f}, V(1)={np.interp(1, sig, V):.3f} (theory {np.exp(-0.5):.3f}), V(3.5)={V[-1]:.3f}")
    noise = smooth_noise(np.random.default_rng(3), 7200, 1.2)      # live jitter sequence, one value per frame
    save("jitter.npz", sig=f32(sig), gam=np.complex64(gam), x=f32(x), I=f32(I), V=f32(V), noise=f32(noise))


# =============================================================== Ch2 ==========
def temporal():
    lam0 = 0.55                                                    # microns
    dL = np.linspace(-6, 6, 2001)
    dlam = np.geomspace(0.001, 0.35, 90)                           # spectral FWHM (microns)
    nu0 = 1 / lam0
    I, env, lc, spec = [], [], [], []
    lam_ax = np.linspace(0.36, 0.80, 400)
    for dl in dlam:
        dnu = dl / lam0 ** 2                                       # FWHM in wavenumber 1/lambda
        sg = dnu / 2.3548
        nu = nu0 + sg * np.linspace(-4, 4, 801)
        S = np.exp(-0.5 * ((nu - nu0) / sg) ** 2)
        S /= S.sum()
        E = (S[None, :] * np.exp(1j * TAU * nu[None, :] * dL[:, None])).sum(1)   # = FT of the spectrum
        I.append(1 + np.real(E))                                    # two-beam interferogram, normalised
        env.append(np.abs(E))
        inside = dL[np.abs(E) >= np.exp(-1)]
        lc.append(inside.max() - inside.min() if inside.size else 0)      # full 1/e width of the envelope
        Sl = np.exp(-0.5 * ((1 / lam_ax - nu0) / sg) ** 2) / lam_ax ** 2    # density per unit wavelength
        spec.append(Sl / Sl.max())
    lc = np.array(lc)
    for dl in (0.001, 0.010, 0.040, 0.100, 0.300):
        i = np.argmin(np.abs(dlam - dl))
        print(f"  temporal: dlam={dlam[i] * 1e3:6.1f} nm  1/e envelope width {lc[i]:7.2f} um  "
              f"(lam^2/dlam = {lam0 ** 2 / dlam[i]:7.2f} um)")
    cols = np.linspace(0.47, 0.63, 15)                              # 15 colours for the rainbow fan
    save("temporal.npz", dL=f32(dL), dlam=f32(dlam), I=f32(np.array(I)), env=f32(np.array(env)), lc=f32(lc),
         lam_ax=f32(lam_ax), spec=f32(np.array(spec)), lam0=lam0, cols=f32(cols))


# =============================================================== Ch3 ==========
def spatial():
    # visibility vs slit separation (van Cittert-Zernike = |FT of the source's angular intensity profile|)
    q = np.linspace(0, 3, 601)                                      # d * theta_s / lambda
    al = np.linspace(-0.5, 0.5, 2001)                               # uniform strip source, in units of theta_s
    V_strip = np.abs(np.exp(1j * TAU * np.outer(q, al)).mean(1))
    th = 0.047 / 206264.8                                           # Betelgeuse, 0.047 arcsec (Michelson & Pease)
    lam = 0.575e-6
    dm = np.linspace(0, 5, 501)
    xx = np.pi * th * dm / lam
    V_disk = np.abs(np.where(xx > 1e-9, 2 * special.j1(xx) / np.maximum(xx, 1e-12), 1.0))
    z0 = 1.2197 * lam / th                                          # first zero of 2 J1(x)/x: x = 3.8317
    print(f"  spatial: strip first zero at q={q[np.argmin(V_strip[:300])]:.3f}; Betelgeuse disk first zero at d={z0:.2f} m")
    save("spatial.npz", q=f32(q), V_strip=f32(V_strip), dm=f32(dm), V_disk=f32(V_disk), d0=z0)


# =============================================================== Ch4 ==========
def speckle():
    M, A = 40, 20.0                                                 # scatterers in a disk of radius 20 wavelengths
    n, U = 720, 0.6                                                 # far-field direction sines in [-U, U]
    u = np.linspace(-U, U, n)

    def scatterers(rng):
        p = []
        while len(p) < M:
            q = rng.uniform(-A, A, 2)
            if np.hypot(*q) <= A:
                p.append(q)
        return np.array(p), rng.uniform(0, TAU, M)

    def field(pos, ph):
        Ax = np.exp(-1j * TAU * np.outer(u, pos[:, 0]))             # (n, M)  e^{-i k xi u}
        Ay = np.exp(-1j * TAU * np.outer(u, pos[:, 1]))
        return (Ay * np.exp(1j * ph)[None, :]) @ Ax.T               # U[v, u] = sum_j e^{i phi_j} e^{-ik(xi u + eta v)}
    rng = np.random.default_rng(4)
    pos0, ph0 = scatterers(rng)
    I0 = np.abs(field(pos0, ph0)) ** 2
    c = slice(n // 4, 3 * n // 4)                                   # displayed crop: [-0.3, 0.3]
    acc = np.zeros_like(I0)
    keep_N = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    avg, C = [], []
    rng2 = np.random.default_rng(5)
    for k in range(1, 65):
        Ik = I0 if k == 1 else np.abs(field(*scatterers(rng2))) ** 2
        acc += Ik
        A_ = acc / k
        C.append(A_.std() / A_.mean())
        if k in keep_N:
            avg.append(A_[c, c] / M)
    C = np.array(C)
    x = I0 / I0.mean()
    edges = np.linspace(0, 5, 21)
    hist, _ = np.histogram(x, bins=edges, density=True)
    print(f"  speckle: contrast N=1 {C[0]:.3f}; N=4 {C[3]:.3f} (1/2); N=16 {C[15]:.3f} (1/4); N=64 {C[63]:.3f} (1/8)")
    save("speckle.npz", I1=f32(I0[c, c] / M), avg=np.array(avg, np.float16), keep_N=np.array(keep_N), C=f32(C),
         hist=f32(hist), edges=f32(edges), pos=f32(pos0), ph=f32(ph0), u=f32(u[c]), A=A)


# =============================================================== Ch5 ==========
def modes():
    N = 128
    x = np.linspace(-1, 1, N)
    a = np.exp(-(x / 0.72) ** 8)                                    # soft-edged slit (aperture amplitude)
    ells = np.geomspace(3.0, 0.11, 48)                              # coherence width (first zero of mu)
    Js, lams, vecs = [], [], []
    for ell in ells:
        mu = np.sinc((x[:, None] - x[None, :]) / ell)               # uniform incoherent source -> sinc (vCZ)
        J = a[:, None] * a[None, :] * mu                            # J(x1, x2) = <U(x1) U*(x2)>  (real here)
        w, v = np.linalg.eigh(J)
        w, v = w[::-1], v[:, ::-1]
        for n_ in range(16):                                        # sign convention, continuous in ell
            ref = a if n_ % 2 == 0 else a * x
            if (v[:, n_] * ref).sum() < 0:
                v[:, n_] *= -1
        Js.append(J)
        lams.append(np.clip(w, 0, None))
        vecs.append(v[:, :16].T)
    lams = np.array(lams)
    frac = lams / lams.sum(1, keepdims=True)
    for i in (0, 16, 26, 34, 47):
        print(f"  modes: ell={ells[i]:.3f}  power fractions {np.round(frac[i, :6], 3)}  "
              f"(modes for 95%: {int(np.searchsorted(np.cumsum(frac[i]), 0.95)) + 1})")
    coef = np.stack([smooth_noise(np.random.default_rng(100 + k), 7200, 7.0, complex_=True) for k in range(8)])
    save("modes.npz", x=f32(x), a=f32(a), ells=f32(ells), J=f32(np.array(Js)), lams=f32(lams),
         vecs=f32(np.array(vecs)), coef=np.complex64(coef))


# =============================================================== Ch6 ==========
def doe():
    n, pad = 128, 4
    yy, xx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2].astype(float)
    r = np.hypot(xx, yy)
    # target far field: a ring and a cross-hair (a typical laser-pointer DOE pattern)
    T = np.exp(-((r - 36) / 1.1) ** 2)
    arm = lambda c_, o: np.exp(-(o / 0.9) ** 2) * (1 / (1 + np.exp(-(np.abs(c_) - 9) / 0.6))) * (1 / (1 + np.exp((np.abs(c_) - 29) / 0.6)))
    T = np.maximum(T, np.maximum(arm(xx, yy), arm(yy, xx)))
    T = np.fft.ifftshift(T)
    rng = np.random.default_rng(6)
    ph = TAU * rng.random((n, n))
    for _ in range(80):                                              # Gerchberg-Saxton: phase-only DOE
        F = np.fft.fft2(np.exp(1j * ph))
        F = np.sqrt(T) * np.exp(1j * np.angle(F))
        ph = np.angle(np.fft.ifft2(F))
    N = n * pad
    gy, gx = np.mgrid[0:n, 0:n].astype(float)

    def farfield(ax=0.0, ay=0.0):
        """|FFT|^2 of the DOE lit by a plane wave tilted by (ax, ay) far-field pixels (padded grid)."""
        u = np.zeros((N, N), complex)
        u[:n, :n] = np.exp(1j * (ph + TAU * (ax * gx + ay * gy) / N))
        return np.fft.fftshift(np.abs(np.fft.fft2(u)) ** 2)
    P0 = farfield()
    # LED: (1) source points = tilts = shifted copies, hexagonal sampling of a disk of radius Rs far-field pixels
    Rs, nr = 7.0, 4
    hpts = []
    for i in range(-nr, nr + 1):
        for j in range(-nr, nr + 1):
            k = -i - j
            if max(abs(i), abs(j), abs(k)) <= nr:
                hpts.append((i + 0.5 * j, j * np.sqrt(3) / 2, max(abs(i), abs(j), abs(k))))
    hpts = np.array(sorted(hpts, key=lambda p: (p[2], np.arctan2(p[1], p[0]))))
    hpts[:, :2] *= Rs / nr
    cum, sums = np.zeros_like(P0), {}
    for k, (ax, ay, _) in enumerate(hpts):
        cum += farfield(ax, ay)
        if k + 1 in (7, 19, 61):
            sums[k + 1] = cum / (k + 1)
    # (2) colours: far-field angle scales with lambda -> magnify the pattern about the centre by lambda/lambda0
    lam0, fw = 0.530, 0.035
    lams = lam0 + fw / 2.3548 * np.linspace(-2.5, 2.5, 25)
    wl = np.exp(-0.5 * ((lams - lam0) / (fw / 2.3548)) ** 2)
    cy = cx = N // 2
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    spec = np.zeros_like(P0)
    for lam, w in zip(lams, wl):
        sc = lam0 / lam
        spec += w * ndimage.map_coordinates(sums[61], [(Y - cy) * sc + cy, (X - cx) * sc + cx], order=1)
    led_img = spec / wl.sum()
    c = slice(N // 2 - 180, N // 2 + 180)
    norm = P0[c, c].mean()
    out = dict(laser=P0[c, c] / norm, k7=sums[7][c, c] / norm, k19=sums[19][c, c] / norm, k61=sums[61][c, c] / norm,
               led=led_img[c, c] / norm)
    ring = np.abs(np.hypot(*np.mgrid[-180:180, -180:180]) - 36 * pad) < 2
    for k, v in out.items():
        print(f"  doe {k:5s}: contrast along the ring {v[ring].std() / v[ring].mean():.2f}")
    save("doe.npz", **{k: f32(v) for k, v in out.items()}, src=f32(hpts[:, :2]), Rs=Rs, lams=f32(lams), wl=f32(wl),
         phase=f32(ph))


if __name__ == "__main__":
    hook()
    jitter()
    temporal()
    spatial()
    speckle()
    modes()
    doe()
