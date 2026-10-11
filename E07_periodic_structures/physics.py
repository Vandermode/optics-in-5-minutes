"""Precompute every physical field shown in E07 (Periodic Structures) -> data/*.npz (run once; scenes only load).

    python physics.py            (JAX on a free GPU for the angular-spectrum maps;
                                                          numpy fallback; ~1 min)

Methods (all deterministic, no random numbers):
* rcwa_te      rigorous coupled-wave analysis for 1D gratings, TE polarisation (E along the grooves), S-matrix
               formulation (Redheffer star product). Validated below: uniform film vs the Airy formula, energy
               conservation for lossless gratings, thin-grating limit vs Bessel efficiencies, convergence.
* tmm_*        1D transfer matrices at normal incidence: Bloch bands cos(kappa*L) = tr(M)/2, stack reflectance,
               the field inside the stack and the Bloch mode of the infinite lattice.
* fan maps     white light through a sinusoidal surface-relief grating (glass -> air): the order amplitudes t_m
               come from rcwa_te for every wavelength, the beam is propagated with the exact angular-spectrum
               method (E01) and the intensities are turned into colour with the CIE 1931 observer (sRGB).
Outputs: data/fan.npz (hook + Ch2 maps), data/stack.npz (Ch0 spectrum, Ch3 bands/reflectance/fields),
         data/rcwa.npz (Ch4 orders/modes, Ch5 convergence + unit-cell library).
"""
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # single-threaded numpy (shared server)
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi


# =============================================================== colour science ======
def cie_xyz(lam_nm):
    """CIE 1931 2-degree colour-matching functions (Wyman, Sloan & Shirley 2013 multi-lobe fit)."""
    l = np.asarray(lam_nm, float)

    def g(mu, s1, s2):
        s = np.where(l < mu, s1, s2)
        return np.exp(-0.5 * ((l - mu) / s) ** 2)
    x = 1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2)
    y = 0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1)
    z = 1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


M_RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
_WB = None


def xyz_to_lin_rgb(xyz):
    """XYZ -> linear sRGB, white-balanced so that an equal-energy spectrum is neutral (R = G = B)."""
    global _WB
    if _WB is None:
        lam = np.arange(380, 781, 1.0)
        _WB = 1.0 / (cie_xyz(lam).sum(0) @ M_RGB.T)
    return (np.asarray(xyz) @ M_RGB.T) * _WB


def srgb_encode(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def spectral_rgb(lam_nm, desat=0.0):
    """Display colour (0..1 sRGB) of a monochromatic line: out-of-gamut handled by adding white."""
    rgb = xyz_to_lin_rgb(cie_xyz(np.atleast_1d(lam_nm)))
    rgb = np.clip(rgb, 0, None) - 0.25 * np.minimum(rgb.min(-1, keepdims=True), 0)
    rgb = rgb / rgb.max(-1, keepdims=True)
    rgb = (1 - desat) * rgb + desat
    return srgb_encode(rgb)


def tone_map(rgb_lin, y_ref, gamma=0.55, gain=1.0, white_mix=0.25):
    """Linear RGB image -> sRGB uint8, compressing luminance (Y/y_ref)^gamma but keeping chromaticity.
    Out-of-gamut (spectral) colours: negative channels are clipped, and white_mix of the deficit is added back
    as white (0 = vivid, 1 = fully desaturated)."""
    neg = np.minimum(rgb_lin.min(-1, keepdims=True), 0)
    rgb = np.clip(rgb_lin, 0, None) - white_mix * neg
    Y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    Yn = np.clip(Y / y_ref, 0, None)
    f = np.where(Yn > 1e-9, gain * Yn ** gamma / np.maximum(Yn, 1e-9), 0.0) / y_ref
    out = rgb * f[..., None]
    m = out.max(-1, keepdims=True)
    out = np.where(m > 1, out / np.maximum(m, 1e-9), out)                   # keep hue when clipping
    return (255 * srgb_encode(out) + 0.5).astype(np.uint8)


# =============================================================== RCWA (TE) ===========
def eps_fourier(profile_eps, nharm):
    """Fourier coefficients eps_n, n = -nharm..nharm, of one period sampled finely (profile_eps: (P,) real/complex)."""
    P = len(profile_eps)
    c = np.fft.fft(profile_eps) / P
    n = np.arange(-nharm, nharm + 1)
    return c[n % P]


def eps_fourier_lamellar(eps_ridge, eps_gap, fill, nharm):
    """Exact coefficients of a ridge of relative width `fill` centred at x = 0 (period 1)."""
    n = np.arange(-nharm, nharm + 1)
    return eps_gap * (n == 0) + (eps_ridge - eps_gap) * fill * np.sinc(n * fill)


def _sqrt_branch(q2):
    q = np.sqrt(q2.astype(complex))
    flip = (q.imag < 0) | ((np.abs(q.imag) < 1e-12) & (q.real < 0))
    q = np.where(flip, -q, q)
    return np.where(np.abs(q) < 1e-9, 1e-9 * (1 + 1j), q)   # exact cut-off (kz = 0) would make the modes degenerate


def layer_modes(eps, kxn):
    """Modes of one layer.  eps: scalar (uniform) or array of Fourier coefficients eps_{-2M..2M}.
    Returns (W, q): columns of W are the mode profiles in the order basis, q = kz / k0 (Im q >= 0)."""
    N = len(kxn)
    if np.isscalar(eps) or np.ndim(eps) == 0:
        return np.eye(N, dtype=complex), _sqrt_branch(eps - kxn ** 2)
    M = (N - 1) // 2
    idx = np.arange(N)[:, None] - np.arange(N)[None, :] + 2 * M          # E[m, p] = eps_{m-p}
    E = np.asarray(eps)[idx]
    A = E - np.diag(kxn ** 2)
    if np.allclose(A, A.conj().T, atol=1e-13):           # lossless: Hermitian -> orthonormal modes (robust
        q2, W = np.linalg.eigh(A)                       # when +-m pairs are degenerate in thin slices)
    else:
        q2, W = np.linalg.eig(A)
    return W.astype(complex), _sqrt_branch(q2)


def interface_S(W1, q1, W2, q2):
    """S-matrix of the interface region1 (below) -> region2 (above); amplitudes referenced at the interface.
    Field vector [S ; S'/(i k0)] = [[W, W], [W q, -W q]] [a+ ; a-]."""
    N = W1.shape[0]
    A, B = W1 * q1[None, :], W2 * q2[None, :]
    L = np.block([[-W1, W2], [A, B]])
    R = np.block([[W1, -W2], [A, B]])
    S = np.linalg.solve(L, R)
    return S[:N, :N], S[:N, N:], S[N:, :N], S[N:, N:]


def prop_S(q, k0d):
    P = np.diag(np.exp(1j * q * k0d))
    Z = np.zeros_like(P)
    return Z, P, P, Z


def star(A, B):
    """Redheffer star product: A (lower element) then B (upper element). S = (S11, S12, S21, S22)."""
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    I = np.eye(A11.shape[0])
    X = np.linalg.solve(I - B11 @ A22, np.eye(A11.shape[0]))
    Y = np.linalg.solve(I - A22 @ B11, np.eye(A11.shape[0]))
    return (A11 + A12 @ X @ B11 @ A21, A12 @ X @ B12, B21 @ Y @ A21, B22 + B21 @ Y @ A22 @ B12)


def rcwa_te(lam, period, layers, n_in, n_out, M, theta=0.0, return_S=False):
    """TE diffraction by a stack of 1D-periodic layers (light travels +z from n_in to n_out).
    layers: list of (eps, thickness) bottom -> top, eps scalar or Fourier coefficients (length 4M+1).
    Returns dict: m, kxn, r, t (complex order amplitudes), R, T (efficiencies)."""
    k0 = TAU / lam
    m = np.arange(-M, M + 1)
    kxn = n_in * np.sin(theta) + m * lam / period
    Wi, qi = layer_modes(n_in ** 2, kxn)
    Wo, qo = layer_modes(n_out ** 2, kxn)
    S = None
    Wp, qp = Wi, qi
    for eps, d in layers:
        W, q = layer_modes(eps, kxn)
        Si = interface_S(Wp, qp, W, q)
        S = Si if S is None else star(S, Si)
        S = star(S, prop_S(q, k0 * d))
        Wp, qp = W, q
    Si = interface_S(Wp, qp, Wo, qo)
    S = Si if S is None else star(S, Si)
    a = (m == 0).astype(complex)
    r, t = S[0] @ a, S[2] @ a
    R = np.real(qi) / np.real(qi[M]) * np.abs(r) ** 2
    T = np.real(qo) / np.real(qi[M]) * np.abs(t) ** 2
    out = dict(m=m, kxn=kxn, r=r, t=t, R=R, T=T, q_in=qi, q_out=qo)
    if return_S:
        out["S"] = S
    return out


def sinusoid_layers(height, n_ridge, n_gap, nlay, M):
    """Staircase approximation of a sinusoidal surface relief (glass ridge profile) with nlay slices."""
    xs = np.arange(4096) / 4096                             # samples at x_j = j/P: FFT coefficients need no phase factor
    prof = 0.5 * (1 + np.cos(TAU * xs))                    # 1 at the ridge centre (x = 0), 0 in the groove
    out = []
    for j in range(nlay):
        level = (j + 0.5) / nlay                            # slice j from the bottom
        e = np.where(prof >= level, n_ridge ** 2, n_gap ** 2).astype(float)
        out.append((eps_fourier(e, 2 * M), height / nlay))
    return out


# =============================================================== transfer matrices ====
def tmm_P(n, k0d):
    """Forward transfer of (E, H) through a uniform layer: [E(z+d); H(z+d)] = P [E(z); H(z)]  (H = n E forward)."""
    c, s = np.cos(n * k0d), np.sin(n * k0d)
    return np.array([[c, 1j * s / n], [1j * n * s, c]])


def tmm_halftrace(nu, n1, n2, f1=0.5):
    """cos(kappa*Lambda) = tr(M)/2 for a period (n1 for f1*Lambda, then n2); nu = Lambda / lambda."""
    d1, d2 = TAU * nu * f1, TAU * nu * (1 - f1)
    return np.cos(n1 * d1) * np.cos(n2 * d2) - 0.5 * (n1 / n2 + n2 / n1) * np.sin(n1 * d1) * np.sin(n2 * d2)


def tmm_stack(nu, n1, n2, npairs, n_in, n_out, f1=0.5):
    """Reflection / transmission amplitudes of npairs periods (n1, n2) between n_in and n_out at nu = Lambda/lambda."""
    T = np.eye(2, dtype=complex)
    cell = tmm_P(n2, TAU * nu * (1 - f1)) @ tmm_P(n1, TAU * nu * f1)
    for _ in range(npairs):
        T = cell @ T
    a = n_out * T[0, 0] - T[1, 0]
    b = n_in * (n_out * T[0, 1] - T[1, 1])
    r = -(a + b) / (a - b)
    t = T[0, 0] * (1 + r) + T[0, 1] * n_in * (1 - r)
    return r, t


def tmm_field(nu, n1, n2, npairs, n_in, n_out, z, f1=0.5):
    """Complex E(z) (z in units of Lambda; stack occupies 0..npairs) for unit incidence from n_in."""
    r, t = tmm_stack(nu, n1, n2, npairs, n_in, n_out, f1)
    E = np.zeros(len(z), complex)
    k = TAU * nu
    left = z < 0
    E[left] = np.exp(1j * n_in * k * z[left]) + r * np.exp(-1j * n_in * k * z[left])
    right = z >= npairs
    E[right] = t * np.exp(1j * n_out * k * (z[right] - npairs))
    v = np.array([1 + r, n_in * (1 - r)])
    bounds = []
    for p in range(npairs):
        bounds += [(p, p + f1, n1), (p + f1, p + 1, n2)]
    for za, zb, n in bounds:
        sel = (z >= za) & (z < zb)
        for i in np.where(sel)[0]:
            E[i] = (tmm_P(n, k * (z[i] - za)) @ v)[0]
        v = tmm_P(n, k * (zb - za)) @ v
    return E, r, t


def bloch_mode(nu, n1, n2, z, f1=0.5):
    """Forward Bloch mode of the infinite lattice (z in units of Lambda, z >= 0): E(z), kappa*Lambda."""
    k = TAU * nu
    cell = tmm_P(n2, k * (1 - f1)) @ tmm_P(n1, k * f1)
    w, V = np.linalg.eig(cell)
    flux = [np.real(V[0, j] * np.conj(V[1, j])) for j in range(2)]
    j = int(np.argmax(flux))
    v = V[:, j] / V[0, j]
    kap = np.angle(w[j])
    E = np.zeros(len(z), complex)
    for i, zz in enumerate(z):
        p = int(np.floor(zz))
        u = v * w[j] ** p
        s = zz - p
        E[i] = (tmm_P(n1, k * s) @ u)[0] if s < f1 else (tmm_P(n2, k * (s - f1)) @ (tmm_P(n1, k * f1) @ u))[0]
    return E, kap


# =============================================================== angular spectrum =====
try:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    _GPU = any(d.platform == "gpu" for d in jax.devices())
except Exception:                                            # pragma: no cover
    jax, jnp, _GPU = None, None, False


def asm_intensity_rows(u0, dx, zs, n_med, lam, crop):
    """|ASM(u0, z)|^2 for every z in zs (exact kz in medium n_med), cropped to columns crop=(i0, i1)."""
    N = len(u0)
    kx = TAU * np.fft.fftfreq(N, d=dx)
    k = TAU * n_med / lam
    kz = np.sqrt((k ** 2 - kx ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    i0, i1 = crop
    if _GPU:
        F = jnp.fft.fft(jnp.asarray(u0, jnp.complex64))
        H = jnp.exp(1j * jnp.asarray(kz, jnp.complex64)[None, :] * jnp.asarray(zs, jnp.float32)[:, None])
        U = jnp.fft.ifft(F[None, :] * H, axis=1)[:, i0:i1]
        return np.asarray(jnp.abs(U) ** 2)
    F = np.fft.fft(u0)
    return np.array([np.abs(np.fft.ifft(F * np.exp(1j * kz * z))[i0:i1]) ** 2 for z in zs])


# ------------------------------------------------------------------ fan maps ----------
N_GLASS = 1.5
LAM_HOOK = 1.5                                               # grating period of the hook / Ch2 maps (um)
FAN = dict(N=16384, dx=0.04, xh=32.0, zlo=-5.0, zhi=40.0, w0=3.0, h=0.30, nlay=10, M=12)


def fan_order_amplitudes(lam, period, M=None):
    """RCWA (TE) transmitted order amplitudes into air for the sinusoidal relief (normal incidence from glass)."""
    M = M or FAN["M"]
    lay = sinusoid_layers(FAN["h"], N_GLASS, 1.0, FAN["nlay"], M)
    res = rcwa_te(lam, period, lay, N_GLASS, 1.0, M)
    return res


def fan_xyz(period, lams, nrow, ncol):
    """CIE XYZ image (nrow x ncol, row 0 = top = largest z) of a white Gaussian beam through the grating."""
    N, dx, xh = FAN["N"], FAN["dx"], FAN["xh"]
    x = (np.arange(N) - N // 2) * dx
    g = np.exp(-(x / FAN["w0"]) ** 2)
    b = int(round(2 * xh / dx / ncol))                      # columns averaged per output pixel
    i0 = N // 2 - (ncol * b) // 2
    i1 = i0 + ncol * b                                       # symmetric crop: x in [-ncol*b*dx/2, +ncol*b*dx/2)
    nz_lo = int(round(nrow * (0 - FAN["zlo"]) / (FAN["zhi"] - FAN["zlo"])))
    z_up = np.linspace(0, FAN["zhi"], nrow - nz_lo)
    z_dn = np.linspace(FAN["zlo"], 0, nz_lo, endpoint=False)
    cmf = cie_xyz(lams)
    dl = (lams[-1] - lams[0]) / (len(lams) - 1) if len(lams) > 1 else 1.0
    xyz = np.zeros((nrow, ncol, 3))
    K = TAU / period
    stats = []
    for li, lam_nm in enumerate(lams):
        lam = lam_nm * 1e-3                                  # um
        res = fan_order_amplitudes(lam, period)
        u_up = np.zeros(N, complex)
        for mm, tm in zip(res["m"], res["t"]):
            if abs(tm) > 1e-6:
                u_up += tm * np.exp(1j * mm * K * x)
        u_up *= g
        I_up = asm_intensity_rows(u_up, dx, z_up, 1.0, lam, (i0, i1))
        # incident beam in the glass: waist at z = 0, so |U(-z)| = |U(z)| (propagating backwards with the
        # evanescent part would blow up)
        I_dn = asm_intensity_rows(g.astype(complex), dx, -z_dn, N_GLASS, lam, (i0, i1))
        I = np.concatenate([I_dn, I_up], 0)[::-1]           # row 0 = top (z = zhi)
        I = I.reshape(I.shape[0], ncol, b).mean(-1)
        xyz += I[..., None] * cmf[li][None, None, :] * dl
        stats.append((lam_nm, res["T"].sum(), res["R"].sum()))
    return xyz, stats


LUM = np.array([0.2126, 0.7152, 0.0722])


def lum_ref(xyz, nrows=5):
    """Peak luminance of the incident beam (bottom rows), measured in the same white-balanced RGB as tone_map."""
    return float((np.clip(xyz_to_lin_rgb(xyz[-nrows:]), 0, None) @ LUM).max())


def fan_rgb(xyz, y_ref, gamma=0.7, gain=1.0):
    return tone_map(xyz_to_lin_rgb(xyz), y_ref, gamma, gain, 0.15)


# =============================================================== main =================
def check_rcwa():
    print("--- RCWA checks")
    # (a) uniform film between glass and air vs the Airy formula
    lam, nf, d = 1.0, 2.0, 0.37
    res = rcwa_te(lam, 0.37, [(nf ** 2, d)], 1.5, 1.0, 5)
    r12, r23 = (1.5 - nf) / (1.5 + nf), (nf - 1.0) / (nf + 1.0)
    ph = np.exp(2j * TAU / lam * nf * d)
    ra = (r12 + r23 * ph) / (1 + r12 * r23 * ph)
    print(f"  film: R_rcwa={res['R'].sum():.12f} R_airy={abs(ra) ** 2:.12f}  R+T={res['R'].sum() + res['T'].sum():.12f}")
    # (b) energy conservation, lossless lamellar grating, several orders
    M = 20
    for per in (0.45, 1.3, 2.5):
        eps = eps_fourier_lamellar(2.4 ** 2, 1.0, 0.45, 2 * M)
        res = rcwa_te(1.0, per, [(eps, 0.8)], 1.46, 1.0, M)
        print(f"  lamellar period {per}: sum R+T = {res['R'].sum() + res['T'].sum():.12f}")
    # (c) thin sinusoidal relief, wide period -> Bessel efficiencies (thin-element limit)
    from scipy.special import jv
    h, per = 0.06, 20.0
    lay = sinusoid_layers(h, 1.5, 1.0, 16, 40)
    res = rcwa_te(1.0, per, lay, 1.5, 1.0, 40)
    phi = TAU * 0.5 * h / 2                                   # half peak-to-peak phase, (n-1) h k0 / 2
    Tm = res["T"][40 - 2: 40 + 3]
    print("  thin relief T_m (m=-2..2):", np.round(Tm, 5), " Bessel*T0tot:",
          np.round(jv(np.arange(-2, 3), phi) ** 2 * res["T"].sum(), 5))


def build_fan():
    t = time.time()
    lams = np.arange(400, 701, 10.0)
    # white reference: the incident beam below the grating (glass), peak luminance
    xyz, stats = fan_xyz(LAM_HOOK, lams, 360, 400)
    y_ref = lum_ref(xyz)
    hook = fan_rgb(xyz, y_ref)
    for s in stats[::6]:
        print(f"  fan L={LAM_HOOK} lam={s[0]:.0f}: T={s[1]:.4f} R={s[2]:.4f} (sum {s[1] + s[2]:.6f})")
    # monochromatic (530 nm) version for Ch2
    xyz_g, _ = fan_xyz(LAM_HOOK, np.array([530.0]), 360, 400)
    green = fan_rgb(xyz_g, lum_ref(xyz_g))
    # period sweep (uniform in K) for Ch2, 16 wavelengths
    lams_s = np.arange(400, 701, 20.0)
    Ks = np.linspace(TAU / LAM_HOOK, TAU / 0.35, 96)
    sweep = np.zeros((len(Ks), 260, 320, 3), np.uint8)
    xyz0, _ = fan_xyz(LAM_HOOK, lams_s, 260, 320)
    y_ref_s = lum_ref(xyz0, 4)
    for i, K in enumerate(Ks):
        xyz_s, _ = fan_xyz(TAU / K, lams_s, 260, 320)
        sweep[i] = fan_rgb(xyz_s, y_ref_s)
    # 0th-order transmission vs period at 550 nm (sanity: energy goes to the 0th order below lambda)
    T0 = [fan_order_amplitudes(0.55, TAU / K)["T"][FAN["M"]] for K in Ks[::8]]
    print("  T0(550 nm) along the sweep:", np.round(T0, 3))
    cols = {k: spectral_rgb(v)[0] for k, v in dict(red=650, green=530, blue=450).items()}
    np.savez_compressed(os.path.join(DATA, "fan.npz"), hook=hook, green=green, sweep=sweep,
                        periods=TAU / Ks, lam_rgb=np.array([650, 530, 450.0]),
                        rgb=np.array([cols["red"], cols["green"], cols["blue"]]),
                        xh=FAN["xh"], zlo=FAN["zlo"], zhi=FAN["zhi"], w0=FAN["w0"], h=FAN["h"])
    print(f"  fan.npz written ({time.time() - t:.1f}s, GPU={_GPU})")


def build_stack():
    t = time.time()
    nbar, dn_max, npairs = 1.9, 0.9, 10
    dns = np.linspace(0, dn_max, 46)
    nus = np.linspace(0.0, 0.42, 1261)
    half = np.zeros((len(dns), len(nus)))
    R = np.zeros_like(half)
    for i, dn in enumerate(dns):
        n1, n2 = nbar - dn / 2, nbar + dn / 2
        half[i] = tmm_halftrace(nus, n1, n2)
        for j, nu in enumerate(nus):
            r, _ = tmm_stack(nu, n1, n2, npairs, nbar, nbar)
            R[i, j] = abs(r) ** 2
    kap = np.where(np.abs(half) <= 1, np.arccos(np.clip(half, -1, 1)), np.nan) / TAU     # kappa in units of K
    # first gap edges (around the zone-edge crossing nu_B = 1 / (2 nbar))
    gap = np.zeros((len(dns), 2))
    nuB = 1 / (2 * nbar)
    for i in range(len(dns)):
        out = np.abs(half[i]) > 1
        js = np.where(out & (np.abs(nus - nuB) < 0.12))[0]
        gap[i] = (nus[js[0]], nus[js[-1]]) if len(js) else (nuB, nuB)
    # fields in the finite stack (dn_max) for a range of frequencies, and the Bloch mode of the infinite lattice
    n1, n2 = nbar - dn_max / 2, nbar + dn_max / 2
    zf = np.linspace(-2.0, npairs + 2.0, 700)
    nu_f = np.linspace(0.12, 0.40, 141)
    Ef = np.array([tmm_field(nu, n1, n2, npairs, nbar, nbar, zf)[0] for nu in nu_f]).astype(np.complex64)
    nu_bl = 0.18
    zb = np.linspace(0, 8, 640)
    Eb, kapb = bloch_mode(nu_bl, n1, n2, zb)
    # hook: visible spectrum of a gentler mirror (dn = 0.4, 10 pairs, Bragg wavelength 470 nm) -> one colour
    lamB, dn_hook = 470.0, 0.4
    Lam = lamB / (2 * nbar)
    lam_v = np.linspace(380, 720, 341)
    Rv = np.array([abs(tmm_stack(Lam / l, nbar - dn_hook / 2, nbar + dn_hook / 2, npairs, nbar, nbar)[0]) ** 2
                   for l in lam_v])
    cmf = cie_xyz(lam_v)
    white = cmf.sum(0)
    c_ref = xyz_to_lin_rgb((Rv[:, None] * cmf).sum(0) / white[1])
    c_tr = xyz_to_lin_rgb(((1 - Rv)[:, None] * cmf).sum(0) / white[1])
    c_ref = srgb_encode(np.clip(c_ref - min(c_ref.min(), 0), 0, None) / max(c_ref.max(), 1e-9))
    c_tr = srgb_encode(np.clip(c_tr - min(c_tr.min(), 0), 0, None) / max(c_tr.max(), 1e-9))
    lam_rgb = spectral_rgb(lam_v, 0.0)
    print(f"  stack: nu_B={nuB:.4f}, gap(dn_max)={gap[-1]}, rel width={(gap[-1][1] - gap[-1][0]) / nuB:.3f}, "
          f"Rmax={R[-1].max():.6f}, R(dn=0)max={R[0].max():.2e}, Bloch kappa*L={kapb:.3f} rad, "
          f"hook mirror Rmax={Rv.max():.3f} at {lam_v[np.argmax(Rv)]:.0f} nm, reflected rgb={np.round(c_ref, 3)}, "
          f"transmitted rgb={np.round(c_tr, 3)}")
    np.savez_compressed(os.path.join(DATA, "stack.npz"), dns=dns, nus=nus, half=half, kap=kap, R=R, gap=gap,
                        nbar=nbar, npairs=npairs, zf=zf, nu_f=nu_f, Ef=Ef, nu_bl=nu_bl, zb=zb,
                        Eb=Eb.astype(np.complex64), kapb=kapb, lam_v=lam_v, Rv=Rv, lamB=lamB, Lam=Lam, dn_hook=dn_hook,
                        c_ref=c_ref, c_tr=c_tr, lam_rgb=lam_rgb)
    print(f"  stack.npz written ({time.time() - t:.1f}s)")


def build_rcwa():
    t = time.time()
    # ---- Ch4: a thick lamellar grating with several orders (TiO2-like ridges on glass, into air)
    lam, per, fill, h, n_r, n_sub = 1.0, 2.5, 0.4, 0.9, 2.4, 1.46
    M = 10
    eps = eps_fourier_lamellar(n_r ** 2, 1.0, fill, 2 * M)
    res = rcwa_te(lam, per, [(eps, h)], n_sub, 1.0, M)
    m = res["m"]
    kxn = res["kxn"]
    W, q = layer_modes(eps, kxn)
    order = np.argsort(-q.real)
    xs = np.linspace(-per, per, 600)                           # two periods, ridge centred at 0 and +-per
    modes = []
    for j in order[:3]:
        prof = (W[:, j][None, :] * np.exp(1j * TAU / lam * kxn[None, :] * xs[:, None])).sum(1)
        prof = prof / prof[np.argmax(np.abs(prof))]
        modes.append(prof.real)
    eps_prof_x = np.linspace(-0.5, 0.5, 400)
    eps_prof = np.where(np.abs(eps_prof_x) <= fill / 2, n_r ** 2, 1.0)
    E_mat = eps[np.arange(2 * M + 1)[:, None] - np.arange(2 * M + 1)[None, :] + 2 * M].real
    # 9 x 9 matrices shown on screen (orders m = -4..4, k0 = 1): uniform glass layer vs the grating layer
    Ms = 4
    sl = slice(M - Ms, M + Ms + 1)
    A_uni = np.diag(n_sub ** 2 - kxn[sl] ** 2)
    A_gr = E_mat[sl, sl] - np.diag(kxn[sl] ** 2)
    print(f"  Ch4 grating: T_m={np.round(res['T'][M - 3:M + 4], 4)}, R_m={np.round(res['R'][M - 3:M + 4], 4)}, "
          f"sum={res['T'].sum() + res['R'].sum():.12f}; top mode n_eff={np.round(q[order[:3]].real, 3)}")
    # ---- Ch5: unit-cell library (sub-wavelength period, TiO2 ridges on glass, TE, normal incidence)
    per_l, h_l, n_l = 0.45, 0.9, 2.4
    Ml = 25
    fills = np.linspace(0.05, 0.95, 91)
    t0, T0, Rsum, Tsum = [], [], [], []
    for f in fills:
        e = eps_fourier_lamellar(n_l ** 2, 1.0, f, 2 * Ml)
        rr = rcwa_te(1.0, per_l, [(e, h_l)], n_sub, 1.0, Ml)
        t0.append(rr["t"][Ml])
        T0.append(rr["T"][Ml])
        Rsum.append(rr["R"].sum())
        Tsum.append(rr["T"].sum())
    t0 = np.array(t0)
    # reference: no ridge (bare glass -> air through the same height of air): phase relative to that
    rr0 = rcwa_te(1.0, per_l, [(1.0, h_l)], n_sub, 1.0, Ml)
    ph = np.unwrap(np.angle(t0 / rr0["t"][Ml]))
    ph = ph - ph[0]
    T0 = np.array(T0)
    print(f"  library: phase span {ph.max() - ph.min():.3f} rad, T0 min {T0.min():.3f} mean {T0.mean():.3f}, "
          f"max |1-(R+T)| = {np.max(np.abs(1 - np.array(Rsum) - np.array(Tsum))):.2e}, "
          f"nonzero orders T={max(np.array(Tsum) - T0):.2e}")
    # convergence of the zero-order amplitude vs number of orders (fill 0.5)
    Ms = np.arange(1, 41)
    e_ref = eps_fourier_lamellar(n_l ** 2, 1.0, 0.5, 2 * 120)
    t_ref = rcwa_te(1.0, per_l, [(e_ref, h_l)], n_sub, 1.0, 120)["t"][120]
    err = []
    for Mm in Ms:
        e = eps_fourier_lamellar(n_l ** 2, 1.0, 0.5, 2 * Mm)
        err.append(abs(rcwa_te(1.0, per_l, [(e, h_l)], n_sub, 1.0, Mm)["t"][Mm] - t_ref))
    err = np.array(err)
    print(f"  convergence |t0(M)-t0(240 orders)|: M=1 {err[0]:.2e}, M=5 {err[4]:.2e}, M=10 {err[9]:.2e}, "
          f"M=20 {err[19]:.2e}, M=40 {err[39]:.2e}")
    np.savez_compressed(os.path.join(DATA, "rcwa.npz"), m=m, T=res["T"], R=res["R"], kxn=kxn, per=per, fill=fill,
                        h=h, n_r=n_r, n_sub=n_sub, lam=lam, q=q[order[:3]], modes=np.array(modes), xs=xs,
                        eps_m=eps.real, E_mat=E_mat, A_uni=A_uni, A_gr=A_gr, eps_prof_x=eps_prof_x, eps_prof=eps_prof,
                        fills=fills, ph=ph, T0=T0, per_l=per_l, h_l=h_l, n_l=n_l, Ms=Ms * 2 + 1, err=err)
    print(f"  rcwa.npz written ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["check", "stack", "rcwa", "fan"]
    if "check" in what:
        check_rcwa()
    if "stack" in what:
        build_stack()
    if "rcwa" in what:
        build_rcwa()
    if "fan" in what:
        build_fan()
