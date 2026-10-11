"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).
    python physics.py          (single-threaded numpy, ~1 CPU-minute, seeded)

Units: lengths in wavelengths (lambda = 1). Sample-plane pixel dx = lambda/4, objective NA = 0.25 (band limit
B = k NA), so the object field is band-limited to |k| <= B = 0.39 rad/px.

data/cells.npz  synthetic transparent cells (pure phase objects): hook images, the 1D cut used in chapter 1
data/holo.npz   off-axis digital holography: tilt sweep (in-line -> off-axis), final hologram, its spectrum
                (three islands), the cropped island, the reconstruction, the ASM refocus stack, and the
                carrier sweep past the sampling limit (island wraps around)
data/gs.npz     Gerchberg-Saxton phase retrieval: known window |U| + measured far field |FFT U| -> snapshots,
                error curve, ambiguity copies (shift / twin / constant phase), stall statistics
data/cgh.npz    computer-generated hologram: phase-only mask whose far field is a LaTeX lambda (GS loop)
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi

# ----------------------------------------------------------------- colour maps (copied from o5common) ----
def lut(stops, n=1024):
    v = np.linspace(0, 1, n)
    return np.stack([np.interp(v, [s[0] for s in stops], [s[1][c] for s in stops]) for c in range(3)], -1).astype(np.uint8)


HEAT = lut([(0.0, (0, 0, 0)), (0.25, (12, 50, 70)), (0.5, (28, 117, 138)), (0.75, (88, 196, 221)),
            (0.92, (255, 255, 120)), (1.0, (255, 255, 255))])
PHASE_HUE = lut([(0.0, (255, 80, 80)), (0.17, (255, 200, 60)), (0.33, (120, 230, 90)), (0.5, (60, 210, 230)),
                 (0.67, (90, 120, 255)), (0.83, (220, 90, 230)), (1.0, (255, 80, 80))], 2048)


def phase_rgb(U, vmax=None, gamma=0.7, floor=0.0):
    """complex field -> uint8 RGB: hue = phase, brightness = |U|^gamma (row 0 = top of the picture)."""
    a = np.abs(U)
    vmax = vmax if vmax is not None else max(np.percentile(a, 99.5), 1e-12)
    b = floor + (1 - floor) * np.clip(a / vmax, 0, 1) ** gamma
    hue = (np.angle(U) % TAU) / TAU
    return (PHASE_HUE[(hue * 2047).astype(int)].astype(np.float32) * b[..., None]).astype(np.uint8)


def to_u8(v):
    return (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)


def log_spec(S, decades, top=None):
    """|S|^2 on a log scale -> [0, 1] (top = log10 of the brightest value shown as 1)."""
    L = np.log10(np.abs(S) ** 2 + 1e-30)
    top = L.max() if top is None else top
    return np.clip((L - (top - decades)) / decades, 0, 1)


# ----------------------------------------------------------------- optics helpers ----------------------------
def kgrid(N, d=1.0):
    k = TAU * np.fft.fftfreq(N, d=d)
    return np.meshgrid(k, k)


def gsmooth(a, s):
    KX, KY = kgrid(a.shape[0])
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-0.5 * (KX ** 2 + KY ** 2) * s ** 2)))


def bandlimit(u, B):
    KX, KY = kgrid(u.shape[0])
    return np.fft.ifft2(np.fft.fft2(u) * (np.hypot(KX, KY) <= B))


def asm(u, dx, z, lam=1.0):
    """Exact scalar angular spectrum propagation by z (either sign); evanescent waves always decay."""
    KX, KY = kgrid(u.shape[0], dx)
    kz2 = (TAU / lam) ** 2 - KX ** 2 - KY ** 2
    H = np.where(kz2 >= 0, np.exp(1j * np.sqrt(np.abs(kz2)) * z), np.exp(-np.sqrt(np.abs(kz2)) * abs(z)))
    return np.fft.ifft2(np.fft.fft2(u) * H)


# ----------------------------------------------------------------- synthetic cells -----------------------------
def one_cell(x, y, c, R0, rng, fine=1.0):
    """Phase (rad) of one cell-like transparent object: dome body, nucleus, nucleoli, granules (all smooth)."""
    th0 = rng.uniform(0, TAU)
    ecc = rng.uniform(0.62, 0.9)
    dx_, dy_ = x - c[0], y - c[1]
    xr = dx_ * np.cos(th0) + dy_ * np.sin(th0)
    yr = (-dx_ * np.sin(th0) + dy_ * np.cos(th0)) / ecc
    ang = np.arctan2(yr, xr)
    Rth = R0 * (1 + sum(rng.uniform(0, 0.08) * np.cos(m * ang + rng.uniform(0, TAU)) for m in range(2, 6)))
    s = np.hypot(xr, yr) / Rth
    body = np.clip(1 - s ** 2, 0, None) ** 0.9
    nc = rng.uniform(-0.25, 0.25, 2) * R0
    nr = R0 * rng.uniform(0.30, 0.40)
    sn = np.hypot(xr - nc[0], (yr - nc[1]) / rng.uniform(0.8, 1.0)) / nr
    nuc = 0.5 * (1 + np.tanh((1 - sn) / 0.08))
    nucl = np.zeros_like(x)
    for _ in range(rng.integers(1, 3)):
        p = nc + rng.uniform(-0.4, 0.4, 2) * nr
        nucl += np.exp(-((xr - p[0]) ** 2 + (yr - p[1]) ** 2) / (2 * 3.6 ** 2))
    gran = np.zeros_like(x)
    for _ in range(rng.integers(14, 22)):
        rr = rng.uniform(0.40, 0.88) * R0
        aa = rng.uniform(0, TAU)
        gran += np.exp(-((xr - rr * np.cos(aa)) ** 2 + (yr - rr * np.sin(aa)) ** 2) / (2 * rng.uniform(2.4, 3.2) ** 2))
    return 1.5 * body + 0.8 * nuc * (s < 1) + 0.85 * nucl + fine * 0.5 * gran * (body > 0.05)


def cells_at(N, spots, seed, smooth=1.6, fine=1.0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(float)
    phi = np.zeros((N, N))
    for (cx, cy, R) in spots:
        phi += one_cell(x, y, (cx, cy), R, rng, fine)
    return gsmooth(phi, smooth)


# ================================================================= parameters ===================================
N = 512
DX = 0.25                       # sample-plane pixel (wavelengths)
NA = 0.25
B = NA * TAU * DX               # band limit in rad/px (0.393)
BG = 0.64 * TAU                 # global (unobservable) background phase: picks a calm blue background
SPOTS = [(140, 150, 56), (330, 118, 50), (258, 276, 62), (410, 318, 48), (126, 370, 52), (318, 428, 46)]
CELL_SEED = 7
ZREC = 110.0                    # the hologram is recorded this far (in wavelengths) beyond the cells
M_FINAL = 116                   # reference tilt: carrier k_c = 2 pi M/N rad/px (on the FFT grid: no leakage)
SPEC_HALF = 160                 # displayed spectrum crop: |k| <= 160 bins (of 256)
ROW = 276                       # chapter-1 cut through the central cell
SPEC_DEC = 11.5                 # decades shown in the log spectra (outside the bands it is numerically zero)


def make_cells():
    phi = cells_at(N, SPOTS, CELL_SEED)
    U0 = bandlimit(np.exp(1j * phi), B) * np.exp(1j * BG)          # in-focus object field
    I0 = np.abs(U0) ** 2                                             # what an in-focus camera records
    print(f"cells: phase 0..{phi.max():.2f} rad, in-focus camera intensity {I0.min():.3f}..{I0.max():.3f}")
    np.savez_compressed(os.path.join(DATA, "cells.npz"),
                        phase_rgb=phase_rgb(U0, 1.04), camera=to_u8(I0 / 1.6),
                        phi_row=phi[ROW].astype(np.float32), row=ROW, phi_max=phi.max(), bg=BG)
    return phi, U0


def make_holo(U0):
    x = np.arange(N)
    Uc = asm(U0, DX, ZREC)                                           # object wave at the camera
    ref = lambda m: np.exp(1j * TAU * m * x / N)[None, :]            # tilted plane reference, |R| = 1
    holo = lambda m: np.abs(Uc + ref(m)) ** 2
    IMAX = 4.4
    # (a) chapter 2 forms |Uc + R|^2 per frame from Uc for any tilt (one line of arithmetic, no physics)
    # (b) the final hologram and its spectrum
    I = holo(M_FINAL)
    S = np.fft.fftshift(np.fft.fft2(I))
    c, h = N // 2, SPEC_HALF
    TOP = np.log10(np.abs(S[c, c - M_FINAL]) ** 2) + 0.3              # brightest shown = the island centres
    spec = log_spec(S, SPEC_DEC, TOP)[c - h:c + h, c - h:c + h]
    KX, KY = np.meshgrid(np.arange(N) - c, np.arange(N) - c)
    Bb = B * N / TAU                                                 # band limit in FFT bins (32)
    isl = np.hypot(KX + M_FINAL, KY) <= 1.04 * Bb                    # crop circle around the U R* island
    island_alpha = isl[c - h:c + h, c - h:c + h]
    # reconstruction: crop, recentre (shift by +M_FINAL bins), inverse FFT -> the object wave at the camera
    Ur = np.fft.ifft2(np.fft.ifftshift(np.roll(S * isl, M_FINAL, axis=1)))
    err = np.linalg.norm(Ur - Uc) / np.linalg.norm(Uc)
    # refocus with the ASM (episode 1): back-propagate by -z
    zs = np.linspace(0, ZREC + 20, 53)

    def deglobal(V):                     # propagation adds the trivial global phase e^{ikz}: keep the background at BG
        return V * np.exp(1j * (BG - np.angle(V.mean())))
    stack = np.stack([phase_rgb(deglobal(asm(Ur, DX, -z))) for z in zs])
    Uf = asm(Ur, DX, -ZREC)
    err_f = np.linalg.norm(Uf - U0) / np.linalg.norm(U0)
    print(f"holo: carrier {TAU * M_FINAL / N:.3f} rad/px = {TAU * M_FINAL / N / B:.2f} B, fringe period "
          f"{N / M_FINAL:.2f} px; reconstruction error {err:.1e} (camera plane), {err_f:.1e} (refocused)")
    # (c) chapter 6: carrier swept past the sampling limit (k_c + B > pi): the islands wrap around
    m_wrap = np.unique(np.round(np.linspace(M_FINAL, 360, 62)).astype(int))
    wrap_spec, wrap_zoom = [], []
    for m in m_wrap:
        Iw = holo(m)
        Sw = np.fft.fftshift(np.fft.fft2(Iw))
        wrap_spec.append(to_u8(log_spec(Sw, SPEC_DEC, TOP)))
        wrap_zoom.append(to_u8(Iw[12:36, 8:32] / IMAX))
    np.savez_compressed(os.path.join(DATA, "holo.npz"),
                        Uc=Uc.astype(np.complex64), imax=IMAX, m_final=M_FINAL, holo=to_u8(I / IMAX), spec=to_u8(spec),
                        island_alpha=island_alpha, spec_half=SPEC_HALF, band_bins=Bb,
                        recon=stack[0], zs=zs, zrec=ZREC, refocus=stack,
                        m_wrap=m_wrap, wrap_spec=np.stack(wrap_spec), wrap_zoom=np.stack(wrap_zoom),
                        n=N, dx=DX, na=NA)


# ----------------------------------------------------------------- Gerchberg-Saxton --------------------------
NG = 256
GS_SPOTS = [(104, 112, 30), (152, 98, 26), (134, 160, 33)]
GS_CELL_SEED = 3
GS_SEED = 12                     # the run shown in chapter 4 (a start that converges; many stall, see ch. 6)
GS_IT = 150
GS_SNAPS = [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 33, 41, 50, 60, 72, 90, 115, 149]
STALL_SEEDS = list(range(8, 16))


def gs_window(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    disk = np.hypot(xx - n / 2, yy - n / 2) < 0.36 * n
    bite = np.hypot(xx - n / 2 - 0.27 * n, yy - n / 2 + 0.22 * n) < 0.15 * n        # lopsided: no twin symmetry
    return (disk & ~bite).astype(float)


def gs_run(Aobj, Afar, iters, seed, snaps=()):
    rng = np.random.default_rng(seed)
    U = Aobj * np.exp(1j * rng.uniform(0, TAU, Aobj.shape))
    errs, shots = [], {}
    for n in range(iters):
        if n in snaps:
            shots[n] = U.copy()
        F = np.fft.fft2(U)
        errs.append(np.linalg.norm(np.abs(F) - Afar) / np.linalg.norm(Afar))
        U = np.fft.ifft2(Afar * np.exp(1j * np.angle(F)))            # far field: keep angles, swap in lengths
        U = Aobj * np.exp(1j * np.angle(U))                          # object plane: keep angles, swap in lengths
    return U, np.array(errs), shots


def make_gs():
    W = gs_window(NG)
    phi = cells_at(NG, GS_SPOTS, GS_CELL_SEED, fine=0.8)
    Ut = W * bandlimit(np.exp(1j * phi), 2.2 * B) * np.exp(1j * BG)
    Ut = W * np.exp(1j * np.angle(Ut + 1e-30))                       # exactly |U| = 1 inside the window
    Afar = np.abs(np.fft.fft2(Ut))
    U, errs, shots = gs_run(W, Afar, GS_IT, GS_SEED, set(GS_SNAPS))

    def disp(V):                                                     # remove the global phase (blind spot 3)
        g = np.vdot(V[W > 0], Ut[W > 0])
        return phase_rgb(V * np.exp(1j * np.angle(g)), 1.0)
    snaps = np.stack([disp(shots[n]) for n in GS_SNAPS])
    final_err = np.linalg.norm((U * np.exp(1j * np.angle(np.vdot(U[W > 0], Ut[W > 0]))) - Ut)[W > 0]) / np.sqrt(W.sum())
    far = log_spec(np.fft.fftshift(Afar), 8.0)
    # blind spots: shifted copy, flipped twin, constant phase -> identical far-field brightness
    shifted = np.roll(Ut, (22, -30), (0, 1))
    twin = np.conj(np.roll(Ut[::-1, ::-1], (1, 1), (0, 1)))
    const = Ut * np.exp(1j * 2.1)
    for name, V in (("shift", shifted), ("twin", twin), ("const", const)):
        d = np.abs(np.abs(np.fft.fft2(V)) - Afar).max() / Afar.max()
        print(f"gs: blind spot '{name}': far-field difference {d:.1e}")
    blind = np.stack([phase_rgb(V, 1.0) for V in (Ut, shifted, twin, const)])
    # stall statistics: the same problem from 8 random starts
    stall = np.stack([gs_run(W, Afar, 300, s)[1] for s in STALL_SEEDS])
    print(f"gs: run {GS_SEED}: far-field error {errs[0]:.3f} -> {errs[-1]:.1e}, object error {final_err:.1e}; "
          f"stalled starts (err > 0.05 after 300 it): {[s for i, s in enumerate(STALL_SEEDS) if stall[i, -1] > 0.05]}")
    np.savez_compressed(os.path.join(DATA, "gs.npz"),
                        window=to_u8(W), truth=phase_rgb(Ut, 1.0), far=to_u8(far), snaps=snaps,
                        snap_it=np.array(GS_SNAPS), err=errs, blind=blind, stall=stall, shift=np.array([22, -30]))


# ----------------------------------------------------------------- computer-generated hologram ---------------
NC, MC = 512, 256                 # far-field grid, mask aperture (zero-padded 2x: speckle grains ~2 px)
CGH_SEED = 1
CGH_IT = 60
CGH_SNAPS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40, 50, 59]
CGH_HALF = 150                    # displayed far-field crop half-width (px)


def lambda_glyph(n, height=0.5, bold=10):
    """LaTeX-style math-italic lambda (Latin Modern Math), rendered 4x supersampled -> [0, 1] coverage."""
    font = "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf"
    S = 4 * n
    im = Image.new("L", (S, S), 0)
    f = ImageFont.truetype(font, int(S * height * 1.25))
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), "\U0001D706", font=f)
    d.text(((S - (bb[2] - bb[0])) / 2 - bb[0], (S - (bb[3] - bb[1])) / 2 - bb[1]), "\U0001D706", font=f, fill=255,
           stroke_width=bold, stroke_fill=255)
    return np.asarray(im, float).reshape(n, 4, n, 4).mean((1, 3)) / 255


def make_cgh():
    tgt = lambda_glyph(NC, 0.42, 14)
    aper = np.zeros((NC, NC))
    a0 = (NC - MC) // 2
    aper[a0:a0 + MC, a0:a0 + MC] = 1
    At = np.fft.ifftshift(np.sqrt(tgt))
    rng = np.random.default_rng(CGH_SEED)
    U = aper * np.exp(1j * rng.uniform(0, TAU, (NC, NC)))
    At *= np.linalg.norm(np.fft.fft2(U)) / np.linalg.norm(At)
    inside = At > 0.3 * At.max()
    eff, masks, fars = [], [], []
    c, h = NC // 2, CGH_HALF
    for n in range(CGH_IT):
        F = np.fft.fft2(U)
        I = np.abs(F) ** 2
        eff.append((I * inside).sum() / I.sum())
        if n in CGH_SNAPS:
            masks.append(phase_rgb(U[a0:a0 + MC, a0:a0 + MC], 1.0))
            fars.append(np.fft.fftshift(I)[c - h:c + h, c - h:c + h])
        U = np.fft.ifft2(At * np.exp(1j * np.angle(F)))              # far field: keep angles, impose the target
        U = aper * np.exp(1j * np.angle(U))                          # mask: phase only
    vmax = np.percentile(np.stack(fars[1:]), 99.7)
    far_u8 = np.stack([to_u8((f / vmax) ** 0.6) for f in fars])
    Ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))        # centred FFT: phase relative to the mask centre
    print(f"cgh: light inside the target {eff[0]:.2f} (random mask) -> {eff[-1]:.2f} after {CGH_IT} iterations")
    np.savez_compressed(os.path.join(DATA, "cgh.npz"),
                        target=to_u8(tgt[c - h:c + h, c - h:c + h]), masks=np.stack(masks), fars=far_u8,
                        snap_it=np.array(CGH_SNAPS), eff=np.array(eff),
                        far_phase=phase_rgb(Ff[c - h:c + h, c - h:c + h] / np.abs(Ff[c - h:c + h, c - h:c + h]).max(), 1.0,
                                            gamma=0.0), half=CGH_HALF)


if __name__ == "__main__":
    phi, U0 = make_cells()
    make_holo(U0)
    make_gs()
    make_cgh()
    for f in sorted(os.listdir(DATA)):
        print(f"  data/{f}: {os.path.getsize(os.path.join(DATA, f)) / 1e6:.1f} MB")
