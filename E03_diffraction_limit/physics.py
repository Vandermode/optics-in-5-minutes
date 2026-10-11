"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).
    OMP_NUM_THREADS=1 python physics.py

Everything is scalar Fourier optics with numpy/scipy FFTs (single worker), deterministic (seeded noise):
  pupil P(f) = disc |f| <= NA/lambda (anti-aliased edge)  ->  amplitude PSF h = IFFT(P)
  incoherent (fluorescence):  PSF = |h|^2,  OTF = FFT(PSF)  (= pupil autocorrelation), cutoff 2 NA/lambda
  coherent (laser):           field image = IFFT(FFT(t) * P), intensity |.|^2, cutoff NA/lambda
Lengths are in wavelengths (lambda = 1) except the hook (micrometres, lambda = 0.5 um, NA = 1.4).

data/hook.npz     bead field image (NA 1.4 oil, 500 nm), bead positions, two-bead merge stack + profiles
data/stamps.npz   points -> cumulative PSF stamps (NA 0.5), PSF icon, transfer function H(kx) (= OTF cut)
data/na.npz       PSF stack for the NA slider (NA 0.25 ... 0.95)
data/abbe.npz     4f grating images: on-axis period sweep, illumination-tilt sweep, tilted period sweep
data/chart.npz    Siemens star: object, coherent image, incoherent image; OTF and MTF cuts
data/deconv.npz   test object, blurred + noise, naive inverse filter, Wiener filter; radial gain curves
data/aberr.npz    field-dependent PSFs (illustrative Seidel coma/astigmatism/curvature) as a mosaic
"""
import os
import time

import numpy as np
import scipy.fft as sfft
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi
W1 = dict(workers=1)


def fft2(a):
    return sfft.fft2(a, **W1)


def ifft2(a):
    return sfft.ifft2(a, **W1)


def fgrid(n, dx):
    """Spatial-frequency grid (cycles per unit length) in FFT layout."""
    f = np.fft.fftfreq(n, d=dx)
    FX, FY = np.meshgrid(f, f)
    return FX, FY, f[1] - f[0]


def pupil(n, dx, na, lam=1.0, shift=(0.0, 0.0)):
    """Anti-aliased circular pupil |f - shift| <= NA/lambda on an n x n FFT grid (spacing dx)."""
    FX, FY, df = fgrid(n, dx)
    R = np.hypot(FX - shift[0], FY - shift[1])
    return np.clip((na / lam - R) / df + 0.5, 0.0, 1.0)


def psf_fft(n, dx, na, lam=1.0):
    """Incoherent PSF |IFFT(pupil)|^2, normalised to peak 1, FFT layout (origin at [0, 0])."""
    h = ifft2(pupil(n, dx, na, lam))
    p = np.abs(h) ** 2
    return p / p.max()


def centred_crop(a_fftlayout, half):
    """Crop (2*half+1)^2 around the origin of an FFT-layout array (returns a centred array)."""
    a = np.fft.fftshift(a_fftlayout)
    c = a.shape[0] // 2
    return a[c - half:c + half + 1, c - half:c + half + 1]


def shifted_sum(otf, n, dx, pts, amps, x0, y0):
    """Image of point sources sum_j a_j delta(r - r_j) through an incoherent system with this OTF (FFT layout).
    Pixel (i, j) of the result sits at (x0 + j*dx, y0 + i*dx)."""
    FX, FY, _ = fgrid(n, dx)
    S = np.zeros((n, n), complex)
    for (x, y), a in zip(pts, amps):
        S += a * np.exp(-1j * TAU * (FX * (x - x0) + FY * (y - y0)))
    return np.real(ifft2(otf * S))


t_all = time.time()

# ================================================================ hook =========
# 500 nm emission, perfect oil-immersion objective NA 1.4; grid 10 nm, 1024^2 (10.24 um), field shown 8 x 4 um
LAM_H, NA_H, DXH, NH = 0.5, 1.4, 0.01, 1024
psf_h = psf_fft(NH, DXH, NA_H, LAM_H)
otf_h = fft2(psf_h)
x0h = -NH // 2 * DXH                                       # pixel 0 at x = -5.12 um
rng = np.random.default_rng(3)
pts = []
while len(pts) < 30:
    p = np.array([rng.uniform(-3.7, 3.7), rng.uniform(-1.75, 1.75)])
    if all(np.hypot(*(p - q)) > 0.62 for q in pts) and np.hypot(*(p - np.array([2.3, 0.7]))) > 0.9:
        pts.append(p)
pts += [np.array([2.3 - 0.075, 0.7]), np.array([2.3 + 0.075, 0.7])]   # one pair 150 nm apart (merged)
pts = np.array(pts)
amps = rng.uniform(0.7, 1.0, len(pts))
img_h = shifted_sum(otf_h, NH, DXH, pts, amps, x0h, x0h)
i0 = NH // 2 - 400
j0 = NH // 2 - 200
field = img_h[j0:j0 + 400, i0:i0 + 800]                    # rows: y from -2 to 2 um (row 0 at y = -2)
# two beads pushed together: separations 600 -> 100 nm (5 nm steps); crop 2.2 x 1.1 um around the pair
seps = np.round(np.linspace(0.6, 0.1, 101), 4)
pair, prof = [], []
for s in seps:
    im = shifted_sum(otf_h, NH, DXH, [(-s / 2, 0.0), (s / 2, 0.0)], [1.0, 1.0], x0h, x0h)
    c = NH // 2
    pair.append(im[c - 55:c + 56, c - 110:c + 111])
    prof.append(im[c, c - 110:c + 111])
pair = np.array(pair, np.float32)
prof = np.array(prof, np.float32)
r_psf = np.fft.fftshift(psf_h)[NH // 2, NH // 2:NH // 2 + 60]   # radial cut (for checks)
print(f"hook: Airy first zero at {DXH * np.argmin(r_psf[:40]) * 1000:.0f} nm (theory {0.61 * LAM_H / NA_H * 1000:.0f} nm); "
      f"field max {field.max():.2f}")
np.savez_compressed(os.path.join(DATA, "hook.npz"), field=field.astype(np.float32), pts=pts, amps=amps, seps=seps,
                    pair=pair, prof=prof, dx=DXH, lam=LAM_H, na=NA_H)

# ================================================================ stamps =======
# NA 0.5 (Airy radius 1.22 lambda); 9 points in a 10 x 10 lambda field; grid lambda/20, 1024^2
NS, DXS, NA_S = 1024, 0.05, 0.5
psf_s = psf_fft(NS, DXS, NA_S)
otf_s = fft2(psf_s)
x0s = -NS // 2 * DXS
spts = np.array([(-3.0, 2.6), (-0.6, 3.1), (2.2, 2.0), (-2.4, -0.2), (0.4, 0.6), (1.7, 0.1),
                 (3.4, -1.6), (-1.0, -2.6), (0.6, -2.8)])
samps = np.array([1.0, 0.55, 0.85, 0.7, 1.0, 0.6, 0.9, 0.5, 0.8])
c = NS // 2
cum = [np.zeros((201, 201), np.float32)]
acc = np.zeros((NS, NS))
for p, a in zip(spts, samps):
    acc = acc + shifted_sum(otf_s, NS, DXS, [p], [a], x0s, x0s)
    cum.append(acc[c - 100:c + 101, c - 100:c + 101].astype(np.float32))
cum = np.array(cum)
icon = centred_crop(psf_s, 50).astype(np.float32)          # +-2.5 lambda
# transfer function cut H(kx) (incoherent OTF along kx, normalised), from a finer frequency grid (lambda/8, 2048^2)
NO, DXO = 2048, 0.125
otf_fine = np.real(fft2(psf_fft(NO, DXO, NA_S)))
otf_fine /= otf_fine[0, 0]
fo = np.fft.fftfreq(NO, d=DXO)
sel = (fo >= 0) & (fo <= 2 * NA_S * 1.08)
kx_over_k = fo[sel]                                        # kx / k  = f * lambda
Hk = np.clip(otf_fine[0, sel], 0, None)
nu = np.clip(kx_over_k / (2 * NA_S), 0, 1)
mtf_th = 2 / np.pi * (np.arccos(nu) - nu * np.sqrt(1 - nu ** 2))
print(f"stamps: max |H - analytic MTF| = {np.abs(Hk - mtf_th).max():.4f}")
np.savez_compressed(os.path.join(DATA, "stamps.npz"), pts=spts, amps=samps, cum=cum, icon=icon, kx=kx_over_k, H=Hk,
                    dx=DXS, na=NA_S)

# ================================================================ NA slider =====
# PSFs for NA = 0.25 ... 0.95 (step 0.005) on lambda/40 pixels (2048^2 FFT, 51.2 lambda), window +-2.6 lambda
NN, DXN = 2048, 0.025
nas = np.round(np.arange(0.25, 0.9501, 0.005), 4)
half = 104                                                # +-2.6 lambda
stack = np.zeros((len(nas), 2 * half + 1, 2 * half + 1), np.float16)
for i, na in enumerate(nas):
    p = psf_fft(NN, DXN, na)
    stack[i] = centred_crop(p, half)
cut = stack[:, half, half:].astype(np.float32)
zeros = [DXN * np.argmax(np.diff(cut[i]) > 0) for i in range(len(nas))]
err = max(abs(z - 0.61 / na) for z, na in zip(zeros, nas))
print(f"NA stack: {len(nas)} PSFs; max |first zero - 0.61 lambda/NA| = {err:.3f} lambda (pixel {DXN})")
np.savez_compressed(os.path.join(DATA, "na.npz"), nas=nas, stack=stack, dx=DXN, half=half)

# ================================================================ Abbe 4f =======
# sinusoidal amplitude grating t = 0.5 + 0.5 cos(2 pi x / d) on an apodised 40-lambda patch; NA 0.5 pupil;
# plane-wave illumination exp(i 2 pi s_in x) (s_in = sin theta_in); image intensity |IFFT(FFT(t u_in) P)|^2
NB, DXB, NA_B = 1024, 1 / 16, 0.5
xb = (np.arange(NB) - NB // 2) * DXB
XB, YB = np.meshgrid(xb, xb)


def taper(x, a=12.0, b=20.0):
    """Raised-cosine (Tukey) window: flat for |x| <= a, zero beyond b (low sidelobes -> crisp order cutoff)."""
    ax = np.abs(x)
    return np.where(ax <= a, 1.0, np.where(ax >= b, 0.0, 0.5 * (1 + np.cos(np.pi * (ax - a) / (b - a)))))


patch = taper(XB) * taper(YB)                             # 40-lambda apodised patch; the crop shows its flat centre
PB = pupil(NB, DXB, NA_B)
cb = NB // 2
hb = 176                                                  # crop +-11 lambda


def abbe_img(d, s_in):
    t = (0.5 + 0.5 * np.cos(TAU * XB / d)) * patch
    u = t * np.exp(1j * TAU * s_in * XB)
    im = np.abs(ifft2(fft2(np.fft.ifftshift(u)) * PB)) ** 2
    im = np.fft.fftshift(im)[cb - hb:cb + hb:2, cb - hb:cb + hb:2]
    return im.astype(np.float16)


S_TILT = -0.46                                            # 0th order just inside the cap edge (-NA)
d1 = np.round(np.linspace(4.0, 1.4, 131), 4)              # on-axis: cutoff d = lambda/NA = 2 (ends where the tilt sweep starts)
tilts = np.round(np.linspace(0.0, S_TILT, 61), 4)         # d = 1.4 fixed, tilt the illumination
d3 = np.round(np.linspace(1.4, 0.85, 81), 4)              # tilted: cutoff d = 1/(0.46 + 0.5) = 1.04
ab1 = np.array([abbe_img(d, 0.0) for d in d1])
ab2 = np.array([abbe_img(1.4, s) for s in tilts])
ab3 = np.array([abbe_img(d, S_TILT) for d in d3])


def contrast(im):
    r = im[im.shape[0] // 2, 40:-40].astype(float)
    return (r.max() - r.min()) / (r.max() + r.min() + 1e-9)


print("abbe: contrast on-axis d=3,2.2,1.8:", [round(contrast(ab1[np.argmin(abs(d1 - d))]), 3) for d in (3.0, 2.2, 1.8)],
      "| tilt 0,-0.3,-0.46:", [round(contrast(ab2[np.argmin(abs(tilts - s))]), 3) for s in (0.0, -0.3, -0.46)],
      "| tilted d=1.2,1.0:", [round(contrast(ab3[np.argmin(abs(d3 - d))]), 3) for d in (1.2, 1.0)])
np.savez_compressed(os.path.join(DATA, "abbe.npz"), d1=d1, tilts=tilts, d3=d3, ab1=ab1, ab2=ab2, ab3=ab3,
                    s_tilt=S_TILT, na=NA_B, d_fixed=1.4, half=hb * DXB)

# ================================================================ chart =========
# Siemens star (36 periods, radius 30 lambda), lambda/12 pixels, 1024^2; NA 0.5
NC, DXC, NA_C, NSPK, RST = 1024, 1 / 12, 0.5, 36, 30.0
SS = 4                                                    # supersampled rasterisation (area-averaged object)
xs = (np.arange(NC * SS) - NC * SS // 2 + 0.5) * DXC / SS
Xs, Ys = np.meshgrid(xs, xs)
star_ss = ((np.cos(NSPK * np.arctan2(Ys, Xs)) > 0) & (np.hypot(Xs, Ys) <= RST)).astype(np.float32)
star = star_ss.reshape(NC, SS, NC, SS).mean(axis=(1, 3))
del star_ss, Xs, Ys
t_amp = star                                              # binary mask: amplitude = intensity transmittance
PC = pupil(NC, DXC, NA_C)
coh = np.abs(np.fft.fftshift(ifft2(fft2(np.fft.ifftshift(t_amp)) * PC))) ** 2
otf_c = np.real(fft2(psf_fft(NC, DXC, NA_C)))
otf_c /= otf_c[0, 0]
inc = np.real(np.fft.fftshift(ifft2(fft2(np.fft.ifftshift(star)) * otf_c)))
hc_ = 384                                                 # crop +-32 lambda
sl = slice(NC // 2 - hc_, NC // 2 + hc_)
print(f"chart: coherent max {coh.max():.2f} (ringing), incoherent max {inc.max():.2f}; "
      f"cutoff radii coherent {NSPK * (1 / NA_C) / TAU:.2f}, incoherent {NSPK * (0.5 / NA_C) / TAU:.2f} lambda")
# OTF / coherent transfer cuts, kx in units of k*NA
fo2 = np.fft.fftfreq(NO, d=DXO)
otf2 = np.real(fft2(psf_fft(NO, DXO, NA_C)))
otf2 /= otf2[0, 0]
order = np.argsort(fo2)
kk = fo2[order] / NA_C                                    # kx / (k NA)
mtf_cut = np.clip(otf2[0, order], 0, None)
keep = np.abs(kk) <= 2.4
def dec2(a):                                              # 2x2 area average -> display resolution (no moire)
    a = a[sl, sl]
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3)).astype(np.float16)


np.savez_compressed(os.path.join(DATA, "chart.npz"), obj=dec2(star), coh=dec2(coh), inc=dec2(inc), kk=kk[keep],
                    mtf=mtf_cut[keep], dx=2 * DXC, na=NA_C, nspk=NSPK, rstar=RST, half=hc_ * DXC)

# ================================================================ deconvolution ==
# test object: the word "OPTICS" + three bar groups (periods 2.0, 1.3 and 0.8 x the incoherent cutoff period
# lambda/2NA); NA 0.5, lambda/10 pixels, 512^2 (51.2 lambda); seeded Gaussian camera noise
ND, DXD, NA_D = 512, 0.1, 0.5
canvas = Image.new("L", (ND * 2, ND * 2), 0)              # draw at 2x, area-average down
dr = ImageDraw.Draw(canvas)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
dr.text((ND, 280), "OPTICS", fill=255, font=font, anchor="mm")
pc = 1 / (2 * NA_D)                                        # incoherent cutoff period lambda/2NA = 1 lambda
BAR_PER = (2.0 * pc, 1.3 * pc, 0.8 * pc)
for per, xc in zip(BAR_PER, (250, 512, 774)):
    ppx = per / DXD * 2                                    # period in 2x pixels
    for k in range(5):
        x_l = xc - 2.5 * ppx + k * ppx
        dr.rectangle([x_l, 470, x_l + ppx / 2 - 1, 780], fill=255)
rng_d = np.random.default_rng(11)
obj = np.asarray(canvas, np.float32).reshape(ND, 2, ND, 2).mean(axis=(1, 3)) / 255.0
obj = obj[::-1]                                           # row 0 at the bottom (scenes flip again)
psf_d = psf_fft(ND, DXD, NA_D)
otf_d = np.real(fft2(psf_d))
otf_d /= otf_d[0, 0]
otf_d = np.clip(otf_d, 0, None)
blur = np.real(ifft2(fft2(obj) * otf_d))
noise = rng_d.normal(0.0, 0.006 * blur.max(), blur.shape)
meas = blur + noise
Fm = fft2(meas)
inside = otf_d > 1e-9
naive = np.real(ifft2(np.where(inside, Fm / np.where(inside, otf_d, 1), 0)))
K = 2e-3                                                  # 1/SNR (noise-to-signal power ratio)
G = otf_d / (otf_d ** 2 + K)
wien = np.real(ifft2(Fm * G))
fd = np.fft.fftfreq(ND, d=DXD)
rsel = (fd >= 0) & (fd <= 2.4 * NA_D)
kxd = fd[rsel] / (2 * NA_D)                               # kx in units of the incoherent cutoff 2 k NA
Hd = otf_d[0, rsel]
print(f"deconv: blur max {blur.max():.3f}, noise sd {noise.std():.4f}, naive range [{naive.min():.1f}, {naive.max():.1f}], "
      f"wiener range [{wien.min():.2f}, {wien.max():.2f}], max Wiener gain {G.max():.1f}")
# fine radial curves for the log plot (analytic MTF of the same circular pupil, verified against the FFT OTF)
nu_f = np.linspace(0, 1.2, 600)
mtf_f = np.where(nu_f < 1, 2 / np.pi * (np.arccos(np.clip(nu_f, 0, 1)) - np.clip(nu_f, 0, 1) * np.sqrt(np.clip(1 - nu_f ** 2, 0, None))), 0)
nu_c = np.clip(kxd, 0, 1)
mtf_c = np.where(kxd < 1, 2 / np.pi * (np.arccos(nu_c) - nu_c * np.sqrt(1 - nu_c ** 2)), 0)
print(f"deconv: max |FFT OTF - analytic MTF| on the kx axis = {np.abs(Hd - mtf_c).max():.4f}")
np.savez_compressed(os.path.join(DATA, "deconv.npz"), obj=obj, meas=meas.astype(np.float32), naive=naive.astype(np.float32),
                    wien=wien.astype(np.float32), blur=blur.astype(np.float32), nu=nu_f, mtf=mtf_f, K=K, kxd=kxd, Hd=Hd,
                    dx=DXD, na=NA_D)

# ================================================================ aberrations ====
# field-dependent PSF: W(rho, phi; h) = W131 h rho^3 cos(phi - phi_h) + W222 h^2 rho^2 cos^2(phi - phi_h) + W220 h^2 rho^2
# (waves), pupil radius 32 px on a 512 grid -> PSF pixel lambda/(16 NA); 5 x 3 field points
NA_A, NP_, RP = 0.5, 512, 32
u = (np.arange(NP_) - NP_ // 2) / RP
U_, V_ = np.meshgrid(u, u)
rho, phi = np.hypot(U_, V_), np.arctan2(V_, U_)
inpup = (rho <= 1).astype(float)
W131, W222, W220 = 0.6, 0.5, 0.25
tiles, fields = [], []
for hy in (0.6, 0.0, -0.6):
    for hx in (-1.0, -0.5, 0.0, 0.5, 1.0):
        h = np.hypot(hx, hy)
        ph = np.arctan2(hy, hx)
        W = W131 * h * rho ** 3 * np.cos(phi - ph) + W222 * h ** 2 * (rho * np.cos(phi - ph)) ** 2 + W220 * h ** 2 * rho ** 2
        E = np.fft.fftshift(sfft.fft2(np.fft.ifftshift(inpup * np.exp(1j * TAU * W)), **W1))
        I = np.abs(E) ** 2
        tiles.append(I)
        fields.append((hx, hy))
peak0 = max(t.max() for t in tiles)
mos = np.zeros((3 * 80, 5 * 80), np.float32)
for k, I in enumerate(tiles):
    r, cc = divmod(k, 5)
    cy, cx = NP_ // 2, NP_ // 2                            # paraxial image point at the tile centre
    tile = I[cy - 40:cy + 40, cx - 40:cx + 40] / peak0
    mos[r * 80:(r + 1) * 80, cc * 80:(cc + 1) * 80] = tile
print(f"aberr: tile peaks (rel. to centre) min {min(t.max() for t in tiles) / peak0:.2f}")
np.savez_compressed(os.path.join(DATA, "aberr.npz"), mosaic=mos[::-1].copy(), fields=np.array(fields),
                    w=np.array([W131, W222, W220]))
print(f"physics done in {time.time() - t_all:.1f}s")
