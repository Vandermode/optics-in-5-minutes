"""Precompute every computed field of E01 (the angular spectrum method) -> data/*.npz.
    python physics.py        (numpy only, ~10 s, one thread)

These are the pilot's in-scene computations (angular_spectrum_method_5min/asm5_scenes.py), moved here unchanged so that
every render piece only loads them. All fields use the exact scalar angular spectrum method (numpy FFTs, exact k_z).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU_ = 2 * np.pi


def asm_rows(u0, dx, zs, lam=1.0):
    """Angular-spectrum propagation of a 1D field u0 (spacing dx) to every z in zs -> complex (nz, N)."""
    k = TAU_ / lam
    kx = TAU_ * np.fft.fftfreq(len(u0), d=dx)
    kz = np.sqrt((k ** 2 - kx ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    F0 = np.fft.fft(u0)
    return np.array([np.fft.ifft(F0 * np.exp(1j * kz * z)) for z in zs])


def slit_map(a=12.0, x_half=25.0, z_max=150.0, nx=1215, nz=607, N=8192, dx=0.1):
    xg = (np.arange(N) - N // 2) * dx
    u0 = (np.abs(xg) <= a / 2).astype(complex)
    zs = np.linspace(0, z_max, nz)
    xs = np.linspace(-x_half, x_half, nx)
    U = asm_rows(u0, dx, zs)[:, np.searchsorted(xg, xs)]
    return xs, zs, np.abs(U)


def lens_map(D=34.0, f=55.0, x_half=25.0, z_max=110.0, nx=1215, nz=607, N=8192, dx=0.1):
    xg = (np.arange(N) - N // 2) * dx
    ap = 0.5 * (1 - np.tanh((np.abs(xg) - D / 2) / 0.6))
    u0 = ap * np.exp(-1j * TAU_ * (np.sqrt(xg ** 2 + f ** 2) - f))   # exact hyperbolic lens phase
    zs = np.linspace(0, z_max, nz)
    xs = np.linspace(-x_half, x_half, nx)
    U = asm_rows(u0, dx, zs)[:, np.searchsorted(xg, xs)]
    return xs, zs, np.abs(U)


def hook():
    """Ch0: 8-wavelength slit, |U(x, z)| for z = 0..240 lambda (rows: +x at top; cols: z)."""
    N, dx = 8192, 0.1
    xg = (np.arange(N) - N // 2) * dx
    u0 = (np.abs(xg) <= 4.0).astype(complex)
    nz, nxr = 1215, 702
    zs, xs = np.linspace(0, 240, nz), np.linspace(-40, 40, nxr)
    A = np.abs(asm_rows(u0, dx, zs)[:, np.searchsorted(xg, xs)]).T[::-1]
    np.savez(os.path.join(DATA, "hook.npz"), A=A)


def diffraction_limit():
    """Ch6: two point-like details 2 and 0.3 wavelengths apart, before and after cutting the spectrum at |k_x| = k."""
    Nn, dxx = 8192, 0.01
    xg = (np.arange(Nn) - Nn // 2) * dxx
    kxg = TAU_ * np.fft.fftfreq(Nn, d=dxx)
    out = {}
    for tag, s_ in (("wide", 2.0), ("close", 0.3)):
        u0 = np.exp(-((xg - s_ / 2) / 0.04) ** 2) + np.exp(-((xg + s_ / 2) / 0.04) ** 2)
        ub = np.abs(np.fft.ifft(np.fft.fft(u0) * (np.abs(kxg) <= TAU_)))
        sel = np.abs(xg) <= 3.0
        out[f"x_{tag}"], out[f"a0_{tag}"], out[f"a3_{tag}"] = xg[sel], np.abs(u0[sel]), ub[sel]
    np.savez(os.path.join(DATA, "difflimit.npz"), **out)


def algorithm():
    """Ch7: step-3 profile, wrap-around vs zero padding, and the slit and lens maps."""
    xs4, _, A4 = slit_map(z_max=60.0, nz=2)
    dxw = 0.05
    Nw = int(20 / dxw)
    xw = (np.arange(Nw) - Nw // 2) * dxw
    beam = lambda x: np.exp(-((x + 3.0) / 2.0) ** 2) * np.exp(1j * TAU_ * np.sin(np.radians(22)) * x)
    zsw = np.linspace(0, 40, 300)
    Anp = np.abs(asm_rows(beam(xw).astype(complex), dxw, zsw))
    Np = 8 * Nw
    xp = (np.arange(Np) - Np // 2) * dxw
    Ap = np.abs(asm_rows(beam(xp).astype(complex), dxw, zsw))[:, Np // 2 - Nw // 2: Np // 2 + Nw // 2]
    _, _, A6 = slit_map(nx=1215, nz=594)
    _, _, A7 = lens_map(nx=1215, nz=594)
    np.savez(os.path.join(DATA, "algorithm.npz"), xs4=xs4, prof=A4[-1], Anp=Anp, Ap=Ap, A6=A6, A7=A7)


if __name__ == "__main__":
    hook()
    diffraction_limit()
    algorithm()
    print("wrote", sorted(os.listdir(DATA)))
