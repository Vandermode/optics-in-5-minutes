# Optics in 5 Minutes

Narrated 3Blue1Brown-style explainers of the big ideas in computational optics and photonics. Each is about five
minutes, 1920×1080 at 60 fps, made with Manim CE and Kokoro-82M narration, with subtitles both burned in and as a
soft track. Each film teaches one idea from scratch, assuming calculus and a little linear algebra. Every physical
field on screen is computed with the method the episode teaches, and each episode's README says which panels are
computed and which are schematic.

**Watching.** This repository holds the *source* only. The finished films are attached to the GitHub Releases (tag
`season1-…`), or can be rebuilt from the source in about 3–9 minutes per episode.

## Season 1: the ten ideas

| # | Folder | Length | The idea |
|---|---|---|---|
| E01 | [`E01_angular_spectrum`](E01_angular_spectrum/) | 5:07 | Propagation is diagonal in the plane-wave basis: FFT, multiply, inverse FFT. |
| E02 | [`E02_lens_fourier`](E02_lens_fourier/) | 5:12 | Far away, or at a lens's focus, each direction lands on its own spot. The 4f machine and phase contrast. |
| E03 | [`E03_diffraction_limit`](E03_diffraction_limit/) | 5:11 | Imaging is convolution with the PSF; the lens is a hole in k-space (λ/2NA). Coherent vs incoherent imaging, Wiener filtering. |
| E04 | [`E04_coherence`](E04_coherence/) | 5:01 | Fields add when phases stay in step, intensities when they wander. Speckle; coherent modes. |
| E05 | [`E05_phase_problem`](E05_phase_problem/) | 5:05 | Cameras lose phase. Off-axis holography and Gerchberg–Saxton recover it; computer-generated holograms design it. |
| E06 | [`E06_flat_optics`](E06_flat_optics/) | 5:15 | Thin elements and kinoforms: phase wraps at 2π. Steps, colour, and where "thin" breaks. |
| E07 | [`E07_periodic_structures`](E07_periodic_structures/) | 5:19 | Symmetry fixes momentum: Snell's law, gratings, Bloch waves and band gaps, RCWA. |
| E08 | [`E08_trapping_light`](E08_trapping_light/) | 5:06 | Waveguide modes as eigenvectors; couplers as 2×2 rotations; ring resonators, Q and critical coupling. |
| E09 | [`E09_maxwell_grid`](E09_maxwell_grid/) | 5:15 | FDTD: E and H leapfrog on the Yee grid. The Courant limit, grid dispersion, and the PML as complex-stretched space. |
| E10 | [`E10_adjoint_design`](E10_adjoint_design/) | 5:15 | The adjoint method: two simulations give every pixel's gradient. Differentiable and end-to-end optics. Season recap. |

The total running time is about 52 minutes. `OUTLINE.md` explains why these ten ideas were chosen, and lists the
candidates for Season 2.

## Setup (once per machine)

You need:
- conda, mamba or micromamba;
- a C compiler (gcc, or `xcode-select --install` on macOS);
- a LaTeX distribution with `dvisvgm`;
- FFmpeg built with libass, which the burned-in subtitles need. conda-forge's FFmpeg lacks it; the Ubuntu and
  Homebrew packages have it.

On Ubuntu: `sudo apt install build-essential texlive texlive-latex-extra dvisvgm ffmpeg`. On macOS:
`xcode-select --install`, `brew install ffmpeg`, and MacTeX.

```bash
./setup_env.sh              # creates the `optics5min` environment, builds ManimPango, downloads the speech models (~800 MB)
conda activate optics5min   # or micromamba/mamba activate; or set O5_PY in config.local.sh
make check-env              # lists anything still missing, with a hint for each
```

`environment.yml` pins everything else: Manim 0.22, Kokoro TTS, faster-whisper, numpy, scipy and JAX. If the
`ffmpeg` found on PATH lacks libass, `make check-env` says so. Point `O5_FFMPEG` at one that has it in
`config.local.sh` (copy `config.local.sh.example`). A GPU is optional; it only speeds up some
`physics.py` runs.

## Building

```bash
make episode E=E03          # data -> narration -> render -> verify; skips every step that is already up to date
make preview E=E03 CH=Ch3Abbe     # low-res previews of chosen chapters, for layout work
make release E=E03          # the shareable copy: dist/<slug>_1080p.mp4 (about half the render's size)
make help                   # all targets (FORCE=1 reruns a step even if it is up to date)
```

Builds always take the fastest route:

- **Incremental.** Each step is a file target, so it reruns only when its inputs changed.
  - Physics reruns when `physics.py` changes.
  - Narration reruns when `narration.json` changes; only the edited sentences are re-synthesised, and the Whisper
    check is skipped when the timing comes out identical.
  - The render reruns when the scenes, data, audio or renderer change.

  A narration edit therefore never recomputes physics.
- **Physics on a GPU, or not at all.** `physics.py` files that only use numpy run locally, in seconds to two
  minutes. The JAX ones (E02, E06, E07, E09, E10) run on the least-busy NVIDIA GPU; `setup_env.sh` installs CUDA
  JAX when it sees one. Measured on an A100: E02 54 s, E06 63 s, E07 60 s, E09 30 s, and E10 170 s (mostly its
  single-core sparse solver). Each uses about one CPU core. On a machine without a GPU, their precomputed data are
  downloaded from the release (`O5_DATA_URL` in `config.local.sh`, bundles made with `make pack-data`). Computing
  them on the CPU is the last resort (8 cores):

  | Episode | `physics.py` on CPU |
  |---|---|
  | E02 | about 50 min |
  | E06 | about 12 min |
  | E07, E10 | about 5 min |

- **Parallel render.** It uses `WORKERS` processes (default 6; keep at most about 8 on a shared server).

Measured on a fresh clone: `make episode E=E01` takes about 7 minutes. The rebuilt film matches the original (51–67
dB PSNR), and the regenerated timing and subtitles are byte-identical. `E` takes a folder name or just its prefix.

## Layout

| Path | Contents |
|---|---|
| `E0N_slug/` | one episode: `narration.json` (script), `physics.py` (computed fields → `data/`), `scenes.py` (one Manim `Chapter` per chapter), `episode.json`, `README.md` |
| `common/` | the shared pipeline: series library `o5common.py`, TTS and subtitles, the windowed fast renderer, `verify.py`, fonts |
| `PRODUCTION_SPEC.md` | quality bar, workflow and acceptance checklist for an episode |
| `CONTRIBUTING.md` | how to edit a narration, add an episode, and share results |
| `docs/FAST_ANIMATION_PIPELINE.md` | how the renderer works and why; scene rules every chapter must follow |

Git tracks only what people write. These small generated files are also tracked, so that reviews show their
changes:
- `audio/timing.json` and `audio/words.json`;
- `subtitles.srt` and `subtitles.ass`.

Everything else is rebuilt by `make` and ignored by git: films, audio, `data/`, caches, models and `dist/`.

## Production notes (Season 1, 2026-10-10/11)

- **Verification.**
  - Every film passed `verify.py`: exact frame count against `round(total·60)`, soft-subtitle packets, −16.2 to
    −16.3 LUFS.
  - Every seam-scan jump was inspected and is an intended transition.
  - The Whisper round trips match the scripts.
- **E01 port.** E01 is the series pilot ported onto `common/`. It is pixel-identical to the original (`--dump`
  test, 144 frames, maximum difference 0).
- **No unpublished data.** Comparisons such as thin element vs full wave (E06, E09, E10) come from small solvers
  written for the episodes (FDFD, FDTD, RCWA), each validated in its README.
