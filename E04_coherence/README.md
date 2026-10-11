# Coherence: when does light interfere? (Optics in 5 Minutes · E04, 5:01)

3Blue1Brown-style explainer (5:01, 1920×1080, 60 fps) for viewers who know calculus and a little linear algebra.
Narration by an open-source TTS voice (Kokoro-82M), subtitles timed to the speech with Whisper.

* `e04_coherence_1080p60.mp4` – subtitles burned in
* `e04_coherence_1080p60_softsubs.mp4` – clean picture + toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass` – subtitles

The idea in one sentence (the recap line): *fields add when their phases stay in step; when phases wander,
intensities add. Partially coherent light is a sum of coherent patterns.*

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a laser spot speckles, an LED spot is smooth; two slits: crisp fringes (laser), none (LED), fringes back with a pinhole (LED) |
| 1 | 0:19 | interference is arrow addition: the camera sees \|a₁+a₂\|² = \|a₁\|²+\|a₂\|²+2\|a₁\|\|a₂\| cos Δφ; Δφ across a screen paints fringes; fast phase jitter averages the cross term away; visibility V |
| 2 | 1:08 | colour (temporal coherence): every colour has its own fringe spacing; summed over a spectrum, fringes survive only near zero delay; envelope = Fourier transform of the spectrum, ℓc ≈ λ²/Δλ; laser / LED / sunlight |
| 3 | 1:48 | size (spatial coherence): a tilted source point shifts the fringes; independent points add shifted copies, which wash out when θs ≈ λ/d (signature shot); a pinhole or distance restores them; van Cittert–Zernike; Michelson & Pease, Betelgeuse (1920) |
| 4 | 2:45 | speckle: each pixel is a random walk of phasors (signature shot); contrast 1; grain ≈ λ/NA; averaging N independent patterns → 1/√N |
| 5 | 3:22 | coherent modes: mutual intensity J(x₁,x₂) = ⟨U(x₁)U*(x₂)⟩ is a Hermitian matrix; eigenvectors = coherent modes, eigenvalues = their powers; I(x) = Σ λn\|φn(x)\|²; one mode for a laser, more for a bigger source |
| 6 | 4:08 | on a computer: three recipes (coherent, incoherent, partially coherent); a diffractive element's far field under a laser (crisp, speckled) and an LED (shifted + stretched copies: blurred, smooth); small-angle shortcut vs a few modes |
| 7 | 4:43 | recap, teaser for E05 (the phase problem: holography and phase retrieval) |

## What is computed, what is schematic
All fields are computed in `physics.py` (numpy, fixed seeds, ~10 s single-threaded) → `data/*.npz`; `scenes.py` only
loads them. Generic demos only: no unpublished group results or numbers.

| Panel | How |
|---|---|
| hook: laser vs LED spot | Gaussian spot on a wall with a uniformly random phase per pixel (seeded), imaged through a circular pupil: laser = \|field ⊛ h\|² (coherent), LED = \|amplitude\|² ⊛ \|h\|² (incoherent) |
| hook: double-slit fringes | Fraunhofer two-slit pattern (a = 10 µm, d = 250 µm, 530 nm). LED = average over a uniform 1 mm source 50 mm away (241 tilts) and a 40 nm FWHM Gaussian spectrum (41 wavelengths); with a 20 µm pinhole the visibility returns (0.94), limited at the sides by the spectrum |
| Ch1 fringes vs jitter, visibility dial | Monte Carlo average (20,000 seeded Gaussian samples) of \|a₁ + a₂e^{i(Δφ+σg)}\|²; V = \|⟨e^{iσg}⟩\| (matches e^{−σ²/2}); the live jitter is a seeded smoothed noise sequence |
| Ch2 interferograms, envelope, ℓc | I(ΔL) = ∫S(ν̃)[1 + cos 2πν̃ΔL]dν̃ for Gaussian spectra (in wavenumber) at 550 nm, FWHM 1–350 nm (90 tables); envelope = \|FT of the spectrum\|; the marked ℓc is the full 1/e width of the envelope (≈ 1.06 λ²/Δλ for a Gaussian spectrum — "roughly", the prefactor depends on the spectral shape). Presets: laser (1.2 nm, a stand-in: the 12 µm window cannot show cm–km), LED 40 nm, sunlight 300 nm (Gaussian model) |
| Ch3 shifted copies and their sum | two-slit cos² fringes shifted by d·α/λ for 11 equally spaced, mutually incoherent source points (exact for that model; the single-slit envelope is omitted); the sum and the visibility dial are the exact average; at θs = λ/d the copies tile one fringe period and the sum is flat |
| Ch3 visibility vs separation | uniform strip source: \|FT\| computed numerically (sinc); Betelgeuse: uniform disk \|2J₁(x)/x\| with 0.047″ at λ = 575 nm → first zero at 3.08 m |
| Ch4 speckle, random walk | far field of 40 seeded random scatterers in a disk of radius 20λ; the random walk draws exactly those 40 unit phasors for the probed direction, so its end-to-end length is that pixel's amplitude; histogram vs e^{−I/⟨I⟩}; contrast of averages of N = 1…64 independent patterns vs 1/√N; the λ/NA bar uses NA = (spot radius)/(distance) |
| Ch5 J matrix, modes, powers | J(x₁,x₂) = a(x₁)a(x₂) sinc((x₁−x₂)/ℓ): a soft-edged slit lit by a uniform incoherent source (van Cittert–Zernike), 128 samples, 48 source sizes, `eigh` → modes and powers; the live field strip is Σ√λn cn(t) φn(x) with seeded smoothed complex Gaussian coefficients (its ensemble average is J) |
| Ch6 diffractive element, laser vs LED | phase-only diffractive optical element (128², 80 Gerchberg–Saxton iterations, seeded random start), far field by a 4× zero-padded FFT; LED = average of the far fields under 61 tilted plane waves (hexagonal sampling of a disk source: each tilt computed by its own FFT, i.e. shifted copies), then a 25-wavelength average (35 nm FWHM at 530 nm) of the λ-scaled pattern (paraxial; the element's phase-depth change with λ is ignored) |

Schematic (drawn, not computed): the hook's laser/LED/slit/pinhole icons; the Ch1 phasor diagram (exact algebra,
illustrative arrows; the spin rate is slowed down); the Ch2 "beam / delayed copy" waves; the Ch3 geometry (rays,
wavefronts, pinhole plate) and the Michelson sketch (mirrors drawn as slits); the Ch4 rough wall, rays and pixel
(the 40 scatterers are drawn along a 1D wall although the computed ones lie in a 2D disk); the Ch6 code card and
source/spectrum insets; the Ch7 recap icons. Models are scalar and paraxial (far field) throughout; the hook and
Ch6 panels are normalised per panel for display.

## Rebuild
```bash
../common/audio.sh .                          # narration (Kokoro TTS) -> Whisper check -> subtitles
python physics.py    # all computed fields -> data/*.npz (seeded, ~10 s)
./preview.sh [Chapter ...]                    # low-res chapter previews (2 Manim processes)
./render.sh --skip-audio                      # final render (3 workers, windowed parallel renderer)
python ../common/verify.py .   # counts, seam scan, loudness, stills -> .verify/
```
Verified (`../common/verify.py`): 18,070 frames, 90 cues, −16.3 LUFS; the only seam-scan jump (79.3 s) is the intended
Chapter-2 transform. Scene-sync note: `scenes.py` anchors animations to spoken words (`W(segment, word)`, from Whisper's `audio/words.json`);
if a word is not found it falls back to the segment start and prints a warning to stderr.

Re-rendered 2026-10-11 after `common/o5common.py` got the whole-frame `Chapter.at()` (FAST_ANIMATION_PIPELINE.md §10): 18,070 frames, all counts OK, −16.3 LUFS.
