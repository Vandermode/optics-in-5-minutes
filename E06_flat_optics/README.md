# Flat Optics: a Lens One Micron Thick (Optics in 5 Minutes · E06, 5:15)

3Blue1Brown-style explainer (5:15, 1920×1080, 60 fps) for someone who knows basic calculus and a little linear
algebra. Narration by an open-source TTS voice (Kokoro-82M, `af_heart`), subtitles timed to the spoken words.

* `e06_flat_optics_1080p60.mp4`: subtitles burned in
* `e06_flat_optics_1080p60_softsubs.mp4`: clean picture plus a toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass`: subtitles

**One idea:** A thin element only delays light. Phase wraps at 2π, so a lens can be a micron thick, until "thin"
stops being true.

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a 9 µm glass lens and a 1 µm patterned plate focus green light to the same spot; in white light the plate smears the colours along its axis (both computed) |
| 1 | 0:23 | glass as a delay: crests bunch up in glass (speed c/n) and lag by (n−1)h, so φ = k₀(n−1)h; thin element approximation (TEA) U_out = U_in·e^{iφ(x)}; a computed live wave through a TEA lens; the model ladder |
| 2 | 1:13 | wrap the phase: a full turn (2π) is free, which takes a height λ/(n−1) ≈ 1.1 µm. **Signature shot:** the lens is sliced into slabs, the full slabs vanish and the curved remainders drop onto the base as a kinoform (Fresnel zones, top view computed); zones shrink from ≈25λ to ≈2.5λ; e^{iφ_kino} = e^{iφ_lens}·e^{−2πim}, so the computed foci and focal-plane profiles are identical |
| 3 | 2:06 | steps cost light: an N-level staircase; computed far-field diffraction orders; the phasor chain (gap to the perfect ramp, head to tail) curls and shortens; η = sinc²(1/N): 41, 81, 95, 99 % for N = 2, 4, 8, 16 |
| 4 | 2:54 | the rainbow problem: sin θ = λ/zone width; f = f₀λ₀/λ (computed on-axis foci: blue 18 µm farther than red), against glass (blue 0.9 µm closer); Abbe numbers −3.45 and ≈ +60; opposite signs let a weak diffractive surface cancel a glass lens's colour error; or correct in software |
| 5 | 3:36 | where "thin" breaks: TEA's column-by-column picture against a full-wave Maxwell solution of 4λ-wide zones with 2λ walls: light crowds along the walls, leaves in jets, and leaves shadows; 100 % (TEA) against 71 % (computed) to the focus order; efficiency against zone width (computed); way out 1: metasurfaces (computed pillar delay library); way out 2: full-wave solvers (Episode 9), shown on the model ladder |
| 6 | 4:26 | on a computer: four lines (wrap, multiply by e^{ik₀(n−1)h}, angular spectrum, \|U\|²) with a computed mini pipeline; pitfalls: sample the finest zone, wrap the heights, simulate the fabricated profile. A 0.24 µm blur costs 25 % of the focused light (computed) |
| 7 | 4:52 | recap (delay → wrap → one micron, until "thin" breaks; the price: steps, colour, fine zones); next: periodic structures |

## What is computed, and how (`physics.py` → `data/*.npz`; scenes only load)
* **Microlens and kinoform** (hook, Ch2, Ch4, Ch6): circular lens, radius 20 µm, f₀ = 43.25 µm at λ₀ = 550 nm (NA 0.42,
  exactly 8 zones, sag 8.8 µm, kinoform height λ₀/(n−1) = 1.1 µm). The glass is BK7-like with Cauchy dispersion,
  n(550 nm) = 1.50 and Abbe number 63.8. The thick lens has the ideal TEA phase, and the kinoform is its height modulo
  λ₀/(n−1). The fields come from TEA (U·e^{ik(n(λ)−1)h}) followed by **exact 2D angular-spectrum propagation**
  (1024² grid over 80 µm, JAX on GPU); the x–z slice through the axis is shown. The aperture has a 2 µm raised-cosine
  edge. White light is 27 wavelengths from 420 to 680 nm with equal energy, mapped to true colour with the CIE 1931
  colour-matching functions (Wyman–Sloan–Shirley fit) and sRGB, with a luminance tone map. Checks: thick-lens and
  kinoform fields at 550 nm are identical (difference 0); kinoform foci at 450/550/650 nm are 53.4/43.0/35.8 µm
  (f₀λ₀/λ = 52.9/43.3/36.6); glass foci are 42.5/43.0/43.3 µm.
* **Ch1 live wave**: a wavelength-scale cylindrical TEA lens. Inside the lens the field is the TEA column picture
  (each column advances with its own index), and above the exit plane it is propagated with 1D ASM. The film shows
  Re{U e^{−iωt}}.
* **Ch1 crests (two paths)**: analytic phase of a plane wave in air and in a glass block (exact).
* **Ch3 orders**: FFT of e^{iφ_N(x)} over one period of the N-level staircase (|c_m|², equal to sinc²(1/N) for m = +1);
  the phasor chains are the exact discrete sums of e^{i(φ_N − φ_ramp)}.
* **Ch5 full wave**: a new **2D finite-difference frequency-domain (FDFD) Maxwell solver** (sparse LU), periodic in x,
  with a stretched-coordinate PML in z and 40 cells per λ. The structure is a glass blazed grating (n = 1.5, wall height
  λ/(n−1) = 2λ) lit at normal incidence from the substrate side. The field shown is TE (E along the grooves). Efficiency
  is the exactly conserved discrete Poynting flux per diffraction order, normalised to the incident flux. Checks: T + R = 1
  to < 10⁻⁵; a flat glass–air interface gives T = 0.9607 (Fresnel 0.9600); converged to ±0.2 % with resolution.
  - 4λ zones: 70.6 % (TE, shown) and 76.2 % (TM) of the incident light go into the focusing order, against TEA's 100 %
    (TEA also ignores the ≈4 % Fresnel reflection).
  - Sweep (TE, solid; TM, dashed) from 2.4λ to 20λ zones: 29→92 % (TE) and 55→92 % (TM).
  - The TEA panel is the same TEA column picture plus an exact tilted plane wave above the teeth.
* **Ch5 metasurface library**: the same FDFD solver, TE, with TiO₂-like ridges (n = 2.4, 0.66 µm tall, period 0.33 µm)
  on glass. It gives the 0th-order transmission phase against ridge width (0 → 1.1 turns for 33–200 nm; the first
  resonance is above that range).
* **Ch4 colour cancellation**: thin-lens powers of the same glass plus a diffractive surface (power ∝ λ) that holds
  5.1 % of the power, chosen so that d(power)/dλ = 0 at 550 nm.
* **Ch6 blur**: the kinoform height map is blurred with a Gaussian of σ = 0.1 µm (FWHM 0.24 µm), then TEA + 2D ASM. The
  power within 1.5 µm of the focus drops by 25 % (0.915 → 0.684 of the incident power).

**Schematic (not data):** the slice-and-drop animation (exact zone geometry, heights drawn ×2), the phase dial and
wave strip, the diffraction-fan rays (exact angles for a 1.35 µm zone), the pillar row and its phase dials (widths read
from the computed library), the model ladder, the code card, and the recap icons.

**Simplifications** (stated here as the spec requires):
* The lens fields use scalar TEA; the full-wave check is 2D (infinitely long grooves and ridges), not 3D.
* sinc²(1/N) assumes an ideal blaze in scalar theory.
* "Pillars" in the metasurface library are 2D ridges.

## Verification (final render)
* `render.sh --skip-audio`: all 47 pieces OK; `verify.py`: 18,888 frames (= round(314.801 s × 60)), duration 314.801 s,
  88 subtitle cues, integrated loudness −16.2 LUFS.
* The seam scan flags one group, 246.35–246.52 s. It is the intended 0.5 s fade-out of the two bright full-wave panels
  (a smooth bell of frame differences) and is not near any piece start. Small-region checks of the moving redraws
  (crests, live waves, the dial, the panel slide, the curve marker) show no snaps.
* `scenes.py` overrides `Chapter.at()` with the whole-frame version (FAST_ANIMATION_PIPELINE.md §10), because
  `common/` still had the pre-fix float wait.
* Whisper round trip: every remaining mismatch is a number, a spelling ("color"), or a word assigned to the
  neighbouring segment. The voice spells "kinoform" as "kin-oh-form", "Abbe" as "Ahbuh", and TEA as "T, E, A".

## Rebuild
```bash
python physics.py      # ~2 min: 2D ASM on a GPU (JAX), FDFD with scipy sparse LU (single core)
../common/audio.sh .                            # Kokoro TTS -> Whisper check -> subtitles
./preview.sh                                    # low-res chapter previews (2 processes)
./render.sh --skip-audio                        # final render, 3 workers
python ../common/verify.py .
```
