# Optics in 5 Minutes · E07 · Periodic Structures: Gratings, Bloch Waves & Band Gaps (5:23)

3Blue1Brown-style explainer (5:23, 1920×1080, 60 fps) for viewers with basic calculus and a little linear algebra.
Narration by an open-source TTS voice (Kokoro-82M, `af_heart`), subtitles timed to the speech.
One idea: **symmetry fixes momentum**. Flat interfaces conserve kx. Periodic ones conserve it only up to multiples of
K = 2π/Λ, and where copies meet, gaps open.

* `e07_periodic_structures_1080p60.mp4`: subtitles burned in
* `e07_periodic_structures_1080p60_softsubs.mp4`: clean picture plus a toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass`: subtitles

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: white light through a fine grating fans into rainbows; a 10-pair layer stack reflects one colour (both computed). "How can mere structure pick a direction, and a colour?" |
| 1 | 0:21 | a flat boundary: recall sin θ = kx/k; k = n·k0, so each medium has its own k-circle; crests must meet at the boundary, so kx is shared (conserved); one vertical line through both circles = Snell's law; tilt past the critical angle and the line misses the air circle: total internal reflection with an evanescent tail |
| 2 | 1:20 | a periodic boundary: anything repeating every Λ is a sum of e^(imKx), K = 2π/Λ; multiplying shifts kx by mK; copies of the line spaced by K; every copy that crosses the circle is a diffraction order (computed beams); red has the smaller circle, so it bends most (rainbow); shrink Λ and the orders leave one by one; at normal incidence only the transmitted zero order propagates into air when Λ < λair; internal resonances can remain, so a uniform-film approximation needs an independent check |
| 3 | 2:25 | Bloch waves and band gaps: shift a stack by one period and nothing changes, so waves gain a fixed phase e^(iκΛ) per period (computed Bloch wave, phasor dials); κ only matters modulo K; light line ω = cκ/n and its copies shifted by K fill one zone; they cross at the zone edge, λ = 2nΛ (Bragg); **signature shot:** the index-contrast slider Δn opens an avoided crossing, the band gap, while the computed reflectance of a 10-pair stack grows a stop band at the same frequencies; inside the gap the field dies off within a few layers |
| 4 | 3:31 | RCWA, one layer at a time: a thick grating's orders (computed); the field in a layer is one amplitude per order; uniform layer: diagonal matrix, each order picks up e^(ikz,m z); grating layer: the Fourier coefficients ε_(m−p) fill the matrix; eig gives the Bloch modes and their kz (computed); chain layers with S-matrices (only decaying exponentials, so stable) |
| 5 | 4:19 | on a computer: the grating equation is one line and RCWA four (expand ε, build the matrix, eig, chain); pitfall: truncation (computed convergence; TM needs Li's factorization rules); payoff: zero-order phase vs ridge width = a metasurface unit-cell library (computed), picked cell by cell into a flat lens |
| 6 | 4:52 | recap: the one-sentence idea with three icons; where it breaks: RCWA needs true periodicity and a metasurface lens only borrows it locally (local periodic approximation); teaser: TIR traps light in a slab, Bragg mirrors between them; next: trapping light, waveguides, modes and resonators |

## What is computed and what is schematic
All numerics are in `physics.py` (deterministic, no random numbers) and are written to `data/*.npz`. The scenes only
load and draw.

**Methods and checks** (printed by `physics.py`):
* `rcwa_te`: a compact numpy RCWA for 1D gratings, TE polarisation (E along the grooves), in the S-matrix formulation
  (Redheffer star product, Hermitian eigen-solver for lossless layers). Checks: a uniform film matches the Airy
  formula to 12 digits; lossless gratings conserve energy, 1 − ΣR − ΣT < 2·10⁻¹⁵; a thin sinusoidal relief reproduces
  the Bessel efficiencies J_m²(φ₀) of the thin-element limit to about 1 %.
* `tmm_*`: 2×2 transfer matrices at normal incidence: Bloch bands from cos κΛ = ½ tr M, stack reflectance, fields
  inside the stack, and the Bloch mode of the infinite lattice (eigenvector of the unit-cell matrix).
* Field maps: RCWA order amplitudes for every wavelength, exact angular-spectrum propagation (E01; JAX on a GPU), and
  intensities turned into colour with the CIE 1931 observer → sRGB.

**Computed panels**
* Hook, left; Ch2 maps: white light (31 wavelengths, 400–700 nm; 16 for the sweep) through a sinusoidal glass relief
  (period 1.5 µm, depth 0.30 µm) into air, Gaussian beam with w₀ = 3 µm. Ch2's green map is the 530 nm case. The
  period sweep has 96 frames from 1.5 µm to 0.35 µm, uniform in K, and is blended per video frame.
* Hook, right: reflectance spectrum of a 10-pair stack (n = 1.7 / 2.1, Bragg wavelength 470 nm) and the colours of
  the reflected and transmitted light.
* Ch1: the crests are exact plane waves (incident, transmitted or evanescent, and reflected crests weighted by the
  TE Fresnel |r|²). The k-space constructions are exact geometry, as in Ch2 and Ch6.
* Ch3: the Bloch wave and its phase per period (n = 1.45 / 2.35, Λ/λ = 0.18); the bands for Δn = 0…0.9 at a fixed
  mean index n̄ = 1.9 with equal layer thicknesses (interpolated in Δn every frame); the reflectance of 10 pairs; the
  field at mid-gap. The colour bar on the frequency axis maps ω to wavelength for Λ = 123.7 nm.
* Ch4: the thick grating's transmitted orders (TiO₂-like n = 2.4 ridges, 40 % fill, height 0.9λ, Λ = 2.5λ, on
  glass); the 9×9 matrices shown are the actual RCWA matrices k₀²E − Kx² of the glass layer and of the grating layer
  (orders −4…4); the three mode profiles and kz values are eigenvectors and eigenvalues of the grating-layer matrix.
* Ch5: the convergence curve |t₀(M) − t₀(241 orders)| and the unit-cell library: zero-order phase vs ridge width
  (Λ = 0.45λ, h = 0.9λ, n = 2.4 on glass, 51 orders; transmission 77–100 %, phase span 1.1 turns). The metalens widths
  are looked up in that library for a hyperbolic lens phase (f = 8λ, 25 cells).
* Ch6: the band-gap icon (bands at Δn = 0.9).

**Schematic** (drawn, not computed): the Fourier-harmonic rows in Ch2 (real parts of e^(imKx)); the light arrows on
the hook's stack; the S-matrix chain in Ch4; the converging wavefronts above the metalens in Ch5 (labelled on
screen); the icons in Ch6 except the band diagram.

**Simplifications** (honest notes)
* All RCWA results are for TE polarisation. Unpolarised white light would average TE and TM; TM needs Li's
  factorization rules, which are mentioned but not implemented.
* The stacks are embedded in a medium of the mean index n̄ = 1.9 (no outer air surfaces), so R = 0 exactly at
  Δn = 0. The hook's mirror uses Δn = 0.4 so that it reflects one colour band; Ch3 sweeps Δn up to 0.9.
* In the field maps every plane-wave component of the beam (divergence about 3°) gets the order amplitudes computed
  for normal incidence, the incident beam below the grating is drawn without reflected light, and only the light
  transmitted into air is shown.
* The zero-order statement in Ch2 concerns normal-incidence transmission into air. The reflected channels in glass and the internal grating modes obey their own propagation conditions. Closing higher transmitted orders does not establish a uniform effective medium; near-wavelength high-contrast gratings can still support resonances ([Karagodsky and Chang-Hasnain, 2012](https://doi.org/10.1364/OE.20.010888)).
* The ridge layer in Ch4's cross-section is drawn 1.25× taller than its true 0.9λ for legibility.

## Rebuild
```bash
python physics.py              # data/*.npz (about 1.5 min; the field maps use JAX on a GPU)
../common/audio.sh .                                    # narration: Kokoro TTS -> Whisper check -> subtitles
PAR=1 ./preview.sh                                      # low-res chapter previews; coordinate across episodes
WORKERS=3 ./render.sh --skip-audio                       # final render, 3 workers
python ../common/verify.py .   # counts, seam scan, loudness, stills -> .verify/
```
Scenes: `scenes.py` (`Ch0Hook` … `Ch6Recap`), shared helpers from `../common/o5common.py`. Narration: `narration.json`
(40 segments, 802 words). Sync uses segment starts plus word timestamps from the Whisper check (`WT()` in `scenes.py`).
`scenes.py` overrides `Chapter.at()` with a whole-frame version (wait exactly k frames), matching the whole-frame behavior now used by the shared helper (FAST_ANIMATION_PIPELINE.md §10). The local override is retained for compatibility with this episode's original render.

## Validation (2026-10-26)

The revised Ch2 narration and labels distinguish the transmitted zero-order condition from a uniform-film
approximation. The physics source is unchanged; the existing production data were reused after confirming
the source hash, and the RCWA Airy-film, energy-conservation and thin-relief checks were rerun successfully.

All seven chapter previews passed exact frame counts and timing checks. Preview inspection also caught a
late Ch3 animation and an overlapping Ch4 label transition; both were corrected before the final render.
The 1080p60 render contains 19,361 frames (expected 19,361), runs 322.684 s, carries 91 subtitle cues, and
measures −16.3 LUFS. No detected image jump coincides with a render-piece boundary. The single scan flag at
21.00–21.10 s is the intended hook fade to black, confirmed frame by frame. Final contact sheets and the
corrected Ch2 frame at 142.5 s were inspected for legibility and subtitle clearance. The Whisper round-trip
flags were reviewed against word timestamps; the revised narration was transcribed in full.

The distribution encode, `../dist/e07_periodic_structures_1080p.mp4` (23,346,685 bytes), was independently
probed after encoding: H.264 at 1920×1080 and 60 fps, the same 19,361 frames and 322.684 s duration, AAC audio,
and 91 cues in a toggleable English subtitle track.

Verification records and stills are generated under `.verify/` and remain local build artifacts.
