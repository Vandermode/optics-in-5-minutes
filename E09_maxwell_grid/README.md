# Maxwell on a Grid: FDTD and the perfectly matched layer (Optics in 5 Minutes, E09)

3Blue1Brown-style explainer (5:15, 1920×1080, 60 fps) for viewers who know basic calculus and a little linear
algebra. Narration by an open-source TTS voice (Kokoro-82M, `af_heart`), subtitles timed to the speech.

* `e09_maxwell_grid_1080p60.mp4`: subtitles burned in
* `e09_maxwell_grid_1080p60_softsubs.mp4`: clean picture plus a toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass`: subtitles (88 cues)

**The idea (recap line).** E updates H and H updates E, one neighbour at a time. A layer of complex-stretched space
absorbs whatever leaves.

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a few-cycle pulse races along a silicon wire, hops into a small ring, circles and leaks back out every lap (computed 2D FDTD); "this is Maxwell's equations, solved on a grid: how?"; title card; the two ideas |
| 1 | 0:30 | Maxwell's two rules in 2D (Ez out of the screen, H in the plane): ∂H/∂t = −(1/μ)∇×E, ∂E/∂t = (1/ε)∇×H; curl = swirl; an E bump makes H circulate, then spreads into a ring (computed); only neighbours matter |
| 2 | 1:09 | the Yee grid: a difference of two samples is the slope at their midpoint (secant vs end/midpoint tangents), so H sits half a cell from E; E on whole time steps, H on half steps; the leapfrog; a 1D update with real numbers and the marching pulse (computed); matter = ε per cell (a staircased ring) |
| 3 | 1:56 | the grid's own rules: news spreads one cell per step in a diamond, light's circle must fit inside it, cΔt ≤ Δx/√2 (Courant); 6% over the limit the field explodes (computed, both runs); the grid's k-circle bulges along the axes (exact Yee dispersion relation) and a pulse on 5 vs 24 cells per λ (computed); 10–20 cells per wavelength |
| 4 | 2:41 | the edge of the world: a hard wall echoes everything, a lossy layer echoes from its edge (29% of the wall's echo, computed); gentler means thicker; "an absorber light cannot see" |
| 5 | 3:03 | the PML (Bérenger; Chew & Weedon, 1994): signature shot, the real axis as a ribbon over the complex x-plane bending into Im x inside the border, x̃ = x + (i/ω)∫σ dx, the wave keeps its wavelength and dies as e^(−(k/ω)∫σ dx); exact continuum solution at varying angle and wavelength (no echo); wall vs lossy layer vs PML side by side (computed: echo 100%, 29%, 0.0004%); the PML's residual echo magnified 10⁵ (computed); σ graded smoothly; grazing and evanescent waves are hardest |
| 6 | 3:59 | on a computer: the four-line time loop; one pulse + a Fourier transform in time gives the whole reflection spectrum of a Bragg mirror, on top of the exact transfer-matrix curve (computed 1D FDTD); cost ∝ cells × steps and the model ladder (TEA → BPM → RCWA → FDTD); FDTD vs the thin-element model on a fine blazed glass relief: 96% vs 57% of the light in the design direction (computed) |
| 7 | 4:47 | recap: FDTD converges only as the cells shrink, curves become staircases, the PML adds a small error; the recap line; teaser: light → structure (next: the adjoint method) |

## Computed vs schematic
Every field panel is computed by `physics.py` with the method the episode teaches: a 2D TMz Yee FDTD
(Ez, Hx, Hy; leapfrog; JAX on one GPU) with a convolutional PML (the stretched coordinate
x → x + (i/ω)∫σ dx in the time domain, σ graded ∝ depth³, Roden & Gedney 2000), or the same scheme in 1D.
Units: λ₀ = c = ε₀ = μ₀ = 1.

| Panel | How it is computed |
|---|---|
| Hook ring resonator | 2D FDTD, n = 3.5 bus + ring (R = 1λ₀, width 0.3λ₀, gap 0.07λ₀), 56 cells/λ₀, Courant 0.5, Gaussian-modulated pulse injected with the slab-mode profile 14λ₀ upstream (its stray radiation leaves before the guided pulse arrives); 1115×309 cells × 13 888 steps, about 3 s on one A100. The light circulating in the ring peaks at about 17% of the bus pulse |
| E bump → ring, H arrows (Ch1) | 2D FDTD from an initial Gaussian Ez, H = 0; H sampled on rings of points |
| 1D leapfrog with numbers (Ch2) | 1D Yee update at cΔt/Δx = 0.5; the worked example uses the stored values |
| Courant blow-up (Ch3) | 2D FDTD in float64 from a smooth bump at S = 0.70 and 0.75: the second grows ×2 per step from round-off |
| Grid k-circle (Ch3) | exact Yee dispersion relation sin²(kₓΔ/2) + sin²(k_yΔ/2) = sin²(πS/N)/S², solved per direction |
| Coarse vs fine pulse (Ch3) | 2D FDTD point-source pulse at 5 and 24 cells per wavelength |
| Wall / lossy layer / PML (Ch4–5) | 2D FDTD, 8λ₀ box + 1λ₀ layer: perfect conductor; step conductivity σ = 10 then a wall; graded CPML. Echo = difference from a reference run in a box too large to reflect within the window |
| PML residual echo ×10⁵ (Ch5) | the same PML run minus the reference run |
| Bragg mirror spectrum (Ch6) | 1D FDTD, 8 pairs n = 2.3/1.45 on glass, one pulse, reflected/incident spectra by FFT; exact curve = transfer matrix for the same grid-rounded layers |
| TEA vs FDTD relief (Ch6) | periodic 2D FDTD (period 3λ₀, blazed height λ₀/(n−1), n = 1.5, E along the grooves), CW phasor from a time DFT; TEA = column-by-column phase delay plus angular-spectrum propagation, with the flat-interface Fresnel amplitude; efficiencies from the order amplitudes 3λ₀ above the relief |

**Validation** (scratch checks of the solver): the PML reflects −87 dB (10 cells) to −111 dB (20 cells) of the
incident amplitude for a point-source pulse; the measured phase velocity on 5 and 10 cells/λ matches the analytic Yee
dispersion to 4 digits along the axis and the diagonal; S = 0.7 is stable over 6000 steps; the Bragg spectrum agrees
with the exact result to 0.003 on average (0.03 at the steep band edges, residual grid dispersion).

**Schematic.** The 3D sketch of E and H (Ch1), the secant/tangent picture and the Yee lattice (Ch2), the diamond and
light circle (Ch3), the ribbon over the complex x-plane (Ch5; the bend follows Im x̃ ∝ depth⁴ like the graded σ, its
scale is illustrative), the model ladder and its cost labels (Ch6, typical orders of magnitude), and the recap icons.
The Ch5 angle/wavelength panel is the exact analytic continuum solution in stretched coordinates, not FDTD.

**Display choices.** Signed fields use the series' diverging map with a signed power law (γ = 0.6–0.75) so weak
waves stay visible. In the hook the colour scale saturates at 32% of the bus pulse (so the ring's light is visible), and |E| below
0.4% of the pulse peak is shown black (it hides a ~10⁻³ residue of the imperfect mode-matched source travelling
along the bus). The waveguides are drawn as outlines. The PML's own echo is shown magnified 10⁵ times.

## Notes
* `scenes.py` replaces `Chapter.at` with the whole-frame version (FAST_ANIMATION_PIPELINE.md §10: it waits exactly k
  frames, so a target on the 1/60 s grid cannot shift a play by one frame at a render seam); `common/o5common.py`
  still has the pre-fix `at()`.
* Local helper copies (not in `common/`): `code_card_ind` (keeps code indentation), `movie_rgba` (time-interpolated
  frame stacks), `warp` (smooth film-time to simulation-time maps), `layer_frame`, `yee_icon`, `ribbon_icon`.

## Rebuild
```bash
CUDA_VISIBLE_DEVICES=<free gpu> python physics.py   # data/*.npz (about 1 min)
../common/audio.sh .                                                          # TTS, Whisper check, subtitles
./preview.sh                                                                  # low-res previews (2 processes)
./render.sh --skip-audio                                                      # final render (3 workers)
python ../common/verify.py .                         # counts, seams, loudness, stills
```
