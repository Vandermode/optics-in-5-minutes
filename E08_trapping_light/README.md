# Trapping Light: Waveguides, Modes & Resonators (Optics in 5 Minutes · E08)

Narrated 3Blue1Brown-style explainer (5:06, 1920×1080, 60 fps) for viewers with basic calculus and a little linear
algebra. Open-source TTS voice (Kokoro-82M), subtitles timed to the speech by a Whisper round trip.

* `e08_trapping_light_1080p60.mp4` – subtitles burned in
* `e08_trapping_light_1080p60_softsubs.mp4` – clean picture + toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass` – subtitles; `narration.json` – script (subtitle text + phonetic "say" text)

**Idea (recap line).** Trapped light settles into modes, the eigenvectors of the cross-section. A resonator stores
light where round trips add in phase.

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a ring a few microns across sorts colours – white in, magenta through, green dropped (computed 2D wave field); "how do you trap light in a wire, and why do only certain colours fit?" |
| 1 | 0:24 | bouncing waves make modes: core/cladding, TIR zigzag = two crossing plane waves (±kx across, β along) = standing wave across, travelling along; transverse round trip 2kx·d + 2φr = 2πm; angle slider clicks at m = 0, 1, 2 (the profile's tail stops blowing up); three live modes with evanescent tails |
| 2 | 1:12 | modes are eigenvectors: E(x)e^{iβz} in the wave equation → [d²/dx² + k0²n(x)²]E = β²E; sampled → tridiagonal (1, −2, 1)/Δx² + diag(k0²n²) = H; eig → mode shapes and β; n2 < n_eff = β/k0 < n1; any input = Σ c_m E_m e^{iβ_m z} ("the angular spectrum method of a waveguide"); multimode beating and self-imaging |
| 3 | 2:04 | two guides: light sloshes (signature shot, BPM); symmetric/antisymmetric supermodes β ± κ; the antisymmetric one falls behind and flips after L = π/2κ; 2×2 rotation (cos κz, i sin κz); half a crossing = 50:50 splitter |
| 4 | 2:52 | resonators: each round trip adds a rotated, weaker copy (phasor chain, signature shot); off resonance a small spiral, on resonance (βL = 2πm ⇔ n_eff L = mλ) one long arrow; sweeping the colour gives evenly spaced peaks; back to the computed ring on resonance |
| 5 | 3:25 | linewidth: fewer surviving round trips → broader peak; Q = λ/Δλ ↔ storage time; the through port: light straight past + light leaking back out (flipped on resonance); critical coupling (coupling = loss) → zero; weaker/stronger → shallower dip |
| 6 | 4:02 | on a computer: 4-line code card (eigh mode solver; split-step BPM = FFT diffraction step + thin phase screen; ring = geometric series); pitfalls: the solver window must hold the tails (squeezed mode computed); BPM only marches forward at gentle angles – fine for couplers, wrong for rings and tight bends |
| 7 | 4:43 | recap (the sentence above); where it breaks → Maxwell's equations; teaser "Maxwell on a grid: FDTD and the perfectly matched layer" |

## What is computed, and how (`physics.py` → `data/*.npz`, single-threaded numpy/scipy, about 70 s)
* **Hook / ch4 / ch6 ring field** – 2D scalar Helmholtz equation (= Maxwell for Ez polarisation) on a 25 nm grid
  (396×402), stretched-coordinate PML, sparse LU (SuperLU). Add-drop ring: core index 2.9 (an effective index for a
  silicon wire) in 1.45, width 0.3 µm, radius 2.5 µm, gaps 0.18 µm; the bus mode is launched from the 1D mode solver.
  A wavelength scan plus a 1D minimisation finds the resonance at 1.5578 µm: drop 99.65 %, through 0.01 %, ring field
  2.9× the bus amplitude. The two off-resonance runs (±21 nm) pass with 99.4–99.6 % transmission. 31 sparse solves.
  The **three "colours" are false colours** (red/green/blue) for these three nearby infrared wavelengths, mixed
  additively; brightness is |E|^0.72 times a travelling crest pattern cos²(arg E − ωt).
* **Ch1 slab modes** – n1 = 2.0, n2 = 1.45, thickness 0.95λ: the transverse-resonance condition 2kx d − 4 atan(γ/kx) =
  2πm solved by root finding (θ = 12.18°, 24.66°, 37.46°; critical 43.53°); the slider's profile is the exact field
  integrated from the lower cladding (its upper tail grows unless θ is a mode angle); the dial shows the same round-trip
  phase; the live fields are Re{E(x) e^{i(βz − ωt)}} of these modes. Checked against the eigen-solver: n_eff agree to 5e-5.
* **Ch2 mode solver** – finite differences, `eigh(D2/dx² + diag(k0² n²))` (36×36 matrix shown, 1000-point grid for the
  profiles). **Self-imaging**: a 6λ-wide guide (17 guided modes, Lπ = 103.7λ); an off-centre spot expanded in the
  eigenmodes, each advanced by e^{iβ_m z} and summed; mirrored image at 307λ (peak 0.90), direct image at 613λ (0.83).
* **Ch3 / ch6 coupler** – split-step BPM: paraxial FFT diffraction step relative to n_ref + thin phase screen
  e^{ik0 Δn Δz} (Δz = 0.05λ, 1024 points over 16λ), two guides n 1.50/1.45, width 1λ, gap 1λ. The supermodes and
  κ come from the eigen-solver: L = π/2κ = 109.2λ; the BPM transfer peaks (97.7 %) at 108.6λ (0.6 % apart).
  The sum column is |S + A e^{−2iκz}|² from the computed supermodes.
* **Ch4–5 resonator curves and phasors** – the analytic round-trip sum (geometric series): stored field
  Σ (q e^{iφ})^n = 1/(1 − q e^{iφ}) with q = ra = 0.86; all-pass through port t = (r − a e^{iφ})/(1 − r a e^{iφ}) with
  round-trip amplitude a = 0.85 and r = √(1 − κ²) (critical coupling at r = a); "light left in the ring" = q^{2n}.
* **Ch6 window pitfall** – the m = 2 slab mode in a ±0.72λ window (n_eff 1.564) vs ±3λ (1.588, the true value).

**Schematic (drawn, not data):** the ray zigzag, the crossing crest families and k-vector diagram (geometry exact for
the m = 1 mode), the round-trip dial's walk (its end points are computed), the ring+bus sketches with moving dots, the
supermode/sum layout, the rotation diagram, the slice diagram, the code card, the recap icons and Maxwell's equations.
Display scales: the Ch1 mode strips and the MMI/coupler maps are compressed along z (not to scale); crest speeds are slowed.

## Notes and compromises
* 2D models stand in for 3D devices (the ring's 2.9 core index is an effective index); demos use different index
  contrasts (ch1–2: 2.0/1.45, coupler: 1.50/1.45 so the paraxial BPM is accurate, ring: 2.9/1.45).
* The code card writes the diffraction step generically as `exp(1j*kz*dz)`; physics.py uses its paraxial form.
* `scenes.py` overrides `Chapter.at()` with the pilot's whole-frame version (`EpChapter`): `common/o5common.py`'s
  `at()` still waits a float duration, which can differ by one frame between workers when a target lies on the frame grid.

## Verification (2026-10-11)
`render.sh --skip-audio`: 50 pieces, all OK, 7.75 min on 3 workers. `verify.py`: 18,377 frames = round(306.278 s × 60),
87 subtitle cues, integrated loudness −16.2 LUFS; seam scan: two hits, both intended (the chapter-3 end fade at 171.6 s
and the start of the wavelength sweep at 198.05 s, where the near-resonant phasor chain swings quickly), none near a
piece start; small-region scans at every live-redraw introduction show no snaps. Whisper check: all 43 segments heard as
written apart from spelling/homophones (colour/color, plane/plain, sign/sine, "50-50").

## Rebuild
```bash
OMP_NUM_THREADS=1 python physics.py   # data/*.npz (about 70 s, one thread)
../common/audio.sh .                                           # Kokoro TTS -> Whisper check -> subtitles
./preview.sh                                                   # optional low-res previews (2 processes)
./render.sh --skip-audio                                       # windowed parallel render, 3 workers
python ../common/verify.py .          # counts, seam scan, loudness, stills
```
