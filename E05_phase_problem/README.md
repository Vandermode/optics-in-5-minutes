# Optics in 5 Minutes · E05 · The Phase Problem: holography and phase retrieval (5:05)

3Blue1Brown-style explainer (5:05, 1920×1080, 60 fps) for viewers with basic calculus and a little linear algebra.
Narration by Kokoro-82M (voice `af_heart`), subtitles timed to the speech by a Whisper round trip.

* `e05_phase_problem_1080p60.mp4` – subtitles burned in
* `e05_phase_problem_1080p60_softsubs.mp4` – clean picture + toggleable English subtitle track
* `subtitles.srt` / `.ass` – subtitles; `narration.json` – script (`text` = subtitle, `say` = phonetic)

**The idea (recap line).** Cameras lose phase. Interference writes it into fringes, and constraints let a computer
recover it, or design it.

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: living cells are nearly invisible to a camera (computed in-focus image), yet the light's phase shows them clearly; "how do you photograph what a camera can't see?" |
| 1 | 0:21 | what a camera throws away: light = arrows (amplitude, phase); a cell delays light, φ = k₀Δn·h, so the arrows turn; the camera records only \|U\|², and the arrows spin at 6×10¹⁴ turns/s |
| 2 | 0:57 | holography: add a reference R that spins in step with U; \|U+R\|² depends on the relative phase (phasors + brightness curve); \|U+R\|² = \|U\|²+\|R\|²+UR*+U*R; in-line (Gabor 1948): all terms overlap; tilted reference (Leith & Upatnieks 1962): fringes, and the cross terms shift in k-space |
| 3 | 1:48 | signature shot: computed off-axis hologram → FFT → three islands (\|U\|²+\|R\|² twice as wide, UR*, twin U*R) → rule k_c ≥ 3B → crop one island, re-centre, inverse FFT → full complex field (hue = phase) → ASM refocus: the cells snap into focus |
| 4 | 2:35 | phase retrieval: known window + transparent sample (\|U\| known), lens + camera give \|Û\|; Gerchberg–Saxton loop; alternating projections between two constraints (schematic); computed run: error falls 10⁰ → 10⁻⁴ as the cells emerge; blind spots (shift, twin, constant phase give the same far field); the lopsided window removes two; diverse data (distances, ptychography) |
| 5 | 3:35 | run it backwards: computer-generated hologram; a phase-only mask whose far field is a λ (same loop, target instead of measurement); speckle from the free far-field phase (E04); gradient descent teaser (E10) |
| 6 | 4:09 | on a computer: holography and GS in 3 lines each; pitfall 1: < 2 pixels per fringe → the island wraps around (computed sweep), so off-axis holography spends pixels on fringes; pitfall 2: GS stalls (8 computed random starts: half stall) → restart or use gradients |
| 7 | 4:43 | recap (camera → fringes → recovered → designed), where it breaks (coherent, stable light; diverse data), next: flat optics |

## What is computed (physics.py → data/*.npz) and what is schematic
All fields are synthetic, seeded and computed with numpy (single thread, ~30 s). Units: wavelengths; sample-plane
pixel λ/4; objective NA 0.25 (band limit B = k·NA).
* **Cells** (`cells.npz`): six synthetic transparent cells (phase-only, 0–3.5 rad: dome, nucleus, nucleoli, granules),
  band-limited by the NA. The hook's "camera image" is the computed in-focus \|U\|² (contrast ±30 % only at
  nucleoli, otherwise flat); chapter 1's arrows use a computed cut through one cell.
* **Off-axis holography** (`holo.npz`): the object wave is propagated 110λ to the camera with the exact ASM
  (episode 1); hologram \|U_c + R\|² with a unit tilted plane reference (carrier 3.62 B, on the FFT grid, fringe period
  4.4 px). Chapter 2's tilt sweep forms \|U_c + R\|² per frame from the stored U_c (one line of arithmetic). The
  spectrum is the FFT of the hologram (log scale, 11.5 decades); the island crop, re-centring and inverse FFT
  reproduce U_c to 1e-14, and the refocus stack is ASM back-propagation (53 planes, 0–130λ; the trivial global
  phase e^{ikz} is removed per plane so the background colour stays fixed). Chapter 6's wrap-around uses
  holograms with carriers up to 360 bins (undersampled fringes alias).
* **Gerchberg–Saxton** (`gs.npz`): three cells inside a lopsided window (disk with a bite), \|U\| = 1 inside;
  the far-field magnitude is \|FFT\|; 150 GS iterations from random phases (start 12 shown; global phase aligned for
  display); blind-spot copies (shift, conjugate flip, constant phase) verified to give the same far field to 1e-16;
  the stall plot is 8 random starts × 300 iterations (4 stall at ~0.13, 4 converge).
* **CGH** (`cgh.npz`): GS between a 256² phase-only mask (zero-padded to 512²) and a target λ (Latin Modern
  math-italic glyph); 60 iterations, light inside the target 3 % → 90 %; speckle and far-field phase zooms are the
  computed final far field (centred FFT).
* **Schematic** (drawn, not computed): the phasor diagram and brightness curve of chapter 2 (exact formula
  \|R+U\|² for |R| = 1, |U| = 0.65), the spectrum discs of chapter 2 (positions follow the true carrier), the
  constraint-set cartoon of chapter 4 (exact alternating projections between a line and a circle, labelled
  "schematic"), the icons for "several distances / overlapping scans", the reference-wave icon and the
  gradient-descent icon. The cells are synthetic, not measured; no group data are shown.
* Simplifications (said or shown honestly): simulation in sample-plane coordinates (no magnification), periodic FFT
  grid, noise-free holograms, scalar fields. GS is shown converging for one start; chapter 6 shows that half of the
  starts stall.

## Rebuild
```bash
cd ~/code/LAFA/animations/Optics5min/E05_phase_problem
python physics.py      # data/*.npz (deterministic, ~30 s, 1 thread)
../common/audio.sh .                            # TTS + Whisper check + subtitles (only if narration.json changed)
./preview.sh [Ch3Islands ...]                   # low-res previews (2 processes)
./render.sh --skip-audio                        # 1080p60 film, 3 workers
python ../common/verify.py .
```
Chapters: `Ch0Hook Ch1Camera Ch2Holography Ch3Islands Ch4Retrieval Ch5Design Ch6Computer Ch7Recap` (scenes.py).
`scenes.py` replaces `Chapter.at()` with a whole-frame version (`_at_whole_frames`, the fix from
FAST_ANIMATION_PIPELINE.md §10) until `common/o5common.py` carries it; it also has a local `code_card2` that keeps
code indentation.
