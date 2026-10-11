# Optics in 5 Minutes · E02 · Why a Lens Computes a Fourier Transform (5:12)

3Blue1Brown-style explainer (5:12, 1920×1080, 60 fps) for viewers with basic calculus and a little linear algebra.
Narration by an open-source TTS voice (Kokoro-82M, `af_heart`), subtitles timed to the speech (Whisper word
timestamps). Recap line: *"Far away, or at a lens's focus, each direction lands on its own spot, so light lays out
its own spectrum."*

* `e02_lens_fourier_1080p60.mp4` – subtitles burned in
* `e02_lens_fourier_1080p60_softsubs.mp4` – clean picture + toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass` – subtitles (85 cues); `narration.json` – script (47 segments, 796 words)

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a laser through a slide with an "F"; the far wall shows the F's Fourier transform, and a lens shows the same pattern one focal length away (both computed); title card |
| 1 | 0:24 | far away, directions become positions: E01's slit map zooms out 1000× (150λ → 150 000λ) into a fan of straight rays; direction θ lands at x = z tan θ; the wall profile overlays the slit's spectrum exactly (Fraunhofer); far means z ≫ D²/λ |
| 2 | 1:11 | the paraxial shortcut: path length ≈ z + (x − x′)²/2z (exact vs parabola plot); Fresnel integral; expanding the square gives a wall phase, the Fourier kernel (kx = k·x/z) and a chirp across the slide; as z grows the chirp flattens and the computed pattern becomes the spectrum |
| 3 | 2:00 | a lens sorts directions: a lens is the opposite chirp e^(−ikx²/2f); live computed waves: a flat wave focuses at f, tilting slides the focus to x = f tan θ, three directions give three spots (signature shot, colour = kx); a slit's spectrum appears in the focal plane; chirps cancel at z = f; slide at the front focal plane → exact Fourier transform |
| 4 | 2:48 | the 4f machine: a second lens transforms back and the image is inverted (FT∘FT: x → −x, ray trace); the middle plane holds the spectrum, so a mask edits it: pinhole → blur, central stop → edges (computed) |
| 5 | 3:20 | Zernike phase contrast: a transparent cell is invisible (big arrow + tiny arrow at 90°, scanning phasor + brightness bar); the unscattered light is one Fourier-plane spot; a quarter-wave delay lines the arrows up and the cell appears (computed) |
| 6 | 3:53 | on a computer: angular spectrum (same grid, near), single-FFT Fresnel (grid Δx′ = λz/NΔx, far; animated grid), Fraunhofer (one FFT); chirps alias: computed sampled chirps through z_c = NΔx²/λ, where ASM and Fresnel trade places |
| 7 | 4:44 | recap: far field and focal plane side by side; the 4f edit; where it breaks (paraxial parabola vs the exact k-circle); a finite lens catches only a cap of directions → next: the diffraction limit |

## What is computed, and how (`physics.py` → `data/*.npz`; units λ = 1)
* **Hook** (`hook.npz`): an 80λ letter-F aperture on an 8192² grid. Far wall: single-FFT Fresnel propagation to
  Z = 10⁶λ (Fresnel number 0.006). Focal plane: thin-lens phase e^(−ik r²/2f), f = 4000λ, then band-limited 2D
  angular-spectrum propagation (Matsushima & Shimobaba 2009) on the GPU (JAX). The two patterns agree to within 3 % of the peak (mean difference < 1e-4).
  Shown with a log intensity scale over ±0.3 rad.
* **Far-field zoom** (`zoom.npz`): 12λ slit, exact band-limited ASM on a 2²¹-point grid (Δx = λ/4), 64 zoom levels
  from z ≤ 150λ to z ≤ 1.5·10⁵λ, 512×256 bin-averaged pixels each (JAX/GPU, 14 s). The display weights |U|² by
  (1 + zλ/D²) to undo the 1/z fall-off. The wall profile at 1.5·10⁵λ matches |Û(k sin θ)| to 0.5 %.
* **Fresnel panel** (`fresnel.npz`): the Fresnel integral ∫ e^(ik(x−x′)²/2z) dx′ over the slit, summed directly
  (4001 points) for 240 distances, plotted against kx = k·x/z next to the slit's spectrum. The chirp curve is
  Re e^(ikx′²/2z) evaluated analytically.
* **Lens fields** (`lens.npz`): 1D x–z fields behind a thin lens (paraxial phase e^(−ikx²/2f), f = 40λ, aperture 24λ):
  plane waves at 29 tilts from 0° to 7° plus −7°, and a 6λ slit, propagated with exact ASM (4096 points, Δx = λ/8).
  Displayed live as Re-crest brightness × |U|^0.55 (crests move as e^(−iωt)); colour = direction
  (cyan kx < 0, white kx = 0, gold kx > 0). Foci land at x = ±4.94λ vs f tan 7° = 4.91λ. The focal-plane read-outs
  are columns of the same fields; the slit's focal profile matches its spectrum at x = f tan θ to 1 %.
* **4f filtering** (`fourf.npz`): 2D FFT → mask → FFT on a 512² letter F; the second forward FFT produces the
  inverted image (checked to 1e-15). Pinhole radius 10 and stop radius 14 frequency samples.
* **Phase contrast** (`zernike.npz`): a synthetic weak phase object (cell, nucleus, organelles; φ ≤ 0.62 rad, seeded).
  Brightfield = |e^(iφ)|² = 1 exactly; phase contrast multiplies the DC sample by e^(iπ/2) (plane-wave illumination
  focuses to one Fourier point); the result correlates 0.999 with 1 + 2φ.
* **Chirp sampling** (`chirp.npz`): N = 32 samples, Δx = 2λ (z_c = NΔx²/λ = 128λ); the exact ASM transfer function on
  the kx grid and the Fresnel kernel on the x grid, with their band-limited (trigonometric) reconstructions from the
  samples, for 300 distances from 0.35 z_c to 2.8 z_c. "sampled/aliased" = adjacent-sample phase step ≤ π. The
  constant phase e^(ikz) of the transfer function is dropped for display (it does not affect sampling, but would
  make the curve strobe during the z sweep). With Δx = 2λ the exact ASM limit, z_c·√(1 − (λ/2Δx)²), is 0.97 z_c.
  The scene evaluates the same formulas (32 samples + one 768-point inverse FFT) at the continuous z of every frame,
  so the sweep has no table steps; they match `chirp.npz` to 1e-14 at its 300 tabulated distances.

**Schematic (drawn, not computed):** the side-view laser/slide/wall/lens set-up in the hook; the direction arrows
and rays in chapter 1 (drawn at the true angles of the map); the path-length geometry and the exact-vs-parabola plot
(exact formulas); the lens outline and delay curve; the 4f layout, the inversion ray and the mask symbols; the phasor
diagram (true ratio of arrow lengths, φ read from the cell along a scan line); the output-grid dots; the k-circle and
parabola in the recap. The incoming-direction arrows in chapter 3 are drawn 3× steeper than the 7° tilt (labelled).
**Simplifications:** the far-field formula U ∝ Û(k sin θ) omits the slowly varying obliquity/1/√z factors; Fresnel,
the thin-lens phase and the "exact Fourier transform" of the 2f system are paraxial (stated on screen and in the recap).

## Rebuild
```bash
cd ~/code/LAFA/animations/Optics5min/E02_lens_fourier
../common/audio.sh .                                         # narration -> audio/, Whisper check, subtitles
CUDA_VISIBLE_DEVICES=<free gpu> python physics.py   # data/*.npz (~1 min, GPU for hook + zoom)
./preview.sh                                                 # low-res previews (2 Manim processes)
./render.sh --skip-audio                                     # 1080p60, 3 workers -> the two MP4s
python ../common/verify.py .        # counts, seam scan, loudness, stills in .verify/
```
Shared pipeline and scene rules: `../PRODUCTION_SPEC.md`, `../docs/FAST_ANIMATION_PIPELINE.md`, `../common/`.
`scenes.py` replaces `Chapter.at` with the whole-frame version (`_at_whole_frames`, the pilot's 2026-10-10 fix), because
`common/o5common.py` still waits a float duration, which can shift a seam by one frame when a target lies on the 1/60 s grid.
