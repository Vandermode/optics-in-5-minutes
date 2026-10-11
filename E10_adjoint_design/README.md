# Designing Light with Gradients: the adjoint method (Optics in 5 Minutes · E10, season finale)

3Blue1Brown-style explainer (5:15, 1920×1080, 60 fps) for viewers with calculus and a little linear algebra.
Narration by an open-source TTS voice (Kokoro-82M, `af_heart`), subtitles timed to the speech.

* `e10_adjoint_design_1080p60.mp4`: subtitles burned in
* `e10_adjoint_design_1080p60_softsubs.mp4`: clean picture plus a toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass`: subtitles

**The idea (recap line):** one forward and one backward simulation give the gradient for every pixel at once.

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a human Y-splitter (86 %) vs a computer-designed pattern in the same 1.5 × 2.4 µm box (99.8 %); light sent backward; title |
| 1 | 0:28 | design as numbers p (0 = glass, 1 = silicon) and a merit F(p); climbing a landscape; finite differences = one simulation per pixel (10⁶ × 1 min ≈ 2 years per step) |
| 2 | 1:08 | a grid simulation at one frequency is A(p) x = b; differentiate; dF/dp = −(∂F/∂x) A⁻¹ (∂A/∂p) x; right-to-left = one solve per pixel, left-to-right = one extra solve: Aᵀλ = (∂F/∂x)ᵀ, dF/dp = −λᵀ (∂A/∂p) x |
| 3 | 1:57 | reciprocity: Aᵀ = A, so λ is light sent backward from the outputs; a nudged pixel scatters a wavelet; slope = −2k₀² Re{λⱼxⱼ} (forward × adjoint); the sensitivity map; 160 iterations: the device takes shape (signature shot) |
| 4 | 2:44 | differentiable optics: phase mask + E01's propagation as a chain; autodiff runs it backward (FFT ↔ FFT⁻¹, H → H*); for propagating waves H* = propagation by −z (computed); gradient descent projects ∇; E05's Gerchberg–Saxton vs gradients |
| 5 | 3:33 | end-to-end: scene → optics → noise → Wiener filter (E03) → error; gradients flow into the optics; extended depth of field: lens PSF blurs with defocus, learned PSF barely changes, filtered images sharp at all depths |
| 6 | 4:10 | in practice: the 3-line loop; binarization and minimum feature size; different starts → different good designs; the model gap: a relief optimized with the thin-element model loses about half its focus in a full-wave simulation; models must be fast and accurate |
| 7 | 4:46 | season recap: the series map (E01–E10) lit word by word, one thread; closing line "Light is linear: in the right basis, the physics is simple; run it backward, and the design is too." |

## What is computed, and how (`physics.py` → `data/*.npz`)
All physical fields are computed with the method being taught; nothing is hand-drawn except where marked schematic.

* **Splitter, FDFD + adjoint (Ch0, Ch1, Ch3, Ch6, Ch7)** (`data/splitter.npz`). 2D frequency-domain scalar
  Helmholtz solver for Ez (e^(−iωt)), written from scratch with `scipy.sparse` + SuperLU. Silicon ε = 12 in glass
  ε = 2.1, λ = 1.55 µm, 30-nm grid (240 × 180 cells), stretched-coordinate PML. The matrix is symmetrized
  (multiplied by s_x s_y) so that Aᵀ = A exactly, so the adjoint solve reuses the forward LU factors.
  * Merit: power in the fundamental mode of both single-mode output guides (0.24 µm wide), from a 1D eigenmode
    solve, normalized to a straight guide.
  * Gradient: dF/dε = −2k₀² Re{λ x} with A λ = ∂F/∂x (Wirtinger). It matches central finite differences on 5
    pixels to a relative error of 2 × 10⁻⁹.
  * Optimization: 160 iterations of Adam with mirror symmetry (equal split), a conic minimum-feature filter
    (R = 120 nm) and tanh projection (β = 2 → 64). **The quoted 99.8 % is the hard-binarized design (every pixel
    all silicon or all glass)**, and it holds at a 15-nm grid (99.77 %). Grey iterates are labelled as such.
  * Human Y-branch: two cosine S-bends in the same box, sub-cell averaged: 86.4 % (86.7 % at a 15-nm grid).
  * Shown: forward and adjoint fields at iteration 0 and for the final design; the exact first-order wavelet of
    one nudged pixel, δx = −A⁻¹(∂A/∂εⱼ)x; the sensitivity map; every iteration's design and |E|; the merit
    history; two more seeded starting guesses giving 99.7 % and 99.3 %.
* **Phase mask by autodiff (Ch4)** (`data/asm.npz`). JAX on one GPU: a 256 × 256 thin phase mask (2λ pixels),
  exact angular-spectrum propagation (E01) to z = 1200λ on a 2× zero-padded grid, 300 Adam steps toward a ∇
  target. The error field at the screen, propagated back by −z (multiplied by H*), equals JAX's reverse-mode
  gradient to 3.5 × 10⁻⁷. Both fields are shown in pink.
* **End-to-end extended depth of field (Ch5)** (`data/edof.npz`). JAX: a radially symmetric pupil phase (Zernike
  radial orders 4–24) and one depth-blind Wiener filter are trained on the expected restoration error over 9
  defocus values (±1.5 waves), with a 1/f² image prior and sensor noise. Shown: PSFs of the plain lens and of the
  learned optics (and their training snapshots), and a synthetic test scene through the whole pipeline (blur,
  noise, filter) at three depths.
* **Model gap (Ch6)** (`data/gap.npz`). A glass relief (n = 1.5, up to 2π of phase) is optimized with JAX under
  the thin-element model plus exact ASM to focus steeply off-axis: 76 % of the light in the focal spot. The same
  relief simulated with the full-wave 2D FDFD solver (λ/30 grid, sub-cell averaged, same normalization): 41 %.
  This is one illustrative toy case, not a general ratio.

**Schematic panels:** the merit landscape and its uphill path in Ch1 (an analytic function, labelled
"schematic"), the matrix block diagrams in Ch2 (the 6 × 6-grid pattern of A is computed; the grid values are
sampled from the computed field), the reciprocity cartoon in Ch3, the computation graphs in Ch4, the pipeline icons
in Ch5 (its images are computed), the model-ladder run times in Ch6 (orders of magnitude), and the series-map icons
in Ch7.

**Simplifications (stated honestly):** 2D scalar (Ez) physics at one wavelength, not a 3D device. No fabrication
tolerance analysis. The EDOF loss is the closed-form linear (Wiener) error, not a trained network. λ in Ch2–Ch3
is the adjoint field, not a wavelength; the film says so on screen.

## Rebuild
```bash
python physics.py          # ~3 CPU-min (FDFD, single thread) + seconds on one GPU (JAX)
../common/audio.sh .                                # Kokoro TTS -> Whisper round-trip check -> subtitles
./preview.sh                                        # low-res chapter previews (optional), 2 processes
./render.sh --skip-audio                            # windowed parallel render, 3 workers
python ../common/verify.py .   # counts, seam scan, loudness, stills -> .verify/
```
Scenes: `scenes.py` (8 chapters, `CHAPTERS` = `episode.json["chapters"]`); shared helpers: `../common/o5common.py`.

Notes: `scenes.py` overrides `Chapter.at` with whole-frame waits (FAST_ANIMATION_PIPELINE.md §10), so a target on
the 1/60-s grid cannot cause a one-frame step at a render seam. Final check (`verify.py`): 18,890 frames
(= 314.836 s · 60), 92 subtitle cues, −16.3 LUFS; every scan hit is an intended fade, none at a piece seam.
