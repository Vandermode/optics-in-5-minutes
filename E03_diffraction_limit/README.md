# The Diffraction Limit: why every image is blurry (5 min)

**Companion lesson:** [interactive course page, Colab notebook, and instructor notes](../teaching/e03-diffraction/README.md). This standalone teaching module runs without the video-production environment.

Optics in 5 Minutes, episode 3. A 3Blue1Brown-style explainer (5:11, 1920×1080, 60 fps) for someone who knows basic
calculus and a little linear algebra. Narration by an open-source TTS voice (Kokoro-82M, `af_heart`), subtitles
timed to the speech. 43 segments, 807 words.

* `e03_diffraction_limit_1080p60.mp4`: subtitles burned in
* `e03_diffraction_limit_1080p60_softsubs.mp4`: clean picture + toggleable English subtitle track
* `subtitles.srt` / `subtitles.ass`: subtitles

The idea (recap line): *Imaging is convolution with the PSF. A lens only keeps directions with sin θ ≤ NA, so details
finer than ~λ/2NA never arrive.*

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: glowing beads through a perfect microscope become discs with rings; two beads closer than ~200 nm merge (computed); title |
| 1 | 0:24 | images are stamps: each point stamps a scaled PSF copy, the stamps add (convolution, I = O ∗ PSF); waves are eigenvectors of blurring, so F[I] = H(kx) F[O]; H = Fourier transform of the PSF; fine stripes fade most |
| 2 | 1:20 | the lens is a hole in k-space: plane waves fill the k-circle (kx = k sin θ); a lens catches only θ ≤ θmax, NA = n sin θmax, the cap \|kx\| ≤ k·NA; evanescent waves never leave; the pupil; **signature shot**: NA slider, cap widens ↔ Airy disc shrinks, r = 0.61 λ/NA; magnification is not resolution |
| 3 | 2:20 | Abbe's grating test (4f, computed): orders at sin θ = ±λ/d; stripes need the 0th and a 1st order; straight on d_min = λ/NA; tilted illumination d_min = λ/2NA, the diffraction limit |
| 4 | 3:10 | coherent vs incoherent: laser fields add, the pupil cuts the field at k·NA; glowing beads add intensities, OTF = pupil overlapping a shifted copy of itself, reaching 2k·NA with fading contrast; Siemens star imaged both ways (computed) |
| 5 | 3:56 | on a computer: image = IFFT(FFT(obj) × OTF); dividing by the OTF explodes the noise; Wiener filter H*/(\|H\|² + 1/SNR); beyond the cutoff the OTF is exactly zero; priors / extra measurements = computational imaging |
| 6 | 4:42 | recap; where it breaks: the PSF changes across the field of a real lens (computed, illustrative aberrations); next: coherence |

## Computed vs schematic
All physical fields are precomputed by `physics.py` (numpy/scipy FFTs, single worker, seeded; ~75 s) into `data/*.npz`;
the scenes only load them. Scalar Fourier optics throughout: pupil P = disc |f| ≤ NA/λ (anti-aliased edge),
amplitude PSF h = IFFT(P), incoherent PSF = |h|², OTF = FFT(PSF) (= pupil autocorrelation), coherent image =
|IFFT(FFT(t)·P)|².

* **Computed**
  * Hook: 32 beads imaged incoherently (λ = 500 nm, perfect NA 1.4 oil objective): image = IFFT(OTF × Σ phase ramps),
    so bead positions are exact (not snapped to pixels). Two-bead merge: 101 separations, 600 → 100 nm (the readout
    stops at 150 nm, one blob). Airy first zero 220 nm vs 0.61λ/NA = 218 nm.
  * Ch1: cumulative stamp images (NA 0.5); PSF icon; H(kx) = OTF cut (agrees with the analytic circular-pupil MTF to
    0.003); the stripe swatches use that computed H.
  * Ch2: PSF for every slider position, NA 0.25–0.95 in steps of 0.005 (141 FFTs of 2048² pupils); the dashed Airy
    ring is 0.61λ/NA (FFT first zeros agree within 0.013λ). The magnification panel reuses the hook's two-bead image.
  * Ch3: Abbe's experiment as 4f imaging of a sinusoidal amplitude grating (apodised 40λ patch), NA 0.5, plane-wave
    illumination straight on (d = 4λ → 1.4λ) and tilted (sin θin = 0 → −0.46, then d = 1.4λ → 0.85λ): 273 images.
    Stripe contrast 1.0 at d = 2.2λ, 0.05 at 1.8λ (cutoff λ/NA = 2λ); tilted 0.79 (theory 0.80), gone below ≈1.04λ.
  * Ch4: coherent vs incoherent images of a 36-period Siemens star (λ/12 pixels, NA 0.5); the traced OTF curve.
  * Ch5: test object (text + bar groups at 2.0, 1.3 and 0.8 × λ/2NA), incoherent blur (NA 0.5), seeded Gaussian
    camera noise (σ = 0.6 % of the peak), naive inverse filter inside the passband, Wiener filter (1/SNR = 2·10⁻³).
    The log-plot curves are the analytic circular-pupil MTF, which matches the FFT OTF to ~1 %.
  * Ch6: field-dependent PSFs |FFT(P·e^{i2πW})|² with Seidel coma, astigmatism and field curvature
    (0.6, 0.5, 0.25 waves at the field edge; illustrative values, labelled so on screen).
* **Schematic** (exact geometry, not simulations): k-circle diagrams (fan of directions, cap, band, order arrows at
  sin θm = sin θin + mλ/d, grating vector 2π/d), the lens ray diagram, the pupil disc, the NA slider, the sliding-disc
  overlap drawing, the phasor dials (fields add vs intensities add), code cards.

## Rebuild
```bash
cd ~/code/LAFA/animations/Optics5min/E03_diffraction_limit
../common/audio.sh .                                   # Kokoro TTS -> Whisper check -> subtitles (cached per segment)
OMP_NUM_THREADS=1 python physics.py   # data/*.npz (~75 s, one core)
./preview.sh                                           # low-res chapter previews (2 processes), prints LATE lines
./render.sh --skip-audio                               # windowed parallel render, 3 workers (~6 min)
python ../common/verify.py .  # counts, seam scan, loudness, stills -> .verify/
```

Note: `scenes.py` derives its chapters from `Ep`, a `Chapter` subclass with the pilot's whole-frame `at()`
(`common/o5common.py` still waits a float duration there, the one-frame-step pitfall of
`FAST_ANIMATION_PIPELINE.md` §10). Live numbers (separation, NA, period d) are copies of pre-built digit glyphs
dropped into the slots of a pre-built template (`SlotNum`), so no Tex is built per frame.
