# Differentiable Optics: From Images to Design

Episode 13 of **Optics in 5 Minutes**. A 5:18 narrated film connecting a phase element's design parameters to its measured intensity, a scalar loss, and a checked gradient update.

The numerical example matches the E13 interactive lesson and Python lab. Its two controls are a linear pupil phase for tilt and a quadratic phase for defocus. The film follows the intensity error backwards through the Fourier transform, checks both parameter derivatives independently, and compares two optimization starts under the same objective.

## Chapters and displayed panels

| Start | Scene | What is shown |
| --- | --- | --- |
| 0:00.000 | `Ch0Hook` | A phase curve and focal intensity change while aperture and power remain fixed. |
| 0:19.233 | `Ch1Forward` | Separate tilt and defocus sweeps; pupil field, unitary propagation, and intensity measurement. |
| 1:03.667 | `Ch2Loss` | Initial and target intensities, signed residuals, and the fixed normalized loss. |
| 1:38.033 | `Ch3Backward` | The computation chain, residual weighting, adjoint propagation, phase sensitivity, and parameter sums. |
| 2:33.667 | `Ch4Check` | Central finite differences and maximum gradient error over a perturbation-size sweep. |
| 3:11.600 | `Ch5Update` | Intensity after each accepted update; loss histories from the default and difficult starts. |
| 3:51.300 | `Ch6Beyond` | Schematic explanations of automatic differentiation, adjoints, curved-lens derivatives, and joint optical/digital design. |
| 4:47.483 | `Ch7Recap` | The design loop and schematic questions about sensor noise, fabrication, and new samples. |

Starts are chapter boundaries at 60 fps, obtained from the midpoint between adjacent chapters' narration. The narration timeline is 317.663 seconds with 43 segments; the corresponding picture target is 19,060 frames.

## Model and objective

The pupil is monochromatic, one-dimensional, scalar, and phase-only. Its normalized coordinate is `x`; the illuminated aperture is `|x| < 1`. A width-eight computational window contains 128 samples, of which 32 lie in the aperture. The amplitude `A` is fixed and normalized so that `sum(A²) = 1`. The real design parameters `a` and `b` are phase coefficients in radians:

```text
phi(x) = a*x + b*x²
u = A*exp(i*phi)
v = F*u
I = |v|²
```

`F` is the centered unitary discrete Fourier transform. All detector samples are retained, giving `sum(I) = 1` for every phase design. The image coordinate is spatial frequency in cycles per pupil half-width. The film shows the central frequencies from −2 to 2; the loss uses the complete detector grid. Intensity curves share one power scale and are never independently peak-normalized.

The target `I_star` is the focused pattern from the same aperture with `(a,b) = (0,0)`. Its squared norm `S = sum(I_star²)` remains fixed. The dimensionless loss is

```text
L = sum((I - I_star)²) / (2*S).
```

Zero loss is reachable in this numerical model. Matching intensity does not establish unique phase recovery or the performance of a fabricated optic.

## Derivatives and numerical evidence

The backward calculation is implemented explicitly; it does not call an automatic-differentiation library. With the normalized residual `r = (I-I_star)/S` and `w = F†(r*v)`, the real phase derivative is `g_phi = 2*Im(conj(u)*w)`. The parameter derivatives are `sum(g_phi*x)` and `sum(g_phi*x²)`. The backward field is a mathematical sensitivity, not a simulated beam travelling backwards through an experiment.

| Quantity | Value |
| --- | ---: |
| Initial parameters `(a,b)` | `(1.5, 2)` rad |
| Initial loss | 0.2605818922 |
| Initial gradient `(dL/da,dL/db)` | `(0.2188956249, −0.04178200304)` per radian |
| First accepted parameters | `(−1.126747499, 2.501384036)` rad |
| Loss after the first update | 0.1957411942 |
| Loss after 40 accepted updates | 0.00004557347 |
| Final parameters | `(0.002555445, 0.314167128)` rad |
| Difficult-start loss after 40 updates, starting at `(3,3)` | 0.2986665550 |

Each trial uses `theta_trial = theta - eta*gradient`. Backtracking starts at `eta = 12`, halves unsuccessful steps, and accepts a candidate only after recomputing a sufficient decrease in the same loss. The first accepted update increases defocus while improving the combined objective. The default run reduces loss by about 5,718 times. The difficult run reaches a nonzero plateau with a gradient norm of about `1.06e-6`, despite the known zero-loss reference. These examples demonstrate local improvement, not a guarantee of a global optimum.

`physics.py` checks the reverse gradient against both central finite differences and independent forward-tangent derivatives at four parameter/target settings. It also checks power conservation, global-phase invariance, the analytic central power of the uniform pupil, doubled pupil sampling, and monotone accepted losses. The displayed finite-difference sweep covers `h = 1e-1` to `1e-10` rad; its best maximum absolute error is approximately `2.9e-12` per radian. The canonical JavaScript model's initial values and both complete optimization histories were separately compared with this cache, agreeing to within `5.2e-15`.

Full values, assumptions, checks, and array shapes are recorded in `data/report.json`. The rendered numerical panels load `data/differentiable.npz`; scene rendering does not propagate fields or optimize designs.

## Computed quantities and schematic extensions

The phase and intensity sweeps, residuals, phase sensitivities, derivative checks, and optimization histories are computed from the stated Fourier model. Smooth transitions between cached sweep frames are presentation interpolation. The optimizer panel advances through actual accepted iterates.

The computation graphs, curved-lens ray path, clipping illustration, reconstruction chain, and closing icons are labelled schematic explanations. No curved lens is optimized, and no sensor-noise model or reconstruction system is trained. The ray discussion assumes a consistent intersection branch away from singular cases; clipping or path switches can invalidate a smooth local derivative.

The numerical cache also includes an optional quantization audit and a loss landscape, which are not shown in this film. The quantization audit optimizes a shifted target `(a,b) = (1,0)` for 80 steps, then rounds the phase to 2, 4, 8, and 16 levels. The resulting losses are approximately 0.181884, 0.051715, 0.002115, and 0.000556, compared with 0.00001047 for the smooth design. This audits the changed forward model; it does not optimize the discrete device. Hard phase rounding has zero ordinary derivative between thresholds and is nondifferentiable at a threshold.

The model omits polarization, dispersion, material constraints, sensor pixel integration, noise, saturation, manufacturing tolerances, and learned reconstruction. A gradient check validates a derivative of the implemented objective; physical-model fidelity requires separate evidence.

## Rebuild

From this episode directory, using the repository's configured environment:

```bash
source ../common/env.sh
../common/audio.sh .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$O5_PY" physics.py
PAR=1 ./preview.sh
WORKERS=3 ./render.sh --skip-audio
"$O5_PY" ../common/verify.py .
```

The wrappers resolve the configured Python, audio models, and media tools. Limit simultaneous renders to the repository's shared CPU budget. From the repository root, `make release E=E13` creates the distribution encode.

Full outputs are `e13_differentiable_optics_1080p60.mp4` with burned-in captions and `e13_differentiable_optics_1080p60_softsubs.mp4` with toggleable captions. The distribution copy is `../dist/e13_differentiable_optics_1080p.mp4`. Generated media, data, and caches are ignored by Git; narration, timing, subtitles, and source remain reviewable.

## Validation record

The physical checks and JavaScript comparison above passed. All eight 15 fps chapter previews matched their expected frame counts with no late narration cues, and their contact sheets were visually inspected. The finished 1080p60 film contains 19,060 frames over 317.667 seconds, with 100 subtitle cues (200 subtitle packets including gaps). Integrated loudness is −16.3 LUFS. The full-frame seam scan found no jumps, and five overlapping frames rendered from independent windows were pixel-identical. The 24 final verification stills were inspected for plot labels, equations, captions and layout. Reports and contact sheets are in `.verify/`.

The distribution copy is 15,774,714 bytes. It was staged into the local bilingual lesson with a matching SHA-256 hash. Chrome for Testing decoded the actual H.264 release at 1920 × 1080; all nine English guide links and a Chinese-page seek displayed nonblank decoded frames with active captions. Byte-range requests and invalid-range responses were checked. These are sampled playback checks, not a claim of a complete manual replay. The course validation record documents the tested scope.
