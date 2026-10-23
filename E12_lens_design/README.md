# Lens Design: Choose What Better Means

Episode 12 of **Optics in 5 Minutes**. A 5:08 narrated film about defining an image-quality objective, searching a lens prescription under fixed constraints, and checking every tested field.

The film uses the exact sequential spherical-surface ray tracer introduced in E11. Its central comparison is deliberately small and reproducible: bend a singlet while preserving its paraxial effective focal length, and optimize one detector position shared by all fields in the chosen objective. An on-axis optimum and a three-field optimum answer different questions.

## Chapters

| Start | Scene | Idea |
| --- | --- | --- |
| 0:00 | `Ch0Hook` | Equal focal length does not imply equal image quality. |
| 0:21 | `Ch1Score` | Define centroid-referenced RMS geometric spot radius. |
| 1:02 | `Ch2Bending` | Choose bending and one shared detector plane as design variables. |
| 1:48 | `Ch3Axis` | Trace a bending scan and minimize the on-axis score. |
| 2:21 | `Ch4Fields` | Compare the on-axis and combined 0°, 3°, 6° objectives. |
| 3:07 | `Ch5Constraints` | Reject intersecting surfaces and lost rays; keep the aperture fixed. |
| 3:43 | `Ch6Validate` | Refine the pupil sample, check intermediate fields, and separate rays from diffraction. |
| 4:36 | `Ch7Recap` | Specify, trace, measure, change, and verify. |

Times refer to the generated narration timeline; scene boundaries are mid-gap and quantized to the output frame rate.

## Physical model and computed results

All lengths in `physics.py` are millimetres. The optical axis is z; positive field angle tilts the input in the x–z plane. The entrance stop is fixed at z = −5 mm with radius 6 mm. The singlet has constant refractive index 1.5, centre thickness 4 mm, and mechanical clear radius 12 mm. Spherical surfaces use signed radii. A shape parameter q = (c₁+c₂)/(c₁−c₂), with c = 1/R, redistributes curvature while preserving the full paraxial thick-lens power corresponding to EFL = 50 mm.

For each candidate bending, one axial image position minimizes the selected field-weighted variance. It is shared by those fields, never separately optimized for each field. The detector position is therefore a design variable; equal EFL does not imply equal back focal distance. The displayed first-order starting design uses its paraxial image plane; both final designs optimize bending and image position.

Each training field uses 96 equal-weight rays in equal-area pupil annuli. The single-field objective is the centroid-referenced squared RMS radius at 0°. The combined objective is the equal-weight mean of that squared radius at 0°, 3°, and 6°. The same entrance stop and material remain fixed. Designs are invalid if an entrance ray is lost, blocked, or totally internally reflected, or if the required edge thickness is below 0.5 mm. Search bounds are −1.2 ≤ q ≤ 1.2.

| Design | q | Image z (mm) | RMS at 0° (µm) | RMS at 3° (µm) | RMS at 6° (µm) | √combined merit (µm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Symmetric start, paraxial plane | 0 | 52.648 | 71.10 | 83.80 | 122.23 | 94.90 |
| On-axis objective | 0.68651 | 51.228 | 15.45 | 22.38 | 66.65 | 41.56 |
| Three-field objective | 0.46724 | 51.215 | 28.62 | 19.89 | 42.36 | 31.67 |

The finite deterministic search is a bounded educational design study, not a proof of a globally optimal optical system. The three-field design accepts a larger central spot to improve the combined score. The final comparison uses the same 0.32 mm square for every spot and centres each cloud on its own centroid. That centring excludes image displacement and distortion from this particular merit; it does not claim those requirements are satisfied.

`data/report.json` records complete results. Verification holds the selected prescriptions and image positions fixed, increases sampling to 1,536 rays per field, and adds 1.5° and 4.5° test fields. The combined advantage survives the denser sample. A 121-point bending scan and 25-point field curve are computed independently of scene rendering.

## Computed versus schematic

- **Computed:** lens profiles; all sequentially refracted ray paths and image intersections; centroid and RMS measurements; bending scans; on-axis and combined design searches; pupil refinement and intermediate-field checks. All use `teaching/tools/ray_optics.py` as the canonical backend, called from this episode's `physics.py`.
- **Computed rejected examples:** a physically intersecting R₁ = 18 mm, R₂ = −18 mm singlet, and a forbidden reduction of the clear opening to 5 mm while the entrance stop remains 6 mm. These are labelled illustrations of constraints, not members of the feasible optimization scan. The latter rejects 31 of 96 rays at 6° rather than silently rewarding their removal.
- **Computed wave reference:** an ideal uniform circular-pupil Airy pattern at λ = 0.55 µm, f = 50 mm, pupil radius 6 mm. Its first dark ring is approximately 2.8 µm. It is explicitly labelled as an ideal reference on a different displayed scale, not the actual aberrated singlet's PSF.
- **Schematic presentation:** moving dots indicate ray traversal at an arbitrary explanatory speed. The recap arrows show a design workflow. The lens-only cross-section enlarges the axial dimension for visibility and is labelled accordingly. Charts, circles, and text identify computed data; they do not add another physical model.

This is monochromatic geometric optics. It excludes diffraction in the ray score, dispersion, polarization, Fresnel power weighting, ghosts, scatter, manufacturing tolerances, and multielement optimization. Ray-retention fractions count samples, not optical power transmission. The episode links the diffraction question to E03 and the role of objective functions and gradients to E10; this bending search does not use a Maxwell adjoint.

## Rebuild

From this directory, using the repository's configured optics environment:

```bash
source ../common/env.sh
../common/audio.sh .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$O5_PY" physics.py
PAR=1 ./preview.sh
WORKERS=3 ./render.sh --skip-audio
"$O5_PY" ../common/verify.py .
```

The environment script resolves `O5_PY` and the model paths. The wrappers do that themselves. Keep simultaneous renders within the repository's shared CPU budget.

Outputs are `e12_lens_design_1080p60.mp4` (burned-in captions) and `e12_lens_design_1080p60_softsubs.mp4` (clean picture with toggleable captions). The distribution encode is `../dist/e12_lens_design_1080p.mp4`. Audio, render caches, numerical data, and media stay untracked. Narration, source, subtitle files, timing, and word alignments are reviewable source artifacts.

## Validation

The narrated track is 308.187 seconds with 42 segments. Whisper's round-trip flags were inspected: differences are English spelling variants, written versus spoken numbers, possessive spelling, and words assigned just outside segment windows; the opening words are present in the full word transcript. Physics checks cover preserved EFL, complete ray retention in the feasible scan, the three-field improvement, and its persistence under denser sampling.

All eight chapter previews have exact frame counts and no late animations. All 33 full-render pieces passed their frame checks. The assembled 1920 × 1080, 60 fps film has exactly 18,491 frames and a reported total duration of 308.187 seconds. Verification found 186 soft-subtitle packets for 93 cues, zero abrupt frame jumps, no piece-boundary discontinuities, and integrated loudness of −16.3 LUFS. Three captioned stills per chapter, plus full-resolution comparison and closing frames, were visually inspected for legibility, layout, and subtitle clearance. The detailed verification report and contact sheets are in the ignored `.verify/` folder.
