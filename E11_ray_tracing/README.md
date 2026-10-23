# E11 · Tracing Light Through a Lens

A 5:01.5 narrated film at 1920 × 1080, 60 fps. One idea: predict a geometric
ray path by intersecting the next physical surface, constructing its normal,
applying vector Snell refraction, and repeating. A finite bundle then tests the
small-angle focus of an actual two-surface singlet.

## Chapters

Times are approximate starts; `audio/timing.json` contains the exact narration
times and defines the frame windows.

| Time | Chapter | Main question |
| --- | --- | --- |
| 0:00 | Hook | Why do rays through one lens miss one common point? |
| 0:23 | A ray state | What information specifies a geometric ray? |
| 1:01 | Surface intersection | Which sphere root belongs to the physical lens? |
| 1:40 | Vector Snell | How does the interface set the outgoing direction? |
| 2:32 | Through both faces | How do two refractions reach a detector plane? |
| 3:10 | Image error | How do aperture, focus, and field change the spot? |
| 3:56 | Model limits | What do geometric landing points leave out? |
| 4:33 | Recap | How does a ray trace become a lens-design test? |

## Computed and schematic visuals

`physics.py` imports the same pure-Python 3-D tracer used by the course,
`teaching/tools/ray_optics.py`. Every lens fan, intersection, local surface
normal, transmitted direction, detector landing point, and quoted image error
is calculated from that model before rendering. Scenes load `data/rays.npz`;
they do not solve a new refraction problem per frame.

The coordinate convention is `(x, y, z)`, with `z` along the optical axis and
all lengths in millimetres. A positive surface radius places its centre after
its vertex. The prescription is `R1 = +50 mm`, `R2 = -50 mm`, centre thickness
`4 mm`, glass index `1.5`, and mechanical clear radius `12 mm`, with air on both
sides. The circular entrance stop stays at `z = -5 mm`. Field angles are in
the `x-z` plane. Meridional drawings show the `(z, x)` projection at equal
physical scales; enlarged focal plots have separately labelled axes.

The independent thick-lens paraxial prediction gives effective focal length
`50.675676 mm`, back focal length `49.324324 mm`, and paraxial image position
`z = 53.324324 mm` measured from the first vertex. These quantities are not
interchanged. At pupil radius `10 mm`, the marginal meridional ray crosses the
axis at `z = 49.971938 mm`. The illustrated on-axis best plane minimizes
centroid RMS over 192 equal-weight, equal-area circular-pupil samples
(8 annuli × 24 azimuths), giving `z = 51.107950 mm` and RMS `0.114622 mm`.
At the paraxial image plane the same samples have RMS `0.341966 mm`.
The subsequent field sweep keeps the detector at that same best on-axis plane;
the displayed spot is recentered for shape comparison and is not a distortion
measurement. All shown rays transmit through both clear apertures.

The unit-circle decomposition of Snell's law is a local coordinate diagram;
its output vectors are computed with the canonical refraction function.
Its normal points into the transmitted medium. A negative square-root
radicand means total internal reflection; this sequential transmitted-path
model reports that outcome rather than following a reflected branch.
The three small status icons are schematic classifications, not additional
simulations. Moving dots indicate progress along a calculated ray path, not
optical time of flight or photon arrival rates.

Chapter 6 includes a **separate ideal wave-optical reference**: the squared
Fourier transform of a uniformly illuminated circular pupil, computed using
the Airy `2 J1(v) / v` amplitude at 550 nm. Curves are peak-normalized to compare
width and have micrometre axes. They are not inferred from the geometric dots
and are not claimed to be the actual aberrated singlet's point-spread function.

The lens tracing omits Fresnel power weights, reflected ghosts, diffraction,
polarization, material dispersion, absorption, and manufacturing errors.
Increasing the number of sampled rays does not add those mechanisms.
Closing a stop can reduce geometric spread while rejecting incident light;
it can increase diffraction blur in the separate wave model.

## Rebuild

From this folder, with the repository's environment configured:

```bash
../common/audio.sh .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 5 "$O5_PY" physics.py
PAR=1 ./preview.sh
WORKERS=3 ./render.sh --skip-audio
nice -n 5 "$O5_PY" ../common/verify.py .
```

Alternatively, use the root `make` targets. `config.local.sh` is machine-local;
source `../common/env.sh` before invoking `$O5_PY` directly. Final media and
computed arrays are ignored by Git. The narrated segment timings, word timings,
and subtitle sources are small generated files and belong with the source.

From the repository root, `THREADS=3 make release E=E11` creates
`dist/e11_ray_tracing_1080p.mp4`: the shareable 1080p60 H.264 film with narration
and a toggleable English subtitle track.

## Validation

`physics.py` checks finite accepted paths, unit directions, the ordering of
marginal / best / paraxial focus, and the reduction of RMS at the selected best
plane. The common production tools check audio, frame counts, seams, loudness,
and subtitle packets. The initial Whisper review found number/spelling
differences and words assigned just before their segment boundaries; the
apparently missing first words are present in `audio/words.json`.

All eight chapter previews matched their target frame counts without late
animation cues. The full 1080p60 render contains **18,093 frames**, exactly the
target, and runs **301.550 s** (narration timing: 301.542 s). All 33 render pieces
matched their frame windows. The 93 subtitle cues produce 186 valid `mov_text`
packets, including gap packets; measured integrated loudness is **-16.3 LUFS**.

The full-frame seam scan found no jumps within 50 ms of a render-piece start.
Its three flagged moments were inspected: the continuous opening zoom at
7.22–7.32 s and the intended chapter-view entrances at 190.28 and 273.68 s.
Twenty-four final-film contact frames and additional boundary/subtitle frames
were reviewed. The vector-Snell formula, total-internal-reflection caption,
paraxial image label, and ideal-pupil reference stay clear of the burned
subtitles. Logs and inspection sheets are local artifacts under `.verify/`.
