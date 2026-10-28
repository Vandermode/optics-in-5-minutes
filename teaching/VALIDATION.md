# Verification record · Optics teaching course

## Course audit and differentiable optics · 2026-10-28

The review followed each lesson from its opening question through the model, controls, explanation, concept checks and standalone lab. E03, E06, E07, E09, E11 and E12 required no additional scientific correction in this pass; the preceding film/core-lab fixes remain intact.

| Lesson | Clarification or correction |
| --- | --- |
| E01 | A zero-distance setting now shows zero propagation in every heatmap row; it no longer silently displays a 1 μm interval. |
| E02 | Input and output intensity images share one colour range for each object type, including the phase-object case. |
| E04 | Explain the sin(a)/a phase average and why visibility can return beyond a π-radian half-range: the wrapped phase distribution changes, rather than longer exposure restoring coherence. |
| E05 | Known and recovered phase share one symmetric radian colour scale. Intensity has its own labelled scale. |
| E08 | A requested mode disappears at cutoff instead of being replaced by another mode. Guided-mode expansion is explicitly restricted to the guided subspace. The lab distinguishes L2 normalization and scalar intensity integrals from β-weighted TE power flux. Through-port and circulating powers use separate plots so the critical-coupling dip remains visible. |
| E10 | Distinguish 4,096 array entries from 2,601 illuminated phase variables and explain the resulting 5,202 central-difference evaluations. Notebook target, initial and optimized intensities share a scale. |
| E13 | Add parameter → pupil phase → field → intensity → loss → gradient → update, with the residual-to-field sensitivity step written explicitly. Separate differentiable optics, automatic differentiation and adjoint methods. Connect to ray intersections and end-to-end optical/digital design with their assumptions stated. |

E13 uses a fixed-power 1D aperture with phase φ(x)=ax+bx² and a unitary Fourier operator. Its reachable target has a specified tilt and zero defocus. The objective is L=Σ(I−I★)²/(2ΣI★²), with the target and denominator fixed throughout a run. Default parameters (a,b)=(1.5,2) give L=0.2605818922 initially and 4.55734749×10⁻⁵ after 40 accepted updates. The initial reverse gradient [0.2188956249, −0.0417820030] agrees with central differences at h=10⁻⁵ rad to 9.61×10⁻¹² maximum absolute error. The initially negative defocus derivative is explained rather than hidden. Starting at (3,3) stalls near 0.298667 even though a zero-loss reference is known; this demonstrates a local optimization limit, not a global-optimality claim.

All thirteen notebooks ran in fresh CPU kernels in empty working directories, including every optional section: 84 code cells and 54 embedded PNG figures. The execution environment was Python 3.11.7, NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.2. E13 has four saved figures and a downloadable archive. Its tests compare JavaScript with an independent NumPy FFT fixture, an analytic uniform-aperture response, forward tangents and finite differences; they also check power, constant-phase invariance, grid refinement, accepted descent and the limitations of hard phase quantization. The numerical suites pass, including 1,720 E13 comparisons and regression checks for zero propagation, common intensity scales and mode cutoff.

All twelve local browser suites passed: 26 core lesson/viewport cases and 92 internal routes; 78 reader layouts; 39 language cases; course and chapter concept maps; 108 terms and 63 keyword links; bilingual search in every chapter; 48 existing film-guide cases; 12 geometric-optics cases; 52 course-logic cases; and six dedicated E13 cases at 1440, 390 and 320 px. The E13 checks include zero updates, default descent, the difficult-start plateau, preserved settings through language changes/reloads, and byte-checked notebook/figure downloads. No browser errors were recorded. The film suite uses isolated embeds and synthetic native media to test controls; it does not revalidate streaming or the unchanged complete films. Revised figures and the narrow-screen gradient comparison were visually inspected. A repeated build preserved all 46 generated pages, navigation/language data files and notebooks byte-for-byte.

The new chapter is bilingual and integrated into the course catalog, concept map, search and term explanations. It includes the interactive model, concept checks, CPU notebook, saved figures and instructor rubric. It has no companion film; catalog metadata omits film controls and routes its term links to the lesson. Existing films are unchanged in this pass. Local notebook execution does not claim authenticated Colab execution or physical device validation.

## Teaching-logic corrections · 2026-10-26

The six corrections align the films, required work, numerical evidence, and experiment order. Revised materials remain a local review version.

| Correction | Evidence and scope |
| --- | --- |
| E04/E07 film and page agreement | E04 now states that fields always add and intensities add when the averaged cross term vanishes. E07 identifies normal-incidence transmission into air and explains that only the zeroth propagating transmitted order does not establish a uniform-film response or exclude internal resonances. Both films were revoiced, rendered, and checked with their revised captions and guide times. |
| Core and optional work | E04’s core compares spectral and angular filtering; coherent-mode propagation remains optional. E07’s core separates order directions from efficiencies; RCWA convergence remains optional. Both notebooks execute through their first optional heading without defining or calling the advanced solvers. Core goals, quizzes, reports, and rubrics match that boundary. |
| E02 propagation through the lens | The notebook independently integrates input → front focal propagation → thin lens → back focal propagation. Lens half-widths of 32, 64, and 102.4 μm give relative field errors of 3.26×10⁻⁴, 7.26×10⁻¹⁰, and 1.09×10⁻¹⁴. Halving lens-plane or input spacing changes the field by less than 1.3×10⁻¹⁴. The object-against-lens case agrees with the Fourier result including its output chirp to 3.2×10⁻¹⁵; omitting that chirp gives 0.506 field error despite identical intensity. |
| E05 measurement fit versus reconstruction | Both errors are evaluated on the same field after each completed update. At update 150, seeds 3, 7, and 11 give Fourier-amplitude residuals of 0.654%, 0.786%, and 0.734%, but phase-aligned field errors of 18.262%, 17.426%, and 14.611%. The figure includes both curves and all three phase reconstructions on common scales. This finite example demonstrates a limitation of residual-only assessment; it does not prove nonuniqueness. |
| E09 → E10 model change | E10 explicitly identifies its scalar Fourier pupil operator, parameters, objective, and derivative checks alongside E09’s time-domain Yee model. Its optional frequency-domain scalar Helmholtz calculation is also identified as a separate model. |
| Prediction before controls | All twelve lessons place the prediction and guided experiment before the controls in both languages, including E03’s custom page. The focused browser suite checks both document order and rendered layout at 1440 and 390 px. |

All six modified notebooks executed successfully in clean CPU kernels, including optional sections: 38 code cells and 26 PNG figures. Their source code and executed outputs agree. The E04 angular-average check agrees with the uniform-strip formula within 4.6×10⁻⁸; narrowing spectral width or angular extent holds the other parameter fixed. New and revised figures were visually inspected. Rebuilding the full bilingual course twice produces identical artifacts and preserves notebook outputs.

The revised E04 film contains 18,154 frames (302.567 seconds) and 90 subtitle cues; E07 contains 19,361 frames (322.683 seconds) and 91 cues. Both are 1920 × 1080 at 60 fps with H.264 video, AAC audio, and measured integrated loudness of −16.3 LUFS. All chapter previews, piece counts, seam scans, and final formula frames were reviewed; detected abrupt changes are intentional transitions.

Both complete distribution files were played through the local lesson pages in Chrome for Testing 156.0.8078.47. They decoded 1080p frames, sought to the revised guide times, loaded every WebVTT cue, and displayed active captions without player or browser errors. This used the actual films, separately from the synthetic native-player fixture in the general browser suite. No new YouTube upload or public-site deployment is part of this review.

## Geometric-optics additions · 2026-10-23

E11 and E12 use a shared three-dimensional, sequential transmitted-ray specification in Python and JavaScript. The added numerical suite passes 768 checks, including 45 independent meridional paths solved with scalar Snell angles and bracketed surface intersections, an ABCD paraxial reference, ray reversal, physical-cap selection, total internal reflection, fixed focal power, common-plane stationarity, pupil refinement, and penalties for lost rays. Agreement between language ports is checked separately from these independent references.

Both standalone notebooks executed in fresh kernels in empty temporary directories with their optional sections included: 15 code cells and nine saved PNG figures in total. The execution environment was Python 3.11.7, NumPy 2.4.6, SciPy 1.17.1, and Matplotlib 3.11.2. The notebook files embed their numerical engine and need no repository imports or external data. This is local kernel verification; no authenticated Colab execution is claimed for these additions.

At the default E12 aperture, the three-field design has q = 0.467236 and one common detector at z = 51.214566 mm from the first vertex. Its sampled merit is 0.00100277 mm². Increasing the pupil sampling to 576 and 2304 rays per field gives 0.00100648 and 0.00100740 mm² at the same detector. Intermediate fields at 1.5° and 4.5° are audited separately. The centroid-relative metric omits distortion and diffraction and does not certify a globally optimal or manufactured lens.

The native-video browser fixture is a generated 32×32 silent WebM test clip. It checks metadata loading and seeking without requiring production media in Git; it does not establish playback quality of either complete film. Local-film previews are excluded from ordinary website export until published media metadata is supplied.

All ten browser suites passed after the additions: 24 lesson/viewport cases, 72 reader cases, 36 language cases, 12 homepage map layouts and 384 destination anchors, 48 chapter-map cases plus extra widths and static routes, 48 term-network cases, all 58 episode keywords, bilingual concept search across all 12 chapters, 48 video-guide cases, and 12 focused geometric-optics cases. Widths span 320–1440 px where applicable. Checks include physical control effects, optimization cancellation, keyboard navigation, language and URL-state retention, native-video seeking, and layouts without JavaScript. No browser errors were recorded. The numerical command passes 9,148 course checks, 20 spectrum checks, 852 search checks, and the 768 geometric comparisons.

The two films were rendered at 1920 × 1080 and 60 fps: E11 has 18,093 frames (301.55 seconds), and E12 has 18,491 frames (308.187 seconds). Each has 93 English subtitle cues and measured integrated loudness of −16.3 LUFS. Every rendered piece has its expected frame count. E11's detected abrupt changes are intentional zooms or chapter entries; E12 has none. Neither film has a piece-boundary discontinuity. Chapter contact sheets and full-resolution formula/comparison frames were visually inspected.

The complete H.264/AAC distribution files were also played in Chrome for Testing 156.0.8078.47 through the lesson pages, served with byte-range support. Both decoded 1080p frames, sought to their guide timestamps, loaded all 93 WebVTT cues, and displayed active English captions with no player or browser errors. Playback was confirmed from decoded frames and nonblank canvas samples, not metadata alone. The local server's prefix/suffix byte ranges and unsatisfiable-range responses were checked against the original movie bytes. These checks cover the local complete films; new YouTube publication remains separate.

## Original ten-episode verification · 2026-10-17

Checked on 2026-10-17. The machine-readable reports are [notebook execution](validation-notebooks.json) and [browser verification](validation-browser.json). These checks validate the stated numerical models; they do not establish experimental device accuracy.

## Notebook execution and figures

All ten notebooks ran from a fresh kernel in an empty temporary working directory, with two CPU numerical-library threads and no repository imports or external datasets. The environment was Python 3.12.13, NumPy 2.4.3, SciPy 1.16.3, and Matplotlib 3.11.2. Every optional section also ran. All 62 code cells completed, all scientific assertions passed, and 40 static PNG outputs were saved and visually inspected. Execution time includes kernel startup and the extensions; learner working time is longer because it includes prediction, parameter changes, and interpretation.

| Notebook | Executed code cells | PNG figures | Total CPU run |
| --- | ---: | ---: | ---: |
| E01 | 6 | 3 | 4.53 s |
| E02 | 6 | 4 | 3.70 s |
| E03 | 7 | 4 | 5.40 s |
| E04 | 6 | 4 | 4.29 s |
| E05 | 6 | 4 | 5.79 s |
| E06 | 6 | 4 | 3.81 s |
| E07 | 6 | 4 | 4.11 s |
| E08 | 6 | 4 | 5.33 s |
| E09 | 6 | 4 | 6.69 s |
| E10 | 7 | 5 | 6.61 s |

Each lesson includes a ZIP and individual PNG files, independent of notebook iframe rendering. Labels and legends were inspected; the new labs use constrained figure layout to prevent adjacent axes from colliding.

## Numerical evidence

The browser-model suite passes 9,112 numerical comparisons and invariants, plus E03's existing regression suite. Independent fixtures use NumPy FFTs, complex 2×2 transfer matrices, SciPy root finding, analytic limits, and central finite differences. Every default and named preset yields finite plotted data.

| Episode | Checks and representative result |
| --- | --- |
| E01 | Identity at z=0, analytic grid-plane-wave phase, propagating-band norm, evanescent decay; errors below 2×10⁻¹⁴ in the notebook examples. Padding is compared independently of sample spacing. |
| E02 | Correct focal coordinates and sample spacing; exact discrete 4f reversal; direct Fresnel kernel versus FFT plus output chirp agrees to 6×10⁻¹⁵ relative field error. |
| E03 | Airy origin/zero, Rayleigh dip, MTF cutoff, source addition, symmetry, units, and zoom invariance. Browser Airy error versus SciPy is below 3.7×10⁻⁶. |
| E04 | Gaussian spectral quadrature, visibility normalization, unchanged contrast for repeated fixed speckle, independent-pattern contrast, positive semidefinite mutual intensity, and all-mode/source-sum agreement within 8×10⁻¹⁵. |
| E05 | Isolated-sideband complex-field reconstruction and refocusing agree within 5×10⁻¹⁵. Overlap, Nyquist violation, and crop truncation deliberately increase error. Seeded GS runs satisfy amplitude constraints and show residual ambiguity. |
| E06 | Wrapped/unwrapped fields agree at the design wavelength. Numerical staircase-order efficiency agrees with sinc²(1/M) within 8×10⁻¹⁰. Axial maximum and paraxial focal prediction are distinguished. |
| E07 | Lossless stack flux, no-contrast limit, Bloch trace, and grating-angle identity. RCWA agrees with uniform-film Airy amplitudes; final retained-order refinement changes reflectance by 1.39×10⁻⁴. Flux conservation is checked separately from convergence. |
| E08 | Effective-index bounds, orthogonality, matrix residual below 2×10⁻¹⁴, analytic slab comparison, supermode power conservation, all-pass ring power balance, and critical coupling. |
| E09 | Reference-subtracted dielectric reflectance is 0.04084 versus analytic 0.04 on the stated grid. PML echo decreases under refinement; stable/unstable 1D steps and a 2D CPML reference comparison are exercised. Echo ratios apply to these specific pulses and windows. |
| E10 | Every phase derivative and every small Helmholtz derivative checked by central differences; relative errors below 7×10⁻¹⁰. Accepted backtracking steps decrease loss. Quantization and smoothing are evaluated afterward. |

## Browser behavior and compatibility

Headless Chromium was used at 1280×900 and 390×900. Checks cover all ten lessons, nonblank computed canvases, labels, horizontal overflow, keyboard slider operation, every new control, presets, reset, concept feedback, PNG downloads, clipboard experiment links, and URL state restoration. E09 and E10 workers stop and restart correctly. No browser console or page errors were observed. All 49 linked local course routes and downloads return HTTP 200. A package-level scan also checked 165 local HTML/Markdown links and assets. The added E09 stored-time selector was checked separately at 10 and 27 periods and after reset.

The course browser suite substitutes a minimal document for YouTube's iframe, so third-party player scripts and permission messages do not affect checks of the lesson code. The report marks external video playback as outside this suite; playback is checked separately against the real player.

The E03 regression preserves its existing model and notebook GitHub path. Legacy root experiment parameters, section anchors, and notebook download remain available. The homepage has a distinct skip-link anchor so it does not conflict with E03's historical #main link. Grating rays use equal coordinate scales; line weights do not imply diffraction efficiency.

## Live Colab verification limit

Local notebook execution and saved-image inspection are complete. The nine new notebooks have not been executed in an authenticated live Colab runtime in this session: an authenticated browser-control connection was unavailable. Their public links and notebook files can be checked independently, but that is not a substitute for a live kernel test or iframe display test.

The earlier E03 live Colab check passed its seven numerical cells, while three of four rich figure outputs still failed to load; its inversion figure displayed. That historical result is recorded in the E03 README. No claim is made that the new publication fixes Colab's frontend. Saved PNG outputs, lesson figure ZIPs, and the Colab Files-sidebar download provide fallbacks.

## Reproduce

See [the course README](README.md) for installation and commands. Numerical fixtures and all tests are included. The manual Pages workflow packages the whole course; the CI workflow checks notebooks and browser behavior without deploying Pages. Publication preserves the existing Site's URL and public audience, with the previous saved version retained by Sites.
