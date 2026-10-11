# Optics in 5 Minutes: series outline

**Season 1: the ten big ideas of computational optics & photonics**

Draft 2026-10-10; Season 1 was produced 2026-10-11 (see `README.md`). E01 is the pilot, `E01_angular_spectrum/`. Beats below are the plan; each episode's README describes what was actually made.

---

## 0. The pitch

The season is ten five-minute films in the style of 3Blue1Brown, one idea each. The viewer needs only calculus
and a little linear algebra, and goes from "light is a wave" to "a computer designs the optics". Every field on
screen is computed with the method the episode teaches. Each episode ends with the algorithm in a few lines and
an honest note on where it breaks.

**Series spine: light is linear.** Each episode finds the representation in which the physics is simple, does the
simple thing there, and changes back:

> plane waves (E01) → focal plane = Fourier plane (E02) → pupil & OTF (E03) → coherent modes (E04) →
> hologram sidebands (E05) → pointwise phase (E06) → diffraction orders & Bloch waves (E07) → guided modes (E08) →
> local Yee updates & complex coordinates (E09) → the transpose, i.e. light run backward (E10)

**Second thread: the model ladder.** The season climbs one level of approximation at a time. Each episode says
why the simpler model failed:

```
paraxial Fresnel (E02) · exact scalar ASM (E01) · thin element TEA (E06) · multi-slice BPM (E08)
· periodic RCWA (E07) · full Maxwell FDTD (E09) · design on top of any of them (E10)
```

---

## 1. The ten

| # | Episode | The idea in one sentence (also the recap line) | Hero shot |
|---|---|---|---|
| E01 ✅ | How Light Spreads: the Angular Spectrum Method | Break a field into plane waves, advance each one's phase, add them back: FFT, multiply, inverse FFT. | slit field + k-circle |
| E02 | Why a Lens Computes a Fourier Transform | Far away, or at a lens's focus, each direction lands on its own spot, so light lays out its own spectrum. | three tilted waves → three focal spots |
| E03 | The Diffraction Limit: Why Every Image Is Blurry | Imaging is convolution with the PSF. A lens only keeps directions with sin θ ≤ NA, so details finer than ~λ/2NA never arrive. | NA cap on the k-circle ↔ Airy disc |
| E04 | Coherence: When Does Light Interfere? | Fields add when phases stay in step; when phases wander, intensities add. Partially coherent light is a sum of coherent patterns. | shifted fringe copies washing out |
| E05 | The Phase Problem: Holography & Phase Retrieval | Cameras lose phase. Interference writes it into fringes, and constraints let a computer recover it, or design it. | hologram spectrum with three islands |
| E06 | Flat Optics: a Lens One Micron Thick | A thin element only delays light. Phase wraps at 2π, so a lens can be a micron thick, until "thin" stops being true. | thick lens slicing into a kinoform |
| E07 | Periodic Structures: Gratings, Bloch Waves & Band Gaps | Flat interfaces conserve kx. Periodic ones conserve it only up to multiples of K = 2π/Λ, and where copies meet, gaps open. | band folding → gap opening |
| E08 | Trapping Light: Waveguides, Modes & Resonators | Trapped light settles into modes, the eigenvectors of the cross-section. A resonator stores light where round trips add in phase. | coupler beating; resonance phasor spiral |
| E09 | Maxwell on a Grid: FDTD & the Perfectly Matched Layer | E updates H and H updates E, one neighbour at a time. A layer of complex-stretched space absorbs whatever leaves. | PML ribbon twisting into complex space |
| E10 | Designing Light with Gradients: the Adjoint Method | One forward and one backward simulation give the gradient for every pixel at once. | forward × adjoint sensitivity map |

### Why these ten

- **Load-bearing.** Each idea is used constantly in computational optics and photonics, and later episodes build on it.
- **One computable picture.** Each idea has a single "aha" visual that can be computed, not just drawn.
- **A ladder.** The season runs free space → imaging → shaping light → structured matter → full Maxwell → design.
  Each episode's limitation sets up the next episode.
- **Close to research practice.** E04, E05, E06, E09 and E10 cover tools computational-optics labs use daily:
  partially coherent illumination, holographic metrology, thin-element vs full-wave modelling, and end-to-end design.

**Just missed the cut** (first in line for Season 2, §6): polarization & the geometric phase, metasurfaces in
depth, the beam propagation method, sampling & band-limited ASM, Fourier ptychography.

**Possible swaps.** For a more imaging-oriented season, replace E08 with Fourier ptychography. For a more
photonics-oriented season, replace E05 with polarization & the geometric phase.

### Arc

```
Act I    Free space       E01 Angular spectrum ─► E02 Lens = Fourier transform
Act II   Seeing           E03 Diffraction limit ─► E04 Coherence ─► E05 Phase problem
Act III  Shaping light    E06 Flat optics ─► E07 Periodic structures ─► E08 Waveguides & resonators
Act IV   Computing light  E09 Maxwell on a grid ─► E10 Designing with gradients
```

Each episode's last line is the next episode's hook question.

---

## 2. Episode grammar

Taken from the pilot: 5:07, 46 narration segments, 770 words, 9 chapters.

| Block | Time | Job |
|---|---|---|
| Hook | 0:00–0:25 | a computed phenomenon and one question |
| Build | 0:25–~3:45 | 3–4 chapters; the picture first, then at most one key equation per chapter |
| On a computer | ~3:45–4:35 | the algorithm in ≤ 4 lines, one numerical pitfall, a computed demo |
| Recap | 4:35–5:00 | the one-sentence idea from §1 word for word, where it breaks, and the teaser |

- **Budget.** 4:45–5:15 per episode, 720–800 narration words (~150 wpm), ~45 segments of at most 2 sentences,
  6–9 chapters. Hold about 1 s of silence before each "aha".
- **Computed, not drawn.** Physical fields come from the method being taught. Schematic panels are labelled as
  schematic in the README, and on screen wherever they could pass for data.
- **Picture before symbol.** A symbol only appears next to the thing it names. Colours stay fixed for the whole series (§3).
- **Honest limits.** Every episode says where its model fails.
- **No unpublished results.** Every film uses generic computed demos; comparisons come from solvers written for the episodes.
- **Narration.** Second person, short sentences, real questions. The phonetic "say" text spells out symbols
  ("kay ex", "zee", "F F T"). Run the Whisper round-trip check.
- **60-s cut.** Each episode also gets a one-minute version (hook, hero shot, one-sentence answer), like the
  existing 1-min ASM film.

---

## 3. Visual language

**Colour code.** This extends the pilot's `asm5_common.py`.

| Quantity | Colour | From |
|---|---|---|
| field U | BLUE | E01 |
| kx, transverse spatial frequency | YELLOW | E01 |
| kz, propagation constant β | TEAL | E01 |
| transfer function, pupil, filter | GREEN | E01 |
| evanescent, forbidden, loss | RED | E01 |
| imaginary part | PURPLE | E01 |
| reference wave, source points | GOLD | E04 |
| matter (n, ε, height h) | GREY / white solids | E06 |
| grating / lattice vector K | ORANGE | E07 |
| adjoint (backward) field | PINK | E10 |
| intensity / phase | `HEAT` LUT / cyclic `PHASE_HUE` | E01 |

**Recurring visual atoms.** Build these once in `Optics5min/common/`, starting from `asm5_common.py`.

| Atom | Introduced | Reused in |
|---|---|---|
| phasor arrows and arrow sums | E01 | E02 phase contrast · E04 fringes, speckle walk · E06 quantization polygon · E08 resonance spiral |
| plane-wave panel with k-arrow | E01 | E02, E07, E08 |
| k-circle | E01 | E02 directions → spots · E03 NA cap · E07 two circles, grating copies · E08 TIR · E09 the grid's distorted circle |
| space ↔ k-space split screen | E01 | E02, E03, E05 |
| dense → diagonal matrix | E01 | E04 coherent modes · E07 RCWA · E08 mode solver · E10 transpose |
| x–z field map; live wave Re{U e^(−iωt)} | E01, metasurface films | all |
| code card (≤ 4 lines) | E01 | all |
| model-ladder bar | E06 | E06–E10 |
| series map (10 nodes, the current one lit) | outro | all |

---

## 4. Episodes

### E01 · How Light Spreads: the Angular Spectrum Method ✅

**Status.** Done. The 5-min film is `Optics5min/E01_angular_spectrum/` (5:07, burned-in and soft
subtitles), ported 2026-10-11 from the original pilot folder (outside this repo).

**Idea.** Break the field into plane waves. Each one advances its own phase e^(i kz z). Add them back up.

**Existing chapters.** Hook (slit) · waves & k · complex arrows · plane waves & the k-circle · the Fourier basis ·
propagation is diagonal · evanescent waves · the algorithm (sampling, zero padding, slit and lens) · recap.

**Optional retrofit (~15 s).** This is cheap now that the pilot has `fast_render.py`.
- A 3-s series ident at 0:00.
- Change the last line to set up E02: *"And when z gets large, these three steps collapse into one Fourier
  transform, which a lens does for free."*
- A 3-s series-map outro.

---

### E02 · Why a Lens Computes a Fourier Transform

**Idea.** Far away, or at a lens's focus, every plane wave lands on its own spot (x = f sin θ), so light lays out
its own spectrum. Two lenses (a 4f system) let you edit that spectrum with a mask.
Builds on E01 (plane waves, kx, the Fourier basis). Sets up E03 (the pupil) and E05 (Fourier-plane filtering).

**Hook (0:00–0:25).** A laser shines through a small transparency (a mesh, a letter "F"). A far wall shows dots
and streaks: the picture's Fourier transform, computed by light. A lens shows the same pattern a few centimetres
away. *"Why does light compute Fourier transforms, for free?"*

- **0:25–1:05 · Far away, directions become positions.** Zoom E01's slit map out from 150λ to 10⁵λ. The shadow
  becomes a broad pattern whose profile overlays E01's spectrum exactly. Each direction (kx = k sin θ) travels in
  a straight line and lands at x = z tan θ, so the far field is ∝ Û(k sin θ) (Fraunhofer). The Fresnel number
  a²/λz separates near from far.
- **1:05–1:50 · The paraxial shortcut.** Overlay a parabola on the top of the k-circle: kz ≈ k − kx²/2k.
  Propagation becomes chirp · FT · chirp. Animate the expansion of (x − x′)² into three terms: the cross term is
  the Fourier kernel, and far away the x′² chirp no longer matters.
- **1:50–2:45 · A lens cancels the chirp.** A thin lens is a quadratic phase e^(−ikx²/2f), the same lens as the
  pilot's demo. **Signature shot:** three plane waves at different tilts (kx in YELLOW) pass through the lens and
  converge to three separate focal points. *"A lens sorts light by direction."* Then the slit's spectrum appears
  in the focal plane. With the object one focal length in front of the lens, the leftover phase vanishes and the
  transform is exact.
- **2:45–3:45 · The 4f machine.** A second lens transforms back, and the image comes out flipped (two FTs give
  x → −x). The middle plane holds the spectrum, so a mask there multiplies it. Demos:
  - a pinhole blurs the image;
  - a central stop keeps only the edges;
  - **Zernike phase contrast** (Nobel Prize 1953). A transparent "cell" is invisible in brightfield. In the phasor
    picture, the weak scattered arrow is at 90° to the strong DC arrow. Shift the DC by 90° and the arrows line up,
    so the cell becomes visible.
- **3:45–4:35 · On a computer.** Three propagators:
  - ASM (E01): same output grid as the input; best up close.
  - Single-FFT Fresnel: one FFT, and the grid grows with distance, Δx′ = λz/(NΔx); best far away.
  - Fraunhofer: one FFT.

  Pitfall: chirps alias. The ASM transfer function is safely sampled for z < z_c = NΔx²/λ, and the Fresnel
  impulse response for z > z_c. Show chirp samples going from smooth to aliased as z passes z_c.
- **4:35–5:00 · Recap.** Where it breaks: Fresnel is paraxial, so at high NA use E01's exact kz. Teaser: *"A real
  lens is finite: it only catches directions up to some angle. That cutoff is why every image is blurry."*

**Compute.**
- Far-field zoom: `asm5_common.asm_rows` with a larger N.
- Focal-plane sorting: ASM through a lens phase.
- 4f filters and phase contrast: 2D FFT of a synthetic phase object.
- Chirp-aliasing panel and propagator comparison (numpy).

**Watch-outs.**
- The lens gives an exact FT only between its focal planes; elsewhere a quadratic phase remains.
- x = z tan θ is exact; the Fresnel form is paraxial.
- Explain why the image is inverted.

---

### E03 · The Diffraction Limit: Why Every Image Is Blurry

**Idea.** Imaging is convolution with the point-spread function (PSF), and the lens is a hole in k-space. It keeps
only kx up to k·NA, so details finer than about λ/2NA never reach the camera.
Builds on E01 (k-circle, evanescent waves) and E02 (lens = FT, 4f). Sets up E04 (intensities add) and E10
(end-to-end imaging).

**Hook.** Fluorescent beads under a perfect microscope: every point becomes a disc with rings, and two beads
closer than ~200 nm merge into one. *"Why can't a perfect lens make a perfect point?"*

- **0:25–1:05 · Images are stamps.** The object is a set of points. Each point stamps a copy of the PSF, and the
  image is the sum (a convolution). Because the system is linear and shift-invariant, it is diagonal in the
  Fourier basis (E01 callback): convolution becomes multiplication, and the PSF's FT is the transfer function.
- **1:05–2:00 · The lens is a hole in k-space.** The plane waves leaving a point fill the k-circle. The lens's
  acceptance angle keeps only the cap kx ≤ k·NA (NA = n sin θmax). Evanescent waves never leave the object
  anyway (E01). **Signature shot:** an NA slider links the k-circle cap to the PSF. As the cap widens, the Airy
  disc shrinks (radius 0.61λ/NA).
- **2:00–2:50 · Abbe's grating test.** A grating of period d diffracts at sin θ = λ/d. To form an image at all,
  the lens must catch the 0th order and at least one 1st order. On-axis illumination gives d_min = λ/NA. Tilting
  the illumination so the 0th order sits at the edge of the cap gives d_min = λ/2NA. Computed with a 4f system:
  the grating's image vanishes as its first orders leave the cap.
- **2:50–3:45 · Coherent vs incoherent light.**
  - Laser light: the fields are filtered by the pupil (cutoff NA/λ).
  - Fluorescence: intensities add, and the OTF is the pupil's autocorrelation. Animate the pupil sliding over
    itself to build a tent. The OTF reaches 2NA/λ, but contrast fades toward the cutoff (MTF).
  - Computed test chart, side by side: coherent ringing vs incoherent smoothness.
- **3:45–4:35 · On a computer.** image = IFFT(FFT(object) × OTF), the same three steps as E01. Undoing the blur by
  dividing by the OTF makes noise explode wherever the OTF ≈ 0 (show on a log plot). Wiener filtering,
  Ĥ*/(|Ĥ|² + 1/SNR), is the stable version. Frequencies beyond the cutoff are gone; only priors or extra
  measurements bring them back. *"That is where computational imaging begins."*
- **4:35–5:00 · Recap.** Where it breaks: off-axis aberrations make the PSF vary across the field (Season 2).
  Teaser: *"We just said intensities add for fluorescence. Why? When does light interfere, and when doesn't it?"*

**Compute.**
- PSF and OTF from a 2D pupil, with an NA sweep (numpy).
- Grating imaged through a 4f system.
- Coherent vs incoherent test chart.
- Wiener deconvolution (numpy).
- Focusing x–z map: `lens_map`.

**Watch-outs.**
- Magnification is not resolution, and pixel size is not the optical cutoff.
- Incoherent imaging's 2× cutoff does not make it "better": its contrast falls.

---

### E04 · Coherence: When Does Light Interfere?

**Idea.** Fields add when their phases stay in step. When the phases wander, the cross term averages away and
intensities add. Partially coherent light is an incoherent sum of coherent patterns.
Builds on E01 (phasors), E02 (far field) and E03 (intensity PSF). Sets up E05 (holography needs coherence).

**Hook.** A laser on a wall gives sparkly speckle; an LED gives a smooth spot. In a double slit, a laser gives crisp
fringes and a light bulb gives none, unless a distant pinhole is added. *"What does a laser have that a bulb
doesn't?"*

- **0:25–1:05 · Interference is arrow addition.** |a₁ + a₂|² = |a₁|² + |a₂|² + 2|a₁||a₂| cos Δφ. With Δφ frozen,
  you get fringes. If one arrow jitters faster than the camera can follow, the cross term averages to zero and
  intensities add. A dial shows the visibility V = (Imax − Imin)/(Imax + Imin).
- **1:05–1:50 · Temporal coherence.** Each colour has its own fringe spacing. Summed over a spectrum, fringes
  survive only near zero path difference. The envelope width is the coherence length, ≈ λ²/Δλ: centimetres to
  kilometres for lasers, ~10 µm for an LED, ~1 µm for sunlight. The envelope is the FT of the spectrum (the basis
  of Fourier-transform spectroscopy and OCT).
- **1:50–2:45 · Spatial coherence.** An extended source is many independent points. Each sends a tilted plane
  wave, and a tilt shifts the fringes (shift theorem). The camera adds the shifted copies, so the fringes blur,
  and they vanish once the source's angular size exceeds ~λ/d. **Signature shot:** a stack of shifted fringe
  patterns summing to grey. van Cittert–Zernike: coherence vs separation is the FT of the source shape. Michelson
  measured Betelgeuse's diameter this way in 1920.
- **2:45–3:25 · Speckle: coherence's dark side.** At a rough surface, each pixel's field is a random walk of arrows.
  The intensity is exponentially distributed with contrast 1, and the grain size is ≈ λ/NA (E03's PSF).
  Averaging N independent patterns lowers the contrast to 1/√N, so less coherence means smoother light.
- **3:25–4:30 · Modes, and on a computer.** The mutual intensity J(x₁, x₂) is a Hermitian matrix. Its eigenvectors
  are coherent modes and its eigenvalues are their powers (the E01 diagonalization again). Code card:
  - coherent light: propagate fields;
  - incoherent light: convolve intensities;
  - partially coherent light: propagate a few modes and add their intensities.

  For an extended source in the paraxial regime, the modes are tilts, so the result is the coherent pattern
  convolved with the scaled source image. Demo: a DOE pattern under a laser vs an LED. At large angles the blur
  depends on direction, and a few coherent modes still capture it.
- **4:30–5:00 · Recap.** Teaser: *"Coherence lets us do something a camera can't: record the phase of light."*

**Compute.**
- Phasor-jitter fringes; broadband interferogram; extended-source washout (numpy).
- J matrix and its eigenmodes in 1D.
- Speckle from a random phase screen and an FFT.
- LED vs laser: coherent pattern blurred by the scaled source image, plus a spectral average.

**Watch-outs.**
- λ²/Δλ is an order of magnitude; the prefactor depends on the spectral shape.
- "Lasers are coherent because they are monochromatic" is only half the story; spatial coherence matters too.
- Incoherent light still diffracts.

---

### E05 · The Phase Problem: Holography & Phase Retrieval

**Idea.** Cameras record |U|² and lose the phase. Interference with a known wave writes the phase into fringes
(holography). Constraints let a computer recover the phase (phase retrieval), or design one (holograms).
Builds on E02 (the Fourier plane) and E04 (interference). Sets up E06 (phase masks) and E10 (gradient-based
holograms).

**Hook.** A living cell, or a glass bead, is nearly invisible to a camera, yet it changes the light a lot. It
changes the angle of the arrows, not their length. *"How do you photograph what a camera can't see?"*

- **0:25–1:00 · What a camera throws away.** Show the field as E01's arrows, coloured by phase hue. The camera
  squares the arrow lengths, and the hue drains to grey. A transparent object only delays light, φ = k₀(n−1)h, so
  all its information sits in the phase. Green light oscillates about 600 trillion times per second; no detector
  can follow the arrow.
- **1:00–2:05 · Holography: add a reference.** |U + R|² = |U|² + |R|² + UR* + U*R, and the cross terms contain U
  itself.
  - In-line holography (Gabor 1948): the twin image U* overlaps the real one.
  - Off-axis holography (Leith & Upatnieks 1962): tilt R, and UR* shifts in k-space.

  **Signature shot:** fringes → FFT → three islands → crop one → inverse FFT → the full complex field (hue) →
  refocus to any depth with ASM (E01). Condition: the carrier frequency must be at least 3× the object bandwidth.
- **2:05–3:05 · No reference? Phase retrieval.** Suppose we know |U| in two planes. Gerchberg–Saxton: guess a
  phase → FFT → impose the measured amplitude → inverse FFT → impose the object constraint → repeat. Show a
  cartoon of two constraint sets with the iterate bouncing between them toward their intersection, beside the
  computed reconstruction sharpening. The weak points are ambiguities (twin, shift, global phase) and
  stagnation. More diverse data fixes them: several distances, ptychography's overlapping scans, or Fourier
  ptychography's tilted illumination (E03's tilt trick).
- **3:05–3:50 · Run it backwards: computer-generated holograms.** Ask for a target image in the far field from a
  phase-only mask (an SLM or a DOE). It is the same loop, with the target in place of the measurement. The mask's
  phase evolves and the projection sharpens. The free phase causes speckle (E04 callback). Modern versions use
  gradient descent (E10).
- **3:50–4:35 · On a computer.**
  - Digital holography: FFT → crop the sideband → inverse FFT → ASM refocus.
  - GS: loop { FFT; set amplitude; inverse FFT; constrain }.
  - Pitfalls: sample each fringe with at least 2 pixels; keep the carrier above 3× the bandwidth; GS stalls, so
    use random restarts or gradients.
- **4:35–5:00 · Recap.** Where it breaks: holography needs coherence and stability; retrieval needs diverse data.
  Teaser: *"We just designed a phase mask on a computer. How do you build one, and what does 'thin' really mean?"*

**Compute.**
- Synthetic phase object and off-axis hologram pipeline (numpy).
- GS retrieval and CGH (numpy, ~100 lines).

**Watch-outs.**
- Phase retrieval is not always unique, especially in 1D.
- Off-axis holography trades field of view for bandwidth.

---

### E06 · Flat Optics: a Lens One Micron Thick

**Idea.** A thin element only delays light, φ = k₀(n−1)h. Phase wraps at 2π, so any lens can be folded into
micron-thin zones. The price is steps, colour errors, and eventually the "thin" assumption itself.
Builds on E01 (the lens demo), E02 and E05 (phase masks). Sets up E07 (fine zones are gratings) and E09 (where
thin breaks).

**Hook.** A lens tens of microns thick and a ring-patterned plate one micron thick focus to the same spot
(computed). Under white light, the flat one smears the colours along the axis. *"How can a micron do a thick
lens's job, and what's the catch?"*

- **0:25–1:05 · Glass as a delay: the thin element approximation (TEA).** Light slows to c/n, so a column of
  height h adds (n−1)h of path, a phase φ = k₀(n−1)h. If the element is thin, every point acts on its own:
  multiply the field by e^(iφ(x)). Show a live wave entering a lens profile: the crests are delayed column by
  column and leave as a curved wavefront. The model-ladder bar appears on the "thin element" level.
- **1:05–1:55 · Wrap the phase.** *"Phase is an angle, so a full turn is free."* Subtract whole turns, which means
  heights modulo λ/(n−1) ≈ 1 µm. **Signature shot:** the thick lens is sliced into slabs λ/(n−1) tall, the slabs
  vanish, and the remainders drop into Fresnel zones (a kinoform). At the design wavelength it focuses to the same
  spot (computed with ASM). Zone widths shrink to ~λ/NA at the rim.
- **1:55–2:40 · Steps cost light.** Fabrication makes N height levels, and the staircase sends light into extra
  diffraction orders. In the phasor picture, the mean quantization-error arrow gives η = sinc²(1/N): 41%, 81%,
  95% and 99% for 2, 4, 8 and 16 levels. Show far-field order bars (computed).
- **2:40–3:20 · The rainbow problem.** A diffractive lens has f(λ) = f₀λ₀/λ, so red focuses closer than blue,
  the opposite of glass. Its Abbe number is ≈ −3.45, vs about +60 for glass. Fixes: hybrid achromats, or
  computational correction (E03, E10). Show R, G and B foci along z (computed).
- **3:20–4:15 · Where "thin" breaks.** TEA assumes the columns don't interact. With zones a few λ wide and
  1-µm walls, light leaks across walls, casts shadows and is guided inside ridges, so real efficiency is lower
  than TEA predicts. Show a full-wave cross-section of a fine kinoform next to TEA's picture (FDTD, a preview of
  E09). Two ways out:
  - metasurfaces, sub-wavelength pillars whose phase is computed rigorously, pillar by pillar (a simple
    schematic of pillars);
  - full-wave solvers (E09).
- **4:15–4:40 · On a computer.** U_out = U_in · exp(i k₀(n−1)h), then ASM. Pitfalls: sample the finest zone
  (Δx ≪ λ/NA), wrap the heights, and model the fabricated (rounded) profile, not the drawing.
- **4:40–5:00 · Recap.** Teaser: *"A lens's outer zones look like a grating. What does periodic structure do to
  light?"*

**Compute.**
- TEA kinoform vs refractive lens: a `lens_map` variant.
- Quantized-blaze orders (FFT); chromatic foci (ASM at 3 wavelengths).
- FDTD vs TEA cross-section: a small full-wave solver written for the episode.
- Metasurface preview: a schematic of pillars.

**Watch-outs.**
- sinc²(1/N) assumes an ideal blaze and scalar theory.
- When scalar theory holds depends on both period/λ and depth (Pommet et al. 1994); do not quote a single
  threshold.

---

### E07 · Periodic Structures: Gratings, Bloch Waves & Band Gaps

**Idea.** Symmetry fixes momentum. A flat interface conserves kx (Snell's law). A periodic surface conserves kx
only up to multiples of K = 2π/Λ (diffraction orders). Where shifted copies of the dispersion meet, the structure
couples them and opens a band gap.
Builds on E01 (k-circle) and E06 (gratings from zones). Sets up E08 (TIR, Bragg mirrors) and E09.

**Hook.** Colour without pigment: a CD, a morpho butterfly, an opal. Computed: white light on a grating fans
into orders, and a Bragg stack reflects a single colour. *"How does mere structure pick a direction, and a
colour?"*

- **0:25–1:05 · Interfaces conserve kx.** Draw two half-circles with radii n₁k₀ and n₂k₀. The crests must match
  along the boundary, so kx is a vertical line through both circles, which gives the refracted k: Snell's law in
  one picture. Slide kx past n₂k₀ and the line misses the small circle: total internal reflection, with an
  evanescent tail (E01). This seeds E08.
- **1:05–2:00 · Periodic surfaces add K.** With a discrete symmetry, kx is conserved only modulo K:
  kx,m = kx,in + mK. Draw copies of the line spaced by K (ORANGE); each one that hits the circle is a propagating
  order. The same K with different circle radii (colours) gives the rainbow fan. Shrink Λ and the orders leave the
  circle one by one. Below Λ < λ/n only the 0th order survives, and the grating acts as a uniform film: the
  regime metasurfaces live in. Optional 8-s extra: the Talbot carpet, where the orders realign at z_T = 2Λ²/λ.
- **2:00–3:00 · Periodic matter: Bloch waves and gaps.** In a stack of layers with period Λ, the modes are Bloch
  waves (e^(iκz) times a periodic function, with κ defined modulo K). Fold the light line ω = ck/n into the
  Brillouin zone; the copies cross at the zone edge, λ = 2nΛ (Bragg). Turn on index contrast and the crossing
  becomes an avoided crossing: a gap of frequencies with no propagating wave, i.e. a perfect mirror.
  **Signature shot:** band folding and the gap opening under a Δn slider, beside the computed stop-band
  reflectance of a 10-pair stack. Teaser: in 2D and 3D these are photonic crystals, and a defect makes a cavity
  (E08).
- **3:00–3:55 · RCWA: diagonalize each layer.** In a uniform layer, the orders are eigenmodes; each one just picks
  up e^(ikz,m z) (E01). In a grating layer, the Fourier coefficients of ε(x) couple the orders into a Toeplitz
  matrix. Its eigenvectors are Bloch modes and its eigenvalues their kz. Propagate each mode with its own phase,
  match fields at the interfaces, and chain the layers with S-matrices, which stay stable because they never
  contain growing exponentials. Show the diagonal matrix (uniform layer) vs the banded one (grating) → eig →
  modes, and the layer stack.
- **3:55–4:35 · On a computer.** The grating equation is 1 line. RCWA is 4: Fourier-expand ε → build the matrix →
  eig → stack S-matrices. Pitfalls: truncation (convergence vs number of orders; Li's factorization rules for TM)
  and staircasing of sloped walls. Demo: RCWA phase vs pillar diameter, the unit-cell library behind every
  metasurface.
- **4:35–5:00 · Recap.** Where it breaks: RCWA needs periodicity, and metasurfaces only borrow it locally (the
  local periodic approximation). Teaser: *"TIR traps light in a slab; Bragg mirrors trap it between them. How do
  you guide and store light?"*

**Compute.**
- k-circle constructions (exact geometry).
- Grating orders vs λ (FFT/TEA, or RCWA).
- Talbot carpet (ASM of a periodic field).
- 1D bands and stack reflectance with a numpy transfer matrix (cos κΛ = ½ tr M).
- Unit-cell library: a compact numpy RCWA (TE).

**Watch-outs.** This is the densest episode. If the script runs past 5:20, split it into 7a "Snell, TIR &
gratings" and 7b "Bloch waves, band gaps & RCWA" rather than speeding up the narration.

---

### E08 · Trapping Light: Waveguides, Modes & Resonators

**Idea.** Light held by total internal reflection settles into modes, the eigenvectors of the cross-section, each
with its own propagation constant β. A resonator stores light wherever round trips add in phase.
Builds on E01 (eigen-decomposition) and E07 (TIR, Bragg mirrors). Sets up E09 (devices need full-wave solvers)
and Season 2 (computing with light).

**Hook.** A fibre carries light around a corner. A photonic chip routes it through wires 0.5 µm wide. A 10-µm ring
picks one colour out of many (computed field). *"How do you trap light in a wire, and why do only certain shapes
fit?"*

- **0:25–1:10 · TIR + interference = modes.** A ray zigzags in a core. The same thing is two crossing plane waves,
  which make a standing wave across the core and a travelling wave along it. Self-consistency: after a round trip
  the wave must repeat itself, so only discrete angles work (round-trip phase = 2πm). An angle slider shows only
  a few angles "clicking" into place. Mode profiles have evanescent tails (RED).
- **1:10–2:00 · Modes are eigenvectors.** [∂x² + k₀²n(x)²] E = β² E. Discretize it into a tridiagonal matrix plus
  a diagonal n², take eig, and the mode profiles appear (they look like particle-in-a-box states). Each has
  n_eff = β/k₀ between the cladding and core indices. Any input is a sum of modes, each advancing as e^(iβz):
  *"the angular spectrum method of a waveguide"* (E01 callback). Computed: multimode beating and self-imaging.
- **2:00–2:50 · Two guides: light that sloshes.** When two guides' tails overlap, the system has symmetric and
  antisymmetric supermodes with propagation constants β ± κ. Light launched in guide 1 is the sum of both. The two
  dephase, so the light crosses after L = π/2κ and then comes back. The coupled-mode equations are a 2×2
  rotation. Computed: a directional coupler (BPM). Half a crossing is a 50:50 splitter, the basis of
  interferometers, switches and optical processors.
- **2:50–3:50 · Resonators: storing light.** Close a guide into a ring, or put Bragg mirrors (E07) at both ends.
  **Phasor shot:** each round trip adds an arrow rotated by the round-trip phase. On resonance (n_eff L = mλ) the
  arrows line up into a big field; off resonance they curl into a small spiral. The linewidth comes from loss:
  Q = λ/Δλ, roughly how many cycles light survives. The through-port shows a Lorentzian dip, and at critical
  coupling (coupling = loss) the dip reaches zero. One complex amplitude with decay rates describes all of this
  (temporal coupled-mode theory, the model behind many resonant metasurface elements).
- **3:50–4:35 · On a computer.**
  - Mode solver: a tridiagonal eig, about 5 lines.
  - Split-step BPM: alternate a diffraction step (FFT, E01) and a refraction step (multiply by e^(ik₀Δn Δz)). It is
    the multi-slice cousin of TEA.
  - Resonators: one ODE.

  Pitfalls: the mode solver's window and boundaries; BPM is paraxial, fine for couplers but wrong for tight bends
  (E09).
- **4:35–5:00 · Recap.** Teaser: *"Gratings, rings, metasurface pillars: past a point you need Maxwell's equations
  themselves. How does a computer solve them?"*

**Compute.**
- Slab mode solver (numpy eig).
- Multimode guide and coupler with a numpy split-step BPM.
- Ring spectrum from an analytic round-trip sum.
- TCMT: one complex ODE per resonator.
- Optional ring field with E09's 2D FDTD.

**Watch-outs.** If the script runs long, split it into 8a "waveguides & couplers" and 8b "resonators".

---

### E09 · Maxwell on a Grid: FDTD & the Perfectly Matched Layer

**Idea.** A changing E curls H, and a changing H curls E. Put E and H on interleaved grids and let them leapfrog
in time, one neighbour at a time. Surround the box with complex-stretched space, which swallows outgoing waves
without reflecting them.
Builds on E01, E06 (where TEA breaks), E07 and E08. Sets up E10.

**Hook.** A femtosecond pulse hits a fine DOE relief (or a ring) and you watch it scatter, ring and leave
(computed FDTD movie). *"Everything so far was an approximation. This is Maxwell's equations, solved directly.
How?"*

- **0:25–1:05 · Maxwell in one sentence.** Use 2D TM fields (Ez, Hx, Hy): ∂H/∂t = −(1/μ)∇×E and
  ∂E/∂t = (1/ε)∇×H. A vector-field animation shows an Ez bump making H circulate and E spreading out: the wave
  pushes itself along. Only local derivatives appear, so the updates are local.
- **1:05–2:00 · The Yee cell and the leapfrog.** A finite difference is most accurate at its midpoint, so H lives
  half a cell and half a time step away from E. Update H from curl E, then E from curl H, and repeat (Yee 1966).
  The material is just ε per cell, so any shape works. Show a grid of dots and arrows with alternating highlights,
  and a 1D pulse marching forward with real numbers.
- **2:00–2:45 · Speed limit and grid errors.**
  - Courant limit: light must not skip a cell in one step, cΔt ≤ Δx/√D. Violate it and the computed field blows up.
  - Numerical dispersion: the grid's k-circle bulges along the grid axes (exact discrete dispersion vs E01's
    circle). Short waves travel at the wrong speed, so use at least 10–20 cells per wavelength in the densest
    material.
- **2:45–3:45 · PML: hiding a wall in complex space.** A hard wall reflects, and a lossy layer reflects at its
  edge. The PML replaces x with x + (i/ω)∫σ dx (Bérenger 1994; Chew & Weedon 1994). It is the same equations under
  a coordinate change, so nothing reflects at any angle in the continuum, and e^(ikx) picks up a factor
  e^(−(k/ω)∫σ), which makes it decay. **Signature shot:** the x-axis ribbon twists into the imaginary direction
  and the wave fades without an echo, with a computed comparison of wall, absorber and PML. On a grid, small
  reflections remain, so grade σ smoothly; grazing and evanescent waves are the hardest cases.
- **3:45–4:35 · On a computer.** Four lines: H += …curl E; E += …curl H; apply the PML; add the source. A TFSF
  source injects a clean incident wave, and a short pulse plus a Fourier transform in time gives every wavelength
  in one run. Cost scales as cells × steps. Model-ladder bar: TEA in ms → BPM → RCWA → FDTD in minutes to hours. Demo: FDTD vs TEA on a fine kinoform
  (E06 callback).
- **4:35–5:00 · Recap.** Teaser: *"Now we can simulate almost anything. What we really want is the reverse: say
  what the light should do, and get the structure."*

**Compute.**
- A new 2D FDTD in JAX/numpy (~150 lines; GPU for large runs).
- CFL blow-up demo; dispersion surface (analytic).
- Wall vs absorber vs PML (2D FDTD).

**Watch-outs.**
- FDTD is not "exact"; it converges with Δx and Δt.
- The PML adds its own error.
- Curved surfaces are staircased.

---

### E10 · Designing Light with Gradients: the Adjoint Method

**Idea.** Two simulations, light forward from the source and light backward from the goal, give the gradient for
every pixel at once, so a computer can optimize millions of parameters. With an autodiff propagator, the backward
pass is literally back-propagation of light.
Builds on E01, E03, E05, E06 and E09. Season finale.
Background: end-to-end optics, e.g. Sitzmann et al., ACM TOG 2018.

**Hook.** Two devices do the same job. One is a human design with neat shapes. The other is an algorithm's
organic, alien pattern, and it is smaller and better (computed 2D splitter). *"How does a computer invent shapes
no human would draw?"*

- **0:25–1:00 · Design is climbing a landscape.** A merit F(p) depends on the parameters (heights, ε pixels), and
  the gradient points uphill. Finite differences need one simulation per pixel; a counter spins up to 1,000,000
  E09 runs. Impossible.
- **1:00–2:05 · The adjoint trick.** Discretized Maxwell is A(p)x = b. The chain rule gives
  dF/dp = −λᵀ(∂A/∂p)x with Aᵀλ = ∂F/∂x: one extra solve, however many parameters there are. Reciprocity makes
  Aᵀ = A, so λ is light sent backward from the detector, carrying "what I wish were different".
  **Signature shot:** the forward field (BLUE) comes from the source and the adjoint field (PINK) from the
  detector. Their product is the sensitivity map. The pixels update and the device morphs.
  *"Light run backward tells every pixel what to do."*
- **2:05–2:55 · Differentiable optics: autodiff does it for you.** The forward graph is U₀ → ×e^(iφ) → FFT → ×H →
  inverse FFT → |·|² → loss. Backward, the FFT becomes the inverse FFT and H becomes H*. For propagating waves,
  the adjoint of propagating by z is propagating by −z. GS (E05) was a heuristic; gradient descent is the
  principled version (Wirtinger derivatives). Demo: a phase mask optimized to project a target (JAX), with the
  loss curve.
- **2:55–3:50 · End-to-end: optics and algorithm as one network.** Scene → optics (PSF) → sensor noise →
  reconstruction (E03's Wiener filter or a network) → loss vs ground truth. Gradients flow through everything,
  and the optics learns a PSF that the algorithm can invert: extended depth of field, or a double-helix PSF whose
  rotation encodes depth. Show the PSF morphing during training (computed with the lom apps).
- **3:50–4:35 · On a computer, and where it breaks.** Code card: `loss = merit(simulate(p)); g = grad(loss)(p);
  p -= lr * g`. In practice:
  - binarization and minimum-feature filters;
  - local optima;
  - the model gap: an optimizer exploits any model error, so a design that is great under TEA can fail in
    reality (E06).

  Hence the need for accurate, fast models (learned surrogates of FDTD) and fabrication digital twins: Season 2.
- **4:35–5:00 · Season recap.** A 20-s montage of the ten hero shots on the series map. Closing line: *"Light is
  linear: in the right basis, the physics is simple; run it backward, and the design is too."*

**Compute.**
- A new 2D FDFD adjoint inverse design (numpy/scipy sparse LU, ~200 lines; CPU is fine).
- JAX ASM + TEA phase-mask optimization.
- EDOF PSFs: pupil phase plus a Wiener filter (computed simply).

**Watch-outs.**
- The adjoint undoes the phase of propagating waves, not the decay of evanescent ones.
- Use Wirtinger derivatives for real losses of complex fields.

---

## 5. Production

**Pipeline.** Unchanged from the pilot:
1. `narration.json` → `tts.py` (Kokoro-82M, voice `af_heart`, 4 CPU threads).
2. `check_audio.py` (faster-whisper round trip, word timestamps) → `make_subs.py`.
3. One Manim scene per chapter, timed from `audio/timing.json`.
4. `fast_render.py` renders parallel pieces (~3 min per 5-min film).
5. FFmpeg: libass subtitles, −16 LUFS.

**Layout.**
```
Optics5min/
  OUTLINE.md                this file
  common/                   style, colour code, visual atoms, ASM/FFT helpers, TTS/subtitle/render scripts (from the pilot)
  E01_angular_spectrum/     the pilot, ported onto common/ (pixel-identical)
  E02_lens_fourier/         narration.json · physics.py (writes data/*.npz) · scenes.py · render.sh · README.md
  …
  E10_adjoint_design/
```

**Per episode.**
1. One-page script from this outline.
2. Compute and cache fields (`physics.py`). Heavy runs go on the GPU; never run long CPU jobs on the shared server.
3. Narration, TTS, Whisper check.
4. Low-res chapter previews.
5. Full render with subtitles.
6. 60-s cut.
7. README with the chapter table and the computed-vs-schematic statement.

**Order.** Produce in series order, since each episode reuses the previous one's visual atoms. E02 reuses nearly
all of E01's code, so it is the cheapest next step. The heaviest compute is in E09 (2D FDTD, plus optional 3D
production fields) and E10 (2D FDFD inverse design). For E07, RCWA sweeps already exist.

**Decisions for you.**
1. **Audience.** Keep the pilot's level (calculus plus a little linear algebra), or assume viewers know the
   Fourier transform (saves ~30 s per episode)?
2. **Unpublished results.** Resolved: none are shown; comparisons come from solvers written for the episodes.
3. **E08.** Keep waveguides & resonators in the ten, or swap in polarization or Fourier ptychography?
4. **Branding.** Series ident and music bed; keep the Kokoro voice or record a human narrator?
5. **E01 retrofit.** Add the 3-s ident, the series-map outro and the E02 teaser line?

---

## 6. Season 2 candidates (ranked)

| # | Topic | Core idea | Assets |
|---|---|---|---|
| 1 | Polarization & the geometric phase | Jones 2-vectors and 2×2 wave plates; the Poincaré sphere; rotating a half-wave plate by α adds phase 2α | E07's RCWA, extended to both polarizations |
| 2 | Metasurfaces in depth | unit-cell libraries, the local periodic approximation, metalens efficiency and bandwidth limits | E06/E07 solvers |
| 3 | The beam propagation method | multi-slice, from one TEA screen to thick media; tomography; why more slices help | E08's split-step BPM |
| 4 | Sampling light | band-limited and adaptive ASM, Rayleigh–Sommerfeld via FFT, shifted and scaled propagation | E01/E02 propagators |
| 5 | Neural surrogates | learning Maxwell from FDTD: locality, a physics baseline plus a learned correction, out-of-distribution honesty | E09's FDTD as ground truth |
| 6 | Fabrication in the loop | digital twins of grayscale lithography and nanoimprint; AFM metrology | new |
| 7 | Fourier ptychography | tilted illumination stitches a synthetic aperture; phase from intensities | E05's phase retrieval |
| 8 | Aberrations & Zernike polynomials | an orthogonal basis on the pupil disc; Strehl ratio; wavefront sensing and adaptive optics | numpy |
| 9 | Computing with light | 4f convolution; MZI meshes (any unitary is a product of 2×2 rotations); diffractive networks | E02's 4f machine |
| 10 | Rays from waves | stationary phase, Fermat's principle, the eikonal; when ray tracing is enough; differentiable lens design | numpy |

**Bonus shorts.** Gaussian beams & the Gouy phase · scattering media & the transmission matrix · optical
diffraction tomography (the Ewald sphere) · nonlinear phase matching · light fields & the Wigner function ·
dispersion & pulses.

---

## 7. References for fact-checking scripts

- Goodman, *Introduction to Fourier Optics*: E01–E05. Saleh & Teich, *Fundamentals of Photonics*: E04, E07, E08.
  Born & Wolf, *Principles of Optics*: E04, E06.
- Voelz & Roggemann, Appl. Opt. 48, 6132 (2009), chirp sampling and the critical distance; Matsushima &
  Shimobaba, Opt. Express 17, 19662 (2009), band-limited ASM: E02.
- Gabor, Nature 161, 777 (1948); Leith & Upatnieks, JOSA 52, 1123 (1962); Gerchberg & Saxton, Optik 35, 237
  (1972); Fienup, Appl. Opt. 21, 2758 (1982): E05.
- Pommet, Moharam & Grann, JOSA A 11, 1827 (1994), the limits of scalar diffraction theory: E06.
- Moharam & Gaylord, JOSA 71, 811 (1981), RCWA; Li, JOSA A 13, 1870 (1996), factorization rules; Joannopoulos
  et al., *Photonic Crystals: Molding the Flow of Light*: E07, E08.
- Yee, IEEE TAP 14, 302 (1966); Bérenger, J. Comput. Phys. 114, 185 (1994); Chew & Weedon, Microw. Opt. Technol.
  Lett. 7, 599 (1994); Taflove & Hagness, *Computational Electrodynamics*: E09.
- Molesky et al., Nat. Photonics 12, 659 (2018), inverse design review; Piggott et al., Nat. Photonics 9, 374
  (2015); Sitzmann et al., ACM TOG 37(4), 114 (2018), end-to-end optics; Pavani et al., PNAS 106, 2995 (2009),
  double-helix PSF: E10.
