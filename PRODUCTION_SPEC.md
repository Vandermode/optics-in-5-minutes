# Optics in 5 Minutes: production spec (Season 1, E02–E10)

How every episode is made, so the nine films share one quality bar. The content plan is `OUTLINE.md` §4. The render
pipeline is `docs/FAST_ANIMATION_PIPELINE.md`, whose scene rules (§5) apply unchanged. The reference implementation
is `E01_angular_spectrum/scenes.py` (the pilot, ported).


> **Update 2026-10-11.** `common/o5common.py` now has the whole-frame `Chapter.at()` (pipeline doc §10), so
> new episodes no longer need a local override. `code_card` now keeps leading-space indentation.

## 1. Folders

| Folder | Title (`series_title`) | Subtitle | Slug / outputs |
|---|---|---|---|
| `E01_angular_spectrum` | The Angular Spectrum Method | (the pilot, ported onto `common/`) | `e01_angular_spectrum_1080p60.mp4` |
| `E02_lens_fourier` | Why a Lens Computes a Fourier Transform | far fields, lenses and the 4f machine | `e02_lens_fourier_1080p60.mp4` |
| `E03_diffraction_limit` | The Diffraction Limit | why every image is blurry | `e03_diffraction_limit_…` |
| `E04_coherence` | Coherence | when does light interfere? | `e04_coherence_…` |
| `E05_phase_problem` | The Phase Problem | holography and phase retrieval | `e05_phase_problem_…` |
| `E06_flat_optics` | Flat Optics | a lens one micron thick | `e06_flat_optics_…` |
| `E07_periodic_structures` | Periodic Structures | gratings, Bloch waves and band gaps | `e07_periodic_structures_…` |
| `E08_trapping_light` | Trapping Light | waveguides, modes and resonators | `e08_trapping_light_…` |
| `E09_maxwell_grid` | Maxwell on a Grid | FDTD and the perfectly matched layer | `e09_maxwell_grid_…` |
| `E10_adjoint_design` | Designing Light with Gradients | the adjoint method | `e10_adjoint_design_…` |

Shared code lives in `common/`. **Do not edit `common/` while episodes are in production.** If a helper needs
changing, copy it into the episode's own `scenes.py` or `physics.py` and report it.

Per-episode files:
- `episode.json`: `{"slug", "title", "chapters": [...]}`. The chapter list is in chapter-index order.
- `narration.json`
- `physics.py` → `data/*.npz`
- `scenes.py`
- `render.sh`, `preview.sh` (wrappers)
- `README.md`
- `audio/`, `subtitles.srt`, `subtitles.ass` (generated)
- the two MP4s
- `.verify/` contact sheets

## 2. Workflow (commands from inside the episode folder)

```bash
../common/new_episode.sh E0N_slug "<Title>"         # scaffold (run from Optics5min/common or with a path)
# 1. script
$EDITOR narration.json                               # ~720-800 words, 40-50 segments, ch = chapter index
../common/audio.sh .                                 # Kokoro TTS -> Whisper check -> subtitles; read every "check" line
# 2. physics
python physics.py           # deterministic (seeded); writes data/*.npz
# 3. scenes + previews (2 processes at a time)
./preview.sh                                         # or ./preview.sh Ch3Foo Ch4Bar ; prints LATE lines
python ../common/sheet.py . $OUT sheet.png 12.0 18.5 ...   # contact sheet; inspect with Read
# 4. final render (3 workers) + verification
./render.sh --skip-audio
python ../common/verify.py .   # counts, seam scan, loudness, stills -> .verify/sheet_*.png
```

`OUT` defaults to `$TMPDIR/o5pv_<slug>` (or `/tmp/o5pv_<slug>`); `preview.sh` prints it. Re-running `audio.sh` only
re-synthesises segments whose `say` changed. A new narration moves every chapter window, so preview again
afterwards.

## 3. Quality bar: script

1. **One idea per episode.** The recap line is the episode's sentence from `OUTLINE.md` §1, and the beats follow
   §4. You may simplify or reorder beats to make the film clearer, but don't add topics.
2. **Audience.** They know calculus and a little linear algebra. They may not have seen E01, so recall what you
   need in one sentence plus one picture: "remember: any field is a sum of plane waves, each with a direction
   sin θ = kx/k".
3. **Picture first.** Show the thing, then name it, then (at most once per chapter) write its equation. Every
   symbol appears next to what it names.
4. **Plain words.** Short sentences (mostly ≤ 20 words), active voice, second person, real questions. Define every
   term the first time ("the numerical aperture, NA: the sine of the widest angle the lens accepts").
   No unexplained acronyms.
5. **Narrate the screen.** Narration and animation stay in sync through `self.at(S("id") + offset)`. Say
   "watch…", "notice…" when something moves.
6. **Hook.** A striking computed visual and one question within about 20 s, then the title card
   (`series_title(ep, title, subtitle)`). Chapter 1 begins by 0:30.
7. **End.** The recap plus the teaser for the next episode (`next_card("…")`; E10 ends the season instead).
8. **Length.** 4:45–5:15 (hard cap 5:30), about 720–800 words, 40–50 segments of at most 2 sentences, 6–9
   chapters. Hold about 1 s of silence before each "aha" (`"pause_after": 0.8` on that segment).
9. **Physics must be right.** Use correct signs, factors and conventions. If something is a simplified picture,
   the README says so.

**Pronunciation (`say` field).** Spell symbols phonetically:

| Symbol | Say |
|---|---|
| kₓ | "kay ex" |
| k_z | "kay zee" |
| λ | "lambda" |
| θ | "theta" |
| φ | "phi" |
| Δx | "delta ex" |
| e^{iφ} | "e to the i phi" |
| NA | "N A" |
| FFT | "F F T" |
| PSF | "P S F" |
| OTF | "O T F" |
| 4f | "four F" |
| TEA | "T E A" |
| FDTD | "F D T D" |
| PML | "P M L" |
| RCWA | "R C W A" |
| DOE | "D O E" |
| LED | "L E D" |
| SLM | "S L M" |
| η | "eta" |
| β | "beta" |
| κ | "kappa" |
| ε | "epsilon" |
| ω | "omega" |
| σ | "sigma" |
| ∇× | "curl" |
| µm | "microns" |
| nm | "nanometres" |
| sinc | "sink" |

- Kokoro drops lone letters such as "A": rephrase.
- Commas shape intonation.
- Read every Whisper `check` line and fix `say` until the heard text matches.
- Subtitle `text` is display-ready Unicode: kx, kz, λ, θ, e^(iφ), |U|², ≈.

## 4. Quality bar: visuals (3Blue1Brown style)

- **Background and fonts.** Black background. Use `Tex`/`MathTex` for text and maths. The `series_title` kicker
  and code cards use Lato and DejaVu Sans Mono, built into the helpers.
- **Colour code** (`o5common`): field U BLUE, kx YELLOW, kz TEAL, transfer/pupil GREEN, evanescent/loss RED,
  imaginary PURPLE, reference/source GOLD, matter GREY_B, grating vector ORANGE, adjoint PINK. Intensity uses
  `heat_rgba`, phase uses `phase_rgba` (hue), signed fields use `diverge_rgba`.
- **Clean frames.** One main visual plus at most 2–3 text items. Remove the old before adding the new.
  Keep breathing room.
- **Frame bounds.** x ∈ [−6.9, 6.9], y ∈ [−3.2, 3.7]. Nothing below `SUB_Y = −3.25` (the subtitle band). The chapter
  tag sits top-left (`TAG = "3\\quad Coherent vs incoherent"`, shown by `self.show_tag()`), so keep the top-left
  corner clear.
- **Font sizes.** Titles 48–64, body 30–38, labels 24–30, never below 22.
- **Purposeful motion.** Use trackers, transforms and live waves (`Re{U e^(−iωt)}`). Avoid static stretches longer
  than about 4 s.
- **Signature shot.** Each episode has the one from `OUTLINE.md`; give it time (5–10 s) and make it beautiful.
- **Computed fields.** Every physical field comes from `physics.py`, which uses the method being taught. Mark data
  panels with a small "(computed)" caption where helpful.

## 5. Technical rules (FAST_ANIMATION_PIPELINE.md §5, condensed)

- Scene state is a pure function of `self.now` and the plays before it. No dt updaters, no accumulated state, no
  wall clock, and no unseeded randomness (precompute random fields in `physics.py` with a fixed seed).
- One `Chapter` subclass per chapter: `CH = index`. Sync with `self.at(t)` and end with `self.finish()`.
  `scenes.py` ends with `CHAPTERS = [...]`, which must equal `episode.json["chapters"]`.
- Per-frame redraws (`always_redraw`, updaters) use `fast_arrow`, `fast_dashed`, `poly_line`, `Line`, `Dot`,
  `Circle`, and copies of pre-built Tex. They never build `Tex`, `MathTex`, `Arrow`, `DashedLine`, `Brace` or
  `SurroundingRectangle` per frame.
- **Images.** Use `m, fade = self.live_image(fn, w, center, h)`, `self.static_image(rgba, w, center, h)` or
  `m, fade, sweep = self.sweep_image(rgba, w, center, h, axis)`. Fade via the tracker, never `FadeIn(m)`. Keep
  images axis-aligned.
- **Heavy work happens once.** It goes in `physics.py`, which writes `.npz` files; `construct()` only loads them.
  `construct()` runs once per render piece, so keep its setup under about 2 s.
- **CPU budget on the shared server.** Use `nice -n 5`. Previews run at most 2 Manim processes at a time (the
  default `PAR=2`). The final render uses `WORKERS=6` by default; use 3 when several episodes render at once.
  Physics runs single-threaded numpy and stays under about 5 CPU-minutes. Anything heavier uses JAX:
  `make physics` puts it on the least-busy GPU, and without a GPU it downloads the data bundle. Never run long
  multi-core CPU jobs.
- `common/o5common.py` API:
  - timing: `Chapter` (`S`, `E`, `at`, `show_tag`, `finish`, `live_image`, `static_image`, `sweep_image`)
  - cards: `series_title`, `next_card`, `code_card`, `caption`, `boxed`
  - images: `heat_rgba`, `grey_rgba`, `diverge_rgba`, `phase_rgba`, `colour_wheel_rgba`
  - propagation: `asm_rows`, `asm2d`
  - drawing: `fast_arrow`, `fast_dashed`, `fade_vgroup`, `graph_line`, `poly_line`, `plot_xy`, `stems`,
    `phasor`, `panel_frame`, `wave_panel`, `k_circle`, `axis_pair`
  - constants: `C_*`, `SUB_Y`, `TAU_`, `smoothstep`, `data_path(name)`

## 6. Acceptance checklist (before reporting done)

1. `audio.sh`: total 4:45–5:30. Every Whisper mismatch is either fixed or harmless (numbers, spelling).
2. Previews:
   - no `LATE` lines;
   - every chapter reports `frames (target …)` with matching numbers;
   - contact sheets (several per chapter) inspected with Read, with no overlaps, clipped text, content in the
     subtitle band, or unreadable labels.
3. `render.sh --skip-audio` completes with every `PIECE … OK`.
4. `verify.py`:
   - `ALL COUNTS OK`;
   - every seam-scan jump is an intended transition (look at the frames near any jump that falls within 50 ms of
     a piece start);
   - loudness about −16 LUFS;
   - `.verify/sheet_*.png` inspected.
5. `README.md` in the pilot's format:
   - title and length;
   - chapter table (time, idea);
   - which panels are computed (and how) and which are schematic;
   - the rebuild command.
6. Final report (≤ 15 lines):
   - the MP4 path and duration;
   - the verify summary;
   - computed vs schematic;
   - any compromise or known issue.
