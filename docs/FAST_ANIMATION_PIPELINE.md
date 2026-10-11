# Fast Manim animation pipeline

Reference for agents that render, extend or debug the narrated 3Blue1Brown-style explainers in
`~/code/LAFA/animations/`. As of 2026-10-10.

**Result.** The reference film `angular_spectrum_method_5min/` renders end to end in **2 min 46 s** on 6 worker
processes. It is 5:07 long, 1920×1080 at 60 fps, 18,398 frames. The original Manim CLI path took about 27 min. Two
checks show that pieces join frame-exactly: pixel diffs across seams, and traces of play timings across the points
where float noise could shift a wait.

**Core idea.** Manim CE (Cairo renderer) is bound by Python and uses one core per process. The pipeline therefore
cuts the timeline into about 40 frame windows of equal measured cost. Each worker process fast-forwards its chapter
scene to its window on a whole-frame clock, draws only its own frames and streams raw pixels into FFmpeg. The
pieces are then joined by stream copy, with no re-encode.

**Status.** The pilot's MP4s were rendered at 22:47 on 2026-10-10. A review later that day fixed three bugs in the
code that those MP4s still contain (section 10, rows marked *fixed 2026-10-10*):
- a one-frame timing step at 274.9 s;
- a snap of the Ch1 wave at 25.8 s;
- a pop, blink and snap of the Ch8 dials at 294–296 s.

Re-render to pick up the fixes.

**Rules an agent must not break** (details in sections 5–6):
1. Scene state is a pure function of global time `self.now` and the plays before it. No dt updaters, no
   accumulated state, no wall clock, no unseeded randomness.
2. Every chapter scene renders exactly its frame window. It syncs with `self.at(...)`, whose waits are whole
   frames, and ends with `self.finish()`. Never compute a wait from the clock yourself, as in
   `self.wait(t - self.now)`.
3. Never introduce a mobject that moves with `self.now` through `Create`, `FadeIn` or `Write`. Animations suspend
   the updaters of the mobject they animate, so it freezes and then snaps. Instead, add it and reveal or fade it
   with a tracker that its redraw reads.
4. Per-frame redraws use the cheap helpers (`fast_arrow`, `fast_dashed`, pre-built text). They never build `Arrow`,
   `DashedLine`, `Tex` or `MathTex`.
5. Re-run the verification checklist (section 9) after any change to the renderer patches, to `at()` or
   `finish()`, or to the scene rules.
6. The server is shared. Keep `nice`, use at most about 8 workers, and render Blender on the GPU, never with long
   CPU renders.

Paths below are relative to `~/code/LAFA/animations/angular_spectrum_method_5min/` unless they are absolute.
`SP=<the pilot's session scratch directory>` (see section 2).


> **Reading this inside the Optics5min repo.** This document was written for the pilot folder
> `angular_spectrum_method_5min/`, where the pipeline was developed. Paths, timings and machine details are from that
> setup (a 112-core shared server; `$SP` venvs). In this repo the same code is factored with the episode folder as a
> parameter:
>
> | Pilot | This repo |
> |---|---|
> | `asm5_common.py` | `common/o5common.py` (identical scene rules; adds series cards and colour maps) |
> | `asm5_scenes.py` | `E01_angular_spectrum/scenes.py` (pixel-identical port) and every `E0N_*/scenes.py` |
> | `fastrender.py <Scene> fa fb prefix` | `common/fastrender.py <episode_dir> <Scene> fa fb prefix` |
> | `fast_render.py`, `render.sh`, `preview.sh`, `sheet.py` | `common/…` (or `make render/preview E=E0N`) |
> | `asm_subtitles.srt/.ass` | `<episode>/subtitles.srt/.ass` |
> | `$SP/manimvenv`, `$SP/ttsvenv`, `/usr/bin/ffmpeg` | the `optics5min` environment (`setup_env.sh`); `$O5_FFMPEG` (`common/o5env.py`) |
> | verification checklist (section 9) | automated by `common/verify.py` (or `make verify E=E0N`) |

---

## 1. Data flow and files

```
narration.json ──tts.py──────────► audio/<id>.wav, audio/narration.wav, audio/timing.json
audio/narration.wav ─check_audio.py─► transcript check (stdout) + audio/words.json
timing.json + words.json ─make_subs.py─► asm_subtitles.srt, asm_subtitles.ass
timing.json ─asm5_common.WIN──────► chapter windows [f0, f1) in global frames (60 fps)
asm5_scenes.py ───────────────────► one Chapter scene per chapter (Ch0Hook … Ch8Recap)

fast_render.py
  plan()  → ~40 pieces (scene, fa, fb) of equal cost, from .cache/piece_speed.json
  pool of N workers, longest piece first → fastrender.py <Scene> fa fb <prefix>
      Manim: skip plays that end before fa → draw [fa, fb) → raw RGBA pipe → ffmpeg
      ffmpeg: one colour conversion → split → <prefix>_clean.ts  +  libass → <prefix>_subs.ts
  meanwhile: narration.wav → loudnorm −16 LUFS → AAC (audio.m4a)
  concat demuxer, stream copy → angular_spectrum_method_5min_1080p60.mp4          (burned-in subtitles)
                              → angular_spectrum_method_5min_1080p60_softsubs.mp4 (clean + mov_text track)
```

| File | Role |
|---|---|
| `narration.json` | voice settings, timing gaps and 46 segments `{id, ch, text, say}` |
| `tts.py` | Kokoro-82M text-to-speech → one WAV per segment, the assembled track and `timing.json` |
| `check_audio.py` | faster-whisper round trip: prints script vs heard per segment, writes `words.json` |
| `make_subs.py` | subtitle cues from the script text, timed by the Whisper word timestamps |
| `asm5_common.py` | timing (`SEG`, `WIN`), colour code, `Chapter` base class, physics and drawing helpers |
| `asm5_scenes.py` | the nine chapter scenes |
| `fastrender.py` | one worker: renders global frames `[fa, fb)` of one scene into two MPEG-TS pieces |
| `fast_render.py` | orchestrator: plan, worker pool, audio, stitch, mux |
| `render.sh` | entry point: TTS → check → subtitles → `fast_render.py` |
| `preview.sh`, `sheet.py` | low-res chapter previews (Manim CLI) and contact sheets |
| `.cache/Tex/` | pre-warmed LaTeX SVG cache (282 files) |
| `.cache/piece_speed.json` | measured fps of every piece in the last run (the cost profile) |

## 2. Environment

| Component | Location | Version / notes |
|---|---|---|
| Manim venv | `$SP/manimvenv` | Python 3.12.2, manim 0.22.0, manimpango 0.7.0, pycairo 1.27.0, numpy 2.4.5; built over conda env `jax` with `--system-site-packages` |
| TTS venv | `$SP/ttsvenv` | kokoro-onnx 0.6.1, onnxruntime 1.31.0, faster-whisper 1.2.1; built from the `jax` env's Python, no system site packages |
| Kokoro model | `$SP/tts/` | `kokoro-v1.0.onnx`, `voices-v1.0.bin` from the kokoro-onnx GitHub release `model-files-v1.0` |
| Whisper model | `HF_HOME=$SP/hf` | `small.en`, int8, CPU, downloaded on first use |
| FFmpeg | `/usr/bin/ffmpeg` 4.4.2 | has libass (`ass` filter) and libx264. The `jax` env's ffmpeg 5.1 has no libass |
| LaTeX | `/usr/bin/latex`, `/usr/bin/dvisvgm` | used by `Tex` and `MathTex` |
| Font | `/usr/share/fonts/truetype/lato` | subtitles (`fontsdir` of the `ass` filter) |
| Analysis Python | `~/anaconda3/envs/jax/bin/python` | numpy/PIL for checks; the base anaconda numpy is broken |
| Machine | 112 cores, 1 TB RAM, shared | A100 GPUs have no NVENC (no hardware video encoding) |

`$SP` is a session scratch directory under `/tmp`, so it can disappear. `render.sh` and `preview.sh` hard-code it.
If it is gone, rebuild the venvs somewhere durable and update `SP` in both scripts:

```bash
J=~/anaconda3/envs/jax
$J/bin/python -m venv --system-site-packages ~/venvs/manimvenv
# ManimPango has no Linux wheel: build it against the jax env's pango (pycairo comes from jax)
PKG_CONFIG_PATH=$J/lib/pkgconfig CFLAGS=-I$J/include LDFLAGS="-Wl,-rpath,$J/lib -L$J/lib" \
    ~/venvs/manimvenv/bin/pip install manim==0.22.0
$J/bin/python -m venv ~/venvs/ttsvenv
~/venvs/ttsvenv/bin/pip install kokoro-onnx soundfile faster-whisper
# then download kokoro-v1.0.onnx + voices-v1.0.bin (release model-files-v1.0) into the model dir
```

## 3. Quick start

Re-render the pilot (from any directory; the script `cd`s to its folder):

```bash
cd ~/code/LAFA/animations/angular_spectrum_method_5min
./render.sh --skip-audio              # picture + mux; reuses audio/ and the subtitles (~2.8 min)
WORKERS=8 ./render.sh --skip-audio    # more workers
./render.sh                           # also regenerate narration, Whisper check and subtitles
```

Outputs, written into the project folder:
- `angular_spectrum_method_5min_1080p60.mp4`: subtitles burned in (53 MB).
- `angular_spectrum_method_5min_1080p60_softsubs.mp4`: clean picture plus an English `mov_text` track (50 MB).

The orchestrator prints one line per finished piece, in the form `[ elapsed s] PIECE <Scene> [fa,fb) frames=<n> OK time=<s> fps=<f>`.
It ends with `frames rendered+encoded in …s; total …s`. Pieces go to `$TMPDIR/asm5fast_*/` and are not deleted
afterwards (about the size of both outputs together). Remove them after checking the result.

**When a piece fails:**
- The error is raised only after the pool has finished every queued piece, because the executor waits on shutdown.
  It includes the tail of the worker's stdout and stderr.
- The old MP4s stay in place.
- A crashed worker leaves its `$TMPDIR/asm5w_*` temp directory behind.

One worker by hand, for debugging. Frame indices are global: 60 fps counted from t = 0 of the film. The prefix's
directory must already exist.

```bash
PY=$SP/manimvenv/bin/python; mkdir -p /tmp/x
$PY fastrender.py Ch3PlaneWaves 6400 6700 /tmp/x/p   # → /tmp/x/p_clean.ts, /tmp/x/p_subs.ts
$PY fastrender.py Ch3PlaneWaves 6400 6700 /tmp/x/p --null   # frames discarded (speed test)
$PY fastrender.py Ch3PlaneWaves 6690 6710 /tmp/x/p --dump   # raw RGBA kept → /tmp/x/p_raw.npy (≈ 8.3 MB/frame)
```

Low-res previews for layout and timing. These use the plain Manim CLI and run 3 chapters at a time:

```bash
export OUT=$SP/asm5pv                    # preview root (give each episode its own)
./preview.sh Ch2Arrows Ch7Algorithm      # → $OUT/<Chapter>/####.png; env RES (960,540), FPS (15)
$SP/manimvenv/bin/python sheet.py $OUT sheet.png 40.0 52.5 61.0 75.2   # contact sheet at global times (s)
```

`preview.sh` prints each chapter's frame count, plus any `LATE` lines and errors from the scene log.

## 4. Narration, TTS and subtitles

```json
{"voice": "af_heart", "speed": 1.03, "gap_s": 0.32, "chapter_pause_s": 0.75, "lead_in_s": 0.9, "tail_s": 2.6,
 "segments": [{"id": "h1", "ch": 0, "text": "<subtitle text>", "say": "<what the voice reads>"}]}
```

`render.sh` has no audio-only mode. The three audio steps, run from the project folder:

```bash
OMP_NUM_THREADS=4 $SP/ttsvenv/bin/python tts.py $SP/tts audio
HF_HOME=$SP/hf OMP_NUM_THREADS=4 $SP/ttsvenv/bin/python check_audio.py audio
~/anaconda3/envs/jax/bin/python make_subs.py audio      # writes the subtitles into the current directory
```

- `text` is the subtitle and `say` is what gets spoken. Spell symbols phonetically in `say`: "kay ex" for kₓ, "zee"
  for z, "F F T". A comma inside a question changes the intonation.
- Keep segments to at most two sentences. `ch` is the chapter index, and the chapter windows are derived from it.
- **`tts.py`** runs Kokoro-82M through kokoro-onnx, on an ONNX Runtime CPU session with 4 intra-op threads.
  - Each segment is trimmed of leading and trailing silence.
  - The timeline is `lead_in_s`, then each segment, with `gap_s` between segments, an extra `chapter_pause_s` at
    each chapter change and `tail_s` at the end.
  - The track is peak-normalised to −1 dBFS.
  - `audio/timing.json` holds `{total, segments: [{id, ch, text, start, end}]}` in seconds of film time. The pilot's
    total is 306.632 s.
  - `asm5_common` reads `timing.json` at import, so previews, `sheet.py` and `fast_render.py` fail until TTS has run
    once.
- **`check_audio.py`** transcribes the track with faster-whisper `small.en` (int8, 4 threads).
  - It prints `script:` (the `text` field) and `heard:` for every segment, and writes the word timestamps to
    `audio/words.json`.
  - Compare `heard:` with the `say` text, since phonetic spellings always differ from `text`. Where a word is
    wrong or missing, rewrite `say` and rerun TTS. Example: Kokoro kept dropping the lone letter "A" in one segment,
    so that sentence was rewritten without it.
- **`make_subs.py`** turns each segment into one or more cues.
  - Long segments split at punctuation into balanced parts of at most 72 characters.
  - Cues are timed by mapping character fractions onto the spoken words. Each cue stays 0.25 s past its last word
    and at least 1 s in total, but ends at least 40 ms before the next cue.
  - Writes `asm_subtitles.srt` and `asm_subtitles.ass`: Lato 42, outline, 120 ms fades, PlayRes 1920×1080.
  - `clean()` holds the episode's symbol rewrites for subtitles, e.g. `k_z` → `kz`.
- New narration changes `timing.json` and therefore every chapter window. Re-preview all chapters after a TTS
  rerun. The cost profile is re-measured on every render, so a stale profile only hurts load balance for one run.

## 5. Scene rules (what makes windowed rendering exact)

A worker builds its scene from the top and **skips** every play that ends before its window. When Manim skips a
play, it jumps the animation to its final state without drawing. A worker's frame therefore equals the
monolithic render's frame only if the scene's state depends on nothing but global time and the plays before it.

1. **One `Chapter` subclass per chapter.**
   - Set `CH` to the chapter index and, optionally, `TAG` to a LaTeX label that `self.show_tag()` fades in at the
     upper left.
   - Number chapters 0..N−1 with no gaps: `chapter_windows()` looks up chapter `c − 1`, and `plan()` indexes
     `SCENES[c]`.
   - `Chapter.setup()` sets the window `f0, f1` (global frames), `t0, t_end` and `always_update_mobjects = True`.
     A subclass that overrides `setup()` must call `super().setup()`.
2. **Continuous motion comes from `self.now`** (global seconds = `t0 + renderer.time`, frame-quantised).
   - Write drift, rotation and phase as pure functions of `self.now`; see `shift_at(t)` in `Ch1Waves`.
   - Never use dt updaters (`add_updater(lambda m, dt: ...)`) and never accumulate state inside an updater.
     Skipped plays run at most one update, so accumulated state would differ between workers.
   - No wall clock and no unseeded randomness.
3. **Discrete changes come from plays.**
   - Animate a `ValueTracker` with `self.play(tracker.animate.set_value(v), run_time=...)`.
   - Read the tracker inside `always_redraw` functions or updaters.
   - A skipped play still lands on the same final value.
4. **Animations suspend the updaters of the mobject they animate.**
   - `Create`, `FadeIn`, `Write` and `Transform` freeze their target for the whole play, then resume its updaters,
     so the target snaps to its current state.
   - This is harmless when the redraw reads trackers that do not change during that play.
   - A redraw that moves with `self.now` must be added with `self.add(m)` and revealed or faded through a tracker
     that it reads itself. Examples:
     - `reveal` in `Ch1Waves`, which draws the curve over `[x0, x0 + r·L]`;
     - `op2` with `fade_vgroup` in `Ch8Recap`;
     - the `op` and `reveal` arguments of `wave_panel`.
5. **Sync to the narration with `self.at(t)`.**
   - `t` is a global time, usually `S("seg_id") + offset`; `S` and `E` give a segment's start and end.
   - `at()` counts the frames to go, `k = ceil((t − now)·60 − 1e-6)`, and waits `(k − 0.5)/60`. That is exactly `k`
     frames on every worker, even when `t` lies on the frame grid (section 6).
   - If the scene is already more than half a frame past `t`, `at()` logs `LATE by …`. Shorten run times until
     previews show no `LATE` lines.
6. **End with `self.finish()`.** It clears all updaters, fades everything out over 0.45 s and pads the scene to
   exactly `f1 − f0` frames. It cannot shorten a chapter whose content overruns the window: the fade-out is then
   cut off at the boundary.
7. **Keep `always_update_mobjects = True` and plain `self.wait(d)` with a constant `d`.**
   - With this setting, every wait renders frame by frame and yields `len(np.arange(0, d, 1/60))` frames, which is
     the count the worker's clock assumes.
   - Without it, Manim may freeze a wait into `int(d·60)` copies of one frame, which can be one frame short. Plain
     updaters also stop during frozen waits.
   - Do not use `self.pause()` (a frozen wait), `wait_until()`, `frozen_frame=` or `stop_condition=`.
   - In Manim 0.22, `png-sequence` output writes a single PNG for a frozen wait. That shifts the preview frame
     indices that `sheet.py` and section 9 rely on.
8. **Keep per-frame work small.** Per-frame Python work is the render time.
   - In `always_redraw` functions and updaters, use `fast_arrow(start, end, color, width, tip)` and
     `fast_dashed(p, q, dash, color, width)` from `asm5_common`. Do not use `Arrow` or `DashedLine`: Manim's
     `Arrow` costs about 10× more.
   - Never build `Tex`, `MathTex`, `Brace` or `SurroundingRectangle` per frame. Build them once, then either use
     `.copy().move_to(...)` per frame or move one persistent mobject with an updater
     (`label.add_updater(lambda m: m.next_to(...))`).
   - Remove redrawn mobjects that are off screen (`self.remove(...)`). Clear updaters on mobjects that have stopped
     moving.
   - Do not copy these places in the pilot, which still build objects per frame:
     - `MathTex` in `k_arrow_in_panel` (the θ, kₓ and k_z labels), in `evanescent_envelope` and in `euler_arrow`;
     - `DashedLine` in `rider` (Ch0).

     Convert them when you reuse them.
9. **Images.**
   - Use `m, fade = self.live_image(fn, width, center, height)`, where `fn(t)` returns an RGBA `uint8` array for
     global time `t`.
   - `live_image` does not add the image to the scene. Call `self.add(m)`, then animate `fade`
     (`self.play(fade.animate.set_value(1))`). Do not use `FadeIn(m)`, because the updater rewrites the alpha
     channel every frame.
   - Keep images axis-aligned and unrotated, because the worker's compositor assumes it.
   - Native-width images automatically get nearest-neighbour resampling.
10. **Compute heavy fields once per process.**
    - `construct()` runs from the top in every piece of its chapter, and the slowest chapters have 7 pieces.
    - Do the heavy work (FFT propagation, maps) outside per-frame functions. Cache long computations to `.npz`.
11. **Layout.** Keep content above `y = SUB_Y = −3.25` scene units. The subtitles sit below that line.
12. **Manim's own outputs are disabled in the worker.** Do not rely on movie files, sections, `add_sound` or
    Manim caching.

Minimal chapter (the scenes file starts with `from asm5_common import *`, which also imports Manim):

```python
class Ch2Example(Chapter):
    CH = 2
    TAG = "2\\quad Example"

    def construct(self):
        S = self.S
        self.show_tag()
        phase_at = lambda t: 0.8 * (t - self.t0)                    # pure function of global time
        reveal, amp = ValueTracker(0.0), ValueTracker(0.0)

        def wave_fn():                                              # moves with self.now → reveal via a tracker
            r = reveal.get_value()
            if r < 1e-3:
                return VMobject()
            return graph_line(lambda x: np.cos(2.6 * x - phase_at(self.now)), -6, -6 + 12 * r, 1.0, 1, 1)
        wave = always_redraw(wave_fn)
        arrow = always_redraw(lambda: fast_arrow([0, -1.5, 0], [amp.get_value(), -1.5, 0], YELLOW, 5, 0.2))
        label = MathTex("k_x", color=C_KX)                           # built once, moved by an updater
        label.add_updater(lambda m: m.next_to([amp.get_value(), -1.5, 0], UP, buff=0.1))
        self.at(S("x1"))
        self.add(wave)
        self.play(reveal.animate.set_value(1.0), run_time=1.2)      # not Create(wave)
        self.add(arrow, label)
        self.at(S("x2") + 0.5)
        self.play(amp.animate.set_value(3.0), run_time=2.0)
        self.finish()
```

## 6. The windowed worker (`fastrender.py`)

```
python fastrender.py <SceneClass> <fa> <fb> <prefix> [--null | --dump]      # f0 <= fa < fb <= f1
```

Setup:
- `config` is set to 1920×1080 at 60 fps, `format = "none"` (Manim writes no files), caching off and the progress
  bar off.
- `.cache/Tex` is copied into a private temporary `media_dir`, which is removed when the worker succeeds. The source
  directory must exist. Missing entries are compiled by the worker and then discarded.
- Four methods are monkey-patched, and `scene.renderer._scene = scene` gives `add_frame` access to `f0`.

| Patched method | Behaviour |
|---|---|
| `CairoRenderer.play` | Compiles the play, then computes its frame count `n = len(np.arange(0, duration, 1/60))` and its first global frame `g = f0 + round(time·60)`. If `g ≥ fb`, raises `EndWindow`. If `g + n ≤ fa`, skips: jumps to the end state and advances the clock by exactly `n/60`, with no static-frame capture and no drawing. Otherwise runs CE's normal path. |
| `CairoRenderer.render` | Returns at once while skipping. For frames before `fa` in a play that straddles the window start, advances the clock one frame without drawing. Otherwise draws the frame and passes `camera.pixel_array` to `add_frame`. |
| `CairoRenderer.add_frame` | Writes the frame only if `fa ≤ g < fb`, raises `EndWindow` at `fb`, and advances the clock by `1/60` per frame. |
| `Camera.display_image_mobject` | Resizes the image to its on-screen box and alpha-blends only that box with numpy. CE instead builds a full-frame RGBA image for every image on every frame (perspective transform plus a full composite). Assumes axis-aligned images. |

The core patches, from `fastrender.py` (`add_frame` shortened):

```python
def play(self, scene, *args, **kwargs):
    self.skip_animations = self._original_skipping_status
    self.update_skipping_status()
    scene.compile_animation_data(*args, **kwargs)
    n = len(np.arange(0, scene.duration, 1 / FPS))           # frames this play produces in a full render
    g = scene.f0 + int(round(self.time * FPS))               # global index of its first frame
    if g >= FB:
        raise EndWindow
    if g + n <= FA:
        self.skip_animations = True                           # entirely before the window: jump to its end
    self.file_writer.add_partial_movie_file(None)
    self.animations_hashes.append(None)
    self.file_writer.begin_animation(not self.skip_animations, animation_index=self.num_plays)
    scene.begin_animations()
    if self.skip_animations:
        self.static_image = None
        scene.play_internal()
        self.time += n / FPS                                  # same whole-frame clock as a rendered play
    else:
        self.save_static_frame_data(scene, scene.static_mobjects)
        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene, mobjects=scene.moving_mobjects)
            self.freeze_current_frame(scene.duration)
        else:
            scene.play_internal()
    self.file_writer.end_animation(not self.skip_animations)
    self.num_plays += 1


def render(self, scene, t, moving_mobjects=None):
    if self.skip_animations:
        return
    g = scene.f0 + int(round(self.time * FPS))
    if g < FA:                                                # straddling play: advance without drawing
        self.time += 1 / FPS
        return
    self.update_frame(scene, moving_mobjects)
    self.add_frame(self.camera.pixel_array)                   # streamed immediately: no copy needed


def add_frame(self, frame, num_frames=1):
    ...
    for _ in range(num_frames):
        g = self._scene.f0 + int(round(self.time * FPS))
        if g >= FB:
            raise EndWindow
        if g >= FA:
            sink.stdin.write(frame.tobytes())                 # (or keep a copy for --dump)
            written += 1
        self.time += 1 / FPS
```

**The clock.**
- CE's own skip path advances the clock by the float duration. The patch advances it by whole frames, so every
  play gets the same frame index `g = f0 + round(time·60)` on every worker.
- The clock values still differ in the last bits, by about 1e-12 s: the render path adds 1/60 per frame, while the
  skip path adds n/60 per play.
- Any duration derived from the clock must therefore be rounded to whole frames, as `at()` and `finish()` do:
  `wait((k − 0.5)/60)` gives exactly `k` frames.
- A raw `wait(t − self.now)` whose target `t` lies on the frame grid can last one frame longer on one worker than
  on another. With ms-rounded times, the grid points are the multiples of 0.05 s. This caused the pilot's
  one-frame step (section 10).
- Constant run times need no care and need not be multiples of 1/60: both paths give a play
  `len(np.arange(0, run_time, 1/60))` frames (about `ceil(run_time·60)`).
- `construct()` stops via `EndWindow` once the window is full. The worker then closes FFmpeg's stdin and waits for
  it to finish.

**Frame sink.** Each worker runs one FFmpeg process:

```
/usr/bin/ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 60 -i - -filter_complex
  "[0:v]scale=out_color_matrix=bt709:out_range=tv:flags=fast_bilinear,format=yuv420p,split=2[va][b];
   [b]setpts=PTS+<fa/60>/TB,ass=asm_subtitles.ass:fontsdir=/usr/share/fonts/truetype/lato,setpts=PTS-<fa/60>/TB[vb]"
  -map [va] <x264> -f mpegts <prefix>_clean.ts   -map [vb] <x264> -f mpegts <prefix>_subs.ts
<x264> = -c:v libx264 -preset superfast -crf 17 -tune animation -threads 2
         -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
```

- The colour conversion runs once, before `split`. The first sink converted once per output with `veryfast`, and
  worker throughput fell by a third (34.5 → 22.6 fps on a Ch3 window).
- Every piece's timestamps start at 0. Shifting PTS by `fa/60` around the `ass` filter makes libass use film time,
  so the global `.ass` file lines up in every piece.
- The pieces are MPEG-TS, each starting on a keyframe, so the concat demuxer can join them by stream copy.

The worker ends with `PIECE <Scene> [fa,fb) frames=<n> OK time=<s> fps=<f>`, or `MISMATCH (expected …)` when the
frame count is wrong.

## 7. Scheduler (`fast_render.py`)

`python fast_render.py [workers]`, run with the Manim venv's Python (`render.sh` adds `TMPDIR` and `nice -n 5`).

1. **Plan.** `plan()` builds a per-frame cost array for each chapter window.
   - Cost is `1/fps` from `.cache/piece_speed.json` wherever an earlier piece covered the frame, and 1/25 s
     elsewhere.
   - Each chapter is cut at equal cumulative cost into `max(1, round(cost / PIECE_COST))` pieces, with
     `PIECE_COST = 22 s`. Pieces never cross chapters.
   - Each piece's planned cost includes `STARTUP = 2.6 s` for process start and fast-forward.
2. **Render.** A `ThreadPoolExecutor(workers)` runs the pieces longest first. Each thread runs one `fastrender.py`
   subprocess under its own `nice -n 5`, added to `render.sh`'s, so workers run at nice 10.
3. **Audio.** In parallel, `loudnorm=I=-16:TP=-1.5:LRA=11` is applied to the narration, then AAC at 192 kb/s,
   48 kHz → `audio.m4a`.
4. **Calibrate.** The measured fps of every piece, `frames / (wall − STARTUP)`, overwrites
   `.cache/piece_speed.json`. The next run plans from it.
5. **Stitch.** The concat demuxer joins the pieces by stream copy:
   - subtitled pieces + audio → `…_1080p60.mp4`;
   - clean pieces + audio + the SRT as `mov_text` (language eng) → `…_softsubs.mp4`.
   Both use `+faststart`. Never add `-shortest`, because the subtitle stream would truncate the video.

## 8. Performance (pilot)

| Stage | Before: Manim CLI | After: windowed, 6 workers |
|---|---|---|
| Frames | 19 min (9 chapters, 3 processes, PNG sequences) | 161.9 s: 41 pieces, frames and encoding in one pass |
| Encoding | 8 min (two slow x264 encodes, burned and clean) | inside the workers (x264 superfast) |
| Stitch + mux | in the encode | 2.5 s (stream copy) |
| Wall clock | ≈ 27 min | **2 min 46 s** (164.4 s in the orchestrator) |

The first windowed run took 3 min 26 s. One Ch2 piece took 181 s at 6.2 fps and held up the run. The fixes were the
fast arrow helpers and cost-profile planning.

Per chapter, from the last run's profile (1080p, worker fps including encoding):

| Chapter | Frames | Pieces | Avg fps | Range fps |
|---|---|---|---|---|
| Ch0Hook | 1379 | 7 | 10.1 | 8.9–12.2 |
| Ch1Waves | 1611 | 2 | 43.3 | 41.4–45.1 |
| Ch2Arrows | 2242 | 5 | 19.0 | 11.6–41.7 |
| Ch3PlaneWaves | 2695 | 6 | 24.1 | 15.2–41.8 |
| Ch4Basis | 2627 | 4 | 29.2 | 20.9–36.3 |
| Ch5Diagonal | 2367 | 4 | 31.4 | 26.2–38.4 |
| Ch6Evanescent | 1473 | 4 | 19.0 | 13.1–35.1 |
| Ch7Algorithm | 2819 | 7 | 19.6 | 10.4–39.1 |
| Ch8Recap | 1185 | 2 | 35.8 | 34.2–37.6 |
| **Total** | **18,398** | **41** | **22.4** | 822 worker-s of rendering + 41 × 2.6 s startup |

Other measurements:
- **Startup.** Start-up plus fast-forward to a chapter's last frame takes 2.2–4.0 s per worker (measured with
  one-frame pieces).
- **Encoding cost.** On a 300-frame Ch3 window, the worker made 34.5 fps with no encoding (`--null`). With the first
  sink (a colour conversion per output, `veryfast`) it made 22.6 fps. This led to the single conversion and
  `superfast`.
- **Fast arrows.** Ch2's arrow rows went from 6.2 to 16.9 fps. Ch7 (segment a3) went from 14.6 to 26.6 fps.
- **Pixel count is not the bottleneck.** A frame takes 186 ms at 1920×1080 and 106 ms at 960×540: 4× the pixels
  costs only 1.75× the time, so Python mobject work dominates.
- **Model.** Wall clock ≈ (Σ render-seconds + pieces × 2.6 s) / workers + stitch time. That gives 155 s for 6
  workers against 162 s measured. 8 workers should take about 2.0–2.2 min (not measured).
- **Load.** About 6–9 cores were busy for under 3 minutes (observed).
- **File size.** 53 MB and 50 MB at superfast CRF 17. The earlier slow-preset encode of the same film was 28 MB.

## 9. Verification checklist

Pilot values are in brackets. Use the clean `_softsubs.mp4` for pixel checks, because burned-in subtitles change
the bottom band.

1. **Pieces.** Every `PIECE` line says `OK`.
2. **Counts.** Check frames, duration and subtitle cues:
   ```bash
   ~/anaconda3/envs/jax/bin/ffprobe -v error -count_frames -show_entries stream=codec_type,nb_read_frames \
       -show_entries format=duration -of compact <file.mp4>
   ```
   - Video frames must equal `round(total·60)` [18,398].
   - The duration must match `timing.json`'s total [306.632 s; the MP4 reports 306.634 s].
   - In the soft-sub file, the subtitle stream's `nb_read_frames` must equal the cue count [86]. Its packet count
     is twice that, because `mov_text` writes an empty packet after each cue.
3. **Seam and pop scan.** Decode the film at 320×180 grey and flag frame-to-frame jumps larger than 3× the local
   median + 1.5:
   ```bash
   /usr/bin/ffmpeg -v error -i <file.mp4> -vf scale=320:180 -f rawvideo -pix_fmt gray - | \
   ~/anaconda3/envs/jax/bin/python -c "
   import sys, numpy as np
   fr = 320 * 180; prev = None; d = []
   while True:
       b = sys.stdin.buffer.read(fr)
       if len(b) < fr: break
       a = np.frombuffer(b, np.uint8).astype(np.float32)
       if prev is not None: d.append(np.abs(a - prev).mean())
       prev = a
   d = np.array(d); t = (np.arange(len(d)) + 1) / 60
   win = 30; pad = np.pad(d, win, mode='edge')
   run = np.array([np.median(pad[i:i + 2 * win + 1]) for i in range(len(d))])
   print(len(d) + 1, 'frames; spikes at', [f'{t[i]:.2f}s' for i in np.where(d > 3 * run + 1.5)[0]])"
   ```
   Hits should be real transitions only. Compare them with the piece starts (`fa/60` in `.cache/piece_speed.json`)
   and look at each one. [12.33 s, 281.25–281.30 s, 286.65–286.68 s, all intended fades]
4. **Small-region snaps.** The full-frame scan misses small glitches: it flagged neither the Ch1 wave snap nor the
   Ch8 dial snap. For every redraw that moves with `self.now`, crop its region around the play that introduces it,
   and look for a single frame-to-frame jump far above its neighbours:
   ```bash
   /usr/bin/ffmpeg -v error -i <softsubs.mp4> -vf "select='between(n\,1540\,1560)',crop=1920:300:0:228,format=gray" \
       -vsync 0 -f rawvideo - | ~/anaconda3/envs/jax/bin/python -c "
   import sys, numpy as np
   a = np.frombuffer(sys.stdin.buffer.read(), np.uint8).reshape(-1, 300, 1920).astype(float)
   print([round(float(np.abs(a[i] - a[i - 1]).mean()), 2) for i in range(1, len(a))])"
   ```
   [Ch1 wave before the fix: 11.1 against about 3 for normal drift; after the fix, 2.9–3.0 throughout]
5. **Exactness.** Run this after touching the renderer patches, `at()` or `finish()`, or the scene rules.
   - **Timing trace** (cheap, no pixels).
     - Copy `fastrender.py` to a scratch file. Set its `HERE` to the project folder, and add
       `print(f"TRACE g={g} n={n}", file=sys.stderr)` in `play` before the `EndWindow` check.
     - Run it with `--null` for a window that renders across a fragile point and for one that starts after it,
       e.g. Ch7 `[16132,16420)` and `[16493,16494)` around `a6`.
     - Every play that both runs reach must start at the same `g` with the same `n`. [before the `at()` fix: 16408
       vs 16407; after: 16407 in both]
   - **Pixels.** Dump two overlapping windows that start at different frames, e.g. `[6680,6710)` and `[6700,6710)`.
     The shared frames must be identical:
     ```python
     import numpy as np
     a, b = np.load("a_raw.npy"), np.load("b_raw.npy")              # [6680,6710) and [6700,6710)
     print(int(np.abs(a[20:].astype(int) - b.astype(int)).max()))  # must be 0
     ```
   - **Optional: against Manim's own renderer.** Render the chapter with
     `OUT=<dir> RES=1920,1080 FPS=60 ./preview.sh <Chapter>`; the PNG index is the global frame minus `f0`.
     - Vector content matches exactly [Ch3 seam: 0].
     - Images can differ, because CE's compositor uses a truncated box plus a perspective transform. Native-size,
       whole-pixel images agree within 1 [Ch7 heat maps].
     - Other images can differ by more: the Ch4 panels (364.5 px wide) and the Ch0 heat map (fractional offset).
6. **Loudness.** `ffmpeg -i <file.mp4> -map 0:a -af ebur128 -f null -` should give integrated loudness of about
   −16 LUFS [−16.3].
7. **Frames.** Extract stills with `ffmpeg -ss <t> -i <file.mp4> -frames:v 1 t<t>.png` at several times per
   chapter. Check the layout, overlap with the subtitle band and clipped text.
8. **Transcript.** The Whisper transcript must match the `say` text (section 4).

## 10. Pitfalls

| Symptom | Cause | Fix (already in the code) |
|---|---|---|
| One-frame timing step at a seam (pilot MP4: frame 16493, 274.9 s) | `at()` waited a float duration. The render clock and the skip clock differ by about 1e-12 s, which flips `len(np.arange(0, d, 1/60))` when the target lies exactly on the frame grid (`a6` = 273.45 s) | `at()` counts whole frames, `k = ceil(x − 1e-6)`, then waits `(k − 0.5)/60` (*fixed 2026-10-10*) |
| A moving redraw stalls during its intro, then jumps (pilot MP4: the Ch1 wave at 25.8 s; the Ch8 dials pop in at 294.4 s, blink, freeze and snap at 295.7 s) | `Create` and `FadeIn` suspend the updaters of the mobject they animate | `self.add(m)` plus a tracker read inside the redraw (`reveal`, `op2`) (*fixed 2026-10-10*) |
| Ghost trails of moving objects in windowed frames | a `get_frame` override returned the live canvas, so Manim's cached `static_image` aliased it | keep CE's copying `get_frame`; pass `camera.pixel_array` straight to `add_frame`, which streams it before the next draw |
| Random `FileNotFoundError: .cache/Tex/<hash>_.svg` with several workers | Manim writes and deletes temporary SVGs next to cached ones, so workers sharing a directory race | each worker uses a private copy of `.cache/Tex` |
| Seams jump; pieces drift by part of a frame | CE's skip path advances the clock by the float duration | the skip path adds `n/60` |
| Fast-forward is slow | skip mode still captured static frame data and called `render` | the patched `play` and `render` do neither while skipping |
| Motion differs between pieces | dt-based or accumulating updater | pure functions of `self.now` (`shift_at`) |
| One piece takes far longer than the rest | `Arrow` and `DashedLine` built every frame (Ch2: 6.2 fps) | `fast_arrow` and `fast_dashed` (16.9 fps), plus cost-profile planning |
| Workers slower with encoding (34.5 → 22.6 fps) | colour conversion per output and a slow preset | one conversion before `split`, `superfast`, `fast_bilinear` |
| Images pop instead of fading | the updater overwrote the alpha that `FadeIn` set | the `live_image` fade tracker |
| An image never appears | `live_image` does not add the mobject to the scene | `self.add(m)`, then animate `fade` |
| Waits freeze; updaters stop | `always_update_mobjects` is False | `Chapter.setup` sets it to True |
| Subtitles shifted in every piece but the first | each piece's PTS starts at 0 | `setpts=PTS+fa/60` before `ass`, minus after |
| `FileNotFoundError: audio/timing.json` in previews or the render | `asm5_common` reads it at import | run TTS first |
| `asm5w_*` temp directories pile up | a crashed worker skips its cleanup | `rm -rf $TMPDIR/asm5w_*` after a crash |
| `No such filter: 'ass'` | the `jax` env's ffmpeg 5.1 lacks libass | use `/usr/bin/ffmpeg` |
| Soft-sub MP4 cut short | `-shortest` with a `mov_text` stream | no `-shortest` |
| Whisper `decode_audio` error (`metadata_errors`) | faster-whisper 1.2.1 against a recent PyAV | pass a 16 kHz float32 numpy array |
| A word missing from the voice | Kokoro drops lone letters (e.g. "A") | rephrase `say`; recheck with Whisper |
| Monitoring with `pgrep -f fastrender` counts the monitor itself | the pattern matches the calling shell | `pgrep -f '[f]astrender'` |

## 11. New episode checklist

`Optics5min/OUTLINE.md` §5 sets the series layout:
- one folder per episode, `Optics5min/E0N_slug/`, holding `narration.json`, `physics.py` (which writes
  `data/*.npz`), `scenes.py`, `render.sh` and `README.md`;
- shared code in `Optics5min/common/`;
- heavy field computations on the GPU;
- a one-minute cut for each episode, which can use the same pipeline.

Until `common/` exists, copy the pilot:

1. Copy these files into the new folder: `narration.json`, `tts.py`, `check_audio.py`, `make_subs.py`,
   `asm5_common.py`, `asm5_scenes.py`, `fastrender.py`, `fast_render.py`, `render.sh`, `preview.sh`, `sheet.py`. Do
   not copy `audio/`, `.cache/piece_speed.json` or the MP4s.
2. Rename the hard-coded names:

   | File | Change |
   |---|---|
   | `fastrender.py` | `import asm5_scenes` and `getattr(asm5_scenes, …)`, `asm_subtitles.ass`, temp prefix `asm5w_`, the ffmpeg path |
   | `fast_render.py` | `import asm5_common`, the `SCENES` list, `OUT_SUBS`, `OUT_SOFT`, `asm_subtitles.srt`, prefix `asm5fast_`, `FF` (the ffmpeg path) |
   | `make_subs.py` | output file names, `clean()` symbol rewrites |
   | `preview.sh` | chapter class list, scene file name, the `OUT` default (`$SP/asm5pv`; give each episode its own) |
   | `sheet.py` | `import asm5_common`, chapter class list, `fps = 15` (must equal the preview FPS) |
   | `render.sh`, `preview.sh` | `SP` (venv and model location) |
   | scenes file | `from asm5_common import *`, the `CHAPTERS` list (unused; keep it in sync) |

   Keep the chapter class lists identical, in chapter order (0..N−1), in `fast_render.py`, `preview.sh` and
   `sheet.py`. `render.sh`'s `FFMPEG` variable is unused, because the worker and the orchestrator hard-code
   `/usr/bin/ffmpeg`.
3. Write `narration.json` and run the three audio steps (section 4). Fix `say` until the transcript matches.
4. Write one `Chapter` per chapter, following section 5. Iterate with `preview.sh` (960×540, 15 fps) and
   `sheet.py` until no `LATE` lines remain.
5. Warm the LaTeX cache from the preview media directories (the cache does not depend on resolution):
   `mkdir -p .cache/Tex && cp -n "$OUT"/media_*/Tex/* .cache/Tex/`.
6. Run `./render.sh --skip-audio`. The first run plans with the default 25 fps and writes the profile, so a second
   run is better balanced.
7. Verify (section 9). Write the README with the chapter table, and say which panels are computed and which are
   schematic.

## 12. Tuning knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| workers | `WORKERS` env in `render.sh`, or argv of `fast_render.py` | 6 | wall clock ≈ (Σ render-s + pieces × 2.6 s) / workers; keep ≤ 8 on the shared server |
| `PIECE_COST` | `fast_render.py` | 22 s | smaller gives better balance but adds more startup overhead |
| `STARTUP` | `fast_render.py` | 2.6 s | added to each piece's planned cost, and subtracted from wall time when calibrating (measured 2.2–4.0 s) |
| default cost | `plan()` in `fast_render.py` | 1/25 s per frame | used for frames without profile data |
| x264 preset, CRF | `fastrender.py` | superfast, 17 | a slower preset gives smaller files but lower worker fps |
| x264 threads | `fastrender.py` | 2 per output | two outputs per worker |
| size, frame rate | `W, H, FPS` in `fastrender.py`; `FPS` in `fast_render.py` | 1920×1080, 60 | must match the `.ass` PlayRes and the scenes' frame maths |
| priority | `nice -n 5` in `render.sh` and again per worker | workers at nice 10 | keep it |

## 13. Why not GPU, and next steps

- **Python dominates.** Most of the cost is Python building and transforming mobjects every frame. Cairo
  rasterisation is a small part (4× the pixels costs 1.75× the time), so a GPU rasteriser would not remove it.
- **GPU renderers.** CE's OpenGL renderer and 3b1b's ManimGL rasterise on the GPU. Both need a headless EGL context
  on this server, and their features differ. ManimGL would also need the scenes ported. Neither was tried.
- **Encoding.** The A100s have no NVENC, so there is no GPU encoding. CPU x264 at `superfast` keeps up anyway.
- **Parallelism instead.** Process parallelism over the timeline gave a 10× speed-up. The only change to scene code
  was following the rules in section 5.

Possible next steps (none done yet):
- re-render the pilot to pick up the 2026-10-10 fixes;
- convert the remaining per-frame `MathTex` and `DashedLine` builds (section 5, rule 8);
- run with 8 workers (about 2.0–2.2 min predicted);
- re-encode the final file once with a slower preset if size matters;
- test OpenGL or ManimGL on one chapter that redraws heavily;
- move the venvs out of `/tmp`;
- factor the scripts into `Optics5min/common/`.
