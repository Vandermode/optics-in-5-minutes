# common/: the shared Optics-in-5-Minutes pipeline

This is the pilot's fast windowed renderer (`../docs/FAST_ANIMATION_PIPELINE.md`), factored so that every episode
folder only contains its own script, physics and scenes. The renderer patches are unchanged, so the exactness
guarantees of the pilot carry over. Paths are parameters.

| File | Role |
|---|---|
| `o5common.py` | (2026-10-11: whole-frame `at()`, as in the pilot's fix; `code_card` keeps indentation)<br> Series library (`load_timing(ep_dir)` first, then `from o5common import *`). <br>• timing: `Chapter` base class (`S`, `E`, `at`, `show_tag`, `finish`, `live_image`, `static_image`, `sweep_image`) <br>• cards: `series_title`, `next_card`, `code_card` <br>• colour code and colour maps: `heat_rgba`, `phase_rgba`, `diverge_rgba`, … <br>• propagation: `asm_rows`, `asm2d` <br>• cheap drawing helpers: `fast_arrow`, `fast_dashed`, `poly_line`, `plot_xy`, `phasor`, `k_circle`, `wave_panel`, … |
| `tts.py` | Kokoro-82M narration → `audio/*.wav`, `narration.wav`, `timing.json`. Unchanged segments are reused from `audio/cache.json`. A segment's `pause_after` adds silence after it. |
| `check_audio.py` | faster-whisper round trip; flags words not heard; writes `audio/words.json` |
| `make_subs.py` | `subtitles.srt` and `subtitles.ass`, timed to the spoken words |
| `fastrender.py` | one worker: global frames `[fa, fb)` of one chapter → clean and subtitled MPEG-TS pieces |
| `fast_render.py` | orchestrator: cost-balanced plan, worker pool (1 BLAS thread per worker), audio, stream-copy stitch → `<slug>_1080p60.mp4` and `<slug>_1080p60_softsubs.mp4` |
| `audio.sh`, `render.sh`, `preview.sh` | entry points (episode folders have `render.sh` and `preview.sh` wrappers) |
| `sheet.py` | contact sheet of preview frames at given film times (red line = subtitle band) |
| `verify.py` | checklist: frame count, soft-sub packets, seam/pop scan, loudness, stills → `.verify/sheet_*.png` |
| `new_episode.sh` | scaffold a new episode folder |

## Environment

One conda environment (`../environment.yml`, built by `../setup_env.sh`) runs everything. Machine-specific paths
(models, FFmpeg, fonts, the Python to use) come from `O5_*` variables with portable defaults (`o5env.py`, `env.sh`).
Override them in `../config.local.sh`, which is gitignored; see `../config.local.sh.example`. `python check_env.py`
reports what is missing.

| What | Default |
|---|---|
| Kokoro model files | `<repo>/.models/kokoro` (`python fetch_models.py`) |
| Whisper cache | `<repo>/.models/hf` |
| FFmpeg with libass | `ffmpeg` on PATH (`O5_FFMPEG`) |
| Fonts | `fonts/` (vendored Lato and DejaVu Sans Mono, registered at import) |

## Typical session (inside an episode folder)

```bash
../common/audio.sh .                    # narration.json -> audio + subtitles
python physics.py
./preview.sh Ch2Foo                     # 960x540@15 previews, 2 processes; prints LATE lines; warms .cache/Tex
./render.sh --skip-audio                # 1080p60, WORKERS=3 by default in episode wrappers
python ../common/verify.py .
```
