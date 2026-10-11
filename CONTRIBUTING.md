# Contributing

Setup is in `README.md`. The quality bar is in `PRODUCTION_SPEC.md`. Every scene must follow the render rules in
`docs/FAST_ANIMATION_PIPELINE.md` §5.

## Ground rules

- **One branch and one pull request per change.** Name branches after the episode, e.g. `e03-narration`,
  `e11-polarization`, or `common-fix-at`.
- **Commit source, never media.** The `.gitignore` already excludes films, audio, `data/`, caches, models and
  `dist/`. Do commit these small generated files when they change:
  - `audio/timing.json` and `audio/words.json`;
  - `subtitles.srt` and `subtitles.ass`.

  Reviewers then see exactly how timing and subtitles moved.
- **No unpublished data.** Films show generic computed demos only. Anything from a group's unpublished results
  needs the owner's explicit OK first.
- **`common/` changes go in their own pull request.** They affect every episode. See "Changing `common/`" below.

## Editing the narration of an existing episode

1. Edit `E0N_slug/narration.json`.
   - `text` is the subtitle; `say` is what the voice reads. Spell symbols phonetically in `say`: "kay ex", "F F T",
     "lambda". The full table is in `PRODUCTION_SPEC.md` §3.
   - Keep segments to at most two sentences.
   - `ch` is the chapter index; `"pause_after": 0.8` adds a beat of silence.
2. Run `make audio E=E0N`. Only the changed segments are re-synthesised. Read every `<-- check` line from the
   Whisper round trip, and fix `say` until the words are heard correctly.
3. A changed sentence moves the timing of everything after it, so every later chapter's window shifts.
   - Run `make preview E=E0N` for all chapters, not only the edited one.
   - Fix any `LATE` line and any animation that no longer matches its sentence. Scenes sync to segment ids with
     `self.at(S("id") + offset)`, so usually only offsets and run times need adjusting in `scenes.py`.
4. Run `make episode E=E0N`. It re-renders and verifies, but does not recompute physics. All counts must be OK, and
   seam-scan jumps must be intended transitions. Look at `.verify/sheet_*.png`.
5. Commit `narration.json`, `audio/timing.json`, `audio/words.json`, `subtitles.*` and any `scenes.py` changes.
   Put the verify summary in the pull request.

## Adding an episode

```bash
make new NAME=E11_polarization TITLE="Polarization and the Geometric Phase"
```

This scaffolds the folder: `episode.json`, `scenes.py` and `physics.py` headers, `render.sh` and `preview.sh`.
Then follow `PRODUCTION_SPEC.md` §2–§6:
1. Write the script (about 720–800 words).
2. `make audio`.
3. Write `physics.py` (seeded, deterministic) and run `make physics`.
4. Write the scenes, one `Chapter` per chapter, iterating with `make preview`.
5. `make render`, then `make verify`.
6. Write the `README.md`: chapters, computed vs schematic, rebuild command.
7. Add a row to the table in the top-level `README.md`.

Season 2 candidates are listed in `OUTLINE.md` §6.

## Changing `common/`

- **The renderer is frame-exact.** Pieces rendered by separate worker processes join without a seam. Any change
  to `fastrender.py`, `Chapter.at()` / `finish()` or the scene rules must pass the exactness test in
  `docs/FAST_ANIMATION_PIPELINE.md` §9.4:
  - render windows across a seam with `--dump`;
  - compare them with a single-pass render; the maximum difference must be 0.
- **Re-render and verify at least one episode** that uses the changed helper. List in the pull request which
  episodes need re-rendering.
- **Don't change `common/` while a render runs.** Each worker imports it at start, so a mid-render edit mixes two
  versions in one film.

## Sharing films

Each episode has one shareable file: `<slug>_1080p.mp4`, a clean 1080p60 picture plus an English subtitle track that
viewers can toggle. It is a slow-preset re-encode of the render, about half the size and visually transparent (E01:
22 MB against 52 MB; 47–51 dB PSNR). Desktop players (VLC, QuickTime, mpv) show the subtitles; browser previews such
as Google Drive's ignore them. For YouTube, upload the file and add the episode's `subtitles.srt` as captions.

```bash
make release                # dist/<slug>_1080p.mp4 for every episode (about 2 min each); or E=E03 for one
gh release create season1-v1 dist/*_1080p.mp4 --title "Season 1" --notes "…"   # attach to a GitHub Release
make pack-data              # dist/data/<slug>_data.tar.gz for every episode (the computed physics, ~200 MB total)
gh release create season1-data dist/data/*.tar.gz --title "Season 1 physics data" --notes "for O5_DATA_URL"
```

Collaborators without a GPU then set `O5_DATA_URL=gh:<owner>/<repo>@season1-data` in `config.local.sh`. `make` then
downloads the JAX-heavy episodes' data instead of spending up to an hour per episode on the CPU. Re-pack and re-upload
a bundle whenever that episode's `physics.py` changes.

