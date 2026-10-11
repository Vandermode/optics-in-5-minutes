# E01 · The Angular Spectrum Method, from scratch (5:07)

A 3Blue1Brown-style explainer (1920×1080, 60 fps) for someone who knows only basic calculus and linear algebra.
The narration is by an open-source TTS voice, with subtitles timed to the speech. It is the series pilot,
ported 2026-10-11 from `../../angular_spectrum_method_5min/` onto the shared pipeline in `../common/`. The port
produces the same pixels (see "Port check" below).

* `e01_angular_spectrum_1080p60.mp4`: subtitles burned in.
* `e01_angular_spectrum_1080p60_softsubs.mp4`: clean picture plus a toggleable English subtitle track.
* `subtitles.srt` and `subtitles.ass`: the subtitles.

## Chapters
| # | time | idea |
|---|------|------|
| 0 | 0:00 | hook: a laser through a slit makes bands on a wall (field computed with the method itself) |
| 1 | 0:23 | waves: wavelength λ, and wavenumber k = 2π/λ as "phase per unit length"; one colour → one k |
| 2 | 0:50 | complex numbers as arrows: Euler's formula; light = the arrow's shadow; shift = rotate = × e^{iφ} |
| 3 | 1:27 | plane waves: wave vector (kx, kz); Helmholtz equation → kx² + kz² = k² (one line of calculus); sin θ = kx/k |
| 4 | 2:12 | a basis of waves: linearity; vectors and dot products; functions are long vectors; Fourier transform = dot products |
| 5 | 2:56 | propagation is diagonal: plane waves are the eigenvectors and e^{ikz z} the eigenvalues; U(z) = F⁻¹ D F U(0), like A = P D P⁻¹ (matrices computed) |
| 6 | 3:35 | evanescent waves: kx > k → kz imaginary → e^{−κz}; the diffraction limit (band-limited field, computed) |
| 7 | 4:00 | the algorithm: FFT → × H(kx) → inverse FFT; sampling Δx < λ/2; zero padding (wrap-around computed both ways); slit and lens demos |
| 8 | 4:47 | recap: exact vs paraxial; break into plane waves, advance each phase, add them back up |

## Files

| File | Role |
|---|---|
| `narration.json` | 46 segments, 770 words |
| `audio/` | the pilot's Kokoro audio. `audio/cache.json` lets `../common/tts.py` reuse these WAVs, so `timing.json` reproduces byte for byte. |
| `physics.py` → `data/*.npz` | every computed field: hook slit map, diffraction-limit curves, step-3 profile, wrap-around vs padded maps, slit and lens maps |
| `scenes.py` | the pilot's nine chapter scenes. Changes from the pilot: the header imports `o5common`, the pilot-only helpers `k_arrow_in_panel` and `pipeline` are now local, and the fields are loaded from `data/` instead of recomputed in every render piece. |
| `episode.json` | slug and chapter order |
| `render.sh`, `preview.sh` | wrappers around `../common/` |

All demo fields are computed with the exact scalar angular spectrum method (numpy FFTs, exact kz): slit, lens,
wrap-around, diffraction limit, and the propagation matrices of chapter 5, which stay in `scenes.py` because they
are tiny. The Fourier-series, spectrum and basis panels are schematic illustrations.

## Rebuild

```bash
python physics.py      # ~1 s
./render.sh --skip-audio                        # ~3 min with WORKERS=6 (default 3 in this wrapper)
python ../common/verify.py .
```

## Port check (2026-10-11)

- **Same pixels.** The pipeline doc's `--dump` exactness test (§9.4) was run against the pilot's own worker on 24
  six-frame windows. The windows cover every chapter, and include the hook sweep, the diffraction-limit curves,
  the wrap-around and padding maps, and the slit and lens maps. All 144 frames have a maximum per-channel
  difference of 0.
- **Same audio and subtitles.** `timing.json`, `subtitles.srt` and `subtitles.ass` are byte-identical to the pilot,
  and `audio/narration.wav` is the pilot's track.
- **Same verification.** Full render (38 pieces, 6 workers, 2.9 min): 18,398 frames (= round(306.632·60)), 86 cues,
  −16.3 LUFS. The seam scan finds the pilot's three intended fades (12.33 s, 281.25 s, 286.65 s) and nothing near a
  piece start.
