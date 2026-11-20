# Gravelvox

Voice AI for the real world, not the quiet room.

Gravelvox is an open-source toolkit for turning messy, noisy, multi-speaker audio into searchable, evaluated, agent-accessible knowledge. It is CPU-first: it runs on a laptop with no API keys, and the plan is to escalate only the segments that need it to a GPU.

> **Status: early.** Today the repository contains one working piece: a command-line transcriber that runs on CPU (faster-whisper with int8 weights) and reports its real-time factor. Everything under "Roadmap" is planned, not built. Measurements will be published with their method and raw data.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.14, on Windows x64, Linux (x64 or arm64, glibc 2.28 or newer) or macOS 14 or newer on Apple silicon. Python 3.15, Intel Macs, Windows on Arm and Alpine Linux are not supported yet, because onnxruntime or CTranslate2 publish no wheels for them.

```bash
git clone https://github.com/kenilhv/gravelvox.git
cd gravelvox
uv sync
uv run gravelvox transcribe path/to/audio.wav
```

The package is not on PyPI yet, so install it from a checkout as shown.

The first run downloads the model weights from Hugging Face: about 490 MB for the default `small` model, or 78 MB for `--model tiny`. Later runs load the cached copy without contacting the Hub. Most formats FFmpeg can read work, including wav, mp3, m4a, aac, flac, ogg, opus and webm; gravelvox decodes them with PyAV, so FFmpeg does not need to be installed.

| Flag | What it does |
|---|---|
| `--model NAME` | Whisper model: tiny, base, small (default), medium, large-v3, turbo, a distil model, a Hugging Face repo id, or a local model directory |
| `--language CODE` | Skip language detection, for example `--language en` |
| `--timestamps` | Prefix each segment with its start and end time |
| `--json` | Print the full transcript as JSON (see below) |
| `-o`, `--output FILE` | Write the transcript to FILE as UTF-8 instead of printing it |
| `--word-timestamps` | Add word-level timing to the `--json` output (slower) |
| `--no-vad` | Turn off the Silero voice-activity filter (on by default) |
| `--beam-size N` | Beam search width from 1 to 32; 1 is greedy decoding, the fastest (default: 5) |
| `--threads N` | CPU threads; 0 (the default) uses CTranslate2's default of 4, or `OMP_NUM_THREADS` if set |
| `--device cpu\|cuda\|auto` | Where to run the model (default: cpu). `cuda` needs NVIDIA's cuBLAS and cuDNN; `auto` falls back to the CPU, with a warning, when they are missing |
| `--compute-type TYPE` | CTranslate2 weight type, for example int8 (the default), int8_float32 or float32 |
| `--model-dir DIR` | Download and cache models in DIR instead of the Hugging Face cache |
| `-q`, `--quiet` | Don't print the timing summary |

Each run prints a one-line timing summary to stderr:

```
local model=tiny device=cpu audio=65.1s load=0.24s transcribe=2.24s rtf=0.034
```

The real-time factor (RTF) is transcription time divided by audio duration, so lower is faster. Transcription time includes decoding the file and voice-activity filtering. Model loading is reported separately as `load`, and a first-run download as `download`.

The `--json` output holds the text, the language (with its probability when it was detected), every segment with its start and end time, text, `avg_logprob`, `no_speech_prob`, `compression_ratio` and sampling `temperature`, the timing, the device and weight type that actually ran, `speech_s` (how much audio the voice-activity filter kept) and any warnings.

### Warnings

gravelvox prints a warning, and still exits 0, when the result may be incomplete:

- the file had damaged packets that were skipped, or became unreadable partway through;
- the text covers less than 60% of the speech the voice-activity filter kept (checked when the filter is on and kept at least 10 seconds). Whisper drops a whole 30-second window when it cannot transcribe it confidently, which happens with heavy noise or competing voices, and it does not report this on its own.

## Exit codes

Runtime errors are printed to stderr as one line starting with `gravelvox:`. Options the argument parser rejects print the usage text and a line starting with `gravelvox transcribe: error:`.

| Code | Meaning |
|---|---|
| 0 | Transcribed (check stderr for warnings) |
| 1 | The audio could not be decoded; the model could not be found, downloaded or loaded (including a mistyped `--model` name or an unsupported `--compute-type`); transcription failed (for example `--device cuda` without NVIDIA's libraries); memory ran out; the output file could not be written; or stdout was closed before all output was written |
| 2 | Usage error: an option the parser rejects (an unknown flag, `--beam-size` or `--threads` out of range, an unknown `--device`), an unknown language code, an audio path that is missing or not a file, or an output folder that does not exist |
| 3 | faster-whisper, PyAV, numpy or a native library they need is missing or failed to load |
| 130 | Interrupted with Ctrl+C |

## Known limitations

- **Memory grows with file length.** faster-whisper holds the whole recording and its features in memory. With the tiny model on Windows, peak working set was 0.8 GB for an 11-minute file and 3.8 GB for a 60-minute file ([method and raw numbers](docs/measurements/2026-11-19-first-real-runs.md)). Split very long recordings for now; processing in blocks is planned.
- **Word timestamps on long recordings.** With `--word-timestamps`, faster-whisper 1.2.1 can drop words where it moves from one 30-second window to the next. An 11-minute test file that transcribes completely without the flag lost 18 of its 1,900 words with it ([method and raw numbers](docs/measurements/2026-11-19-first-real-runs.md)).
- **Confidence fields are per window.** `avg_logprob`, `no_speech_prob` and `compression_ratio` are computed once per decoding window of up to 30 seconds, so segments decoded together share the same values.
- **Small run-to-run differences.** Sampling is seeded, so the same command gives the same text. On hybrid CPUs (performance and efficiency cores), CTranslate2's arithmetic can still differ slightly between runs, which occasionally moves a segment boundary or changes a probability in the third decimal. Compare text or word error rate, not exact JSON.
- **Joined MP4 and M4A files.** Two MP4-family files joined end to end are read as the first file only, with no warning, because the container hides the rest. Joined MP3 and ADTS AAC files are read to the end.
- **Non-speech input.** With `--no-vad`, Whisper can invent a word or two on silence or noise. Keep the voice-activity filter on unless you need every sample transcribed.

## Troubleshooting

- **PyAV 19 and faster-whisper.** faster-whisper 1.2.1 passes an argument to `av.open()` that PyAV 19 removed ([SYSTRAN/faster-whisper#1589](https://github.com/SYSTRAN/faster-whisper/issues/1589)), and its decoder stops silently at the first damaged packet. gravelvox decodes audio itself and hands faster-whisper the samples, so neither problem affects it.
- **Windows: "An Application Control policy has blocked this file".** Smart App Control or another App Control policy blocked a native library in the virtual environment, and gravelvox exits with code 3. On the machine this project is developed on, that happened the first time two of PyAV's DLLs loaded in a fresh environment, and the same command worked when run again. Whether to change that Windows setting is your decision; gravelvox does not need it changed. The same policy can block a tool's launcher, such as `mypy.exe`, on every run; start the tool through the interpreter instead, for example `uv run python -m mypy`.
- **`--device cuda` fails with `cublas64_12.dll is not found`.** CTranslate2 found an NVIDIA GPU but not the CUDA 12 cuBLAS and cuDNN 9 libraries it needs. Install them, or use `--device cpu` (the default) or `--device auto`.

## Development

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The default test run never downloads a model. `test_cli.py` and `test_local.py` use fakes; `test_audio.py` runs the real PyAV decoder on generated files (including joined, damaged and mixed-sample-rate MP3s), and `test_dependencies.py` checks the faster-whisper, CTranslate2 and onnxruntime functions the engine relies on, so CI catches a dependency release that changes them. An optional end-to-end test runs the real `tiny` model on a speech file you provide:

```bash
GRAVELVOX_E2E_AUDIO=path/to/speech.wav uv run pytest -k e2e
```

In PowerShell: `$env:GRAVELVOX_E2E_AUDIO = "path\to\speech.wav"; uv run pytest -k e2e`.

Releases reach PyPI from a version tag through trusted publishing; see [docs/releasing.md](docs/releasing.md).

## Repository layout

| Path | Status | Purpose |
|---|---|---|
| `packages/engine` | Working | `gravelvox-engine`: audio decoding, the speech-to-text engine and the `gravelvox` CLI |
| `docs/adr` | Started | Architecture decision records |
| `docs/runpod-friction-log.md` | Template only | Dated, reproducible notes from building on Runpod; no entries yet |
| `docs/measurements` | Started | Method and raw numbers behind the figures in this README |
| `apps/gateway` | Planned | Public API: OpenAI-compatible transcription and WebSocket streaming |
| `deploy/runpod` | Planned | Load-balancing worker, GPU escalation worker, measured experiments |
| `packages/router` | Planned | Noise-aware cascade router: CPU first, GPU only when it pays off |
| `packages/eval` | Planned | Eval harness: WER vs SNR, retrieval and agent evals, gating CI |

## Roadmap (planned, in order)

1. Streaming speech-to-text over WebSocket on a Runpod load-balancing endpoint, with measured cold start, session limits and cost per audio-hour.
2. A noise-robustness benchmark: WER at clean, 20, 10, 5 and 0 dB SNR.
3. An edge gateway with OpenAI-compatible endpoints and async ingest.
4. A noise-aware cascade router with a published cost-versus-WER curve, including the negative result if a simple heuristic wins.
5. Hybrid search over transcripts, an MCP server, and Python and TypeScript SDKs.
6. A voice agent that answers questions about recordings, and a synthetic field-audio dataset.

## Principles

- Consent first: don't process recordings without the speakers' consent. The repository ignores common audio and video files by default.
- Every published number comes with its method and raw data.

## License

Copyright 2026 Gravelvox contributors. Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
