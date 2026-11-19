# 2. Decode audio ourselves, fail with clear errors, and test the real dependencies

- Status: accepted
- Date: 2026-11-19
- Amends: [ADR 1](0001-provider-protocol-and-cpu-first-engine.md)

## Context

The first end-to-end runs on a real machine, with real audio files (text-to-speech clips: clean, with added noise, and deliberately damaged), failed in ways the unit tests could not see, because the tests faked faster-whisper entirely:

1. Windows Smart App Control blocked two of PyAV's DLLs the first time they loaded. The engine reported that `ImportError` as "faster-whisper is not installed".
2. The resolver picked PyAV 19, which removed an argument that faster-whisper 1.2.1 (the latest release) still passes to `av.open()`. Decoding failed with a `TypeError`.
3. faster-whisper's decoder stops without an error at the first damaged packet. Two MP3 files joined end to end, a tag in the middle of an MP3, or a corrupted stretch in an M4A or AAC file produced a transcript of only the first part, with exit code 0.
4. Bad input and bad options reached the user as raw tracebacks: files that are not audio, unknown language codes, typos in the model name, `--device cuda` without NVIDIA's libraries.
5. Under heavy noise Whisper drops a whole 30-second window when it is not confident about it, and reports nothing. Its temperature fallback samples without a seed, so the same command gave different text on each run.

## Decision

- Decode audio in `gravelvox_engine.audio`, packet by packet with PyAV, and pass faster-whisper the samples. A packet that fails to decode is skipped and counted, and the resampler is rebuilt when the sample rate or channel layout changes partway through a file. The engine no longer calls faster-whisper's decoder, so it needs no cap on the PyAV version.
- Validate the language code, then decode, then load the model, so mistakes fail in under a second and before any download.
- Turn each failure into a typed error with one line of output, and map the types to exit codes: runtime failures 1, usage errors 2, dependency problems 3, Ctrl+C 130. Tell "not installed" apart from "installed but failed to load".
- Report what would otherwise be silent. Warn about skipped packets, about a file that became unreadable partway, and about text that covers less than 60% of the speech the voice-activity filter kept. Record the device and weight type that actually ran, and each segment's sampling temperature.
- Seed CTranslate2's sampling (seed 0 by default), so the temperature fallback gives the same text on every run.
- Look for the model in the local cache first and download only when it is missing; report download time separately from load time.
- `--device auto` falls back to the CPU, with a warning, when the GPU's libraries are missing.
- Import PyAV, numpy and faster-whisper only when a transcription starts. A library that is missing or fails to load is then reported as a dependency error with exit code 3, and `--help`, `--version` and argument errors work without them.
- Keep tests that run the real dependencies without a model: the PyAV decoder on generated files, and the faster-whisper, CTranslate2 and onnxruntime functions the engine calls. CI is set up to run them on Linux, Windows and macOS, and a weekly job re-resolves every dependency to its newest release.

## Consequences

- A dependency release that breaks decoding, the voice-activity model or a function the engine calls fails CI instead of failing for users.
- gravelvox now owns about 100 lines of decoding code that faster-whisper used to provide, and has to keep them working across PyAV releases.
- The provider is named `local` instead of `local-cpu`, because it can run on a GPU.
- ADR 1 calls the confidence fields per-segment. They are computed per decoding window of up to 30 seconds, which limits how finely a future router can use them.
- The broad `except Exception` blocks around model download, model load and inference trade tracebacks for readable messages, and the original exception stays chained. The decoder catches only FFmpeg, OS and value errors, so any other failure there still surfaces as a traceback. `MemoryError` and `KeyboardInterrupt` are handled separately.
- Two gaps inherited from faster-whisper remain and are documented in the README: memory grows with file length, and `--word-timestamps` can drop words at window boundaries on long files.
