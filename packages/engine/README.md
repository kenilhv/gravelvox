# gravelvox-engine

The speech-to-text engine behind Gravelvox: a CPU-first transcriber built on faster-whisper (CTranslate2, int8 weights by default).

- It decodes audio itself with PyAV, packet by packet, so two MP3 or ADTS AAC files joined end to end, and MP3, M4A or AAC files with damaged packets, are read to the end instead of being cut short without an error.
- Failures come back as typed errors with one clear message, and results the engine can tell are incomplete (skipped packets, a file that became unreadable partway, low text coverage) carry warnings.
- A small provider protocol lets other backends (a Runpod GPU worker, OpenAI-compatible APIs) plug in later.

Status: early. One provider (`local`) and one CLI command.

## Install

gravelvox-engine is not published on PyPI yet. Install it from a checkout of the repository, with Python 3.11 to 3.14:

```bash
pip install ./packages/engine
```

Inside the repository you can also use `uv sync` and prefix the commands below with `uv run`.

## Command line

```bash
gravelvox transcribe path/to/audio.m4a --timestamps
gravelvox transcribe path/to/audio.m4a --json -o transcript.json
```

The first run downloads the model from Hugging Face (about 490 MB for the default `small`, 78 MB for `--model tiny`). See the [repository README](https://github.com/kenilhv/gravelvox#readme) for every flag, the exit codes and the known limitations.

## Python

```python
from pathlib import Path

from gravelvox_engine.stt import FasterWhisperProvider, TranscribeOptions

provider = FasterWhisperProvider("small")
transcript = provider.transcribe(Path("visit.m4a"), TranscribeOptions(language="en"))

print(transcript.text)
for segment in transcript.segments:
    print(f"{segment.start:7.2f} {segment.end:7.2f} {segment.text.strip()}")
for warning in transcript.warnings:
    print("warning:", warning)
```

Errors are subclasses of `gravelvox_engine.errors.GravelvoxError`: `AudioDecodeError`, `ModelLoadError`, `TranscriptionError`, `InvalidOptionError` and `DependencyError`.

## License

Apache-2.0.
