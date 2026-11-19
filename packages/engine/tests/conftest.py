from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

from gravelvox_engine.types import Segment, Timing, Transcript


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A real one-second WAV file of digital silence (16 kHz, mono, 16-bit)."""
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)
    return path


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        text="the quote covers the roof and the gutters",
        language="en",
        language_probability=0.98,
        segments=(
            Segment(start=0.0, end=1.5, text=" the quote covers the roof", avg_logprob=-0.2),
            Segment(start=1.5, end=3.0, text=" and the gutters", avg_logprob=-0.4),
        ),
        timing=Timing(model_load_s=1.0, transcribe_s=0.75, audio_s=3.0),
        provider="fake",
        model="tiny",
    )


class _FailingImport:
    """An import hook that makes importing one module raise a chosen error."""

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise self.error
        return None


@pytest.fixture
def block_import(monkeypatch):
    """Make `import <name>` fail, the way a missing or blocked native library does."""

    def block(name: str, error: BaseException) -> None:
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_FailingImport(name, error), *sys.meta_path])

    return block
