"""The protocol every speech-to-text backend implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from gravelvox_engine.types import Transcript


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    language: str | None = None  # None means detect
    vad: bool = True
    word_timestamps: bool = False
    beam_size: int = 5


@runtime_checkable
class STTProvider(Protocol):
    """A speech-to-text backend: the local one today; a Runpod GPU worker and others later."""

    name: str

    def transcribe(self, audio: Path, options: TranscribeOptions | None = None) -> Transcript: ...
