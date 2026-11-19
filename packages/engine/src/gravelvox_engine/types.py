"""Typed results shared by every speech-to-text provider."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass(frozen=True, slots=True)
class Segment:
    """One decoded segment.

    faster-whisper computes avg_logprob, no_speech_prob and compression_ratio once per
    decoding window (up to 30 s), so segments decoded in the same window share them.
    temperature is the sampling temperature that produced the window: 0.0 normally, higher
    when the decoder fell back to sampling because a greedy result looked wrong.
    """

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    words: tuple[Word, ...] = ()


@dataclass(frozen=True, slots=True)
class Timing:
    """Wall-clock timing for one transcription, in seconds."""

    model_load_s: float
    transcribe_s: float
    audio_s: float
    download_s: float = 0.0

    @property
    def rtf(self) -> float | None:
        """Real-time factor: transcription time divided by audio duration (lower is faster).

        Includes audio decoding and voice-activity filtering; excludes model download and
        loading.
        """
        if self.audio_s <= 0:
            return None
        return self.transcribe_s / self.audio_s


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None  # None when nothing was transcribed and no language was given
    language_probability: float | None  # None unless the language was detected
    segments: tuple[Segment, ...]
    timing: Timing
    provider: str
    model: str
    device: str | None = None  # the device that actually ran the model
    compute_type: str | None = None  # the weight type CTranslate2 actually used
    speech_s: float | None = None  # audio left after the voice-activity filter
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timing"]["rtf"] = self.timing.rtf
        return data
