"""Speech-to-text providers."""

from gravelvox_engine.stt.base import STTProvider, TranscribeOptions
from gravelvox_engine.stt.local import FasterWhisperProvider

__all__ = ["FasterWhisperProvider", "STTProvider", "TranscribeOptions"]
