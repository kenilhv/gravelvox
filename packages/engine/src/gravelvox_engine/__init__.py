"""Gravelvox engine: CPU-first speech-to-text for noisy, real-world audio."""

from importlib.metadata import PackageNotFoundError, version

from gravelvox_engine.types import Segment, Timing, Transcript, Word

try:
    __version__ = version("gravelvox-engine")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = ["Segment", "Timing", "Transcript", "Word", "__version__"]
