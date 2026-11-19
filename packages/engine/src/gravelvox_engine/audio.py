"""Audio decoding: any file FFmpeg can read, to 16 kHz mono float32 samples.

gravelvox decodes packet by packet with PyAV instead of calling faster_whisper.decode_audio.
That function stops without an error at the first damaged packet, which is common in MP3
and AAC files that were cut, joined or carry a tag mid-stream, and it aborts when the sample
rate changes partway through a file. Here a damaged packet is skipped and counted, and the
resampler is rebuilt whenever the input format changes.

PyAV and numpy are imported when decoding starts, not when this module is imported, so a
native library that fails to load is reported as a dependency error instead of a traceback
at start-up.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gravelvox_engine.errors import AudioDecodeError, DependencyError, MissingDependencyError

if TYPE_CHECKING:
    import av
    import numpy as np

SAMPLE_RATE = 16_000  # every Whisper model expects 16 kHz mono audio


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    samples: np.ndarray  # float32, mono, SAMPLE_RATE
    skipped_packets: int = 0  # packets FFmpeg could not decode and that were left out
    stopped_early: str | None = None  # why reading stopped before the end, if it did

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE


def require_native_libraries() -> None:
    """Import PyAV and numpy, turning a failure into a DependencyError."""
    try:
        importlib.import_module("av")
        importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        if exc.name in ("av", "numpy"):
            raise MissingDependencyError(f"{exc.name} is not installed; run `uv sync`") from exc
        raise DependencyError(f"a library gravelvox needs cannot import: {exc}") from exc
    except (ImportError, OSError) as exc:  # for example a DLL blocked by an application policy
        raise DependencyError(f"a native library failed to load: {exc}") from exc


def decode_audio(path: Path | str) -> DecodedAudio:
    """Decode the first audio stream of `path`.

    Raises AudioDecodeError if the file cannot be opened, has no audio stream, or yields no
    samples, and DependencyError if PyAV or numpy cannot be loaded. MemoryError is left alone
    so callers can report it as such.
    """
    require_native_libraries()
    import av
    import numpy as np

    try:
        container = av.open(str(path), mode="r")
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise AudioDecodeError(f"could not open {path}: {exc}") from exc

    chunks: list[np.ndarray] = []
    skipped = 0
    stopped_early = None
    resampler = _Resampler()
    with container:
        if not container.streams.audio:
            raise AudioDecodeError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        try:
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except (av.error.FFmpegError, ValueError):  # ValueError: unknown codec
                    skipped += 1
                    continue
                chunks.extend(resampler.feed(frames))
        except (av.error.FFmpegError, ValueError) as exc:  # the container is damaged from here
            if not chunks:
                raise AudioDecodeError(f"could not decode {path}: {exc}") from exc
            stopped_early = getattr(exc, "strerror", None) or str(exc)
        chunks.extend(resampler.flush())

    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    if samples.size == 0:
        failed = f"; none of its {skipped} audio packets could be decoded" if skipped else ""
        raise AudioDecodeError(f"no audio samples found in {path}{failed}")
    return DecodedAudio(
        samples=samples.astype(np.float32) / 32768.0,
        skipped_packets=skipped,
        stopped_early=stopped_early,
    )


class _Resampler:
    """Converts frames to 16 kHz mono s16, starting a new resampler when the input changes."""

    def __init__(self) -> None:
        self._resampler: av.AudioResampler | None = None
        self._input: tuple[str, str, int] | None = None

    def feed(self, frames: Iterable[av.AudioFrame]) -> Iterator[np.ndarray]:
        import av

        for frame in frames:
            key = (frame.format.name, frame.layout.name, frame.sample_rate)
            resampler = self._resampler
            if resampler is None or key != self._input:
                yield from self.flush()
                resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
                self._resampler, self._input = resampler, key
            for out in resampler.resample(frame):
                yield out.to_ndarray().reshape(-1)

    def flush(self) -> Iterator[np.ndarray]:
        """Emit what the current resampler still holds and forget it."""
        if self._resampler is not None:
            for out in self._resampler.resample(None):
                yield out.to_ndarray().reshape(-1)
        self._resampler = None
        self._input = None
