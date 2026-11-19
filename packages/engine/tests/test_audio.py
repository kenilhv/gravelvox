"""The real PyAV decoder, on files generated in the test (no model, no fixtures on disk)."""

from __future__ import annotations

import struct
import sys
import wave
import zlib
from pathlib import Path

import av
import numpy as np
import pytest

from gravelvox_engine.audio import SAMPLE_RATE, decode_audio
from gravelvox_engine.errors import AudioDecodeError, DependencyError, MissingDependencyError


def _tone(rate: int, seconds: float) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)


def _write_wav(path: Path, *, rate: int, channels: int, seconds: float) -> Path:
    samples = np.repeat(_tone(rate, seconds)[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return path


def _write_mp3(path: Path, *, rate: int, seconds: float) -> Path:
    codec = "libmp3lame" if "libmp3lame" in av.codecs_available else "mp3"
    if codec not in av.codecs_available:
        pytest.skip("this PyAV build has no MP3 encoder")
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(
            _tone(rate, seconds)[None, :], format="s16", layout="mono"
        )
        frame.sample_rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)
    return path


def _write_png(path: Path) -> Path:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)  # 1x1, 8-bit grayscale
    pixels = zlib.compress(b"\x00\x80")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels))
    path.write_bytes(path.read_bytes() + chunk(b"IEND", b""))
    return path


def test_resamples_44k_stereo_to_16k_mono(tmp_path):
    decoded = decode_audio(_write_wav(tmp_path / "tone.wav", rate=44_100, channels=2, seconds=1.0))
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.ndim == 1
    assert abs(len(decoded.samples) - SAMPLE_RATE) <= SAMPLE_RATE // 100  # within 10 ms
    assert np.abs(decoded.samples).max() > 0.1
    assert decoded.skipped_packets == 0


def test_reads_silence_as_zeros(silent_wav):
    decoded = decode_audio(silent_wav)
    assert decoded.duration_s == 1.0
    assert not np.any(decoded.samples)


@pytest.mark.parametrize("content", [b"", b"RIFF", b"definitely not audio " * 100])
def test_rejects_files_that_are_not_audio(tmp_path, content):
    path = tmp_path / "broken.wav"
    path.write_bytes(content)
    with pytest.raises(AudioDecodeError):
        decode_audio(path)


def test_rejects_a_file_without_an_audio_stream(tmp_path):
    with pytest.raises(AudioDecodeError, match="no audio stream"):
        decode_audio(_write_png(tmp_path / "picture.png"))


def test_rejects_a_wav_with_no_samples(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
    with pytest.raises(AudioDecodeError, match="no audio samples"):
        decode_audio(path)


def test_rejects_a_wav_header_that_claims_zero_channels(tmp_path):
    # PyAV raises a plain ValueError ("cannot decode unknown codec") for every packet here.
    data = b"\x00\x00" * 1600
    fmt = struct.pack("<HHIIHH", 1, 0, 16_000, 32_000, 2, 16)  # PCM, 0 channels
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path = tmp_path / "zero_channels.wav"
    path.write_bytes(header + body)
    with pytest.raises(AudioDecodeError):
        decode_audio(path)


def test_reads_past_the_join_of_two_mp3_files(tmp_path):
    # faster_whisper.decode_audio stops at the join and returns only the first file.
    part = _write_mp3(tmp_path / "part.mp3", rate=16_000, seconds=1.0).read_bytes()
    joined = tmp_path / "joined.mp3"
    joined.write_bytes(part + part)
    assert decode_audio(joined).duration_s == pytest.approx(2.0, abs=0.15)


def test_survives_a_sample_rate_change_partway_through(tmp_path):
    # faster_whisper.decode_audio aborts here with "Frame does not match AudioFifo parameters".
    first = _write_mp3(tmp_path / "a.mp3", rate=22_050, seconds=1.0).read_bytes()
    second = _write_mp3(tmp_path / "b.mp3", rate=44_100, seconds=1.0).read_bytes()
    mixed = tmp_path / "mixed.mp3"
    mixed.write_bytes(first + second)
    assert decode_audio(mixed).duration_s == pytest.approx(2.0, abs=0.15)


def test_skips_damaged_packets_instead_of_stopping(tmp_path):
    data = bytearray(_write_mp3(tmp_path / "clip.mp3", rate=16_000, seconds=3.0).read_bytes())
    middle = len(data) // 2
    data[middle : middle + 400] = b"\xff" * 400  # garbage that looks like frame headers
    damaged = tmp_path / "damaged.mp3"
    damaged.write_bytes(bytes(data))
    assert decode_audio(damaged).duration_s > 2.5


def test_a_blocked_pyav_library_is_a_dependency_error(block_import, silent_wav):
    blocked = ImportError(
        "DLL load failed while importing _core: an Application Control policy blocked this file"
    )
    block_import("av", blocked)
    with pytest.raises(DependencyError, match="native library failed to load: DLL load failed"):
        decode_audio(silent_wav)


def test_missing_pyav_is_reported_as_not_installed(monkeypatch, silent_wav):
    monkeypatch.setitem(sys.modules, "av", None)  # makes the import fail
    with pytest.raises(MissingDependencyError, match="av is not installed"):
        decode_audio(silent_wav)
