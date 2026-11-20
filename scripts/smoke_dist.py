"""Smoke-test an installed gravelvox-engine distribution.

Run it in an environment that holds only the built wheel or source distribution:

    uv run --isolated --no-project --with dist/*.whl scripts/smoke_dist.py

It fails if the package is imported from the source tree instead of the installed copy, if a
file is missing from the distribution, or if the native audio stack does not work.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import wave
from importlib import metadata, resources
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    import gravelvox_engine
    from gravelvox_engine.audio import decode_audio
    from gravelvox_engine.stt import FasterWhisperProvider, STTProvider

    installed = Path(gravelvox_engine.__file__).resolve()
    if REPO in installed.parents:
        raise SystemExit(f"imported from the source tree, not the distribution: {installed}")

    version = metadata.version("gravelvox-engine")
    assert gravelvox_engine.__version__ == version, gravelvox_engine.__version__
    assert resources.files("gravelvox_engine").joinpath("py.typed").is_file(), "py.typed missing"
    assert isinstance(FasterWhisperProvider(), STTProvider)

    command = shutil.which("gravelvox")
    assert command, "the gravelvox command was not installed"
    shown = subprocess.run([command, "--version"], capture_output=True, text=True, check=True)
    assert shown.stdout.strip() == f"gravelvox {version}", shown.stdout

    # The native stack: decode one second of generated audio through PyAV and numpy.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tone.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(b"\x00\x01" * 16_000)
        decoded = decode_audio(path)
    assert decoded.duration_s == 1.0, decoded.duration_s

    print(f"gravelvox-engine {version} works from {installed.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
