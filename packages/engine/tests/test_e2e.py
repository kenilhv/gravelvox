"""Optional end-to-end test with a real model; skipped unless GRAVELVOX_E2E_AUDIO is set.

The first run downloads the `tiny` model from Hugging Face.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gravelvox_engine.stt import FasterWhisperProvider

AUDIO = os.environ.get("GRAVELVOX_E2E_AUDIO")


@pytest.mark.skipif(not AUDIO, reason="set GRAVELVOX_E2E_AUDIO to a speech file to run")
def test_e2e_tiny_model_transcribes_real_speech():
    transcript = FasterWhisperProvider("tiny").transcribe(Path(AUDIO))
    assert transcript.text
    assert transcript.timing.audio_s > 0
    assert transcript.timing.rtf is not None
