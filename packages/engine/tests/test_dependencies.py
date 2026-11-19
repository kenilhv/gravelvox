"""The faster-whisper, CTranslate2 and onnxruntime surface the engine relies on, without a model.

Most unit tests fake faster-whisper, so on their own they cannot notice a dependency release
that changes it. These tests run the real packages.
"""

from __future__ import annotations

import inspect

import faster_whisper
import numpy as np


def test_the_bundled_silero_vad_finds_no_speech_in_silence():
    from faster_whisper.vad import get_speech_timestamps

    assert get_speech_timestamps(np.zeros(2 * 16_000, dtype=np.float32)) == []


def test_the_model_names_the_readme_mentions_exist():
    names = set(faster_whisper.available_models())
    assert {"tiny", "base", "small", "medium", "large-v3", "turbo"} <= names


def test_the_language_list_used_for_early_validation_exists():
    codes = faster_whisper.tokenizer._LANGUAGE_CODES
    assert "en" in codes and "es" in codes and len(codes) > 90


def test_the_parameters_the_engine_passes_still_exist():
    def parameters(function):
        return set(inspect.signature(function).parameters)

    assert {"device", "compute_type", "cpu_threads"} <= parameters(faster_whisper.WhisperModel)
    assert {"language", "beam_size", "vad_filter", "word_timestamps"} <= parameters(
        faster_whisper.WhisperModel.transcribe
    )
    assert {"local_files_only", "cache_dir"} <= parameters(faster_whisper.download_model)
    assert "supported_languages" in dir(faster_whisper.WhisperModel)


def test_sampling_can_be_seeded():
    import ctranslate2

    ctranslate2.set_random_seed(0)
