from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from gravelvox_engine.audio import DecodedAudio
from gravelvox_engine.errors import (
    DependencyError,
    InvalidOptionError,
    MissingDependencyError,
    ModelLoadError,
    TranscriptionError,
)
from gravelvox_engine.stt import FasterWhisperProvider, STTProvider, TranscribeOptions, local

ENGLISH = [(0.0, 1.5, " the quote"), (1.5, 3.0, " covers the roof")]


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel so tests never download a model.

    Class attributes shape the next run: segments to return, VAD output length, and errors
    on load, on transcribe() for a given device, or partway through the segment stream.
    """

    instances: list[FakeWhisperModel] = []
    segments: list[tuple[float, float, str]] = ENGLISH
    duration_after_vad = 3.0
    supported_languages = ["en", "es", "fr"]
    fail_with: Exception | None = None
    transcribe_errors: dict[str, Exception] = {}
    stream_error: Exception | None = None

    def __init__(self, path, **kwargs):
        if FakeWhisperModel.fail_with is not None:
            raise FakeWhisperModel.fail_with
        self.path = path
        self.kwargs = kwargs
        self.calls = []
        device = "cuda" if kwargs["device"] in ("cuda", "auto") else "cpu"
        self.model = SimpleNamespace(device=device, compute_type="int8_float32")
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        error = FakeWhisperModel.transcribe_errors.get(self.model.device)
        if error is not None:
            raise error
        language = kwargs["language"] or "en"
        probability = 1 if kwargs["language"] else 0.97
        info = SimpleNamespace(
            language=language,
            language_probability=probability,
            duration=3.0,
            duration_after_vad=FakeWhisperModel.duration_after_vad,
        )
        return self._stream(), info

    @staticmethod
    def _stream():
        for index, (start, end, text) in enumerate(FakeWhisperModel.segments):
            if index == 1 and FakeWhisperModel.stream_error is not None:
                raise FakeWhisperModel.stream_error
            words = [SimpleNamespace(start=start, end=start + 0.4, word=" w", probability=0.9)]
            yield SimpleNamespace(
                start=start,
                end=end,
                text=text,
                avg_logprob=-0.2,
                no_speech_prob=0.01,
                compression_ratio=1.1,
                temperature=0.0,
                words=words if index == 0 else None,
            )


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """A stand-in faster_whisper module, plus a decoder that returns 3 s of silence."""
    FakeWhisperModel.instances = []
    FakeWhisperModel.segments = ENGLISH
    FakeWhisperModel.duration_after_vad = 3.0
    FakeWhisperModel.supported_languages = ["en", "es", "fr"]
    FakeWhisperModel.fail_with = None
    FakeWhisperModel.transcribe_errors = {}
    FakeWhisperModel.stream_error = None

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    module.available_models = lambda: ["tiny", "small"]
    module.cached = {"tiny", "small"}
    module.downloads = []
    module.decoded = []
    module.seeds = []

    def download_model(name, local_files_only=False, cache_dir=None):
        module.downloads.append((name, local_files_only))
        if local_files_only and name not in module.cached:
            raise FileNotFoundError(f"{name} is not cached")
        return f"/cache/{name}"

    def decode(path):
        module.decoded.append(path)
        return DecodedAudio(samples=np.zeros(3 * 16_000, dtype=np.float32))

    module.download_model = download_model
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(local, "decode_audio", decode)
    monkeypatch.setattr(local, "_seed", module.seeds.append)
    return module


def test_provider_satisfies_protocol():
    assert isinstance(FasterWhisperProvider(), STTProvider)


def test_maps_segments_words_timing_and_device(fake_faster_whisper, silent_wav):
    provider = FasterWhisperProvider("tiny", cpu_threads=4)
    transcript = provider.transcribe(silent_wav, TranscribeOptions(word_timestamps=True))

    assert transcript.text == "the quote covers the roof"
    assert (transcript.language, transcript.language_probability) == ("en", 0.97)
    assert (transcript.provider, transcript.model) == ("local", "tiny")
    assert (transcript.device, transcript.compute_type) == ("cpu", "int8_float32")
    assert transcript.speech_s == 3.0
    assert transcript.warnings == ()
    assert [s.temperature for s in transcript.segments] == [0.0, 0.0]
    assert transcript.segments[0].words[0].word == " w"
    assert transcript.segments[1].words == ()
    assert transcript.timing.audio_s == 3.0
    assert transcript.timing.download_s == 0.0

    assert fake_faster_whisper.decoded == [silent_wav]
    assert fake_faster_whisper.downloads == [("tiny", True)]  # cached: no Hub request
    model = FakeWhisperModel.instances[0]
    assert model.path == "/cache/tiny"
    assert model.kwargs == {"device": "cpu", "compute_type": "int8", "cpu_threads": 4}
    samples, kwargs = model.calls[0]
    assert len(samples) == 3 * 16_000
    assert kwargs == {"language": None, "beam_size": 5, "vad_filter": True, "word_timestamps": True}


def test_loads_the_model_once(fake_faster_whisper, silent_wav):
    provider = FasterWhisperProvider("tiny")
    first = provider.transcribe(silent_wav)
    second = provider.transcribe(silent_wav)
    assert len(FakeWhisperModel.instances) == 1
    assert first.timing.model_load_s >= 0.0
    assert second.timing.model_load_s == 0.0


def test_sampling_is_seeded_before_loading_and_before_each_run(fake_faster_whisper, silent_wav):
    provider = FasterWhisperProvider("tiny", seed=7)
    provider.transcribe(silent_wav)
    provider.transcribe(silent_wav)
    assert fake_faster_whisper.seeds == [7, 7, 7]


def test_downloads_a_model_that_is_not_cached(fake_faster_whisper, silent_wav):
    fake_faster_whisper.cached = set()
    FasterWhisperProvider("small").transcribe(silent_wav)
    assert fake_faster_whisper.downloads == [("small", True), ("small", False)]


def test_a_local_model_directory_is_used_directly(fake_faster_whisper, silent_wav, tmp_path):
    FasterWhisperProvider(str(tmp_path)).transcribe(silent_wav)
    assert FakeWhisperModel.instances[0].path == str(tmp_path)
    assert fake_faster_whisper.downloads == []


def test_an_unknown_model_name_lists_the_valid_ones(fake_faster_whisper, silent_wav):
    with pytest.raises(ModelLoadError, match="unknown model 'smal'; use one of tiny, small"):
        FasterWhisperProvider("smal").transcribe(silent_wav)


def test_a_missing_model_directory_is_named(fake_faster_whisper, silent_wav, tmp_path):
    missing = str(tmp_path / "models" / "mine")
    with pytest.raises(ModelLoadError, match="model directory not found"):
        FasterWhisperProvider(missing).transcribe(silent_wav)


def test_a_download_failure_names_the_model(fake_faster_whisper, silent_wav, monkeypatch):
    def offline(name, local_files_only=False, cache_dir=None):
        raise OSError("outgoing traffic has been disabled")

    monkeypatch.setattr(fake_faster_whisper, "download_model", offline)
    with pytest.raises(ModelLoadError, match="could not download model 'tiny': outgoing"):
        FasterWhisperProvider("tiny").transcribe(silent_wav)


def test_a_model_load_failure_names_the_model(fake_faster_whisper, silent_wav):
    FakeWhisperModel.fail_with = ValueError("Invalid compute type: bogus")
    with pytest.raises(ModelLoadError, match="could not load model 'tiny': Invalid compute"):
        FasterWhisperProvider("tiny", compute_type="bogus").transcribe(silent_wav)


def test_language_codes_are_case_insensitive(fake_faster_whisper, silent_wav):
    transcript = FasterWhisperProvider("tiny").transcribe(
        silent_wav, TranscribeOptions(language=" EN ")
    )
    assert FakeWhisperModel.instances[0].calls[0][1]["language"] == "en"
    assert transcript.language == "en"
    assert transcript.language_probability is None  # given, not detected


def test_an_unknown_language_fails_before_decoding(fake_faster_whisper, silent_wav):
    fake_faster_whisper.tokenizer = SimpleNamespace(_LANGUAGE_CODES=("en", "es", "fr"))
    with pytest.raises(InvalidOptionError, match="unsupported language code 'xx'; use one of"):
        FasterWhisperProvider("tiny").transcribe(silent_wav, TranscribeOptions(language="xx"))
    assert fake_faster_whisper.decoded == []


def test_a_language_the_model_lacks_is_rejected(fake_faster_whisper, silent_wav):
    FakeWhisperModel.supported_languages = ["en"]  # an English-only model such as tiny.en
    with pytest.raises(InvalidOptionError, match="does not support language 'de'"):
        FasterWhisperProvider("tiny").transcribe(silent_wav, TranscribeOptions(language="de"))
    assert FakeWhisperModel.instances[0].calls == []


def test_no_language_is_reported_when_nothing_was_transcribed(fake_faster_whisper, silent_wav):
    FakeWhisperModel.segments = []
    transcript = FasterWhisperProvider("tiny").transcribe(silent_wav)
    assert (transcript.text, transcript.language, transcript.language_probability) == (
        "",
        None,
        None,
    )


def test_text_is_joined_without_inventing_spaces(fake_faster_whisper, silent_wav):
    FakeWhisperModel.segments = [(0.0, 1.0, "你好"), (1.0, 2.0, "世界")]
    assert FasterWhisperProvider("tiny").transcribe(silent_wav).text == "你好世界"


def test_times_are_rounded_and_kept_inside_the_audio(fake_faster_whisper, silent_wav):
    FakeWhisperModel.segments = [(0.0, 1.120000000000005, " a"), (1.12, 3.4, " b")]
    segments = FasterWhisperProvider("tiny").transcribe(silent_wav).segments
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.12), (1.12, 3.0)]


def test_warns_when_text_covers_little_of_the_speech(fake_faster_whisper, silent_wav):
    FakeWhisperModel.duration_after_vad = 60.0  # the model produced 3 s of text from 60 s
    transcript = FasterWhisperProvider("tiny").transcribe(silent_wav)
    assert len(transcript.warnings) == 1
    assert "covers 3 s of the 60 s of speech" in transcript.warnings[0]
    no_vad = FasterWhisperProvider("tiny").transcribe(silent_wav, TranscribeOptions(vad=False))
    assert no_vad.warnings == ()


def test_warns_about_skipped_packets(fake_faster_whisper, silent_wav, monkeypatch):
    damaged = DecodedAudio(samples=np.zeros(48_000, dtype=np.float32), skipped_packets=2)
    monkeypatch.setattr(local, "decode_audio", lambda path: damaged)
    transcript = FasterWhisperProvider("tiny").transcribe(silent_wav)
    assert transcript.warnings == (
        "skipped 2 damaged audio packet(s); the transcript may have small gaps",
    )


def test_missing_gpu_libraries_are_explained(fake_faster_whisper, silent_wav):
    FakeWhisperModel.transcribe_errors = {
        "cuda": RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    }
    with pytest.raises(TranscriptionError, match="cublas64_12.dll.*use --device cpu"):
        FasterWhisperProvider("tiny", device="cuda").transcribe(silent_wav)


def test_device_auto_falls_back_to_the_cpu_without_gpu_libraries(fake_faster_whisper, silent_wav):
    FakeWhisperModel.transcribe_errors = {
        "cuda": RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    }
    transcript = FasterWhisperProvider("tiny", device="auto").transcribe(silent_wav)
    assert transcript.device == "cpu"
    assert transcript.text == "the quote covers the roof"
    assert "the CPU was used instead" in transcript.warnings[0]
    assert [m.kwargs["device"] for m in FakeWhisperModel.instances] == ["auto", "cpu"]


def test_a_failure_partway_through_the_stream_is_wrapped(fake_faster_whisper, silent_wav):
    FakeWhisperModel.stream_error = RuntimeError("bad state")
    with pytest.raises(TranscriptionError, match="transcription failed: bad state") as caught:
        FasterWhisperProvider("tiny").transcribe(silent_wav)
    assert "--device cpu" not in str(caught.value)


def test_running_out_of_memory_is_not_disguised(fake_faster_whisper, silent_wav, monkeypatch):
    def too_big(path):
        raise MemoryError

    monkeypatch.setattr(local, "decode_audio", too_big)
    with pytest.raises(MemoryError):
        FasterWhisperProvider("tiny").transcribe(silent_wav)


def test_missing_faster_whisper_is_reported_as_not_installed(monkeypatch, silent_wav):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes the import fail
    with pytest.raises(MissingDependencyError, match="uv sync"):
        FasterWhisperProvider().transcribe(silent_wav)


def test_a_native_library_that_fails_to_load_is_not_reported_as_missing(block_import, silent_wav):
    # On Windows, an application control policy blocking one of PyAV's DLLs looks like this.
    block_import("faster_whisper", ImportError("DLL load failed while importing link: blocked"))
    with pytest.raises(DependencyError, match="failed to load: DLL load failed") as caught:
        FasterWhisperProvider().transcribe(silent_wav)
    assert not isinstance(caught.value, MissingDependencyError)


def test_a_dll_that_the_system_refuses_to_load_is_a_dependency_error(block_import, silent_wav):
    # CTranslate2 loads its DLLs through ctypes, which raises OSError rather than ImportError.
    blocked = OSError("[WinError 4551] An Application Control policy has blocked this file")
    block_import("faster_whisper", blocked)
    with pytest.raises(DependencyError, match="failed to load: .*WinError 4551"):
        FasterWhisperProvider().transcribe(silent_wav)


def test_a_missing_dependency_of_faster_whisper_names_that_module(block_import, silent_wav):
    block_import("faster_whisper", ModuleNotFoundError("No module named 'av'", name="av"))
    with pytest.raises(DependencyError, match="No module named 'av'") as caught:
        FasterWhisperProvider().transcribe(silent_wav)
    assert not isinstance(caught.value, MissingDependencyError)
