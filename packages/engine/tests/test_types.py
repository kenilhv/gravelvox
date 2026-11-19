from gravelvox_engine.types import Timing


def test_rtf_is_transcribe_time_over_audio_time():
    assert Timing(model_load_s=5.0, transcribe_s=1.5, audio_s=6.0).rtf == 0.25


def test_rtf_is_none_for_empty_audio():
    assert Timing(model_load_s=0.0, transcribe_s=0.1, audio_s=0.0).rtf is None


def test_to_dict_includes_rtf_and_segments(sample_transcript):
    data = sample_transcript.to_dict()
    assert data["timing"]["rtf"] == 0.25
    assert [s["text"] for s in data["segments"]] == [
        " the quote covers the roof",
        " and the gutters",
    ]
    assert data["provider"] == "fake"


def test_to_dict_has_a_stable_set_of_keys(sample_transcript):
    data = sample_transcript.to_dict()
    assert list(data) == [
        "text",
        "language",
        "language_probability",
        "segments",
        "timing",
        "provider",
        "model",
        "device",
        "compute_type",
        "speech_s",
        "warnings",
    ]
    assert list(data["timing"]) == ["model_load_s", "transcribe_s", "audio_s", "download_s", "rtf"]
    assert list(data["segments"][0]) == [
        "start",
        "end",
        "text",
        "avg_logprob",
        "no_speech_prob",
        "compression_ratio",
        "temperature",
        "words",
    ]
    assert data["warnings"] == ()
