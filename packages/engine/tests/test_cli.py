from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gravelvox_engine import cli
from gravelvox_engine.errors import (
    AudioDecodeError,
    DependencyError,
    InvalidOptionError,
    MissingDependencyError,
    ModelLoadError,
    TranscriptionError,
)
from gravelvox_engine.types import Timing


class FakeProvider:
    name = "fake"

    def __init__(self, transcript=None, error=None):
        self.transcript = transcript
        self.error = error
        self.calls = []

    def transcribe(self, audio, options=None):
        self.calls.append((audio, options))
        if self.error is not None:
            raise self.error
        return self.transcript


@pytest.fixture
def fake_provider(monkeypatch, sample_transcript):
    provider = FakeProvider(sample_transcript)
    monkeypatch.setattr(cli, "make_provider", lambda args: provider)
    return provider


def _use(monkeypatch, transcript=None, error=None):
    monkeypatch.setattr(cli, "make_provider", lambda args: FakeProvider(transcript, error))


def test_prints_text_and_a_timing_summary(fake_provider, silent_wav, capsys):
    assert cli.main(["transcribe", str(silent_wav)]) == cli.EXIT_OK
    out, err = capsys.readouterr()
    assert out.splitlines() == ["the quote covers the roof", "and the gutters"]
    assert err == "fake model=tiny audio=3.0s load=1.00s transcribe=0.75s rtf=0.250\n"


def test_the_summary_shows_device_and_download_and_quotes_odd_model_names(
    monkeypatch, sample_transcript, silent_wav, capsys
):
    timing = Timing(model_load_s=1.0, transcribe_s=0.75, audio_s=3.0, download_s=12.34)
    transcript = replace(sample_transcript, model="C:/my models/tiny", device="cpu", timing=timing)
    _use(monkeypatch, transcript)
    cli.main(["transcribe", str(silent_wav)])
    assert capsys.readouterr().err == (
        'fake model="C:/my models/tiny" device=cpu audio=3.0s download=12.3s load=1.00s '
        "transcribe=0.75s rtf=0.250\n"
    )


def test_timestamps_prefix_each_segment(fake_provider, silent_wav, capsys):
    cli.main(["transcribe", str(silent_wav), "--timestamps", "--quiet"])
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == "[00:00.00 -> 00:01.50] the quote covers the roof"
    assert err == ""


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00.00"), (59.999, "01:00.00"), (61.5, "01:01.50"), (3603.31, "1:00:03.31")],
)
def test_clock_rounds_cleanly_and_shows_hours(seconds, expected):
    assert cli._clock(seconds) == expected


def test_json_output_is_the_full_transcript(fake_provider, silent_wav, capsys):
    cli.main(["transcribe", str(silent_wav), "--json", "-q"])
    data = json.loads(capsys.readouterr().out)
    assert data["text"] == "the quote covers the roof and the gutters"
    assert data["timing"]["rtf"] == 0.25
    assert len(data["segments"]) == 2


def test_output_file_is_utf8_without_a_byte_order_mark(fake_provider, silent_wav, tmp_path, capsys):
    target = tmp_path / "out.json"
    assert cli.main(["transcribe", str(silent_wav), "--json", "-o", str(target), "-q"]) == 0
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert json.loads(raw)["model"] == "tiny"
    assert capsys.readouterr().out == ""


def test_output_folder_must_exist(fake_provider, silent_wav, tmp_path, capsys):
    target = tmp_path / "missing" / "out.txt"
    assert cli.main(["transcribe", str(silent_wav), "-o", str(target)]) == cli.EXIT_USAGE
    assert "output folder not found" in capsys.readouterr().err
    assert fake_provider.calls == []


def test_options_reach_the_provider(fake_provider, silent_wav):
    argv = ["transcribe", str(silent_wav), "--language", "en", "--no-vad", "--json"]
    cli.main([*argv, "--word-timestamps", "--beam-size", "1", "-q"])
    _, options = fake_provider.calls[0]
    assert options.language == "en"
    assert options.vad is False
    assert options.word_timestamps is True
    assert options.beam_size == 1


def test_word_timestamps_without_json_warns(fake_provider, silent_wav, capsys):
    cli.main(["transcribe", str(silent_wav), "--word-timestamps", "-q"])
    assert "--word-timestamps only changes the --json output" in capsys.readouterr().err


def test_engine_warnings_are_printed(monkeypatch, sample_transcript, silent_wav, capsys):
    _use(monkeypatch, replace(sample_transcript, warnings=("skipped 2 damaged audio packet(s)",)))
    assert cli.main(["transcribe", str(silent_wav), "-q"]) == cli.EXIT_OK
    assert capsys.readouterr().err == "gravelvox: warning: skipped 2 damaged audio packet(s)\n"


def test_missing_audio_file_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["transcribe", str(tmp_path / "nope.wav")]) == cli.EXIT_USAGE
    assert "audio file not found" in capsys.readouterr().err


def test_a_directory_is_not_an_audio_file(tmp_path, capsys):
    assert cli.main(["transcribe", str(tmp_path)]) == cli.EXIT_USAGE
    assert "not a file" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path length limit")
def test_paths_longer_than_260_characters_are_found(tmp_path):
    long_path = tmp_path / ("recording_" + "x" * 240 + ".wav")
    assert len(str(long_path)) > 260
    with open("\\\\?\\" + os.path.abspath(long_path), "wb") as handle:
        handle.write(b"RIFF")
    resolved = cli._resolve_audio(long_path)
    assert resolved is not None and resolved.is_file()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (InvalidOptionError("unsupported language code 'xx'"), cli.EXIT_USAGE),
        (MissingDependencyError("faster-whisper is not installed"), cli.EXIT_DEPENDENCY),
        (DependencyError("faster-whisper is installed but failed to load"), cli.EXIT_DEPENDENCY),
        (ModelLoadError("could not load model 'nope'"), cli.EXIT_ERROR),
        (AudioDecodeError("could not decode broken.wav"), cli.EXIT_ERROR),
        (TranscriptionError("transcription failed: bad state"), cli.EXIT_ERROR),
    ],
)
def test_engine_errors_become_one_line_and_an_exit_code(
    monkeypatch, silent_wav, capsys, error, code
):
    _use(monkeypatch, error=error)
    assert cli.main(["transcribe", str(silent_wav)]) == code
    out, err = capsys.readouterr()
    assert out == ""
    assert err == f"gravelvox: {error}\n"


def test_ctrl_c_exits_130_without_a_traceback(monkeypatch, silent_wav, capsys):
    _use(monkeypatch, error=KeyboardInterrupt())
    assert cli.main(["transcribe", str(silent_wav)]) == cli.EXIT_INTERRUPTED
    assert capsys.readouterr().err == "gravelvox: interrupted\n"


def test_running_out_of_memory_is_reported(monkeypatch, silent_wav, capsys):
    _use(monkeypatch, error=MemoryError())
    assert cli.main(["transcribe", str(silent_wav)]) == cli.EXIT_ERROR
    assert "ran out of memory" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("error", "platform"),
    [(BrokenPipeError(), "linux"), (OSError(errno.EINVAL, "Invalid argument"), "win32")],
)
def test_a_closed_stdout_pipe_exits_quietly(
    monkeypatch, fake_provider, silent_wav, capsys, error, platform
):
    def closed_pipe(transcript, args):
        raise error

    discarded = []
    monkeypatch.setattr(cli.sys, "platform", platform)
    monkeypatch.setattr(cli, "_write_output", closed_pipe)
    monkeypatch.setattr(cli, "_discard_stdout", lambda: discarded.append(True))
    assert cli.main(["transcribe", str(silent_wav)]) == cli.EXIT_ERROR
    assert discarded == [True]
    assert capsys.readouterr().err == ""


def test_other_output_errors_are_not_mistaken_for_a_closed_pipe(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert not cli._is_closed_pipe(OSError(errno.EINVAL, "Invalid argument"))
    assert not cli._is_closed_pipe(OSError(errno.ENOSPC, "No space left on device"))


def test_no_segments_prints_nothing_but_still_summarizes(
    monkeypatch, sample_transcript, silent_wav, capsys
):
    _use(monkeypatch, replace(sample_transcript, text="", segments=()))
    assert cli.main(["transcribe", str(silent_wav)]) == cli.EXIT_OK
    out, err = capsys.readouterr()
    assert out == ""
    assert "rtf=" in err


@pytest.mark.parametrize(
    "flags",
    [
        ["--beam-size", "0"],
        ["--beam-size", str(cli.MAX_BEAM_SIZE + 1)],
        ["--beam-size", "two"],
        ["--threads", "-1"],
        ["--threads", str((os.cpu_count() or 64) + 1)],
        ["--device", "bogus"],
    ],
)
def test_bad_option_values_are_rejected_by_the_parser(silent_wav, capsys, flags):
    with pytest.raises(SystemExit) as exc:
        cli.main(["transcribe", str(silent_wav), *flags])
    assert exc.value.code == cli.EXIT_USAGE
    assert "Traceback" not in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("gravelvox ")


def test_timestamps_and_json_are_mutually_exclusive(silent_wav):
    with pytest.raises(SystemExit) as exc:
        cli.main(["transcribe", str(silent_wav), "--json", "--timestamps"])
    assert exc.value.code == 2


def test_make_provider_passes_every_option(monkeypatch):
    args = cli.build_parser().parse_args(
        ["transcribe", "a.wav", "--model", "base", "--device", "auto", "--compute-type", "float32"]
        + ["--threads", "2", "--model-dir", str(Path("models"))]
    )
    provider = cli.make_provider(args)
    assert (provider.model, provider.device, provider.compute_type) == ("base", "auto", "float32")
    assert (provider.cpu_threads, provider.download_root) == (2, str(Path("models")))


def test_starting_the_cli_loads_no_native_libraries():
    # --help, --version and argument errors must work even if a native library cannot load.
    probe = (
        "import sys, gravelvox_engine.cli; "
        "print([m for m in ('av', 'numpy', 'faster_whisper', 'ctranslate2') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_a_blocked_native_library_exits_3_with_one_line(block_import, silent_wav, capsys):
    blocked = ImportError(
        "DLL load failed while importing _core: an Application Control policy blocked this file"
    )
    block_import("av", blocked)
    assert cli.main(["transcribe", str(silent_wav), "--model", "tiny"]) == cli.EXIT_DEPENDENCY
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("gravelvox: ")
    assert len(err.splitlines()) == 1
    assert "Application Control policy" in err
