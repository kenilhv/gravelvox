"""Command-line interface: `gravelvox transcribe AUDIO`."""

from __future__ import annotations

import argparse
import contextlib
import errno
import io
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from gravelvox_engine import __version__
from gravelvox_engine.errors import DependencyError, GravelvoxError, InvalidOptionError
from gravelvox_engine.stt import FasterWhisperProvider, STTProvider, TranscribeOptions
from gravelvox_engine.types import Transcript

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_DEPENDENCY = 3
EXIT_INTERRUPTED = 130

MAX_BEAM_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gravelvox",
        description="Gravelvox: CPU-first speech-to-text for noisy, real-world audio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser("transcribe", help="Transcribe an audio file on this machine.")
    transcribe.add_argument("audio", type=Path, help="Audio file (wav, mp3, m4a, flac, ...).")
    transcribe.add_argument(
        "--model",
        default="small",
        help="Model name (tiny, base, small, ...), Hugging Face repo id, or model directory "
        "(default: small).",
    )
    transcribe.add_argument("--language", help="Language code, e.g. en (default: detect).")
    transcribe.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silero voice-activity filter (default: on).",
    )
    transcribe.add_argument(
        "--beam-size",
        type=_int_in_range(1, MAX_BEAM_SIZE),
        default=5,
        help=f"Beam search width, 1 to {MAX_BEAM_SIZE}; 1 is greedy and fastest (default: 5).",
    )
    transcribe.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Add word-level timing to the --json output (slower).",
    )
    transcribe.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Where to run the model (default: cpu). cuda needs NVIDIA's cuBLAS and cuDNN.",
    )
    transcribe.add_argument(
        "--compute-type",
        default="int8",
        help="CTranslate2 weight type, e.g. int8, int8_float32, float32 (default: int8).",
    )
    transcribe.add_argument(
        "--threads",
        type=_int_in_range(0, os.cpu_count() or 64),
        default=0,
        help="CPU threads; 0 uses CTranslate2's default of 4, or OMP_NUM_THREADS if set.",
    )
    transcribe.add_argument("--model-dir", help="Download and cache models in this folder.")
    output = transcribe.add_mutually_exclusive_group()
    output.add_argument(
        "--timestamps", action="store_true", help="Prefix each line with segment times."
    )
    output.add_argument("--json", action="store_true", help="Print the full transcript as JSON.")
    transcribe.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the transcript to this file (UTF-8) instead of stdout.",
    )
    transcribe.add_argument(
        "-q", "--quiet", action="store_true", help="Don't print the timing summary."
    )
    return parser


def make_provider(args: argparse.Namespace) -> STTProvider:
    return FasterWhisperProvider(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
        download_root=args.model_dir,
    )


def format_text(transcript: Transcript, *, timestamps: bool = False) -> str:
    lines = []
    for segment in transcript.segments:
        text = segment.text.strip()
        if timestamps:
            text = f"[{_clock(segment.start)} -> {_clock(segment.end)}] {text}"
        lines.append(text)
    return "\n".join(lines)


def format_summary(transcript: Transcript) -> str:
    timing = transcript.timing
    parts = [transcript.provider, f"model={_quote(transcript.model)}"]
    if transcript.device:
        parts.append(f"device={transcript.device}")
    parts.append(f"audio={timing.audio_s:.1f}s")
    if timing.download_s:
        parts.append(f"download={timing.download_s:.1f}s")
    parts.append(f"load={timing.model_load_s:.2f}s")
    parts.append(f"transcribe={timing.transcribe_s:.2f}s")
    parts.append(f"rtf={timing.rtf:.3f}" if timing.rtf is not None else "rtf=n/a")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    _prefer_utf8_output()
    args = build_parser().parse_args(argv)

    audio = _resolve_audio(args.audio)
    if audio is None:
        problem = "not a file" if args.audio.exists() else "audio file not found"
        return _fail(f"{problem}: {args.audio}", EXIT_USAGE)
    if args.output is not None and not args.output.parent.is_dir():
        return _fail(f"output folder not found: {args.output.parent}", EXIT_USAGE)
    if args.word_timestamps and not args.json:
        _warn("--word-timestamps only changes the --json output")

    options = TranscribeOptions(
        language=args.language,
        vad=args.vad,
        word_timestamps=args.word_timestamps,
        beam_size=args.beam_size,
    )
    try:
        transcript = make_provider(args).transcribe(audio, options)
    except KeyboardInterrupt:
        return _fail("interrupted", EXIT_INTERRUPTED)
    except MemoryError:
        return _fail("ran out of memory; long recordings need several GB (see README)", EXIT_ERROR)
    except InvalidOptionError as exc:
        return _fail(exc, EXIT_USAGE)
    except DependencyError as exc:
        return _fail(exc, EXIT_DEPENDENCY)
    except GravelvoxError as exc:
        return _fail(exc, EXIT_ERROR)

    for warning in transcript.warnings:
        _warn(warning)
    try:
        _write_output(transcript, args)
    except OSError as exc:
        if args.output is not None:
            return _fail(f"could not write {args.output}: {exc}", EXIT_ERROR)
        if _is_closed_pipe(exc):
            _discard_stdout()
            return EXIT_ERROR
        raise
    return EXIT_OK


def _write_output(transcript: Transcript, args: argparse.Namespace) -> None:
    if args.json:
        text = json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False)
    else:
        text = format_text(transcript, timestamps=args.timestamps)
    if args.output is not None:
        args.output.write_text(f"{text}\n" if text else "", encoding="utf-8", newline="\n")
    elif text:
        print(text)
        sys.stdout.flush()  # surface a closed pipe here, not during interpreter shutdown
    if not args.quiet:
        print(format_summary(transcript), file=sys.stderr)


def _resolve_audio(path: Path) -> Path | None:
    """Return a path that refers to an existing file, or None."""
    if path.is_file():
        return path
    if os.name == "nt":
        # Without long-path support Windows refuses paths of 260+ characters; the
        # extended-length form works for Python and FFmpeg alike.
        full = os.path.abspath(path)
        if not full.startswith("\\\\?\\"):
            unc = full.startswith("\\\\")
            extended = Path(("\\\\?\\UNC\\" + full[2:]) if unc else ("\\\\?\\" + full))
            if extended.is_file():
                return extended
    return None


def _is_closed_pipe(exc: OSError) -> bool:
    # On Windows a write to a pipe whose reader has gone can raise EINVAL instead of EPIPE.
    return isinstance(exc, BrokenPipeError) or (
        sys.platform == "win32" and exc.errno == errno.EINVAL
    )


def _discard_stdout() -> None:
    # The reader went away (for example `| head`). Point stdout at devnull so the
    # interpreter's final flush doesn't raise a second error.
    with contextlib.suppress(OSError, ValueError):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())


def _fail(message: object, code: int) -> int:
    print(f"gravelvox: {message}", file=sys.stderr)
    return code


def _warn(message: str) -> None:
    print(f"gravelvox: warning: {message}", file=sys.stderr)


def _int_in_range(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}, got {value}"
            )
        return value

    return parse


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False) if any(c.isspace() for c in value) else value


def _clock(seconds: float) -> str:
    """Format seconds as MM:SS.ss, or H:MM:SS.ss from one hour on."""
    centis = max(0, round(seconds * 100))
    minutes, centis = divmod(centis, 6000)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes:02d}:{centis / 100:05.2f}"
    return f"{hours}:{clock}" if hours else clock


def _prefer_utf8_output() -> None:
    # Windows consoles default to a legacy code page, and transcripts are often not ASCII.
    if os.environ.get("PYTHONIOENCODING"):
        return  # the user chose an encoding explicitly
    for stream, errors in ((sys.stdout, "replace"), (sys.stderr, "backslashreplace")):
        if isinstance(stream, io.TextIOWrapper):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8", errors=errors)
