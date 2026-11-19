"""Local provider: faster-whisper (CTranslate2), on the CPU with int8 weights by default."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from gravelvox_engine.audio import decode_audio
from gravelvox_engine.errors import (
    DependencyError,
    InvalidOptionError,
    MissingDependencyError,
    ModelLoadError,
    TranscriptionError,
)
from gravelvox_engine.stt.base import TranscribeOptions
from gravelvox_engine.types import Segment, Timing, Transcript, Word

_GPU_LIBRARY_ERROR = re.compile(r"cublas|cudnn|cuda", re.IGNORECASE)
# Warn when the text covers less than this share of the speech the VAD kept. Whisper drops
# a whole 30 s window when it is not confident about it, and says nothing.
_LOW_COVERAGE = 0.6
_MIN_SPEECH_FOR_COVERAGE_S = 10.0


class FasterWhisperProvider:
    """Runs a Whisper model on this machine through faster-whisper.

    Audio is decoded first, so a bad file fails before any model download. The model loads
    lazily on the first call and is reused. A model that is already cached is loaded without
    contacting the Hugging Face Hub; download time and load time are reported separately.
    Sampling is seeded, so the temperature fallback gives the same text on every run.
    """

    name = "local"

    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        download_root: str | None = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self.seed = seed
        self._whisper: Any = None
        self._load_device = device  # "auto" becomes "cpu" if the GPU turns out to be unusable

    def transcribe(self, audio: Path, options: TranscribeOptions | None = None) -> Transcript:
        options = options or TranscribeOptions()
        faster_whisper = _import_faster_whisper()
        language = _normalize_language(options.language, _known_language_codes(faster_whisper))

        started = time.perf_counter()
        decoded = decode_audio(audio)
        decode_s = time.perf_counter() - started
        warnings = _decode_warnings(decoded.skipped_packets, decoded.stopped_early)

        download_s, load_s = self._ensure_loaded(faster_whisper)
        if language is not None and language not in self._whisper.supported_languages:
            raise InvalidOptionError(
                f"model {self.model!r} does not support language {language!r}; "
                f"it supports: {', '.join(self._whisper.supported_languages)}"
            )

        started = time.perf_counter()
        try:
            raw_segments, info = self._run(decoded.samples, language, options)
        except MemoryError:
            raise
        except Exception as exc:
            if not (self._load_device == "auto" and _GPU_LIBRARY_ERROR.search(str(exc))):
                raise TranscriptionError(_transcription_failure(exc)) from exc
            warnings.append(
                "--device auto chose the GPU, but NVIDIA's cuBLAS or cuDNN libraries are "
                "missing, so the CPU was used instead"
            )
            self._whisper, self._load_device = None, "cpu"
            more_download_s, more_load_s = self._ensure_loaded(faster_whisper)
            download_s, load_s = download_s + more_download_s, load_s + more_load_s
            started = time.perf_counter()
            try:
                raw_segments, info = self._run(decoded.samples, language, options)
            except MemoryError:
                raise
            except Exception as retry_exc:
                raise TranscriptionError(_transcription_failure(retry_exc)) from retry_exc
        transcribe_s = decode_s + (time.perf_counter() - started)

        duration = float(info.duration)
        segments = tuple(_to_segment(raw, duration) for raw in raw_segments)
        speech_s = round(float(info.duration_after_vad), 3)
        if options.vad:
            warnings.extend(_coverage_warning(segments, speech_s))
        detected = language is None and bool(segments)
        return Transcript(
            text="".join(segment.text for segment in segments).strip(),
            language=info.language if (language is not None or segments) else None,
            language_probability=round(info.language_probability, 4) if detected else None,
            segments=segments,
            timing=Timing(
                model_load_s=load_s,
                transcribe_s=transcribe_s,
                audio_s=duration,
                download_s=download_s,
            ),
            provider=self.name,
            model=self.model,
            device=getattr(getattr(self._whisper, "model", None), "device", self._load_device),
            compute_type=getattr(
                getattr(self._whisper, "model", None), "compute_type", self.compute_type
            ),
            speech_s=speech_s,
            warnings=tuple(warnings),
        )

    def _run(self, samples: Any, language: str | None, options: TranscribeOptions) -> Any:
        _seed(self.seed)
        raw_segments, info = self._whisper.transcribe(
            samples,
            language=language,
            beam_size=options.beam_size,
            vad_filter=options.vad,
            word_timestamps=options.word_timestamps,
        )
        # faster-whisper decodes text lazily, so consuming the generator is most of the work.
        return list(raw_segments), info

    def _ensure_loaded(self, faster_whisper: Any) -> tuple[float, float]:
        """Load the model if needed; return (download seconds, load seconds)."""
        if self._whisper is not None:
            return 0.0, 0.0
        path, download_s = self._locate_model(faster_whisper)
        _seed(self.seed)
        started = time.perf_counter()
        try:
            self._whisper = faster_whisper.WhisperModel(
                path,
                device=self._load_device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        except MemoryError:
            raise
        except Exception as exc:  # CTranslate2 raises ValueError and RuntimeError here
            raise ModelLoadError(f"could not load model {self.model!r}: {exc}") from exc
        return download_s, time.perf_counter() - started

    def _locate_model(self, faster_whisper: Any) -> tuple[str, float]:
        """Return a local model directory and the seconds spent downloading it."""
        model = self.model
        if Path(model).is_dir():
            return model, 0.0
        if _looks_like_path(model):
            raise ModelLoadError(f"model directory not found: {model}")
        names = faster_whisper.available_models()
        if "/" not in model and model not in names:
            raise ModelLoadError(
                f"unknown model {model!r}; use one of {', '.join(names)}, a Hugging Face "
                "repo id such as Systran/faster-whisper-small, or a local model directory"
            )
        try:
            cached = faster_whisper.download_model(
                model, local_files_only=True, cache_dir=self.download_root
            )
            return cached, 0.0
        except Exception:  # not in the cache yet
            pass
        started = time.perf_counter()
        try:
            path = faster_whisper.download_model(model, cache_dir=self.download_root)
        except Exception as exc:  # huggingface_hub raises many types, all with clear messages
            raise ModelLoadError(f"could not download model {model!r}: {exc}") from exc
        return path, time.perf_counter() - started


def _import_faster_whisper() -> Any:
    try:
        import faster_whisper
    except ModuleNotFoundError as exc:
        if exc.name == "faster_whisper":
            raise MissingDependencyError(
                "faster-whisper is not installed; run `uv sync` (or `pip install faster-whisper`)"
            ) from exc
        raise DependencyError(f"faster-whisper is installed but cannot import: {exc}") from exc
    except (ImportError, OSError) as exc:  # for example a native library that failed to load
        raise DependencyError(f"faster-whisper is installed but failed to load: {exc}") from exc
    return faster_whisper


def _known_language_codes(faster_whisper: Any) -> tuple[str, ...] | None:
    # faster-whisper only exposes the list through a loaded model (supported_languages). Its
    # tokenizer module holds the same tuple, which lets a typo fail before any decoding.
    codes = getattr(getattr(faster_whisper, "tokenizer", None), "_LANGUAGE_CODES", None)
    return tuple(codes) if codes else None


def _normalize_language(language: str | None, known: tuple[str, ...] | None) -> str | None:
    if language is None:
        return None
    code = language.strip().lower()
    if not code or (known is not None and code not in known):
        hint = f"; use one of: {', '.join(known)}" if known else ""
        raise InvalidOptionError(f"unsupported language code {language!r}{hint}")
    return code


def _looks_like_path(model: str) -> bool:
    return (
        "\\" in model
        or model.startswith((".", "/", "~"))
        or re.match(r"^[A-Za-z]:", model) is not None
        or model.count("/") > 1
    )


def _seed(seed: int) -> None:
    try:
        import ctranslate2
    except (ImportError, OSError):  # reported by _import_faster_whisper
        return
    ctranslate2.set_random_seed(seed)


def _decode_warnings(skipped_packets: int, stopped_early: str | None) -> list[str]:
    warnings = []
    if skipped_packets:
        warnings.append(
            f"skipped {skipped_packets} damaged audio packet(s); the transcript may have small gaps"
        )
    if stopped_early:
        warnings.append(f"stopped reading the file early: {stopped_early}")
    return warnings


def _coverage_warning(segments: tuple[Segment, ...], speech_s: float) -> list[str]:
    covered = sum(segment.end - segment.start for segment in segments)
    if speech_s < _MIN_SPEECH_FOR_COVERAGE_S or covered >= _LOW_COVERAGE * speech_s:
        return []
    return [
        f"the text covers {covered:.0f} s of the {speech_s:.0f} s of speech the voice-activity "
        "filter kept; the model skipped audio it could not transcribe confidently (a larger "
        "model may help)"
    ]


def _transcription_failure(exc: Exception) -> str:
    message = f"transcription failed: {exc}"
    if _GPU_LIBRARY_ERROR.search(str(exc)):
        message += (
            " (the NVIDIA libraries CTranslate2 needs on the GPU, cuBLAS and cuDNN, are missing"
            " or incompatible; install them or use --device cpu)"
        )
    return message


def _to_segment(raw: Any, duration: float) -> Segment:
    def clamp(t: float) -> float:
        return round(min(max(t, 0.0), duration), 3)

    words = tuple(
        Word(
            start=clamp(w.start), end=clamp(w.end), word=w.word, probability=round(w.probability, 4)
        )
        for w in (raw.words or ())
    )
    return Segment(
        start=clamp(raw.start),
        end=clamp(raw.end),
        text=raw.text,
        avg_logprob=_round(raw.avg_logprob),
        no_speech_prob=_round(raw.no_speech_prob),
        compression_ratio=_round(raw.compression_ratio),
        temperature=_round(getattr(raw, "temperature", None)),
        words=words,
    )


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)
