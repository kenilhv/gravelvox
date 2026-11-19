# 1. A provider protocol and a CPU-first engine

- Status: accepted
- Date: 2026-11-18
- Amended by: [ADR 2](0002-own-decoder-clear-errors-and-real-dependency-tests.md). The provider is now named `local`, gravelvox decodes audio itself, and the confidence fields are per decoding window, not per segment.

## Context

Gravelvox has to run on a laptop with no API keys, and later burst to GPUs (a Runpod worker) or call hosted APIs. Each stage (voice activity, transcription, and later diarization and routing) must be swappable and measurable on its own, so that its cost and accuracy can be reported separately.

## Decision

- Every speech-to-text backend implements one small protocol: `STTProvider.transcribe(audio, options) -> Transcript`.
- `Transcript` is a typed, immutable result. It keeps per-segment confidence fields (`avg_logprob`, `no_speech_prob`, `compression_ratio`) and a `Timing` record that separates model loading from transcription, so the real-time factor is always computed the same way.
- The first backend, `local-cpu`, uses faster-whisper (CTranslate2) with int8 weights and its built-in Silero voice-activity filter.
- The CLI is a thin wrapper over a provider and holds no transcription logic of its own.

## Consequences

- New backends (a Runpod GPU worker, OpenAI-compatible APIs) only need to return a `Transcript`.
- The confidence fields are already available to a future cascade router.
- Reported RTF includes audio decoding and voice-activity filtering but not model loading; every benchmark must say so.
- Only batch file transcription exists for now; streaming needs its own interface later.
