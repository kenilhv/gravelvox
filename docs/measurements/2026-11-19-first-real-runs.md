# First real runs (2026-11-19)

This note gives the method and the raw numbers behind the figures quoted in the README. They are observations from one laptop on synthetic speech. They show that the pipeline works and where its limits are; they are not a benchmark and say nothing about accuracy on field recordings.

## Setup

- **Machine:** laptop with an Intel Core i9-13900HX (24 cores, 32 threads; a hybrid CPU with performance and efficiency cores), 32 GB RAM, Windows 11.
- **Software:** Python 3.12.10, gravelvox-engine 0.0.1, faster-whisper 1.2.1, CTranslate2 4.9.0, PyAV 19.0.2, onnxruntime 1.32.0.
- **Settings:** `--model tiny`, CPU, `int8` weights (CTranslate2 reports `int8_float32`), beam size 5, voice-activity filter on, default thread count (4), seed 0. The model was already in the local cache.

## Test audio

All speech is synthetic, so no person was recorded: Windows text-to-speech (`System.Speech`) with the voices "Microsoft Zira Desktop" and "Microsoft David Desktop", written as 16 kHz mono 16-bit WAV.

| File | Length | Words | Content |
|---|---|---|---|
| `rep.wav` | 37.3 s | 109 | Text 1, Zira voice |
| `homeowner.wav` | 27.1 s | 81 | Text 2, David voice |
| `conversation.wav` | 65.1 s | 190 | Text 1, a 0.7 s pause, then text 2 |
| `long10.wav` | 655.4 s | 1,900 | 10 copies of `conversation.wav`, 0.5 s of silence between copies |
| `long55.wav` | 3,607 s | 10,450 | 55 copies, built the same way |

Text 1:

> Thanks for having me out today. I took a look at the roof from the driveway, and the shingles on the north side are starting to curl. That usually means moisture is getting underneath them. Before we talk about price, I want to make sure I understand what matters most to you. Is it mainly the leak over the garage, or are you also thinking about the gutters and the downspouts? Most homeowners in this neighborhood replace everything at once, because the crew is already here and the cleanup only happens once. I can walk you through both options, and you can decide what makes sense for your family.

Text 2:

> Honestly, the leak is the main thing. It started after the big storm last spring, and every time it rains we have to move the boxes in the garage. My wife wants the gutters done too, but I would like to see the numbers first. We also got a quote from another company last week, so I want to compare the warranty and how long the work will take. If you can finish before the holidays, that would help a lot.

The speech was generated in PowerShell like this (one voice shown):

```powershell
Add-Type -AssemblyName System.Speech
$bits = [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen
$channels = [System.Speech.AudioFormat.AudioChannel]::Mono
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, $bits, $channels)
$speech = New-Object System.Speech.Synthesis.PromptBuilder
$speech.StartVoice("Microsoft Zira Desktop"); $speech.AppendText($text1); $speech.EndVoice()
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("rep.wav", $format); $synth.Speak($speech); $synth.Dispose()
```

## Word error rate

WER is (substitutions + deletions + insertions) divided by the number of reference words, from a word-level Levenshtein alignment. Both texts are normalized first: Unicode NFKC, lower case, apostrophes dropped, other punctuation replaced by spaces.

| File | Flags | Reference words | Substitutions | Deletions | Insertions | WER |
|---|---|---|---|---|---|---|
| `conversation.wav` | default | 190 | 0 | 0 | 0 | 0.0% |
| `rep.wav` | default | 109 | 0 | 0 | 0 | 0.0% |
| `homeowner.wav` | default | 81 | 0 | 0 | 0 | 0.0% |
| `long10.wav` | default | 1,900 | 0 | 0 | 0 | 0.0% |
| `long10.wav` | `--word-timestamps` | 1,900 | 0 | 18 | 0 | 0.95% |

With `--word-timestamps`, 17 consecutive words went missing at about 108 to 113 seconds ("the big storm last spring, and every time it rains we have to move the boxes in"), and one more word near 649 seconds. Both places are where faster-whisper moves from one 30-second decoding window to the next. The segment before the gap ends in a single word, " the", timed from 107.84 s to 112.72 s. Earlier runs of the same command, before the decoder change described in ADR 2, lost 17 or 18 words in the same places.

## Speed

Five consecutive runs of `gravelvox transcribe conversation.wav --model tiny --json -q`, each in a new process:

| Run | Model load (s) | Transcribe (s) | RTF |
|---|---|---|---|
| 1 | 0.243 | 2.236 | 0.034 |
| 2 | 0.224 | 2.176 | 0.033 |
| 3 | 0.223 | 2.208 | 0.034 |
| 4 | 0.200 | 2.069 | 0.032 |
| 5 | 0.196 | 2.043 | 0.031 |

The median is 2.18 s for 65.09 s of audio, a real-time factor of 0.033. The text was identical in all five runs. `long10.wav` took 39.6 s for 655.4 s of audio, a real-time factor of 0.060.

## Memory

Peak working set and peak commit of the Python process, read with the Windows `GetProcessMemoryInfo` call in the same process after `gravelvox_engine.cli.main` returned. One run per file.

| File | Audio | Peak working set | Peak commit |
|---|---|---|---|
| `long10.wav` | 10.9 min | 836 MB | 2,297 MB |
| `long55.wav` | 60.1 min | 3,812 MB | 5,732 MB |

## Raw data

```json
{
  "conversation_runs": [
    {"model_load_s": 0.243, "transcribe_s": 2.236, "rtf": 0.034},
    {"model_load_s": 0.224, "transcribe_s": 2.176, "rtf": 0.033},
    {"model_load_s": 0.223, "transcribe_s": 2.208, "rtf": 0.034},
    {"model_load_s": 0.200, "transcribe_s": 2.069, "rtf": 0.032},
    {"model_load_s": 0.196, "transcribe_s": 2.043, "rtf": 0.031}
  ],
  "wer": {
    "conversation": {"wer": 0.0, "sub": 0, "del": 0, "ins": 0, "ref_words": 190},
    "rep": {"wer": 0.0, "sub": 0, "del": 0, "ins": 0, "ref_words": 109},
    "homeowner": {"wer": 0.0, "sub": 0, "del": 0, "ins": 0, "ref_words": 81},
    "long10": {"wer": 0.0, "sub": 0, "del": 0, "ins": 0, "ref_words": 1900, "audio_s": 655.4, "transcribe_s": 39.6},
    "long10_word_timestamps": {"wer": 0.0095, "sub": 0, "del": 18, "ins": 0, "ref_words": 1900, "audio_s": 655.4, "transcribe_s": 40.9}
  },
  "memory": {
    "long10": {"audio_s": 655.4, "peak_working_set_mb": 835.7, "peak_commit_mb": 2297.0},
    "long55": {"audio_s": 3606.95, "peak_working_set_mb": 3811.8, "peak_commit_mb": 5732.3}
  }
}
```
