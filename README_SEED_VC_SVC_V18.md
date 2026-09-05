# Vocal Pitch Analyzer v1.8 - Seed-VC SVC

Based on snapshot:

```text
vocal_pitch_prototype_v1_source_snapshot_20260905_232534.zip
```

Target:

```text
C:\dev\vocal_pitch_prototype_v1
```

## What this patch adds

The existing RubberBand key shifter remains available.

A new engine is added:

```text
AI 고음질 - Seed-VC SVC
```

### Full-song pipeline

```text
Original song
  ↓
BS-RoFormer
  ├─ vocals.wav
  └─ instrumental.wav
        ↓
vocals → Seed-VC SVC / F0 conditioned / semitone shift
instrumental → RubberBand same semitone shift
        ↓
RMS level matching
        ↓
remix
        ↓
WAV / FLAC / MP3 / M4A
```

The goal is to avoid applying one time-domain pitch shifter directly to the complete
mix, especially to the vocal.

## Seed-VC reference modes

Default:

```text
같은 가수 음색 유지 - 보컬 stem에서 자동 참조
```

The program scans the separated vocal stem and picks an approximately 12-second
vocal-active section automatically.

Optional:

```text
별도 참조 음성 파일 사용
```

This can be used later for voice/timbre conversion experiments.

## Recommended first test

```text
Engine          AI 고음질 - Seed-VC SVC
Source          현재 선택한 원본 전체 음원
Key             -4
Reference       같은 가수 음색 유지 - 자동 참조
Diffusion       30 steps
CFG             0.70
FP16            ON
Output          WAV
```

Compare this output directly with:

```text
빠른 DSP - FFmpeg RubberBand
```

using the same song and semitone value.

## Installation

Applying this patch does NOT automatically download Seed-VC or its models.

After `APPLY_PATCH.bat`, run:

```text
SETUP_SEED_VC_SVC_GPU.bat
CHECK_SEED_VC_SVC.bat
```

Seed-VC upstream recommends Python 3.10 on Windows, so the setup creates:

```text
.venv_svc
```

separately from:

```text
.venv
.venv_separator
```

The external repository is cloned to:

```text
tools\seed-vc
```

and pinned to:

```text
51383efd921027683c89e5348211d93ff12ac2a8
```

The setup uses the CUDA 13.0 PyTorch wheel index for the isolated SVC runtime.

## First inference

Seed-VC downloads its model/checkpoint dependencies automatically on first inference.
The 44.1 kHz F0-conditioned SVC path also uses RMVPE.

The first conversion can therefore take much longer than later conversions.

## Git / snapshot collector

These are excluded automatically:

```text
.venv_svc
tools\seed-vc
```

so the Seed-VC source/runtime/model files do not accidentally enter your Git commit or
project source snapshot.

## License note

The external Seed-VC repository is GPL-3.0 and is NOT bundled inside this patch.
The setup script clones it separately. If you later distribute a build containing
Seed-VC, review the Seed-VC GPL-3.0 obligations before distribution.
