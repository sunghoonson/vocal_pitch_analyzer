# Vocal Pitch Analyzer v1.9 - RVC + RMVPE

Base snapshot:

```text
vocal_pitch_prototype_v1_source_snapshot_20260906_001139.zip
```

Target:

```text
C:\dev\vocal_pitch_prototype_v1
```

Existing engines remain intact:

```text
빠른 DSP - FFmpeg RubberBand
AI 고음질 - Seed-VC SVC
```

New engine:

```text
AI 음색 변환 - RVC + RMVPE
```

## Goal

Seed-VC same-singer mode intentionally preserves the original singer's timbre.

RVC mode is for a different goal:

```text
female source vocal
  ↓
RMVPE F0 tracking
  ↓
pitch -4
  ↓
male / target RVC voice model
  ↓
male-sounding converted vocal
```

For full-song conversion:

```text
Original
  ↓
BS-RoFormer
  ├─ Vocals
  │    ↓
  │  RMVPE + RVC target model
  │
  └─ Instrumental
       ↓
     same semitone DSP shift
       ↓
converted vocal + shifted instrumental
       ↓
final remix
```

## Required voice model

RVC itself does NOT define "male".

You must select a male / target-voice RVC model:

```text
MODEL.pth
```

A matching feature index is strongly recommended:

```text
added_....index
```

If no index is selected, this integration safely forces effective Index Rate to 0
instead of failing.

Use models you have permission to use.

## GUI defaults

```text
F0          RMVPE
Index Rate  0.75
Protect     0.33
RMS Mix     1.00
Speaker ID  0
```

The existing Key control is reused:

```text
-4
```

means the RVC vocal F0 is shifted down four semitones, and in full-song mode the
instrumental is shifted down the same amount.

## RVC runtime

The external RVC runtime is isolated:

```text
.venv_rvc
tools\rvc
```

Pinned source commit:

```text
81eed5e8f68b6bed1789f682fe78cdd324495afc
```

The setup follows the current RVC RTX 50-series Windows path:

```text
Python 3.12
torch 2.7.1 + cu128
torchaudio 2.7.1 + cu128
RMVPE
HuBERT
```

This is intentionally separate from:

```text
.venv
.venv_separator
.venv_svc
```

## Install

After applying this patch:

```text
SETUP_RVC_RMVPE_GPU.bat
CHECK_RVC_RMVPE.bat
```

Then start the app.

## First test

Recommended:

```text
변환 방식     AI 음색 변환 - RVC + RMVPE
변환 소스     현재 선택한 원본 전체 음원
RVC 모델      male_voice.pth
Index         matching added_*.index
키            -4
Index Rate    0.75
Protect       0.33
RMS Mix       1.00
Speaker ID    0
출력          WAV
```

## Logs

RVC subprocess output:

```text
logs\rvc_rmvpe_last.log
```

BS-RoFormer output:

```text
logs\vocal_separator_last.log
```

## Git / project snapshots

The patch excludes:

```text
.venv_rvc
tools\rvc
rvc_models
*.index
```

from Git/project source snapshots.
