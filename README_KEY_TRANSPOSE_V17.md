# Vocal Pitch Analyzer v1.7 - Key Transpose / Audio Export

Target:

```text
C:\dev\vocal_pitch_prototype_v1
```

Requires the v1.6 tabbed UI patch.

## New tab

```text
키 변환 / 음원 추출
```

## Input source

You can choose:

```text
현재 선택한 원본 전체 음원
분리된 보컬 stem
```

For normal karaoke-style key changes, use the original full mix.

## Key change

```text
-12 ~ +12 semitone
```

Examples:

```text
-4  = four semitones lower
+3  = three semitones higher
```

Playback tempo stays at 1.0.

## Formant

Default:

```text
Formant 보존 ON
```

This tries to keep the original vocal timbre while shifting pitch.

Turn it OFF when you intentionally want the timbre itself to move more dramatically
with the pitch.

## Engine

This patch uses the FFmpeg `rubberband` audio filter.

The app checks whether the installed FFmpeg actually contains the filter.

Run:

```text
CHECK_KEY_SHIFT.bat
```

before the first test.

## Output

Supported audio export:

```text
WAV 24-bit
FLAC
MP3
M4A/AAC
```

## Transposed note subtitles

If a pitch analysis is currently loaded, the default option:

```text
현재 분석 결과가 있으면 변환된 음계 자막도 함께 생성
```

creates note subtitles next to the new audio.

Example:

```text
original highest note: G5
key shift: -4
subtitle highest note: D#5
```

The subtitle timing does not change because the key shift keeps tempo unchanged.

The existing grouped-subtitle settings are reused:

- ASS / SRT format
- Korean / English note labels
- max group time
- max notes
- silence split
- notes per line
- current-note highlight

## Apply

Run:

```text
APPLY_PATCH.bat
```

The installer backs up the current `main.py` and adds:

```text
audio_transposer.py
CHECK_KEY_SHIFT.bat
```

No change is made to:

- BS-RoFormer
- Vocal Activity Gate
- Pitch Engine v3
- CUDA / PyTorch
- existing vocal cache
