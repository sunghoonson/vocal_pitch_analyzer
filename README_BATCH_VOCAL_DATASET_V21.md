# Vocal Pitch Analyzer v2.1 - Batch Vocal Dataset Preparation

Requires v2.0 RVC model training integration.

New section in:

```text
RVC 모델 학습
```

## Workflow

```text
Downloaded songs / videos
    ↓
select many files at once
    ↓
BS-RoFormer vocal separation
    ↓
one-by-one sequential GPU processing
    ↓
<song>_vocals.wav
    ↓
RVC training dataset folder
```

The batch tool reuses the existing vocal separator settings:

```text
BS-RoFormer model
separator cache
CUDA autocast
```

Existing separator cache is reused, so songs already separated by the analyzer may
finish almost immediately.

## Why sequential instead of parallel?

Several BS-RoFormer instances running at once can consume large amounts of VRAM and
cause CUDA out-of-memory errors. The batch dataset tool intentionally processes one
song at a time.

## Output

When several selected files have unique names:

```text
Song A.mp4  -> Song A_vocals.wav
Song B.m4a  -> Song B_vocals.wav
```

If two selected source files have the same filename stem, an 8-character source hash is
added so one result does not overwrite the other.

By default, after selecting songs the suggested output is:

```text
<first selected song folder>\_rvc_vocals
```

You can choose another folder.

## Existing outputs

Default behavior does not overwrite an existing non-empty `_vocals.wav`. It counts the
file as an existing result. Enable:

```text
기존 _vocals.wav 덮어쓰기
```

when you intentionally want to regenerate it.

## RVC training handoff

This option defaults ON:

```text
완료 후 이 폴더를 RVC 학습 데이터셋으로 자동 지정
```

So after batch separation the output folder immediately becomes the dataset in the
RVC training section below.

## Logs

```text
logs\batch_vocal_extract_last.log
```

The normal per-song separator log still exists:

```text
logs\vocal_separator_last.log
```

## Stop behavior

The button:

```text
현재 곡 완료 후 중지
```

does not kill the separator in the middle of a song. It waits for the current source
to finish, then stops before starting the next source. Already created vocal WAV files
remain intact.
