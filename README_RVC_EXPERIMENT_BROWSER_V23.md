# Vocal Pitch Analyzer v2.3 - RVC Existing Experiment Browser

## New UI

The `RVC 모델 학습` tab now automatically scans:

```text
C:\dev\vocal_pitch_prototype_v1\tools\rvc\logs\
```

and shows existing experiments in a combo box.

Example:

```text
male_voice_01 | 250ep | data 37 | CKPT✓ MODEL✓ INDEX✓
female_voice_01 | 200ep | data 22 | CKPT✓ MODEL✓ INDEX✓
```

`mute` and unrelated empty log folders are excluded.

## Selecting an existing experiment

Selecting an experiment automatically fills:

```text
모델 이름
학습 데이터셋 경로 (when recorded in v2.2 manifest)
```

and displays:

```text
experiment name
G/D checkpoint status
estimated saved epoch
dataset path
dataset file count
final inference .pth status
Feature Index status
```

If an existing experiment with G/D checkpoints is selected while the training mode
is still `새 모델 학습`, the UI automatically switches to:

```text
기존 학습 이어하기
```

to reduce accidental overwrite mistakes.

You can then manually switch to:

```text
데이터 추가 후 파인튜닝
```

when new data has been added.

## Dataset path source

v2.2+ experiments store:

```text
tools\rvc\logs\<experiment>\vocal_pitch_dataset_manifest.json
```

The browser reads its:

```text
dataset_dir
files
```

fields.

Therefore a v2.2+ experiment can restore the original dataset folder automatically.

For older experiments without a manifest, the browser can still detect checkpoint/model
state and estimate the source count from preprocessed filenames, but cannot know the
original source path. In that case choose the dataset folder manually once.

## Epoch display

The browser does not load large G/D training checkpoints merely to paint the UI.

It estimates the most recently exported epoch from files like:

```text
tools\rvc\assets\weights\<experiment>_e250_....pth
```

If such an intermediate inference weight is unavailable, the list displays:

```text
?ep
```

The actual resume/fine-tune pipeline still reads the epoch from the real G checkpoint
before starting, so training safety does not depend on this UI estimate.

## Refresh

Use:

```text
새로고침
```

after manually changing experiment folders/files.

The list is also refreshed automatically after a training run finishes.
