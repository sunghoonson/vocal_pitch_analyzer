# Vocal Pitch Analyzer v2.0 - RVC Model Training

Requires the v1.9 RVC + RMVPE patch.

New main tab:

```text
RVC 모델 학습
```

The integrated workflow is single-speaker RVC v2 / 40k / RMVPE:

```text
Target male voice dataset
  ↓
RVC preprocessing / slicing
  ↓
RMVPE F0 extraction on CUDA
  ↓
HuBERT feature extraction on CUDA
  ↓
RVC v2 model training
  ↓
Feature Index training
  ↓
rvc_models\<name>\
  ├─ <name>.pth
  └─ ...added....index
```

After training completes, the generated `.pth` and `.index` are automatically selected
in the existing `키 변환 / 음원 추출` RVC engine.

## Dataset

Use one target voice only.

Recommended characteristics:

- one male/target speaker
- clean vocal or speech
- little background noise
- no instrumental accompaniment
- no second speaker
- roughly 10 minutes or more is a useful starting point

The selected folder is non-recursive: put the audio files directly in that folder.

Supported input file extensions are the formats already understood by the RVC audio
loader, including WAV, FLAC, MP3, M4A and common video/audio containers.

## Defaults

```text
Model name    male_voice_01
RVC version   v2
Sample rate   40k
F0            RMVPE
Epochs        200
Batch size    8
Save every    10 epochs
Workers       8
GPU ID        0
GPU cache     OFF
```

The trainer uses the official v2 40k pretrained generator/discriminator.

## One-time training asset setup

The v1.9 inference installation already provides RVC, HuBERT and RMVPE.

Training additionally needs:

```text
assets\pretrained_v2\f0G40k.pth
assets\pretrained_v2\f0D40k.pth
logs\mute\...
```

Run:

```text
SETUP_RVC_TRAINING_ASSETS.bat
CHECK_RVC_TRAINING.bat
```

## Logs

Integrated training log:

```text
logs\rvc_training_last.log
```

RVC also keeps experiment-specific logs/checkpoints under:

```text
tools\rvc\logs\<experiment>\
```

## Output

Permanent model copies are placed outside the external RVC repository:

```text
rvc_models\<experiment>\
```

This protects the trained model if `tools\rvc` is later reinstalled.

`rvc_models` remains ignored by Git/project source snapshots because trained voice
models can be large and may contain voice characteristics derived from private
datasets.
