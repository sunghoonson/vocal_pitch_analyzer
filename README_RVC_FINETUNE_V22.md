# Vocal Pitch Analyzer v2.2 - RVC Fine-tune / Resume

## 추가되는 학습 방식

### 1. 새 모델 학습

처음부터 새 RVC 모델을 만듭니다.

같은 모델 이름에 기존 G/D 체크포인트가 있으면 실수로 덮어쓰지 않도록 중단합니다.

### 2. 기존 학습 이어하기 (같은 데이터)

데이터셋이 바뀌지 않았을 때 사용합니다.

```text
기존 G/D checkpoint
+ 기존 preprocess/F0/HuBERT
→ 이어서 RVC 학습
→ Feature Index 재생성
```

전처리, RMVPE, HuBERT를 다시 실행하지 않으므로 재개가 빠릅니다.

`목표 Epochs`는 추가량이 아니라 최종 epoch입니다.

```text
현재 200 epoch
100 epoch 추가
→ 목표 Epochs = 300
```

### 3. 데이터 추가 후 파인튜닝

같은 타깃 가수의 새 데이터를 추가할 때 사용합니다.

선택하는 데이터셋 폴더에는 반드시:

```text
기존 데이터 + 신규 데이터
```

가 모두 있어야 합니다.

파이프라인:

```text
기존 G/D checkpoint 보존
→ 안전 백업
→ 기존 derived 데이터만 제거
   0_gt_wavs
   1_16k_wavs
   2a_f0
   2b-f0nsf
   3_feature768
   filelist/config
→ 기존+신규 전체 데이터 전처리 재생성
→ RMVPE 재생성
→ HuBERT 재생성
→ filelist/config 재생성
→ 기존 G/D checkpoint에서 이어 학습
→ Feature Index 재생성
```

원본 데이터셋 음원은 삭제하지 않습니다.

## 왜 특징 데이터를 다시 만드나?

RVC 전처리는 입력 파일의 정렬 순서에 따라 숫자 기반 출력 이름을 만듭니다.
기존 결과를 그대로 둔 채 파일만 추가하면 이전 번호와 새 번호가 충돌하거나
오래된 F0/HuBERT 특징이 남을 수 있습니다.

v2.2의 파인튜닝 모드는 G/D 학습 상태만 보존하고 derived 데이터는 전체 재구축합니다.

## 안전 백업

파인튜닝 시작 전:

```text
rvc_finetune_backups\<experiment>\<timestamp>\
```

에 최신 G/D 체크포인트와 현재 inference 모델/Index/config/filelist을 가능한 범위에서 백업합니다.

## 데이터셋 기록

전체 특징 생성 후:

```text
tools\rvc\logs\<experiment>\vocal_pitch_dataset_manifest.json
```

을 저장합니다.

다음 파인튜닝부터는 이전 데이터가 현재 폴더에서 빠진 경우 중단합니다.

v2.2 이전에 만든 모델은 manifest가 없으므로 첫 파인튜닝에서는 기존
`0_gt_wavs` 파일의 숫자 prefix를 이용해 기존 원본 개수를 추정합니다.

## 이어학습에 필요한 파일

```text
tools\rvc\logs\<experiment>\G_*.pth
tools\rvc\logs\<experiment>\D_*.pth
```

가 필요합니다.

최종 변환용:

```text
rvc_models\<experiment>\<experiment>.pth
```

만 남아 있다면 optimizer를 포함한 기존 학습 상태로 이어갈 수 없습니다.

## 현재 male_voice_01 예시

기존:

```text
male_voice_01
30곡
200 epoch
```

고음이 많은 동일 가수 보컬 5곡을 기존 `_rvc_vocals` 폴더에 추가했다면:

```text
학습 방식: 데이터 추가 후 파인튜닝
데이터셋: 기존 30곡 + 신규 5곡이 함께 있는 폴더
모델 이름: male_voice_01
목표 Epochs: 300
```

으로 실행합니다.
