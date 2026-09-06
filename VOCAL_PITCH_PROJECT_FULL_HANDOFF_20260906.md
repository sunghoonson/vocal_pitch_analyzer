# Vocal Pitch Analyzer / AI Vocal & Remix Project
## 전체 개발 이력 · 개발 환경 · 가상환경 · 기능 구조 · 현재 상태

> 최종 정리 시점: 2026-09-06  
> 프로젝트 루트: `C:\dev\vocal_pitch_prototype_v1`  
> 개발 환경: Windows + Visual Studio Code + Python 가상환경 분리  
> 주 GPU: NVIDIA GeForce RTX 5070 Ti 16GB  
> 현재 계열: **v3.1 AI Remix / ACE-Step 1.5** + 설치/업로드 관련 후속 핫픽스

---

# 1. 프로젝트 개요

이 프로젝트는 처음에는 **노래에서 보컬 음정을 분석하고 사람이 보기 쉬운 음계로 표시하는 PySide6 GUI 프로토타입**으로 시작했다.

초기 목표는 단순했다.

- 음원/영상 파일 선택
- FFmpeg로 분석 가능한 오디오 준비
- `librosa.pyin()` 기반 F0 추정
- F0 → MIDI note → 한국식 옥타브/계이름 변환
- 최저음/최고음 계산
- 시간대별 note 표시
- CSV 저장
- PySide6 GUI에서 비동기 분석

프로젝트가 진행되면서 사용 목적이 크게 확장되었다. 현재는 다음을 하나의 프로그램에서 다루는 형태가 되었다.

```text
Pitch Analysis
AI Vocal Separation
Lead Vocal Selection
Lead Melody Analysis
Pitch Shift / Key Change
Seed-VC
RVC + RMVPE
RVC Training / Fine-tuning
Harmony / Artifact Guard
Instrument Smart Shift
AI Remix / Arrangement
```

현재 프로젝트는 단순한 Pitch Analyzer라기보다 **노래 분석 + 보컬 분리 + 보컬 변환 + RVC 학습 + 자연스러운 키 변경 + 생성형 AI 재편곡을 통합한 로컬 음악 AI 워크스테이션**에 가깝다.

---

# 2. 개발 하드웨어 및 기본 환경

## 2.1 운영체제 / 장비

```text
OS      : Windows
GPU     : NVIDIA GeForce RTX 5070 Ti
VRAM    : 16GB
RAM     : 32GB
Editor  : Visual Studio Code
Shell   : PowerShell / CMD
```

RTX 50 시리즈 GPU를 사용하기 때문에 RVC 계열에서는 구형 CUDA 조합보다 **PyTorch 2.7.1 + CUDA 12.8** 계열을 기준으로 맞추는 방향을 사용했다.

실제 RVC 학습 중 `nvidia-smi`를 통해 CUDA 사용을 확인했고, 대표 관찰치는 다음과 같았다.

```text
VRAM       : 약 11.5GB / 16.3GB
GPU Util   : 약 28 ~ 76%
Power      : 약 106 ~ 111W
```

즉 RVC 학습은 CPU fallback이 아니라 실제 RTX 5070 Ti CUDA 추론/학습으로 동작했다.

---

# 3. 프로젝트 루트와 VS Code 작업 방식

프로젝트 기본 루트:

```text
C:\dev\vocal_pitch_prototype_v1
```

VS Code에서는 이 폴더를 프로젝트 root로 열고 작업했다.

메인 GUI 개발/실행 기준 Python은:

```text
C:\dev\vocal_pitch_prototype_v1\.venv\Scripts\python.exe
```

이다.

PowerShell 예:

```powershell
cd C:\dev\vocal_pitch_prototype_v1
.\.venv\Scripts\Activate.ps1
python main.py
```

또는 직접:

```powershell
C:\dev\vocal_pitch_prototype_v1\.venv\Scripts\python.exe main.py
```

프로젝트가 커지면서 모든 AI 엔진을 하나의 Python 환경에 넣으면 PyTorch, Transformers, Diffusers, NumPy, Gradio, audio-separator 등에서 충돌 가능성이 커졌기 때문에 **AI 엔진별 가상환경 분리**가 프로젝트 구조의 중요한 원칙이 되었다.

---

# 4. 현재 가상환경 구조

현재 대표 가상환경은 다음과 같다.

```text
.venv
.venv_separator
.venv_svc
.venv_rvc
.venv_remix
```

역할을 한 줄로 정리하면:

```text
.venv           → PySide6 GUI / Pitch / 전체 orchestration
.venv_separator → BS-RoFormer / Demucs / audio-separator
.venv_svc       → Seed-VC
.venv_rvc       → RVC / RMVPE / RVC Training
.venv_remix     → ACE-Step 1.5
```

이 환경들은 의도적으로 서로 분리했다.

---

# 5. `.venv` — 메인 GUI 환경

메인 프로그램 환경이다.

Python 계열:

```text
Python 3.12.x
```

실제 설치 로그에서 확인된 예:

```text
Python 3.12.7
```

대표 라이브러리/역할:

```text
PySide6     → GUI
numpy       → 수치 처리
librosa     → pYIN / audio analysis
soundfile   → WAV read/write
pyqtgraph   → Pitch graph
FFmpeg      → media/audio conversion
RubberBand  → high quality pitch shift
```

메인 환경은 대형 AI 모델을 직접 전부 로드하는 환경이 아니라, 각 서브엔진을 호출하고 결과를 GUI에 연결하는 **orchestrator** 역할을 한다.

---

# 6. `.venv_separator` — 보컬/악기 분리 환경

사용 엔진:

```text
audio-separator
BS-RoFormer
Demucs
```

대표 BS-RoFormer 모델:

```text
model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

기본 역할:

```text
원곡
↓
BS-RoFormer
├─ vocals.wav
└─ instrumental.wav
```

대표 캐시:

```text
cache\separator_models
cache\vocal_stems
```

대표 로그:

```text
logs\vocal_separator_last.log
```

중요한 한계:

> BS-RoFormer는 `보컬 vs 반주`는 잘 나누지만 `Lead Vocal vs Harmony vs Backing Vocal`을 완벽하게 분리하는 모델은 아니다.

이 점이 이후 RVC artifact 문제의 핵심 원인이 되었다.

---

# 7. `.venv_svc` — Seed-VC 환경

Seed-VC repository:

```text
https://github.com/Plachtaa/seed-vc.git
```

프로젝트에서 사용한 고정 commit:

```text
51383efd921027683c89e5348211d93ff12ac2a8
```

Python 계열:

```text
Python 3.10
```

Seed-VC의 역할:

```text
보컬 입력
↓
Singing Voice Conversion
↓
원래 timing / delivery를 최대한 유지하면서 음성 변환
```

프로젝트에서는:

```text
f0-condition     = True
auto-f0-adjust  = False
```

중심으로 사용했다.

실사용 결과에서는 특정 target singer timbre로 강하게 바뀌기보다 원 가수 특성이 많이 유지되는 경향이 있어, 이후 **RVC가 주된 target timbre 변환 엔진**으로 확장되었다.

---

# 8. `.venv_rvc` — RVC / RMVPE 환경

RVC repository:

```text
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
```

프로젝트에서 사용한 고정 commit:

```text
81eed5e8f68b6bed1789f682fe78cdd324495afc
```

대표 환경:

```text
Python       : 3.12
PyTorch      : 2.7.1
Torchaudio   : 2.7.1
CUDA         : 12.8
Pitch method : RMVPE
```

RTX 5070 Ti 대응을 위해 CUDA 12.8 계열을 사용했다.

대표 RVC 추론 옵션:

```text
Index Rate   : 0.75
Protect      : 0.33
RMS Mix Rate : 1.0
Speaker ID   : 0
F0 Method    : RMVPE
```

RVC CLI는 직접 파일 script를 실행할 때 package shadow/circular import 문제가 있었기 때문에:

```text
python -m infer.cli
```

형태로 수정했다.

Index 파일이 없는 경우에는 wrapper가:

```text
effective index rate = 0
```

으로 처리한다.

---

# 9. `.venv_remix` — ACE-Step 1.5 AI Remix 환경

추가 시점:

```text
v3.1
```

ACE-Step repository:

```text
https://github.com/ace-step/ACE-Step-1.5.git
```

로컬 위치:

```text
C:\dev\vocal_pitch_prototype_v1\tools\ACE-Step-1.5
```

전용 Python environment:

```text
C:\dev\vocal_pitch_prototype_v1\.venv_remix
```

처음에는:

```text
pip install -e
```

방식으로 설치하려 했으나 다음 오류가 발생했다.

```text
Could not find a version that satisfies the requirement nano-vllm
No matching distribution found for nano-vllm
```

원인은 ACE-Step의 `nano-vllm` dependency가 PyPI package가 아니라 repository 내부 local source이고, 해당 mapping이 `uv` 기반으로 구성되어 있기 때문이었다.

따라서 최종 설치 방식은:

```text
uv sync
```

로 변경했다.

중요:

```text
.venv
.venv_separator
.venv_svc
.venv_rvc
```

는 ACE-Step 설치 때문에 수정하지 않는다.

---

# 10. 초기 Pitch Analyzer

프로젝트의 가장 초기 핵심은 `librosa.pyin()`을 이용한 F0 분석이었다.

대표 설정:

```text
pYIN confidence threshold : 약 0.25
Bridge                    : 80ms
Smooth                    : 5
Hysteresis                : 20 cents
Minimum merge             : 35ms
```

분석 결과는 MIDI note로 변환한다.

한국식 표기 예:

```text
A4 → 2옥타브 라
C5 → 3옥타브 도
```

초기에는 단순 pYIN 값만 사용했지만 다음과 같은 문제가 발생했다.

```text
무음 구간 F0
숨소리
반주 누출
짧은 이상치
화음 F0
최저/최고음 오검출
```

이 문제를 줄이기 위해 Activity Gate가 추가되었다.

---

# 11. Activity Gate Engine

대표 Activity Gate 설정:

```text
Adaptive RMS margin : 32 dB
Floor               : -55 dBFS
Hysteresis          : 4 dB
Island              : 80 ms
Gap                 : 100 ms
```

최저음/최고음 계산 대표 조건:

```text
duration   >= 100 ms
confidence >= 0.35
```

즉 pYIN이 한두 frame 잡은 극단값을 음역으로 바로 인정하지 않는다.

---

# 12. CSV / SRT / ASS 음계 자막

추가 모듈:

```text
subtitle_generator.py
```

기능:

```text
Pitch CSV
SRT
ASS
```

대표 ASS 동작:

- 여러 note를 한 줄로 묶음
- 현재 note 강조
- 약 6 note / line
- 대표 font size 약 44

분석 결과를 단순 수치가 아니라 영상 위 음계 자막으로 사용할 수 있게 되었다.

---

# 13. 미디어 입력 / FFmpeg

`media_input.py`를 통해 영상도 직접 분석할 수 있게 했다.

대표 입력:

```text
WAV
MP3
M4A
MP4
AAC
WEBM
MKV
MOV
WMA
OPUS
```

FFmpeg는 프로젝트 전반에서 다음 역할을 한다.

```text
video → audio
sample rate conversion
WAV conversion
stem mixing
ACE-Step source preparation
```

---

# 14. RubberBand Pitch Shift

`audio_transposer.py`를 통해 고품질 key shift를 추가했다.

대표 범위:

```text
-12 ~ +12 semitone
```

지원 개념:

```text
tempo 유지
formant option
quality mode
```

초기에는 전체 음원 자체를 한 번에 shift하는 DSP 방식이었다.

이후 보컬/반주를 분리해 서로 다른 처리를 하는 방향으로 발전했다.

---

# 15. BS-RoFormer 기반 전체곡 처리

기본 구조:

```text
원곡
↓
BS-RoFormer
├─ Vocals
└─ Instrumental
```

이 구조가 이후 Seed-VC, RVC, Instrument Smart Shift의 기반이 되었다.

---

# 16. Seed-VC 전체곡 변환

전체곡 Seed-VC 흐름:

```text
원곡
↓
BS-RoFormer
├─ Vocals
│   ↓
│ Seed-VC
│
└─ Instrumental
    ↓
 RubberBand
    ↓
같은 semitone

↓
Final Mix
```

보컬과 반주가 같은 key로 이동하도록 구성했다.

---

# 17. RVC + RMVPE 통합

Seed-VC보다 target singer timbre 변환이 더 중요한 사용 목적이 생겨 RVC를 추가했다.

기본 구조:

```text
Vocals
↓
RMVPE F0
↓
RVC
↓
Target timbre
```

Instrumental은 같은 semitone으로 RubberBand 처리 후 다시 mix한다.

---

# 18. RVC 모델 학습 기능

프로젝트 내부에서 RVC 모델을 직접 학습할 수 있게 별도 탭을 추가했다.

탭:

```text
RVC 모델 학습
```

기본 구성:

```text
Single Speaker
RVC v2
40k
RMVPE
```

대표 초기 기본값:

```text
Experiment : male_voice_01
Epoch      : 200
Batch      : 8
Save every : 10
Workers    : 8
GPU        : 0
Cache GPU  : OFF
```

학습 단계:

```text
1. preprocess
2. RMVPE F0 extraction
3. HuBERT feature extraction
4. filelist/config
5. RVC training
6. feature index generation
```

대표 실행 구조:

```text
python -m train.preprocess
python -m train.dataset.extract_f0
python -m train.dataset.extract_hubert_feature
python -m train.train
python -m train.train_index
```

pretrained:

```text
assets\pretrained_v2\f0G40k.pth
assets\pretrained_v2\f0D40k.pth
```

---

# 19. 실제 `male_voice_01` 학습 사례

실제 테스트 가수:

```text
한동근
```

experiment:

```text
male_voice_01
```

초기 dataset은 BS-RoFormer로 추출한 약 30개 vocal stem으로 시작했다.

초기 200 epoch 학습을 완료했고 CUDA 사용을 확인했다.

이후 dataset을 늘리고 fine-tune하면서 250 epoch까지 진행했다.

---

# 20. Batch Vocal Dataset Extraction

RVC 학습 dataset을 만들기 위해 여러 곡을 한 번에 처리하는 기능을 추가했다.

구조:

```text
여러 원곡
↓
BS-RoFormer 순차 처리
↓
_rvc_vocals
```

대표 dataset path:

```text
F:\HDD\nvideo\downloads\한동근\_rvc_vocals
```

GPU OOM을 피하기 위해 여러 곡을 동시에 돌리는 것보다 **한 곡씩 순차적으로 처리**하는 구조를 사용했다.

---

# 21. RVC Training Mode 확장

학습 mode는 다음 세 가지로 발전했다.

```text
1. 새 모델 학습
2. 기존 학습 이어하기
3. 데이터 추가 후 파인튜닝
```

## 21.1 새 모델 학습

새 experiment를 만든다.

## 21.2 기존 학습 이어하기

기존:

```text
G_*.pth
D_*.pth
```

training checkpoint를 읽고 기존 preprocess/F0/HuBERT feature를 그대로 재사용한다.

## 21.3 데이터 추가 후 파인튜닝

기존 G/D checkpoint는 보존하되 dataset feature를 다시 만든다.

중요 규칙:

> 현재 선택하는 dataset folder에는 기존 데이터와 신규 데이터가 모두 들어 있어야 한다.

예:

```text
기존 37곡
+ 신규 4곡
= 총 41곡
```

---

# 22. Experiment Browser

기존 RVC experiment를 쉽게 선택할 수 있도록 browser를 추가했다.

표시 예:

```text
male_voice_01 | 250ep | data 37 | CKPT✓ MODEL✓ INDEX✓
```

확인 항목:

```text
experiment name
dataset manifest
dataset count
G checkpoint
D checkpoint
estimated epoch
final model
feature index
mtime
```

---

# 23. Fine-tune Backup

Fine-tune 전에 상태를 보호한다.

대표 backup path:

```text
rvc_finetune_backups\<experiment>\<timestamp>\
```

dataset 추가 fine-tune 시 다시 만드는 대표 항목:

```text
0_gt_wavs
1_16k_wavs
2a_f0
2b-f0nsf
3_feature768
filelist.txt
config.json
```

G/D checkpoint는 유지한 채 다음 epoch부터 이어간다.

---

# 24. RVC Index stale 문제

Fine-tune 후 final `.pth`는 최신인데 `.index`가 오래된 문제가 있었다.

원인:

- old index가 존재하면 rebuild skip 가능
- copy가 old mtime을 유지할 수 있음

해결:

```text
데이터 추가 fine-tune
→ old index backup/remove
→ feature rebuild
→ index rebuild
```

같은 data에서 단순 epoch resume하는 경우는 불필요하게 index를 매번 다시 만들지 않는다.

---

# 25. Chorus / Harmony에서 RVC artifact 발생

실제 변환에서 다음 구간이 문제였다.

```text
chorus
harmony
double vocal
backing vocal
```

대표 artifact:

```text
warble
robotic
squeak
pitch jump
mush
```

근본 원인은 다음 구조였다.

```text
BS-RoFormer Vocals
=
Lead
+ Harmony
+ Backing
+ Double
```

반면 RVC/RMVPE는 기본적으로 한 순간에 하나의 dominant F0가 있는 monophonic 입력에서 가장 안정적이다.

여러 음높이가 동시에 존재하면 RMVPE가 서로 다른 F0 사이를 이동하면서:

```text
pitch jump
octave jump
warble
RVC artifact
```

가 생길 수 있다.

---

# 26. Harmony Guard

첫 대응은 화음 위험 구간을 분석해 RVC 대신 pitch-only result를 섞는 방식이었다.

```text
Vocals
├─ RVC
└─ RubberBand pitch-only
     ↓
Harmony risk
     ↓
Blend
```

하지만 실제 곡에서 너무 많은 구간이 fallback되어 target timbre가 거의 사라지는 문제가 발생했다.

실제 예:

```text
곡 길이   : 약 211s
fallback  : 약 146s
mean risk : 약 0.735
```

즉 input Harmony Guard만으로는 지나치게 공격적이었다.

---

# 27. Artifact Guard / Diagnostics

v2.5 계열에서 output 자체를 검증하는 Artifact Guard를 추가했다.

대표 metric:

```text
F0 mismatch
voiced loss
confidence drop
harmonic coverage
chroma similarity
differential jumps
```

관련 로그:

```text
logs\rvc_harmony_guard_last.json
logs\rvc_adaptive_guard_last.log
logs\rvc_adaptive_guard_last.json
```

---

# 28. Manual Timeline Bypass

v2.6에서는 사용자가 artifact 구간을 직접 지정할 수 있게 했다.

입력 예:

```text
12.5 - 15.0
01:23.500 - 01:26.200
1:02:03.0 ~ 1:02:05.5
```

해당 구간은 RVC 결과 대신 pitch-only result를 강하게 사용한다.

Auto Guard를 끈 상태에서도 manual bypass는 사용할 수 있게 했다.

---

# 29. Lead Vocal Selector

문제의 근본 원인이 Lead와 Harmony가 섞인 입력이라는 점이 명확해져 v2.7에서 Lead Vocal Selector를 추가했다.

모듈:

```text
rvc_lead_selector.py
```

입력:

```text
BS-RoFormer vocals stem
```

출력:

```text
lead_candidate.wav
nonlead_residual.wav
```

분석 요소:

```text
pYIN voiced probability
dominant-F0 harmonic coverage
stereo center / mid-side
spectral tonality
spectral flatness
RMS level
```

대표 Lead confidence weight:

```text
pYIN              : 0.36
harmonic coverage : 0.30
center            : 0.14
tonality          : 0.12
level             : 0.08
```

강도:

```text
gentle
balanced
strict
```

기본:

```text
balanced
```

최종 구조:

```text
BS-RoFormer Vocals
↓
Lead Selector
├─ Lead
│   ↓
│ RVC
│
└─ Non-lead / Harmony / Backing
    ↓
Pitch Shift only

↓
Final Vocal Mix
```

즉 **Lead만 RVC**, Harmony/Backing은 RVC를 거치지 않는다.

---

# 30. Lead Selector 실제 테스트 결과

실제 곡에서 확인된 예:

```text
곡 길이              : 약 234.4s
활성 보컬            : 약 181.6s
Lead 선택            : 약 158.5s
Lead 선택률          : 약 67.6%
Mean Lead Confidence : 약 0.596
Lead Energy Ratio    : 약 48.9%
```

즉 synthetic test만이 아니라 실제 separated vocal에서도 Lead candidate가 만들어지는 것을 확인했다.

---

# 31. Training Lead Dataset Cleaner

추론만 Lead Selector를 써도 training data가 Harmony에 오염되어 있으면 모델 자체가 이미 잘못 학습될 수 있다는 문제가 있었다.

그래서 v2.8에서 학습 전 dataset도 정제한다.

```text
_rvc_vocals
↓
Training Lead Dataset Cleaner
↓
_rvc_lead_vocals
↓
preprocess
↓
RMVPE
↓
HuBERT
↓
RVC Training
```

원본:

```text
_rvc_vocals
```

은 보존한다.

정제 output:

```text
_rvc_lead_vocals
```

검토 대상:

```text
_rvc_lead_vocals_review
```

manifest:

```text
.rvc_lead_clean_manifest.json
```

로그:

```text
logs\rvc_training_lead_cleaner_last.log
logs\rvc_training_lead_cleaner_last.json
```

---

# 32. Training Cleaner Quality Gate

Lead Selector 결과가 너무 빈약한 파일을 학습에 넣지 않도록 다음을 검사한다.

```text
Lead selected seconds
Lead selected ratio
Lead energy ratio
Mean Lead confidence
```

불확실한 파일은 review folder로 보내 사람이 직접 확인할 수 있게 했다.

---

# 33. Fine-tune + Lead Cleaner lineage 문제

Lead Cleaner 적용 후 파일명이 바뀐다.

```text
기존:
song_vocals.wav

정제:
song_vocals_lead.wav
```

기존 fine-tune validation은 파일명을 직접 비교했기 때문에 같은 원본의 정제본을 “기존 파일이 빠졌다”고 오판했다.

해결:

```text
.rvc_lead_clean_manifest.json
```

의:

```text
source_name
source_path
```

를 사용해 **정제 파일명 자체가 아니라 원본 데이터 계보(lineage)**로 비교하도록 수정했다.

예:

```text
이전 raw dataset         : 37곡
현재 Lead-clean lineage : 41곡
```

이면:

```text
기존 37곡 모두 존재
신규 4곡 추가
```

로 판단한다.

---

# 34. Lead Melody Analysis

Lead Selector는 RVC뿐 아니라 Pitch Analysis에도 적용하는 것이 맞다고 판단했다.

이유:

> 원하는 것은 화음 전체 음계가 아니라 메인 보컬의 멜로디 음계이기 때문.

v2.9 구조:

```text
BS-RoFormer Vocals
├─ pYIN F0
└─ Lead Frame Analysis
      ↓
Lead confidence mask

pYIN confidence
+ RMS Activity Gate
+ Lead Melody Gate
↓
Final melody notes
```

중요한 설계:

> `lead_candidate.wav` 자체를 다시 pYIN 분석하지 않는다.

원래 separated vocal에서 F0를 검출하고, Lead Selector는 **어떤 frame을 note로 인정할지만 결정**한다.

대표 threshold:

```text
gentle   : 0.24
balanced : 0.30
strict   : 0.38
```

결과적으로 다음이 Lead melody 기준으로 통일된다.

```text
processed pitch graph
note list
minimum note
maximum note
CSV
SRT
ASS
```

---

# 35. Instrument Smart Shift

기존에는 Instrumental 전체에 같은 pitch shift를 걸었다.

문제:

```text
Kick
Snare
Hi-hat
Cymbal
```

같은 타악기까지 -4 semitone 내려가면 둔탁해질 수 있다.

그래서 v3.0에서 **Instrument Smart Shift**를 추가했다.

구조:

```text
원곡
↓
BS-RoFormer
├─ Vocals
└─ Instrumental
      ↓
   Demucs 4-stem
      ├─ Drums
      ├─ Bass
      ├─ Other
      └─ Residual
```

기본 처리:

```text
Drums    → original pitch
Bass     → target semitone
Other    → target semitone
Residual → target semitone
```

GUI:

```text
☑ AI 4-stem 분리로 반주 키 변경
☑ Drums / Percussion은 원래 Pitch 유지
```

기본 Demucs model:

```text
htdemucs_ft.yaml
```

Demucs는 원곡 전체가 아니라 **BS-RoFormer Instrumental에만 적용**한다.

즉:

```text
Vocal boundary  = BS-RoFormer
Instrument type = Demucs
```

로 역할을 분리한다.

캐시:

```text
cache\instrument_stems
```

로그:

```text
logs\instrument_smart_shift_last.log
logs\instrument_smart_shift_last.json
```

Demucs가 실패하면 전체 변환을 죽이지 않고 기존:

```text
Instrumental 전체 RubberBand
```

방식으로 fallback한다.

---

# 36. AI Remix / Arrangement

v3.1에서는 단순 key shift를 넘어 **생성형 재편곡** 기능을 추가했다.

엔진:

```text
ACE-Step 1.5
```

새 탭:

```text
AI 리믹스 / 재편곡
```

기존 key shift와 차이:

```text
Key Shift
→ 원래 편곡/리듬/연주 유지
→ 음정만 이동

AI Remix
→ 리듬 재생성
→ 악기 재구성
→ 반주 재편곡
→ 믹스 변경
→ 보컬 표현도 생성형으로 변경 가능
```

---

# 37. AI Remix 스타일 프리셋

현재 GUI 대표 preset:

```text
Blues
7080 한국 가요
Modern Ballad
R&B
Jazz
City Pop
Rock
Acoustic
Orchestral
사용자 지정
```

이것들은 각각 별도 AI 모델이 아니다.

하나의 ACE-Step 범용 모델에 서로 다른 Prompt를 보내는 **Prompt Preset**이다.

예: `7080 한국 가요`

```text
1970s to 1980s Korean popular music arrangement,
warm analog tape sound,
vintage electric piano,
acoustic guitar,
lush strings,
melodic bass,
restrained live drums,
nostalgic emotional mood
```

따라서 현재 AI Remix에서는 별도 style model을 새로 학습하는 것보다:

```text
Prompt
Lyrics
Cover Strength
Seed
```

조합을 잘 설정하는 것이 중요하다.

---

# 38. AI Remix 입력 파라미터

## 38.1 Style

장르 방향.

## 38.2 Additional Prompt

세부 편곡 지시.

예:

```text
slower tempo,
stronger vintage Korean ballad feeling,
prominent acoustic guitar,
warm string section,
restrained live drums,
minimal modern synthesizers,
warm analog tape saturation
```

## 38.3 Lyrics

무엇을 부를지 결정한다.

가사를 비워두면 실제 테스트에서 instrumental/멜로디 중심 결과가 나올 수 있다.

원곡 가사를 유지하려면 원 가사를 직접 넣는 것이 좋다.

## 38.4 Vocal Language

원곡 가사 언어와 맞추는 것이 기본이다.

예:

```text
Japanese song → 日本語
Korean song   → 한국어
```

## 38.5 Cover Strength

기본:

```text
0.45
```

개념:

```text
낮음 → 장르 변화가 더 자유로움
중간 → 원곡/새 편곡 균형
높음 → 원곡 구조/성격을 더 강하게 유지
```

## 38.6 Seed

같은 조건에서도 서로 다른 variation을 만들 수 있다.

```text
Random Seed ON  → 여러 결과 비교
Random Seed OFF → 동일 seed 재현
```

---

# 39. AI Remix와 RVC 역할 분리

ACE-Step:

```text
어떻게 편곡할지
어떤 악기를 사용할지
어떤 groove로 연주할지
```

RVC:

```text
최종 보컬 timbre를 어떤 target voice로 바꿀지
```

권장 flow:

```text
원곡
↓
ACE-Step AI Remix
↓
여러 Seed 중 마음에 드는 편곡 선택
↓
생성 결과를 현재 음원으로 사용
↓
RVC
semitone = 0
↓
Target Singer Timbre
```

`semitone = 0`이면 새 편곡 key는 유지하면서 보컬 timbre만 바꿀 수 있다.

---

# 40. ACE-Step 로컬 API 구조

중요:

> 현재 AI Remix는 외부 회사 서버로 음원을 업로드하는 방식이 아니다.

구조:

```text
PySide6 GUI
    │
    │ HTTP localhost
    ↓
127.0.0.1:8001
    │
    ↓
ACE-Step Local API Server
    │
    ↓
.venv_remix
    │
    ↓
ACE-Step model
    │
    ↓
RTX 5070 Ti
```

API를 쓰는 이유:

- 메인 GUI dependency와 ACE-Step dependency 분리
- 모델을 한 번 GPU에 로드한 뒤 계속 재사용
- 여러 Seed 생성 시 매번 model startup 방지
- PySide6 main process 안정성 유지

즉 `API`는 외부 SaaS가 아니라 **내 프로그램과 내 PC의 AI 엔진 사이 통신 규격**이다.

---

# 41. ACE-Step 설치 오류와 해결

처음 설치:

```text
pip install -e C:\dev\vocal_pitch_prototype_v1\tools\ACE-Step-1.5
```

오류:

```text
nano-vllm
```

해결:

```text
uv sync
```

전용 `.venv_remix`만 다시 만든다.

---

# 42. ACE-Step absolute path API 오류

첫 Remix 실제 테스트에서는 API server가 정상으로 올라왔지만:

```text
HTTP 400
absolute audio file paths are not allowed
```

오류가 발생했다.

기존 구현:

```text
src_audio_path = F:\...\song.mp4
```

처럼 절대경로를 JSON으로 보냈다.

ACE-Step API는 보안상 임의 Windows absolute path 접근을 막는다.

수정 후:

```text
Local source
↓
multipart/form-data
src_audio=<actual file>
↓
localhost ACE-Step
```

형태로 변경했다.

---

# 43. MP4 / M4A AI Remix Source 처리

AI Remix source가 다음 container라면:

```text
MP4
M4A
AAC
WEBM
MKV
MOV
WMA
M4V
```

프로젝트 FFmpeg로 먼저:

```text
44.1 kHz
Stereo
PCM WAV
```

로 추출한다.

그 후 해당 WAV를 localhost API에 multipart upload한다.

일반 audio:

```text
WAV
MP3
FLAC
OGG
OPUS
```

는 직접 upload한다.

---

# 44. 현재 GUI 기능 단위

현재 프로그램의 주요 기능은 대략 다음과 같이 나뉜다.

## 44.1 분석

```text
Pitch Analysis
AI Vocal Separation
Lead Melody Analysis
Activity Gate
Min / Max note
CSV
SRT
ASS
```

## 44.2 키 변환 / 음원 추출

```text
RubberBand DSP
Seed-VC
RVC + RMVPE
Lead Vocal Selector
Harmony / Artifact Guard
Manual Bypass
Instrument Smart Shift
```

## 44.3 AI 리믹스 / 재편곡

```text
ACE-Step 1.5 Cover
Style presets
Custom Prompt
Lyrics
Vocal language
Cover Strength
Random / Fixed Seed
Result → Current Audio
```

## 44.4 RVC 모델 학습

```text
New Model
Resume
Fine-tune Add Data
Training Lead Dataset Cleaner
Experiment Browser
Index Generation
```

---

# 45. 대표 로그 / 캐시

## Separator

```text
cache\separator_models
cache\vocal_stems
logs\vocal_separator_last.log
```

## Lead Selector

```text
cache\rvc_lead_selector\last_lead_candidate.wav
cache\rvc_lead_selector\last_nonlead_residual.wav

logs\rvc_lead_selector_last.log
logs\rvc_lead_selector_last.json
```

## Artifact Guard

```text
logs\rvc_harmony_guard_last.json
logs\rvc_adaptive_guard_last.log
logs\rvc_adaptive_guard_last.json
```

## Training Lead Cleaner

```text
logs\rvc_training_lead_cleaner_last.log
logs\rvc_training_lead_cleaner_last.json
```

## Instrument Smart Shift

```text
cache\instrument_stems
logs\instrument_smart_shift_last.log
logs\instrument_smart_shift_last.json
```

## ACE-Step

```text
logs\ace_step_server.log
logs\ai_remix_last.log
logs\ai_remix_last.json
```

---

# 46. RVC 모델 / 체크포인트 구조

대표 final model path:

```text
rvc_models\<experiment>\
```

예:

```text
rvc_models\male_voice_01\male_voice_01.pth
```

Index:

```text
added_*.index
```

Training checkpoint:

```text
tools\rvc\logs\<experiment>\G_*.pth
tools\rvc\logs\<experiment>\D_*.pth
```

중요:

> inference용 final `.pth`만으로는 training resume를 할 수 없다.

Resume/Fine-tune에는 G/D training checkpoint가 필요하다.

---

# 47. 현재 권장 RVC Dataset 전략

초기:

```text
_rvc_vocals
```

현재 권장:

```text
_rvc_vocals
↓
Training Lead Dataset Cleaner
↓
_rvc_lead_vocals
↓
RVC Training
```

이유:

- chorus harmony contamination 감소
- backing vocal contamination 감소
- RMVPE one-F0 assumption에 더 적합
- Lead timbre 중심 학습 가능성 증가

단, 현재 Lead Selector는 heuristic이므로 완벽한 neural semantic separator는 아니다.

---

# 48. 현재 프로젝트의 중요한 한계

## 48.1 Lead Selector

현재는:

```text
F0
harmonic coverage
stereo center
tonality
RMS
```

등을 이용한 heuristic 기반이다.

완전한 neural Lead/Backing separator는 아니다.

## 48.2 RVC

동시 다성 vocal 입력에 여전히 취약하다.

완화책:

```text
Lead Selector
Artifact Guard
Manual Bypass
Harmony Guard
```

## 48.3 ACE-Step

생성형 모델이므로 deterministic DSP가 아니다.

결과는 다음에 크게 영향을 받는다.

```text
Prompt
Lyrics
Cover Strength
Seed
```

실전에서는 같은 조건으로 2~4개 variation을 생성하고 가장 좋은 결과를 선택하는 방식이 적합하다.

---

# 49. 현재 권장 AI Remix 테스트 방법

예: `7080 한국 가요`

첫 baseline:

```text
Style             : 7080 한국 가요
Additional Prompt : 비움
Cover Strength    : 0.45
Lyrics            : 원 가사를 유지하려면 입력
Vocal Language    : 원곡 언어
Random Seed       : ON
```

먼저 여러 Seed를 비교한다.

예:

```text
A: 7080 / 0.45 / Seed random
B: 7080 / 0.45 / Seed random
C: 7080 / 0.45 / Seed random
```

그 후 필요한 부분만 Prompt로 강화한다.

---

# 50. 프로젝트 전체 파이프라인 요약

## 50.1 기존 곡을 자연스럽게 Key Shift + Voice Conversion

```text
원곡
↓
BS-RoFormer
├─ Vocals
│   ↓
│ Lead Vocal Selector
│ ├─ Lead
│ │   ↓
│ │ RVC / Seed-VC
│ │
│ └─ Harmony / Backing
│     ↓
│   Pitch Shift only
│
└─ Instrumental
    ↓
  Demucs
  ├─ Drums → original pitch
  ├─ Bass → target pitch
  ├─ Other → target pitch
  └─ Residual → target pitch

↓
Final Mix
```

## 50.2 완전히 다른 장르로 AI 재편곡

```text
원곡
↓
ACE-Step 1.5 Cover
↓
Blues / 7080 / R&B / Jazz / City Pop ...
↓
새 AI Remix WAV
↓
[선택]
RVC 0 semitone
↓
Target Singer Timbre
```

---

# 51. 개발 버전 흐름

```text
v1.x
- Pitch Analyzer
- Activity Gate
- Subtitle
- RubberBand

v1.8
- Seed-VC

v1.9
- RVC + RMVPE

v2.0
- RVC Training

v2.1
- Batch Vocal Dataset Extraction

v2.2
- Resume / Fine-tune

v2.3
- Experiment Browser
- Index refresh fixes

v2.4
- Harmony Guard

v2.5
- Artifact Guard
- Diagnostics

v2.6
- Artifact Priority
- Manual Bypass

v2.7
- Lead Vocal Selector

v2.8
- Training Lead Dataset Cleaner

v2.9
- Lead Melody Analysis
- Fine-tune Lead Lineage Fix

v3.0
- Instrument Smart Shift

v3.1
- ACE-Step AI Remix / Arrangement
- ACE-Step uv setup hotfix
- ACE-Step multipart src_audio upload hotfix
```

---

# 52. 새 PC에서 다시 구축할 때 권장 순서

```text
1. Windows / NVIDIA driver 준비
2. Python 3.12 설치
3. Git 설치
4. FFmpeg / RubberBand 준비
5. C:\dev\vocal_pitch_prototype_v1 생성
6. 메인 .venv 생성
7. PySide6 / numpy / librosa / soundfile / pyqtgraph 등 설치
8. .venv_separator 생성
9. audio-separator 설치
10. BS-RoFormer model 준비
11. Demucs model 준비
12. .venv_svc 생성
13. Seed-VC checkout / dependency 설치
14. .venv_rvc 생성
15. RVC checkout
16. PyTorch 2.7.1 + cu128
17. RMVPE / HuBERT / pretrained assets 준비
18. .venv_remix 생성
19. ACE-Step checkout
20. uv sync
21. ACE-Step first model load/download
22. main.py 실행
```

핵심은 **환경을 서로 합치지 않는 것**이다.

---

# 53. 빠른 상태 체크 명령

## Main

```powershell
C:\dev\vocal_pitch_prototype_v1\.venv\Scripts\python.exe -c "import main; print('OK')"
```

## RVC

```powershell
C:\dev\vocal_pitch_prototype_v1\.venv_rvc\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Separator

```powershell
C:\dev\vocal_pitch_prototype_v1\.venv_separator\Scripts\audio-separator.exe --version
```

## ACE-Step

```powershell
C:\dev\vocal_pitch_prototype_v1\.venv_remix\Scripts\python.exe -c "import torch, acestep; print(torch.__version__, torch.cuda.is_available())"
```

---

# 54. 유지보수 시 보존해야 할 원칙

프로젝트 변경 시 다음을 우선 보존해야 한다.

```text
기존 기능 동작
QSettings 저장값
기존 단축키
로그
가상환경 분리
GPU worker thread
fallback path
원본 dataset
training checkpoint
```

특히 하나의 AI 엔진을 업데이트한다고 다른 가상환경의 dependency를 강제로 덮어쓰지 않는다.

---

# 55. 현재 프로젝트가 할 수 있는 것

현재 상태에서 다음을 수행할 수 있다.

- 음원/영상에서 Pitch 분석
- 한국식 옥타브/계이름 표시
- 최저음/최고음 계산
- Lead Vocal 기준 melody filtering
- CSV/SRT/ASS 생성
- AI 보컬/반주 분리
- Key transpose
- Seed-VC 변환
- RVC + RMVPE 변환
- RVC model training
- Training resume
- Dataset 추가 fine-tuning
- RVC feature index 생성
- Lead-only training dataset 정제
- RVC artifact 감지 및 fallback
- Manual timeline bypass
- Lead vs Backing 기반 RVC 처리
- Demucs 기반 반주 stem 분리
- Drums pitch 유지형 Instrument Smart Shift
- ACE-Step 기반 Blues / 7080 / R&B / Jazz / City Pop 등의 생성형 AI Remix
- AI Remix 결과를 다시 현재 RVC pipeline으로 연결

---

# 56. 프로젝트의 현재 의미

이 프로젝트는 처음에는:

```text
"이 노래의 음역이 어떻게 되는가?"
```

를 분석하는 프로그램이었다.

현재는:

```text
노래를 분석하고
보컬을 분리하고
메인 보컬을 골라내고
키를 바꾸고
다른 target timbre로 변환하고
RVC 모델을 직접 학습하고
반주를 악기별로 자연스럽게 처리하고
아예 다른 장르로 AI 재편곡한 뒤
그 결과를 다시 원하는 보컬 timbre로 바꾸는
```

통합 로컬 음악 AI 프로젝트가 되었다.

현재 프로젝트의 핵심 설계 철학은 다음과 같다.

```text
1. 각 AI 엔진은 자신이 가장 잘하는 역할만 맡긴다.
2. 서로 충돌하는 dependency는 가상환경으로 분리한다.
3. 원본/중간 산출물을 최대한 보존한다.
4. 실패하면 기존 안정 경로로 fallback한다.
5. 로그를 남겨 원인 추적이 가능하게 한다.
6. GPU 장시간 작업은 GUI main thread를 막지 않는다.
7. RVC 입력과 학습은 Lead Vocal 중심으로 정제한다.
8. 생성형 AI와 deterministic DSP를 목적에 따라 구분한다.
```

---

# 57. 다음 단계 후보

현재 구조에서 자연스럽게 이어질 수 있는 기능:

```text
1. Neural Lead / Backing Vocal Separator
2. AI Remix → RVC one-click pipeline
3. Batch Remix Variations
   - Seed A/B/C/D 자동 생성
4. AI Remix Prompt Preset Editor
5. 7080 세부 preset
   - 7080 Folk
   - 7080 Ballad
   - 80s Synth Pop
   - 90s Korean Ballad
6. ACE-Step Style LoRA
7. AI Remix 결과 자동 비교
8. Stem-level Remix Editor
9. RVC Model A/B Comparison
10. Dataset Quality Scoring
```

---

# 58. 최종 한 줄 요약

현재 `vocal_pitch_prototype_v1`은 다음 기술을 한 Windows GUI 안에서 통합한 프로젝트다.

```text
PySide6
+ librosa pYIN
+ FFmpeg
+ RubberBand
+ BS-RoFormer
+ Demucs
+ Seed-VC
+ RVC
+ RMVPE
+ HuBERT
+ ACE-Step 1.5
```

그리고 그 목적은 단순 음계 분석을 넘어:

```text
Analysis
→ Separation
→ Lead Selection
→ Voice Conversion
→ Training
→ Instrument Processing
→ Generative Remix
```

까지 이어지는 전체 음악 AI workflow를 로컬 PC에서 구현하는 것이다.

---

## 문서 용도

이 문서는 다음 용도로 사용한다.

- 새 ChatGPT 대화로 프로젝트를 인계할 때
- 다른 개발 세션에서 진행 상황을 빠르게 복구할 때
- 새 PC에 개발 환경을 재구축할 때
- 현재 가상환경/AI 엔진 역할을 확인할 때
- 향후 리팩토링 전 전체 구조를 파악할 때
- 프로젝트 README보다 더 자세한 기술 핸드오프 문서가 필요할 때

