# Vocal Pitch Analyzer v1.3 - AI Vocal Separation Patch

대상 프로젝트 루트:

```text
C:\dev\vocal_pitch_prototype_v1
```

## 이번 패치의 핵심

기존:

```text
원본 믹스
 ↓
pYIN
 ↓
Pitch Engine v2
```

추가:

```text
MP3 / M4A / MP4
       ↓
audio-separator
       ↓
BS-RoFormer
       ↓
vocals.wav
       ↓
Pitch Engine v2
       ↓
보컬 중심 음계 / 옥타브
```

## 왜 별도 가상환경인가?

AI 분리에는 PyTorch/CUDA 등 무거운 의존성이 필요합니다.

기존 메인 앱의:

```text
.venv
```

를 변경하지 않고 별도로:

```text
.venv_separator
```

를 생성합니다.

따라서 separator 설치가 실패하거나 CUDA 패키지 문제가 생겨도
PySide6/librosa 기반 기존 앱 환경은 그대로 유지됩니다.

## 1. 패치 적용

압축을 풀고:

```text
APPLY_PATCH.bat
```

실행.

기존 main.py는 자동 백업됩니다.

## 2. AI 보컬 분리 환경 설치

프로젝트 루트에서:

```text
SETUP_VOCAL_SEPARATOR_GPU.bat
```

실행.

자동으로:

```text
Python 3.12
 ↓
.venv_separator 생성
 ↓
audio-separator[gpu] 0.47.0 설치
 ↓
환경 확인
 ↓
BS-RoFormer 기본 모델 다운로드
```

를 수행합니다.

다운로드 용량이 크므로 첫 설치에는 시간이 걸릴 수 있습니다.

## 3. 환경 확인

```text
CHECK_VOCAL_SEPARATOR.bat
```

실행.

NVIDIA GPU가 정상 사용되는지 확인합니다.

## 4. 프로그램 사용

```text
RUN.bat
```

실행.

GUI의 분석 대상:

```text
보컬 분리 후 분석 (권장)
원본 전체 믹스 분석
```

중 선택할 수 있습니다.

기본값은:

```text
보컬 분리 후 분석
```

입니다.

## 5. 기본 모델

```text
model_bs_roformer_ep_317_sdr_12.9755.yaml
```

BS-RoFormer 2-stem vocal separation 모델을 사용합니다.

프로그램은 `--single_stem Vocals`로 실행하기 때문에
instrumental 파일은 만들지 않고 분석에 필요한 vocals.wav만 생성합니다.

## 6. 분리 캐시

기본 ON:

```text
분리된 vocals.wav 캐시 사용
```

캐시 위치:

```text
cache\vocal_stems\
```

모델 캐시:

```text
cache\separator_models\
```

같은 원본 파일이 수정되지 않았고 같은 모델을 사용한다면
두 번째 분석부터 AI 분리 과정을 건너뜁니다.

캐시를 지우려면:

```text
CLEAR_VOCAL_CACHE.bat
```

실행.

## 7. 분리 WAV 직접 확인

분석 완료 후:

```text
분리 보컬 WAV 저장
```

버튼이 활성화됩니다.

이 파일을 직접 재생해서:

- 보컬이 충분히 선명한지
- 베이스/기타/신스가 얼마나 남았는지
- 코러스가 함께 남는지

확인할 수 있습니다.

## 8. 결과 CSV

```text
*_pitch_segments_v13.csv
*_raw_pitch_v13.csv
```

둘 모두 `analysis_source` 열을 추가합니다.

값:

```text
separated_vocals
original_mix
```

으로 어떤 입력을 분석한 것인지 바로 알 수 있습니다.

## 9. 로그

AI 분리 마지막 실행 로그:

```text
logs\vocal_separator_last.log
```

분리가 실패하면 이 파일을 전달하면 원인 분석에 매우 유용합니다.

## 10. 첫 테스트 권장 순서

이전과 동일한 곡을 사용합니다.

### A. 원본 전체 믹스 분석

```text
분석 대상:
원본 전체 믹스 분석
```

결과 저장.

### B. 보컬 분리 후 분석

```text
분석 대상:
보컬 분리 후 분석
```

결과 저장.

### 비교할 것

- 최저음
- 저음 C2~A2 검출량
- 보컬 실제 음역에 피치가 집중되는지
- Raw 유성 시간
- 보정 피치 시간
- 커버리지
- 음표 구간 수
- pitch curve 연속성

## 11. 나에게 전달하면 좋은 것

보컬 분리 후 분석이 완료되면:

```text
*_pitch_segments_v13.csv
*_raw_pitch_v13.csv
```

그리고 가능하면:

```text
*_vocals.wav
```

또는 GUI 완료 스크린샷을 보내주세요.

분리 실패 시:

```text
logs\vocal_separator_last.log
CHECK_VOCAL_SEPARATOR.bat 출력
```

이 가장 중요합니다.

## 현재 한계

BS-RoFormer는 "보컬 vs 반주"를 분리합니다.

따라서:

```text
메인 보컬
코러스
백업 보컬
듀엣 상대 보컬
```

은 모두 vocals stem 안에 남을 수 있습니다.

즉 다음 정확도 개선 단계는:

```text
vocals.wav
 ↓
메인 멜로디 추적
 ↓
주 보컬 F0 선택
```

입니다.
