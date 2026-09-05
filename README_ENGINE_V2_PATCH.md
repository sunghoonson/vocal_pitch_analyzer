# Vocal Pitch Analyzer - Engine v2 Patch

대상 루트:

```text
C:\dev\vocal_pitch_prototype_v1
```

## 목적

v1에서 완성곡을 분석할 때 음표가 지나치게 적게 남는 문제를 개선합니다.

v1의 문제:

```text
pYIN
 ↓
신뢰도 0.65 이상만 통과
 ↓
중간 frame 하나가 끊겨도 segment 분리
 ↓
70ms 미만 segment 삭제
 ↓
결과가 지나치게 적음
```

v2:

```text
pYIN Raw F0
 ↓
Raw 데이터는 항상 보존
 ↓
기본 threshold 0.25
 ↓
80ms 이하 짧은 dropout 보간
 ↓
median pitch smoothing
 ↓
음표 전환 hysteresis
 ↓
짧은 note는 삭제하지 않고 주변 note로 병합
 ↓
Note Segment
```

## 패치 적용

압축을 풀고:

```text
APPLY_PATCH.bat
```

실행.

자동 대상:

```text
C:\dev\vocal_pitch_prototype_v1
```

기존 파일은:

```text
backup_before_engine_v2_YYYYMMDD_HHMMSS
```

폴더에 자동 백업됩니다.

## 새 기본값

```text
유성음 신뢰도       0.25
짧은 dropout 연결  80 ms
Pitch smoothing     5 frames
Hysteresis          20 cent
짧은 note 병합      35 ms
```

## GUI 변경

새 통계:

```text
Raw 유성
임계값 통과
보정 피치
커버리지
음표 구간 수
```

그래프:

- Raw pYIN
- Processed v2

두 데이터를 동시에 볼 수 있습니다.

## CSV

### 음표 CSV 저장

사람이 보기 쉬운 note segment:

```text
*_pitch_segments_v2.csv
```

### Raw CSV 저장

모든 frame 단위 분석 진단:

```text
*_raw_pitch_v2.csv
```

Raw CSV에는 다음이 들어갑니다.

```text
time_sec
raw_hz
raw_midi_float
raw_note
raw_voiced_flag
voiced_probability
accepted_by_threshold
processed_hz
processed_midi_float
processed_note
분석 옵션들
```

## 추천 테스트

이전에 분석했던 같은 곡을 기본값 그대로 다시 분석하세요.

완료 후 비교:

```text
v1:
음표 구간 수
검출된 피치 시간

v2:
Raw 유성
임계값 통과
보정 피치
커버리지
음표 구간 수
```

v2에서 음표가 훨씬 많아지는 것이 정상입니다.

## 나에게 전달하면 좋은 파일

결과를 확인하려면 다음 두 파일이 가장 유용합니다.

```text
*_pitch_segments_v2.csv
*_raw_pitch_v2.csv
```

그리고 분석 완료 GUI 스크린샷도 같이 있으면 좋습니다.

## 중요한 한계

이 패치는 "보컬 분리" 패치가 아닙니다.

완성곡:

```text
보컬 + 베이스 + 기타 + 피아노 + 신스
```

를 그대로 pYIN에 넣기 때문에, v2는 보컬 데이터를 덜 버리지만 반주 피치가 섞이는
문제까지 해결하지는 않습니다.

다음 단계는:

```text
FFmpeg
 ↓
BS-RoFormer 보컬 분리
 ↓
vocals.wav
 ↓
Engine v2
```

구조가 권장됩니다.
