# Vocal Pitch Analyzer - Prototype v1

## 목표

1차 프로토타입은 다음 기능까지만 검증합니다.

- MP3 / WAV / FLAC / OGG 열기
- mono 22.05 kHz로 로드
- `librosa.pyin()` 기반 F0(기본주파수) 추출
- 신뢰도가 낮은 프레임 제거
- Hz -> MIDI -> C4/A4 등의 음계 변환
- `2옥타브 라` 같은 한국식 표기
- 시간-음높이 그래프
- 연속 음표 구간 표
- 최고음 / 최저음
- CSV 저장
- 분석은 별도 QThread에서 실행하여 GUI 멈춤 방지

> 주의: 이 버전에는 **보컬 분리 기능이 아직 없습니다.**
> 완성곡 MP3를 직접 넣을 수는 있지만 기타/피아노/베이스 등의 반주를
> 보컬 F0로 잘못 검출할 수 있습니다.
> 가능하면 우선 vocal-only stem으로 테스트하세요.
> 다음 단계에서 BS-RoFormer 계열 보컬 분리를 추가하는 것을 권장합니다.

---

## 1. 권장 환경

- Windows 10/11 x64
- VS Code
- Python 3.12 x64
- 인터넷 연결 (최초 pip 설치 시)

프로젝트 폴더 자체에 `.venv`를 생성하므로 다른 Python 프로젝트와 충돌하지 않습니다.

---

## 2. 가장 쉬운 설치

프로젝트 폴더에서:

```text
SETUP_ENV.bat
```

더블클릭.

이 배치 파일은 자동으로:

1. Python 3.12 확인
2. `.venv` 생성
3. pip 업데이트
4. `requirements.txt` 설치

를 수행합니다.

설치 후:

```text
RUN.bat
```

을 실행하세요.

---

## 3. VS Code에서 열기

VS Code:

```text
File -> Open Folder
```

에서 이 프로젝트 폴더를 선택합니다.

`.vscode/settings.json`이 `.venv` 인터프리터를 지정하도록 포함되어 있습니다.

그래도 우측 아래 Python 버전이 다르다면:

```text
Ctrl + Shift + P
Python: Select Interpreter
```

선택 후:

```text
.venv\Scripts\python.exe
```

를 선택하세요.

`F5`로 실행할 수도 있습니다.

---

## 4. PowerShell에서 수동 설치하고 싶을 때

```powershell
cd "프로젝트 폴더"

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

python main.py
```

PowerShell 실행 정책 때문에 Activate.ps1이 차단되어도 가상환경 자체는 정상입니다.
그 경우 활성화하지 않고 직접:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main.py
```

처럼 실행하면 됩니다.

---

## 5. 기본 옵션

### 최저 검출 주파수
기본:

```text
65.4 Hz
```

대략 C2.

### 최고 검출 주파수
기본:

```text
1396.9 Hz
```

대략 F6.

일반적인 가요 보컬 범위를 꽤 넓게 포함하기 위한 값입니다.

### 유성음 신뢰도
기본:

```text
0.65
```

값을 높이면:
- 오검출 감소 가능
- 실제 보컬 일부도 빠질 수 있음

값을 낮추면:
- 더 많은 음을 잡음
- 반주/잡음 오검출 증가 가능

혼합 음원에서는 0.7~0.8도 시험해보세요.

---

## 6. 음계 표기

예:

```text
A4 = 440 Hz = 2옥타브 라
C5 ≈ 523.25 Hz = 3옥타브 도
```

Cent 값은 해당 평균 피치가 가장 가까운 평균율 음에서 얼마나 벗어났는지 나타냅니다.

예:

```text
A4 +12 cent
```

이면 A4보다 약간 높은 피치입니다.

---

## 7. 현재 분석 방식의 한계

pYIN은 기본적으로 한 시점의 대표적인 F0를 추적합니다.

따라서 완성된 노래에는:

```text
보컬
기타
피아노
베이스
신스
```

등 여러 음이 동시에 존재하기 때문에 보컬만 정확히 골라낼 수는 없습니다.

그래서 개발 순서는:

```text
v1
음원 -> pYIN -> 피치 그래프

v2
음원 -> 보컬 분리 -> pYIN -> 피치 그래프

v3
음원 -> 보컬 분리 -> 피치 분석
                  -> 가사 forced alignment
                  -> 음절별 음계 연결
```

을 권장합니다.

---

## 8. 다음 단계 후보

### 2단계
- BS-RoFormer 기반 vocals / instrumental 자동 분리
- GPU(RTX) 사용
- 원본 / vocals 중 분석 대상 선택

### 3단계
- 가사 입력
- WhisperX 또는 별도 forced alignment
- 단어/음절별 타임스탬프

### 4단계
- `사랑해`
  - 사: A4
  - 랑: B4 -> C5
  - 해: B4
- 식으로 음절과 실제 피치 구간 연결
- 고음 구간 검색
- 음역 통계
- 비브라토 분석
- MIDI 내보내기

---

## 프로젝트 구조

```text
vocal_pitch_prototype_v1/
├─ .vscode/
│  ├─ launch.json
│  └─ settings.json
├─ main.py
├─ pitch_analyzer.py
├─ requirements.txt
├─ SETUP_ENV.bat
├─ CHECK_ENV.bat
├─ RUN.bat
└─ README_SETUP.md
```
