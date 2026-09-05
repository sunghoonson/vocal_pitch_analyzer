# M4A / MP4 입력 지원 패치

대상 프로젝트 루트:

```text
C:\dev\vocal_pitch_prototype_v1
```

## 추가 지원 형식

- M4A
- MP4
- AAC
- WEBM
- MKV
- MOV
- WMA
- OPUS
- M4V

기존 MP3 / WAV / FLAC / OGG 지원도 유지됩니다.

## 동작 방식

```text
M4A / MP4 / AAC / WEBM / MKV / MOV ...
                    ↓
                 FFmpeg
                    ↓
        임시 mono PCM WAV / 22050 Hz
                    ↓
             기존 librosa pYIN
                    ↓
              음계/옥타브 분석
                    ↓
              임시 WAV 자동 삭제
```

MP4/MKV/MOV 파일은 영상 자체를 분석하는 것이 아니라 첫 번째 오디오 스트림만
추출합니다.

## 패치 적용

압축을 아무 곳에나 풀고:

```text
APPLY_PATCH.bat
```

을 실행합니다.

기본 프로젝트 경로는 자동으로:

```text
C:\dev\vocal_pitch_prototype_v1
```

을 사용합니다.

기존 `main.py`, `pitch_analyzer.py`는 프로젝트 루트 아래의

```text
backup_before_m4a_mp4_patch_YYYYMMDD_HHMMSS
```

폴더로 먼저 백업됩니다.

## FFmpeg 설치

프로젝트 루트에서:

```text
SETUP_FFMPEG.bat
```

실행.

Windows Package Manager(winget)의 `Gyan.FFmpeg` 패키지를 설치합니다.

설치 직후에는 기존 VS Code/PowerShell이 예전 PATH를 들고 있을 수 있으므로
VS Code를 한 번 완전히 닫았다가 다시 여는 것을 권장합니다.

이후:

```text
CHECK_FFMPEG.bat
```

으로 확인합니다.

### 로컬 FFmpeg도 지원

시스템에 설치하지 않고 다음 위치에 직접 `ffmpeg.exe`를 두어도 됩니다.

```text
C:\dev\vocal_pitch_prototype_v1\tools\ffmpeg\ffmpeg.exe
```

프로그램은 다음 순서로 FFmpeg를 찾습니다.

1. `tools\ffmpeg\ffmpeg.exe`
2. `tools\ffmpeg.exe`
3. 프로젝트 루트의 `ffmpeg.exe`
4. Windows PATH의 `ffmpeg`

## 테스트 순서

1. `CHECK_FFMPEG.bat`
2. `RUN.bat`
3. M4A 또는 MP4 선택
4. `분석 시작`
5. 진행 상태에 `FFmpeg로 오디오를 추출하고 WAV로 변환했습니다.` 표시 확인

## 주의

이 패치는 **입력 형식 지원 확장 패치**입니다.

아직 보컬/반주 분리는 하지 않으므로 완성곡을 분석하면 반주 피치가 섞일 수
있습니다. 다음 단계에서는 BS-RoFormer 계열 보컬 분리를 추가하는 것이 좋습니다.
