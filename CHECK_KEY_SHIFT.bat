@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PROJECT_ROOT=C:\dev\vocal_pitch_prototype_v1"

echo ============================================================
echo Key Shift / RubberBand Check
echo ============================================================
echo.

if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [ERROR] Main .venv not found.
    pause
    exit /b 1
)

pushd "%PROJECT_ROOT%"
".venv\Scripts\python.exe" -c "from audio_transposer import find_ffmpeg,rubberband_filter_available; print('FFmpeg:',find_ffmpeg()); print('RubberBand:',rubberband_filter_available())"
set "RC=%ERRORLEVEL%"
popd

echo.
if not "%RC%"=="0" (
    echo [ERROR] Check failed.
) else (
    echo [OK] Check completed.
)

if /i not "%~1"=="/nopause" pause
exit /b %RC%
