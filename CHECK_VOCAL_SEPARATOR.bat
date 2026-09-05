@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "SEP=%CD%\.venv_separator\Scripts\audio-separator.exe"

echo ============================================================
echo Vocal Separator Environment Check
echo ============================================================
echo.

if not exist "%SEP%" (
    echo [ERROR] audio-separator was not found:
    echo   %SEP%
    echo.
    echo Run SETUP_VOCAL_SEPARATOR_GPU.bat first.
    pause
    exit /b 1
)

echo [1] audio-separator version
"%SEP%" --version

echo.
echo [2] Environment / GPU / FFmpeg
"%SEP%" --env_info

echo.
echo [3] Default vocal model information
"%SEP%" --list_models --list_filter vocals --list_limit 8

echo.
echo ============================================================
echo Check the output above.
echo.
echo For NVIDIA acceleration, look for CUDA/GPU detection.
echo The program can still run on CPU, but separation will be much slower.
echo ============================================================
pause
