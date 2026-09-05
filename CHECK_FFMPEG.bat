@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo FFmpeg check
echo ============================================================
echo.

if exist "%CD%\tools\ffmpeg\ffmpeg.exe" (
    echo [OK] Local FFmpeg:
    echo   %CD%\tools\ffmpeg\ffmpeg.exe
    "%CD%\tools\ffmpeg\ffmpeg.exe" -version | findstr /b "ffmpeg version"
    pause
    exit /b 0
)

where ffmpeg >nul 2>nul
if not errorlevel 1 (
    echo [OK] FFmpeg from PATH:
    where ffmpeg
    ffmpeg -version | findstr /b "ffmpeg version"
    pause
    exit /b 0
)

echo [ERROR] FFmpeg was not found.
echo Run SETUP_FFMPEG.bat or place ffmpeg.exe here:
echo   %CD%\tools\ffmpeg\ffmpeg.exe
pause
exit /b 1
