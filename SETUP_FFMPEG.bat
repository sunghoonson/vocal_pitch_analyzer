@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Vocal Pitch Analyzer - FFmpeg setup
echo ============================================================
echo.

where ffmpeg >nul 2>nul
if not errorlevel 1 (
    echo [OK] FFmpeg is already available:
    where ffmpeg
    ffmpeg -version | findstr /b "ffmpeg version"
    echo.
    pause
    exit /b 0
)

where winget >nul 2>nul
if errorlevel 1 (
    echo [ERROR] winget was not found.
    echo.
    echo Install FFmpeg manually and place ffmpeg.exe here:
    echo   %CD%\tools\ffmpeg\ffmpeg.exe
    echo.
    echo Or install FFmpeg so that the "ffmpeg" command is available in PATH.
    pause
    exit /b 1
)

echo [INFO] Installing Gyan.FFmpeg with winget...
echo.
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo [ERROR] winget installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Installation command completed.
echo.
echo IMPORTANT:
echo Close and reopen VS Code/terminal once so the updated PATH is loaded.
echo Then run CHECK_FFMPEG.bat.
echo ============================================================
pause
