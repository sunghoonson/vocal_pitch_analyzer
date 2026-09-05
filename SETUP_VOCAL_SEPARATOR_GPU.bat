@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "SEP_VENV=%CD%\.venv_separator"
set "MODEL_DIR=%CD%\cache\separator_models"
set "MODEL=model_bs_roformer_ep_317_sdr_12.9755.yaml"

echo ============================================================
echo Vocal Pitch Analyzer - AI Vocal Separator GPU Setup
echo ============================================================
echo.
echo Separate environment:
echo   %SEP_VENV%
echo.
echo This environment is intentionally separate from the main .venv.
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher ^(py.exe^) was not found.
    echo Install Python 3.12 x64 first.
    pause
    exit /b 1
)

py -3.12 --version
if errorlevel 1 (
    echo [ERROR] Python 3.12 was not found.
    pause
    exit /b 1
)

echo [1/5] Creating isolated separator environment...
if not exist "%SEP_VENV%\Scripts\python.exe" (
    py -3.12 -m venv "%SEP_VENV%"
    if errorlevel 1 goto :error
) else (
    echo [INFO] Existing .venv_separator found.
)

echo.
echo [2/5] Upgrading pip...
"%SEP_VENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [3/5] Installing audio-separator GPU package...
echo This can be a large download because PyTorch/CUDA components are included.
"%SEP_VENV%\Scripts\python.exe" -m pip install --upgrade "audio-separator[gpu]==0.47.0"
if errorlevel 1 goto :error

echo.
echo [4/5] Checking separator environment...
if not exist "%SEP_VENV%\Scripts\audio-separator.exe" (
    echo [ERROR] audio-separator.exe was not created.
    goto :error
)

"%SEP_VENV%\Scripts\audio-separator.exe" --env_info
if errorlevel 1 (
    echo.
    echo [WARN] env_info returned an error.
    echo Installation may still exist, but GPU/FFmpeg configuration needs checking.
)

echo.
echo [5/5] Downloading default BS-RoFormer model...
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

"%SEP_VENV%\Scripts\audio-separator.exe" ^
  --model_filename "%MODEL%" ^
  --model_file_dir "%MODEL_DIR%" ^
  --download_model_only

if errorlevel 1 (
    echo.
    echo [WARN] Model pre-download failed.
    echo The program will retry downloading it on first separation.
)

echo.
echo ============================================================
echo [OK] Vocal separator setup completed.
echo.
echo Run:
echo   CHECK_VOCAL_SEPARATOR.bat
echo.
echo Then start the app with:
echo   RUN.bat
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] Vocal separator setup failed.
echo.
echo Main app .venv was NOT modified.
echo You can delete .venv_separator and run this setup again.
echo ============================================================
pause
exit /b 1
