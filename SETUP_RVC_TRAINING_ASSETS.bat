@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "RVC_PY=%ROOT%.venv_rvc\Scripts\python.exe"
set "RVC_DIR=%ROOT%tools\rvc"

echo ============================================================
echo Vocal Pitch Analyzer - RVC Training Assets Setup
echo ============================================================
echo.

if not exist "%RVC_PY%" (
    echo [ERROR] .venv_rvc is missing.
    echo.
    echo First run:
    echo   SETUP_RVC_RMVPE_GPU.bat
    echo.
    pause
    exit /b 1
)

if not exist "%RVC_DIR%\train\train.py" (
    echo [ERROR] RVC training source is missing.
    pause
    exit /b 1
)

echo [1/2] Checking huggingface_hub...
"%RVC_PY%" -m pip install -U "huggingface_hub==0.28.1"
if errorlevel 1 goto :error

echo.
echo [2/2] Downloading pretrained + mute assets...
"%RVC_PY%" "%ROOT%dev_tools\setup_rvc_training_assets.py" "%RVC_DIR%"
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [OK] RVC training assets setup completed.
echo.
echo Next:
echo   CHECK_RVC_TRAINING.bat
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] RVC training asset setup failed.
echo ============================================================
pause
exit /b 1
