@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "RVC_PY=%ROOT%.venv_rvc\Scripts\python.exe"

echo ============================================================
echo RVC Training Environment Check
echo ============================================================
echo.

if not exist "%RVC_PY%" (
    echo [ERROR] .venv_rvc is missing.
    pause
    exit /b 1
)

pushd "%ROOT%"
"%RVC_PY%" -c "import torch; from rvc_trainer import training_assets_status; ok,msg=training_assets_status(); print(msg); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); raise SystemExit(0 if ok and torch.cuda.is_available() else 1)"
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" goto :error

echo.
echo ============================================================
echo [OK] RVC training environment check passed.
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] RVC training environment is not ready.
echo.
echo Run:
echo   SETUP_RVC_TRAINING_ASSETS.bat
echo ============================================================
pause
exit /b 1
