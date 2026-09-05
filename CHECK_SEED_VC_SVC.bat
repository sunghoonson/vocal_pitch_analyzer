@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "SVC_PY=%ROOT%.venv_svc\Scripts\python.exe"
set "SEED_DIR=%ROOT%tools\seed-vc"

echo ============================================================
echo Seed-VC SVC Environment Check
echo ============================================================
echo.

if not exist "%SVC_PY%" (
    echo [ERROR] .venv_svc is missing.
    echo Run SETUP_SEED_VC_SVC_GPU.bat first.
    pause
    exit /b 1
)

if not exist "%SEED_DIR%\inference.py" (
    echo [ERROR] tools\seed-vc\inference.py is missing.
    echo Run SETUP_SEED_VC_SVC_GPU.bat again.
    pause
    exit /b 1
)

"%SVC_PY%" -c "import sys,torch,torchaudio,librosa,transformers,soundfile; print('Python:',sys.version); print('PyTorch:',torch.__version__); print('TorchAudio:',torchaudio.__version__); print('CUDA available:',torch.cuda.is_available()); print('CUDA runtime:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('librosa:',librosa.__version__); print('transformers:',transformers.__version__)"
if errorlevel 1 goto :error

echo.
echo [INFO] Seed-VC inference help smoke test...
pushd "%SEED_DIR%"
"%SVC_PY%" inference.py --help >nul
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" goto :error

echo.
echo ============================================================
echo [OK] Seed-VC SVC runtime check passed.
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] Seed-VC environment check failed.
echo ============================================================
pause
exit /b 1
