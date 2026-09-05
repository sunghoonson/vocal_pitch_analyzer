@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "RVC_PY=%ROOT%.venv_rvc\Scripts\python.exe"
set "RVC_DIR=%ROOT%tools\rvc"

echo ============================================================
echo RVC + RMVPE Environment Check
echo ============================================================
echo.

if not exist "%RVC_PY%" (
    echo [ERROR] .venv_rvc is missing.
    echo Run SETUP_RVC_RMVPE_GPU.bat first.
    pause
    exit /b 1
)

if not exist "%RVC_DIR%\infer\cli.py" (
    echo [ERROR] RVC infer\cli.py is missing.
    pause
    exit /b 1
)

if not exist "%RVC_DIR%\assets\hubert_base\pytorch_model.bin" (
    echo [ERROR] HuBERT model is missing.
    pause
    exit /b 1
)

if not exist "%RVC_DIR%\assets\rmvpe\rmvpe.pt" (
    echo [ERROR] RMVPE model is missing.
    pause
    exit /b 1
)

"%RVC_PY%" -c "import sys,torch,torchaudio,numpy,librosa,soundfile; print('Python:',sys.version); print('PyTorch:',torch.__version__); print('TorchAudio:',torchaudio.__version__); print('CUDA available:',torch.cuda.is_available()); print('CUDA runtime:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('NumPy:',numpy.__version__); print('librosa:',librosa.__version__)"
if errorlevel 1 goto :error

echo.
echo [INFO] RVC CLI smoke test...
pushd "%RVC_DIR%"
"%RVC_PY%" infer\cli.py --help >nul
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" goto :error

echo.
echo ============================================================
echo [OK] RVC + RMVPE runtime check passed.
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] RVC + RMVPE environment check failed.
echo ============================================================
pause
exit /b 1
