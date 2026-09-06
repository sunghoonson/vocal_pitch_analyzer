@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "RVC_VENV=%ROOT%.venv_rvc"
set "RVC_DIR=%ROOT%tools\rvc"
set "PINNED_COMMIT=81eed5e8f68b6bed1789f682fe78cdd324495afc"
set "REPO_URL=https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"

echo ============================================================
echo Vocal Pitch Analyzer - RVC + RMVPE CUDA Setup
echo ============================================================
echo.
echo RVC isolated environment:
echo   %RVC_VENV%
echo.
echo RVC source:
echo   %RVC_DIR%
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is required.
    pause
    exit /b 1
)

py -3.12 -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.12 x64 was not found.
    pause
    exit /b 1
)

if not exist "%ROOT%tools" mkdir "%ROOT%tools"

if not exist "%RVC_DIR%\.git" (
    echo [1/7] Cloning RVC...
    git clone "%REPO_URL%" "%RVC_DIR%"
    if errorlevel 1 goto :error
) else (
    echo [1/7] Existing RVC repository found.
)

echo [2/7] Pinning RVC source...
pushd "%RVC_DIR%"
git fetch origin
if errorlevel 1 (
    popd
    goto :error
)
git checkout --detach "%PINNED_COMMIT%"
if errorlevel 1 (
    popd
    goto :error
)
popd

if not exist "%RVC_VENV%\Scripts\python.exe" (
    echo [3/7] Creating .venv_rvc with Python 3.12...
    py -3.12 -m venv "%RVC_VENV%"
    if errorlevel 1 goto :error
) else (
    echo [3/7] Existing .venv_rvc found.
)

echo [4/7] Updating pip / build tools...
"%RVC_VENV%\Scripts\python.exe" -m pip install -U ^
  "pip>=25" "setuptools>=75,<81" "wheel>=0.45,<1"
if errorlevel 1 goto :error

echo.
echo [5/7] Installing RTX 50 / CUDA 12.8 PyTorch pair...
"%RVC_VENV%\Scripts\python.exe" -m pip install ^
  torch==2.7.1+cu128 torchaudio==2.7.1+cu128 ^
  --index-url https://download.pytorch.org/whl/cu128 ^
  --extra-index-url https://pypi.org/simple
if errorlevel 1 goto :error

echo.
echo [6/7] Installing RVC inference dependencies...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src='%RVC_DIR%\requirments_cu128_py312.txt'; $dst='%TEMP%\rvc_requirements_official_py312.txt'; Get-Content -LiteralPath $src | Where-Object { $_ -notmatch '^\s*--index-url' -and $_ -notmatch '^\s*--extra-index-url' } | Set-Content -LiteralPath $dst -Encoding utf8; Write-Host ('Requirements: ' + $dst)"
if errorlevel 1 goto :error

"%RVC_VENV%\Scripts\python.exe" -m pip install ^
  --index-url https://pypi.org/simple ^
  -r "%TEMP%\rvc_requirements_official_py312.txt"
if errorlevel 1 goto :error

"%RVC_VENV%\Scripts\python.exe" -m pip install ^
  "huggingface_hub==0.28.1"
if errorlevel 1 goto :error

echo.
echo [7/7] Downloading HuBERT + RMVPE assets...
"%RVC_VENV%\Scripts\python.exe" ^
  "%ROOT%dev_tools\setup_rvc_assets.py" ^
  "%RVC_DIR%"
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [OK] RVC + RMVPE installation completed.
echo.
echo Next:
echo   CHECK_RVC_RMVPE.bat
echo.
echo RVC voice models are NOT bundled.
echo Select your own/licensed .pth model in the GUI.
echo A matching .index file is recommended.
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] RVC + RMVPE setup failed.
echo Check the messages above.
echo ============================================================
pause
exit /b 1
