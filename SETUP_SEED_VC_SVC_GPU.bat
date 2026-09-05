@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "SVC_VENV=%ROOT%.venv_svc"
set "SEED_DIR=%ROOT%tools\seed-vc"
set "PINNED_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8"
set "REPO_URL=https://github.com/Plachtaa/seed-vc.git"

echo ============================================================
echo Vocal Pitch Analyzer - Seed-VC SVC CUDA Setup
echo ============================================================
echo.
echo Seed-VC is installed in an isolated environment:
echo   %SVC_VENV%
echo.
echo Seed-VC source:
echo   %SEED_DIR%
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is required.
    pause
    exit /b 1
)

py -3.10 -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 was not found.
    echo.
    echo Seed-VC upstream recommends Python 3.10 on Windows.
    echo Install Python 3.10 x64, then run this BAT again.
    pause
    exit /b 1
)

if not exist "%ROOT%tools" mkdir "%ROOT%tools"

if not exist "%SEED_DIR%\.git" (
    echo [1/6] Cloning Seed-VC...
    git clone "%REPO_URL%" "%SEED_DIR%"
    if errorlevel 1 goto :error
) else (
    echo [1/6] Existing Seed-VC repository found.
)

echo [2/6] Pinning Seed-VC source...
pushd "%SEED_DIR%"
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

if not exist "%SVC_VENV%\Scripts\python.exe" (
    echo [3/6] Creating .venv_svc with Python 3.10...
    py -3.10 -m venv "%SVC_VENV%"
    if errorlevel 1 goto :error
) else (
    echo [3/6] Existing .venv_svc found.
)

echo [4/6] Updating pip...
"%SVC_VENV%\Scripts\python.exe" -m pip install -U pip setuptools wheel
if errorlevel 1 goto :error

echo.
echo [5/6] Installing current CUDA PyTorch runtime...
echo       CUDA 13.0 wheel index
"%SVC_VENV%\Scripts\python.exe" -m pip install -U ^
  --index-url https://download.pytorch.org/whl/cu130 ^
  torch torchvision torchaudio
if errorlevel 1 goto :error

echo.
echo [6/6] Installing Seed-VC dependencies...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src='%SEED_DIR%\requirements.txt'; $dst='%TEMP%\seed_vc_requirements_no_torch.txt'; Get-Content -LiteralPath $src | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio)(\s|=|$)' } | Set-Content -LiteralPath $dst -Encoding utf8; Write-Host ('Filtered requirements: ' + $dst)"
if errorlevel 1 goto :error

"%SVC_VENV%\Scripts\python.exe" -m pip install -r "%TEMP%\seed_vc_requirements_no_torch.txt"
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [OK] Seed-VC SVC environment installation completed.
echo.
echo Next:
echo   CHECK_SEED_VC_SVC.bat
echo.
echo The first actual SVC conversion will download model files
echo from Hugging Face automatically.
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] Seed-VC setup failed.
echo Check the messages above.
echo ============================================================
pause
exit /b 1
