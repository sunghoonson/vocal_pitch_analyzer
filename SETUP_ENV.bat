@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Vocal Pitch Analyzer v1 - Python 3.12 environment setup
echo ============================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher ^(py.exe^) was not found.
    echo Install Python 3.12 x64 first, then run this file again.
    pause
    exit /b 1
)

py -3.12 --version
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.12 was not found.
    echo Install Python 3.12 x64 and enable the Python Launcher.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [1/3] Creating .venv ...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo.
    echo [1/3] Existing .venv found.
)

echo.
echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [3/3] Installing libraries ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [OK] Setup complete.
echo VS Code interpreter:
echo   %CD%\.venv\Scripts\python.exe
echo.
echo Run RUN.bat or press F5 in VS Code.
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Setup failed. Check the messages above.
pause
exit /b 1
