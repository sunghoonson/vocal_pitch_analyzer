@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv was not found.
    echo Run SETUP_ENV.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Program ended with an error.
    pause
)
