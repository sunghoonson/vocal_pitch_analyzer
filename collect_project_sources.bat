@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem ============================================================
rem Vocal Pitch Analyzer - Source Snapshot Collector
rem Project root:
rem   C:\dev\vocal_pitch_prototype_v1
rem
rem Output:
rem   C:\dev\_project_snapshots\vocal_pitch_prototype_v1\
rem
rem Creates:
rem   1) source snapshot ZIP
rem   2) concatenated source LOG.TXT
rem   3) project tree MD
rem ============================================================

cd /d "%~dp0"

set "ROOT_DIR=%CD%"
set "SCRIPT_PATH=%ROOT_DIR%\dev_tools\collect_project_sources.py"
set "SETTINGS_PATH=%ROOT_DIR%\dev_tools\collect_project_sources_settings.json"

echo ============================================================
echo Vocal Pitch Analyzer - Source Snapshot
echo ============================================================
echo [INFO] Root:
echo        %ROOT_DIR%
echo.

if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Collector script not found:
    echo         %SCRIPT_PATH%
    echo.
    pause
    exit /b 1
)

set "PYTHON_CMD="

if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%ROOT_DIR%\.venv\Scripts\python.exe"
    goto RUN_COLLECTOR
)

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    goto RUN_COLLECTOR
)

where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto RUN_COLLECTOR
)

echo [ERROR] Python was not found.
echo         Install Python or restore the project's .venv.
pause
exit /b 1


:RUN_COLLECTOR
echo [INFO] Python:
echo        %PYTHON_CMD%
echo.

%PYTHON_CMD% "%SCRIPT_PATH%" ^
  --root "%ROOT_DIR%" ^
  --settings "%SETTINGS_PATH%"

set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo [ERROR] Collector failed with exit code %EXITCODE%.
) else (
    echo [OK] Collector finished.
)

pause
exit /b %EXITCODE%
