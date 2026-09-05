@echo off
setlocal
cd /d "%~dp0"

echo This will delete ONLY separated vocal stem cache:
echo   %CD%\cache\vocal_stems
echo.
echo Separator model files will NOT be deleted.
echo.
choice /C YN /M "Delete vocal stem cache"
if errorlevel 2 exit /b 0

if exist "%CD%\cache\vocal_stems" (
    rmdir /s /q "%CD%\cache\vocal_stems"
)

echo [OK] Vocal stem cache cleared.
pause
