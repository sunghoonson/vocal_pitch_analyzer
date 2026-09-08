@echo off
setlocal EnableExtensions

set "ROOT=C:\dev\vocal_pitch_prototype_v1"
set "APK=%ROOT%\tools\s24_native_mic\S24NativeMic-debug.apk"
set "ADB=%ROOT%\tools\platform-tools\adb.exe"

if not exist "%ADB%" set "ADB=adb"

echo ============================================================
echo VPA S24 Native Mic - Install / Launch
echo ============================================================
echo.

if not exist "%APK%" (
    echo [ERROR] APK not found:
    echo   %APK%
    echo Run BUILD_S24_NATIVE_MIC.bat first.
    pause
    exit /b 1
)

"%ADB%" devices
if errorlevel 1 goto :error

echo.
echo [1/3] Install...
"%ADB%" install -r "%APK%"
if errorlevel 1 goto :error

echo.
echo [2/3] ADB reverse audio port...
"%ADB%" reverse tcp:8791 tcp:8791
if errorlevel 1 goto :error

echo.
echo [3/3] Launch...
"%ADB%" shell am start -n local.vocalpitch.s24mic/.MainActivity --ez auto_start true
if errorlevel 1 goto :error

echo.
echo [OK] Native Mic launched.
echo First run: allow microphone permission on the Galaxy.
pause
exit /b 0

:error
echo.
echo [ERROR] Install/launch failed.
pause
exit /b 1
