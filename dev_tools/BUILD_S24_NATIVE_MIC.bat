@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=C:\dev\vocal_pitch_prototype_v1"
set "APP=%ROOT%\android\s24_native_mic"
set "OUTDIR=%ROOT%\tools\s24_native_mic"
set "GRADLE_VERSION=8.9"
set "LOCAL_GRADLE=%APP%\.gradle_dist\gradle-%GRADLE_VERSION%"

echo ============================================================
echo VPA S24 Native Mic - Android APK Build
echo ============================================================
echo.

if not exist "%APP%\settings.gradle" (
    echo [ERROR] Android project not found:
    echo   %APP%
    pause
    exit /b 1
)

rem ------------------------------------------------------------
rem Java 17+: prefer JAVA_HOME, then Android Studio bundled JBR.
rem ------------------------------------------------------------

set "JAVA_EXE="

if defined JAVA_HOME (
    if exist "%JAVA_HOME%\bin\java.exe" (
        set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
    )
)

if not defined JAVA_EXE (
    if exist "C:\Program Files\Android\Android Studio\jbr\bin\java.exe" (
        set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
        set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
    )
)

if not defined JAVA_EXE (
    echo [ERROR] Java 17+ not found.
    echo Install Android Studio, then run this BAT again.
    pause
    exit /b 1
)

echo [JAVA] %JAVA_EXE%
"%JAVA_EXE%" -version
if errorlevel 1 goto :error

rem ------------------------------------------------------------
rem Android SDK.
rem ------------------------------------------------------------

set "SDK="

if defined ANDROID_SDK_ROOT (
    if exist "%ANDROID_SDK_ROOT%\platforms" set "SDK=%ANDROID_SDK_ROOT%"
)

if not defined SDK (
    if defined ANDROID_HOME (
        if exist "%ANDROID_HOME%\platforms" set "SDK=%ANDROID_HOME%"
    )
)

if not defined SDK (
    if exist "%LOCALAPPDATA%\Android\Sdk\platforms" (
        set "SDK=%LOCALAPPDATA%\Android\Sdk"
    )
)

if not defined SDK (
    echo [ERROR] Android SDK not found.
    echo Open Android Studio once and install Android SDK Platform 35.
    pause
    exit /b 1
)

set "ANDROID_HOME=%SDK%"
set "ANDROID_SDK_ROOT=%SDK%"
echo [SDK] %SDK%

if not exist "%SDK%\platforms\android-35\android.jar" (
    echo.
    echo [ERROR] Android SDK Platform 35 is not installed.
    echo Android Studio:
    echo   Tools ^> SDK Manager ^> Android 15 / API 35
    pause
    exit /b 1
)

rem ------------------------------------------------------------
rem Download a local Gradle distribution if needed.
rem No permanent system Gradle installation required.
rem ------------------------------------------------------------

if not exist "%LOCAL_GRADLE%\bin\gradle.bat" (
    echo.
    echo [1/3] Download Gradle %GRADLE_VERSION%...
    mkdir "%APP%\.gradle_dist" >nul 2>nul

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$zip='%APP%\.gradle_dist\gradle-%GRADLE_VERSION%-bin.zip';" ^
      "Invoke-WebRequest -UseBasicParsing 'https://services.gradle.org/distributions/gradle-%GRADLE_VERSION%-bin.zip' -OutFile $zip;" ^
      "Expand-Archive -Force $zip '%APP%\.gradle_dist';" ^
      "Remove-Item -Force $zip"
    if errorlevel 1 goto :error
) else (
    echo [1/3] Local Gradle already exists.
)

echo.
echo [2/3] Build debug APK...
pushd "%APP%"
call "%LOCAL_GRADLE%\bin\gradle.bat" --no-daemon assembleDebug
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" goto :error

set "APK=%APP%\app\build\outputs\apk\debug\app-debug.apk"

if not exist "%APK%" (
    echo [ERROR] Build finished but APK not found:
    echo   %APK%
    goto :error
)

echo.
echo [3/3] Copy APK for PC integration...
mkdir "%OUTDIR%" >nul 2>nul
copy /y "%APK%" "%OUTDIR%\S24NativeMic-debug.apk" >nul
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [OK] APK built:
echo   %OUTDIR%\S24NativeMic-debug.apk
echo.
echo Next:
echo   - Run the main VPA app
echo   - Phone Mic tab
echo   - Native Mic APK install
echo   - Native Mic launch / screen OFF
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Android APK build failed.
pause
exit /b 1
