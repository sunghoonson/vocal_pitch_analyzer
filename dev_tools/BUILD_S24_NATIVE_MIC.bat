@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=C:\dev\vocal_pitch_prototype_v1"
set "APP=%ROOT%\android\s24_native_mic"
set "OUTDIR=%ROOT%\tools\s24_native_mic"
set "GRADLE_VERSION=8.9"
set "LOCAL_GRADLE=%APP%\.gradle_dist\gradle-%GRADLE_VERSION%"

echo ============================================================
echo VPA S24 Native Mic - Android APK Build v4.4c
echo ============================================================
echo.

if not exist "%APP%\settings.gradle" (
    echo [ERROR] Android project not found:
    echo   %APP%
    exit /b 1
)

rem ------------------------------------------------------------
rem Java detection
rem
rem v4.4a had a CMD parser bug in the "for /f + PowerShell" Java
rem major-version extraction. The failure appeared as:
rem   '') was unexpected at this time.
rem
rem v4.4c removes that fragile parser completely.
rem Android Studio bundled JBR is preferred and Gradle/AGP will do
rem the authoritative Java compatibility check.
rem ------------------------------------------------------------

set "JAVA_EXE="
set "JAVA_SOURCE="

if defined JAVA_HOME (
    if exist "!JAVA_HOME!\bin\java.exe" (
        set "JAVA_EXE=!JAVA_HOME!\bin\java.exe"
        set "JAVA_SOURCE=JAVA_HOME"
    )
)

if not defined JAVA_EXE (
    if exist "C:\Program Files\Android\Android Studio\jbr\bin\java.exe" (
        set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
        set "JAVA_EXE=C:\Program Files\Android\Android Studio\jbr\bin\java.exe"
        set "JAVA_SOURCE=Android Studio JBR"
    )
)

if not defined JAVA_EXE (
    if exist "C:\Program Files\Android\Android Studio\jre\bin\java.exe" (
        set "JAVA_HOME=C:\Program Files\Android\Android Studio\jre"
        set "JAVA_EXE=C:\Program Files\Android\Android Studio\jre\bin\java.exe"
        set "JAVA_SOURCE=Android Studio JRE"
    )
)

if not defined JAVA_EXE (
    for /f "delims=" %%J in ('where java.exe 2^>nul') do (
        if not defined JAVA_EXE (
            set "JAVA_EXE=%%J"
            set "JAVA_SOURCE=PATH"
        )
    )
)

if not defined JAVA_EXE (
    echo [ERROR] Java not found.
    echo.
    echo Install Android Studio, then run this build again.
    exit /b 1
)

echo [JAVA SOURCE] !JAVA_SOURCE!
echo [JAVA] !JAVA_EXE!
"!JAVA_EXE!" -version
if errorlevel 1 goto :error

if not defined JAVA_HOME (
    for %%J in ("!JAVA_EXE!") do set "JAVA_BIN_DIR=%%~dpJ"
    for %%J in ("!JAVA_BIN_DIR!..") do set "JAVA_HOME=%%~fJ"
)

echo [JAVA_HOME] !JAVA_HOME!

rem ------------------------------------------------------------
rem Android SDK detection
rem ------------------------------------------------------------

set "SDK="

if defined ANDROID_SDK_ROOT (
    if exist "!ANDROID_SDK_ROOT!\platforms" (
        set "SDK=!ANDROID_SDK_ROOT!"
    )
)

if not defined SDK (
    if defined ANDROID_HOME (
        if exist "!ANDROID_HOME!\platforms" (
            set "SDK=!ANDROID_HOME!"
        )
    )
)

if not defined SDK (
    if exist "%LOCALAPPDATA%\Android\Sdk\platforms" (
        set "SDK=%LOCALAPPDATA%\Android\Sdk"
    )
)

if not defined SDK (
    echo.
    echo [ERROR] Android SDK not found.
    echo.
    echo Open Android Studio once and install:
    echo   Android SDK Platform 35
    echo.
    echo Default expected path:
    echo   %LOCALAPPDATA%\Android\Sdk
    exit /b 1
)

set "ANDROID_HOME=!SDK!"
set "ANDROID_SDK_ROOT=!SDK!"

echo [SDK] !SDK!

if not exist "!SDK!\platforms\android-35\android.jar" (
    echo.
    echo [ERROR] Android SDK Platform 35 is not installed.
    echo.
    echo Android Studio:
    echo   Tools ^> SDK Manager
    echo   SDK Platforms ^> Android 15.0 / API 35
    exit /b 1
)

rem ------------------------------------------------------------
rem Write local.properties explicitly.
rem This avoids Gradle SDK-path ambiguity.
rem ------------------------------------------------------------

set "SDK_ESCAPED=!SDK:\=\\!"
> "!APP!\local.properties" echo sdk.dir=!SDK_ESCAPED!

echo [LOCAL.PROPERTIES] !APP!\local.properties

rem ------------------------------------------------------------
rem Local Gradle distribution
rem ------------------------------------------------------------

if not exist "!LOCAL_GRADLE!\bin\gradle.bat" (
    echo.
    echo [1/3] Download Gradle !GRADLE_VERSION!...
    mkdir "!APP!\.gradle_dist" >nul 2>nul

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$zip='!APP!\.gradle_dist\gradle-!GRADLE_VERSION!-bin.zip';" ^
      "Invoke-WebRequest -UseBasicParsing 'https://services.gradle.org/distributions/gradle-!GRADLE_VERSION!-bin.zip' -OutFile $zip;" ^
      "Expand-Archive -Force $zip '!APP!\.gradle_dist';" ^
      "Remove-Item -Force $zip"
    if errorlevel 1 goto :error
) else (
    echo [1/3] Local Gradle already exists.
)

echo.
echo [2/3] Build debug APK...
pushd "!APP!"
call "!LOCAL_GRADLE!\bin\gradle.bat" --no-daemon assembleDebug
set "RC=!ERRORLEVEL!"
popd

if not "!RC!"=="0" goto :error

set "APK=!APP!\app\build\outputs\apk\debug\app-debug.apk"

if not exist "!APK!" (
    echo [ERROR] Build finished but APK not found:
    echo   !APK!
    goto :error
)

echo.
echo [3/3] Copy APK for PC integration...
mkdir "!OUTDIR!" >nul 2>nul
copy /y "!APK!" "!OUTDIR!\S24NativeMic-debug.apk" >nul
if errorlevel 1 goto :error

echo.
echo ============================================================
echo [OK] APK built:
echo   !OUTDIR!\S24NativeMic-debug.apk
echo ============================================================
exit /b 0

:error
echo.
echo [ERROR] Android APK build failed.
exit /b 1
