@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

rem ============================================================
rem Vocal Pitch Analyzer - Git Commit / Push Helper
rem Repository:
rem   C:\dev\vocal_pitch_prototype_v1
rem Expected remote:
rem   origin -> https://github.com/sunghoonson/vocal_pitch_analyzer.git
rem
rem Commit title: yyyy-MM-dd HH:mm:ss
rem Commit body : optional one-line input
rem Push target : currently checked-out branch
rem ============================================================

set "EXPECTED_REMOTE_URL=https://github.com/sunghoonson/vocal_pitch_analyzer.git"

rem Prevent Git from opening Vim/editor.
set "GIT_EDITOR=true"
set "VISUAL=true"
set "EDITOR=true"
set "GIT_SEQUENCE_EDITOR=true"

cd /d "%~dp0" 2>nul

git --version >nul 2>&1
if errorlevel 1 goto GIT_NOT_FOUND

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 goto NOT_GIT_REPO

for /f "delims=" %%R in ('git rev-parse --show-toplevel') do set "REPO_ROOT=%%R"
cd /d "%REPO_ROOT%"

rem ------------------------------------------------------------
rem Remote repair / setup
rem ------------------------------------------------------------

git remote get-url origin >nul 2>&1
if not errorlevel 1 goto ORIGIN_READY

rem Earlier this project accidentally used "origine".
git remote get-url origine >nul 2>&1
if errorlevel 1 goto ADD_ORIGIN

echo [INFO] Remote "origine" found.
echo [INFO] Renaming "origine" to "origin"...
git remote rename origine origin
if errorlevel 1 goto REMOTE_FAILED
goto ORIGIN_READY

:ADD_ORIGIN
echo [INFO] Remote "origin" is missing.
echo [INFO] Adding:
echo        %EXPECTED_REMOTE_URL%
git remote add origin "%EXPECTED_REMOTE_URL%"
if errorlevel 1 goto REMOTE_FAILED

:ORIGIN_READY
for /f "delims=" %%U in ('git remote get-url origin') do set "REMOTE_URL=%%U"

echo.
echo [INFO] Remote:
echo        origin = %REMOTE_URL%

rem ------------------------------------------------------------
rem Branch
rem ------------------------------------------------------------

set "BRANCH="
for /f "delims=" %%B in ('git symbolic-ref --quiet --short HEAD 2^>nul') do set "BRANCH=%%B"

if not defined BRANCH goto DETACHED_HEAD

rem Remove stale Vim swap file from an old aborted commit.
if exist "%REPO_ROOT%\.git\.COMMIT_EDITMSG.swp" (
    echo [INFO] Removing stale Vim commit swap:
    echo        %REPO_ROOT%\.git\.COMMIT_EDITMSG.swp
    del /f /q "%REPO_ROOT%\.git\.COMMIT_EDITMSG.swp" >nul 2>&1
)

rem Commit title: current date/time.
for /f "delims=" %%M in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"') do set "COMMIT_TITLE=%%M"

echo.
echo ============================================================
echo [INFO] Repository : %REPO_ROOT%
echo [INFO] Branch     : %BRANCH%
echo [INFO] Remote     : origin
echo [INFO] Commit     : %COMMIT_TITLE%
echo ============================================================
echo.

echo [STEP 1] git status --short
git status --short

echo.
echo [STEP 2] git add -A
git add -A
if errorlevel 1 goto ADD_FAILED

rem Check whether there is anything staged.
git diff --cached --quiet
if errorlevel 1 goto DO_COMMIT

echo [INFO] No staged changes. Commit skipped.
goto PUSH_CURRENT_BRANCH


:DO_COMMIT
echo.
echo [INPUT] Optional commit body.
echo         Type/paste one line and press Enter.
echo         Press Enter immediately to use only the timestamp title.
echo.

set "COMMIT_BODY="
set /p "COMMIT_BODY=> "

echo.
echo [STEP 3] git commit
echo [INFO] Title: %COMMIT_TITLE%

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $title=$env:COMMIT_TITLE; $body=$env:COMMIT_BODY; if ([string]::IsNullOrWhiteSpace($body)) { & git -c core.editor=true commit -m $title } else { & git -c core.editor=true commit -m $title -m $body }; exit $LASTEXITCODE"

if errorlevel 1 goto COMMIT_FAILED


:PUSH_CURRENT_BRANCH
echo.
echo [STEP 4] git push -u origin HEAD:%BRANCH%
git push -u origin "HEAD:%BRANCH%"
if errorlevel 1 goto PUSH_FAILED

echo.
echo ============================================================
echo [OK] Commit / push completed.
echo [OK] Repository : %REPO_ROOT%
echo [OK] Branch     : %BRANCH%
echo [OK] Remote     : origin
echo [OK] Title      : %COMMIT_TITLE%
echo ============================================================
pause
exit /b 0


:GIT_NOT_FOUND
echo [ERROR] Git is not installed or is not in PATH.
pause
exit /b 1


:NOT_GIT_REPO
echo [ERROR] This folder is not a Git repository.
echo        %CD%
pause
exit /b 1


:DETACHED_HEAD
echo [ERROR] Could not detect the current branch.
echo         You may be in detached HEAD state.
echo.
git status
echo.
pause
exit /b 1


:REMOTE_FAILED
echo [ERROR] Could not configure the origin remote.
echo.
git remote -v
echo.
pause
exit /b 1


:ADD_FAILED
echo [ERROR] git add failed.
pause
exit /b 1


:COMMIT_FAILED
echo [ERROR] git commit failed.
pause
exit /b 1


:PUSH_FAILED
echo.
echo [ERROR] git push failed.
echo.
echo [INFO] Current remote:
git remote -v
echo.
echo Possible causes:
echo   - GitHub authentication/token required
echo   - network problem
echo   - remote branch has commits not present locally
echo.
pause
exit /b 1
