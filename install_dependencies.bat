@echo off
setlocal
title "AI Music Detector - Automated Dependency Installer"
cd /d "%~dp0"

echo ======================================================================
echo    AI MUSIC DETECTOR - DEPENDENCY INSTALLER LAUNCHER
echo ======================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_dependencies.ps1" %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Installation exited with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
pause
