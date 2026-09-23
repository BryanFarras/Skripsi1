@echo off
setlocal enabledelayedexpansion
title "AI Music Detector - Visualizer and Forensics Dashboard"
cd /d "%~dp0"

echo ======================================================================
echo    AI MUSIC DETECTOR - FORENSIC VISUALIZER LAUNCHER
echo ======================================================================
echo.

:: Detect Python executable
set "PYTHON_EXE="

:: 1. Try standard python in PATH
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :FOUND_PYTHON
)

:: 2. Try Python Launcher (py)
py -3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py -3"
    goto :FOUND_PYTHON
)

:: 3. Try bundled embedded python in workspace
if exist "..\ACE-Step-1.5\python_embeded\python.exe" (
    set "PYTHON_EXE=..\ACE-Step-1.5\python_embeded\python.exe"
    goto :FOUND_PYTHON
)

:: Python not found error
echo [ERROR] Python was not found on your system!
echo Please make sure Python is installed and added to your PATH.
echo.
pause
exit /b 1

:FOUND_PYTHON
echo [*] Using Python: %PYTHON_EXE%
echo [*] Starting backend server and launching browser...
echo.

%PYTHON_EXE% run_server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server exited with an error code: %ERRORLEVEL%
    pause
)
