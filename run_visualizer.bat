@echo off
title AI Music Detector - Visualizer & Backend
cd /d "%~dp0"

echo ===================================================
echo   AI MUSIC DETECTOR - FORENSIC VISUALIZER LAUNCHER
echo ===================================================
echo.
echo Launching local server and web dashboard...
echo.

python run_server.py

pause
