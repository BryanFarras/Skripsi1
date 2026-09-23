# AI Music Detector - PowerShell Launcher
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   AI MUSIC DETECTOR - FORENSIC VISUALIZER LAUNCHER" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Find Python
$PythonCmd = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PythonCmd = "py -3"
} elseif (Test-Path "..\ACE-Step-1.5\python_embeded\python.exe") {
    $PythonCmd = "..\ACE-Step-1.5\python_embeded\python.exe"
}

if (-not $PythonCmd) {
    Write-Host "[ERROR] Python was not found on your system." -ForegroundColor Red
    Write-Host "Please ensure Python 3 is installed."
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host "[*] Using Python: $PythonCmd" -ForegroundColor Green
Write-Host "[*] Starting server and launching browser..." -ForegroundColor Green
Write-Host ""

& $PythonCmd run_server.py
