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
if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
    $PythonCmd = "$PSScriptRoot\.venv\Scripts\python.exe"
} elseif (Test-Path "$PSScriptRoot\..\..\.venv\Scripts\python.exe") {
    $PythonCmd = "$PSScriptRoot\..\..\.venv\Scripts\python.exe"
} elseif (Test-Path "..\..\.venv\Scripts\python.exe") {
    $PythonCmd = "..\..\.venv\Scripts\python.exe"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    # Check py -3.12, py -3.11, py -3.10 first
    $pyCheck = & py -3.12 -c "import sys; print(1)" 2>$null
    if ($pyCheck -eq "1") {
        $PythonCmd = "py -3.12"
    } else {
        $PythonCmd = "py -3"
    }
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
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
