# ==============================================================================
# AI Music Detector - Automated Dependency Installer & Environment Setup
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$CpuOnly = $false,
    [ValidateSet("cu121", "cu118")]
    [string]$CudaVersion = "cu121",
    [switch]$SkipPyTorch = $false,
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Helper display functions
function Write-Header($text) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "   $text" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($step, $text) {
    Write-Host "[$step] $text" -ForegroundColor Yellow
}

function Write-Success($text) {
    Write-Host "[OK] $text" -ForegroundColor Green
}

function Write-Warn($text) {
    Write-Host "[WARN] $text" -ForegroundColor Yellow
}

function Write-Err($text) {
    Write-Host "[ERROR] $text" -ForegroundColor Red
}

Write-Header "AI MUSIC DETECTOR - DEPENDENCY INSTALLER AND ENVIRONMENT SETUP"

# ------------------------------------------------------------------------------
# STEP 1: Locate or Create Virtual Environment
# ------------------------------------------------------------------------------
Write-Step "1/6" "Detecting Python and Virtual Environment..."

$PythonBin = $null
$PipBin = $null
$ActiveVenv = $null

if ($VenvPath -and (Test-Path "$VenvPath\Scripts\python.exe")) {
    $ActiveVenv = (Resolve-Path $VenvPath).Path
} elseif (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    $ActiveVenv = (Resolve-Path "$ScriptDir\.venv").Path
} elseif (Test-Path "$ScriptDir\..\..\.venv\Scripts\python.exe") {
    $ActiveVenv = (Resolve-Path "$ScriptDir\..\..\.venv").Path
}

if ($ActiveVenv) {
    $PythonBin = "$ActiveVenv\Scripts\python.exe"
    $PipBin = "$ActiveVenv\Scripts\pip.exe"
    Write-Success "Found existing virtual environment: $ActiveVenv"
} else {
    Write-Host "[*] No existing virtual environment found. Searching for system Python..."
    $SysPython = $null
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $SysPython = "python"
    } elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
        $SysPython = "py -3"
    }

    if (-not $SysPython) {
        Write-Err "Python was not found on your system PATH."
        Write-Host "Please install Python 3.10+ from https://www.python.org/ or via winget: winget install Python.Python.3.11"
        exit 1
    }

    $NewVenv = "$ScriptDir\.venv"
    Write-Host "[*] Creating new virtual environment at: $NewVenv" -ForegroundColor Cyan
    & $SysPython -m venv "$NewVenv"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$NewVenv\Scripts\python.exe")) {
        Write-Err "Failed to create virtual environment."
        exit 1
    }

    $ActiveVenv = (Resolve-Path $NewVenv).Path
    $PythonBin = "$ActiveVenv\Scripts\python.exe"
    $PipBin = "$ActiveVenv\Scripts\pip.exe"
    Write-Success "Created virtual environment: $ActiveVenv"
}

$PyVersion = & $PythonBin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Success "Python Executable: $PythonBin (v$PyVersion)"

# ------------------------------------------------------------------------------
# STEP 2: Check FFmpeg Availability
# ------------------------------------------------------------------------------
Write-Step "2/6" "Checking FFmpeg audio decoder..."

$FfmpegAvailable = $false
if (Get-Command "ffmpeg" -ErrorAction SilentlyContinue) {
    $FfmpegAvailable = $true
} elseif (Test-Path "$ActiveVenv\Scripts\ffmpeg.exe") {
    $FfmpegAvailable = $true
}

if ($FfmpegAvailable) {
    Write-Success "FFmpeg is available on system PATH."
} else {
    Write-Warn "FFmpeg was NOT detected in PATH."
    Write-Host "    FFmpeg is required for decoding MP3, M4A, FLAC, and multi-stem audio files." -ForegroundColor Gray
    Write-Host "    To install FFmpeg on Windows quickly:" -ForegroundColor Gray
    Write-Host "      winget install Gyan.FFmpeg" -ForegroundColor Cyan
    Write-Host "      or: choco install ffmpeg" -ForegroundColor Cyan
    Write-Host "    (The installer will continue, but install FFmpeg before processing MP3/stems)." -ForegroundColor Gray
}

# ------------------------------------------------------------------------------
# STEP 3: Upgrade Pip, Setuptools, and Wheel
# ------------------------------------------------------------------------------
Write-Step "3/6" "Upgrading pip, setuptools, and wheel..."
& $PythonBin -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Success "Package management tools upgraded."
} else {
    Write-Warn "Notice during pip upgrade; continuing..."
}

# ------------------------------------------------------------------------------
# STEP 4: PyTorch and Neural Backend (Demucs Engine)
# ------------------------------------------------------------------------------
Write-Step "4/6" "Configuring PyTorch and Neural Backend (Demucs Engine)..."

$TorchInstalled = $false
try {
    $TorchCheck = & $PythonBin -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}')" 2>$null
    if ($TorchCheck) {
        $Parts = $TorchCheck.Trim().Split('|')
        $TorchVer = $Parts[0]
        $CudaOk = $Parts[1]
        Write-Success "PyTorch $TorchVer already installed (CUDA Available: $CudaOk)."
        $TorchInstalled = $true
    }
} catch {
    $TorchInstalled = $false
}

if (-not $TorchInstalled -or (-not $SkipPyTorch -and -not $TorchInstalled)) {
    # Detect NVIDIA GPU
    $HasNvidia = $false
    if (-not $CpuOnly) {
        if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
            $HasNvidia = $true
        } else {
            $GpuList = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
            foreach ($gpu in $GpuList) {
                if ($gpu.Name -like "*NVIDIA*" -or $gpu.Name -like "*GeForce*" -or $gpu.Name -like "*RTX*" -or $gpu.Name -like "*GTX*") {
                    $HasNvidia = $true
                    break
                }
            }
        }
    }

    if ($HasNvidia -and -not $CpuOnly) {
        Write-Host "[*] NVIDIA GPU detected! Installing PyTorch with CUDA acceleration ($CudaVersion)..." -ForegroundColor Green
        $TorchIndexUrl = "https://download.pytorch.org/whl/$CudaVersion"
        & $PipBin install torch torchaudio --index-url $TorchIndexUrl
    } else {
        if ($CpuOnly) {
            Write-Host "[*] CPU-only mode selected. Installing standard PyTorch wheels..." -ForegroundColor Cyan
        } else {
            Write-Host "[*] No NVIDIA GPU detected. Installing standard CPU PyTorch wheels..." -ForegroundColor Cyan
        }
        & $PipBin install torch torchaudio
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Err "PyTorch installation encountered an error."
        exit 1
    }
    Write-Success "PyTorch installation complete."
}

# ------------------------------------------------------------------------------
# STEP 5: Install Requirements (Backend, DSP, TreeSHAP, Demucs)
# ------------------------------------------------------------------------------
Write-Step "5/6" "Installing core dependencies from requirements.txt..."

$ReqFile = "$ScriptDir\requirements.txt"
if (-not (Test-Path $ReqFile)) {
    $ReqFile = "$ScriptDir\backend\requirements.txt"
}

Write-Host "[*] Target requirements: $ReqFile" -ForegroundColor Cyan
& $PipBin install -r $ReqFile

if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to install dependencies from $ReqFile."
    exit 1
}
Write-Success "All requirements installed successfully."

# ------------------------------------------------------------------------------
# STEP 6: Verification and Diagnostic Check
# ------------------------------------------------------------------------------
Write-Step "6/6" "Running environment verification diagnostics..."

& $PythonBin "$ScriptDir\verify_env.py"

if ($LASTEXITCODE -ne 0) {
    Write-Err "Verification reported missing or incompatible dependencies."
    exit 1
}

# ------------------------------------------------------------------------------
# SUMMARY & NEXT STEPS
# ------------------------------------------------------------------------------
Write-Header "INSTALLATION COMPLETED SUCCESSFULLY!"

Write-Host "You are ready to run the AI Music Detector." -ForegroundColor Green
Write-Host ""
Write-Host "To launch the web dashboard and backend server, run:" -ForegroundColor Cyan
Write-Host "    .\start_app.ps1" -ForegroundColor Yellow
Write-Host "or double-click:" -ForegroundColor Cyan
Write-Host "    start_app.bat" -ForegroundColor Yellow
Write-Host ""
