# ==============================================================================
# AI Music Detector - Automated Dependency Installer & Environment Setup
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$CpuOnly = $false,
    [ValidateSet("cu121", "cu124", "cu118")]
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

function Get-CompatiblePythonCommand {
    $candidates = @(
        @{ Exe = "py"; Arg = "-3.12" },
        @{ Exe = "py"; Arg = "-3.11" },
        @{ Exe = "py"; Arg = "-3.10" },
        @{ Exe = "python3.12"; Arg = "" },
        @{ Exe = "python3.11"; Arg = "" },
        @{ Exe = "python3.10"; Arg = "" },
        @{ Exe = "python"; Arg = "" },
        @{ Exe = "py"; Arg = "-3" }
    )

    foreach ($c in $candidates) {
        $exe = $c.Exe
        $arg = $c.Arg
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }

        try {
            $ver = if ($arg) {
                & $exe $arg -c "import sys; print(sys.version.split()[0])" 2>$null
            } else {
                & $exe -c "import sys; print(sys.version.split()[0])" 2>$null
            }

            if ($ver) {
                $ver = $ver.Trim()
                $parts = $ver.Split('.')
                if ($parts.Count -ge 2) {
                    $maj = [int]$parts[0]
                    $min = [int]$parts[1]
                    if ($maj -eq 3 -and $min -ge 10 -and $min -le 12) {
                        return @{
                            Exe = $exe
                            Arg = $arg
                            Version = $ver
                        }
                    }
                }
            }
        } catch {}
    }
    return $null
}

Write-Header "AI MUSIC DETECTOR - DEPENDENCY INSTALLER AND ENVIRONMENT SETUP"

# ------------------------------------------------------------------------------
# STEP 1: Locate or Create Virtual Environment
# ------------------------------------------------------------------------------
Write-Step "1/6" "Detecting Python and Virtual Environment..."

$PythonBin = $null
$PipBin = $null
$ActiveVenv = $null

# Check potential existing virtual environments
$CandidatesVenv = @()
if ($VenvPath) { $CandidatesVenv += $VenvPath }
if (Test-Path "$ScriptDir\.venv\Scripts\python.exe") { $CandidatesVenv += "$ScriptDir\.venv" }
if (Test-Path "$ScriptDir\..\..\.venv\Scripts\python.exe") { $CandidatesVenv += "$ScriptDir\..\..\.venv" }

foreach ($v in $CandidatesVenv) {
    if (-not $v) { continue }
    $pyCandidate = "$v\Scripts\python.exe"
    if (Test-Path $pyCandidate) {
        try {
            $ver = & $pyCandidate -c "import sys; print(sys.version.split()[0])" 2>$null
            if ($ver) {
                $ver = $ver.Trim()
                $parts = $ver.Split('.')
                $maj = [int]$parts[0]
                $min = [int]$parts[1]
                if ($maj -eq 3 -and $min -ge 10 -and $min -le 12) {
                    $ActiveVenv = (Resolve-Path $v).Path
                    Write-Success "Found compatible virtual environment: $ActiveVenv (Python v$ver)"
                    break
                } else {
                    Write-Warn "Virtual environment at $v uses Python v$ver (incompatible: requires Python 3.10 - 3.12)."
                }
            }
        } catch {}
    }
}

if ($ActiveVenv) {
    $PythonBin = "$ActiveVenv\Scripts\python.exe"
    $PipBin = "$ActiveVenv\Scripts\pip.exe"
} else {
    Write-Host "[*] Searching for a compatible system Python (Python 3.10, 3.11, or 3.12)..." -ForegroundColor Cyan
    $CompatPython = Get-CompatiblePythonCommand

    if ($CompatPython) {
        $cExe = $CompatPython.Exe
        $cArg = $CompatPython.Arg
        $cVer = $CompatPython.Version
        Write-Success "Found compatible system Python: $cExe $cArg (v$cVer)"

        $NewVenv = "$ScriptDir\.venv"
        if (Test-Path $NewVenv) {
            Write-Host "[*] Removing incompatible virtual environment at: $NewVenv" -ForegroundColor Yellow
            try {
                Remove-Item -Recurse -Force $NewVenv -ErrorAction SilentlyContinue
            } catch {
                Write-Warn "Could not automatically remove old .venv. Please close running Python processes and try again."
            }
        }

        Write-Host "[*] Creating new virtual environment with Python v$cVer at: $NewVenv" -ForegroundColor Cyan
        if ($cArg) {
            & $cExe $cArg -m venv "$NewVenv"
        } else {
            & $cExe -m venv "$NewVenv"
        }

        if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$NewVenv\Scripts\python.exe")) {
            Write-Err "Failed to create virtual environment using $cExe $cArg."
            exit 1
        }

        $ActiveVenv = (Resolve-Path $NewVenv).Path
        $PythonBin = "$ActiveVenv\Scripts\python.exe"
        $PipBin = "$ActiveVenv\Scripts\pip.exe"
        Write-Success "Created compatible virtual environment: $ActiveVenv"
    } else {
        # Check whatever system python is installed to show a helpful message
        $AnyVer = $null
        try {
            $AnyVer = & python -c "import sys; print(sys.version.split()[0])" 2>$null
        } catch {}
        if (-not $AnyVer) {
            try { $AnyVer = & py -c "import sys; print(sys.version.split()[0])" 2>$null } catch {}
        }

        Write-Host ""
        Write-Err "Incompatible Python version detected ($AnyVer)."
        Write-Host "    Machine Learning and Audio DSP libraries (PyTorch, TorchAudio, Demucs," -ForegroundColor Yellow
        Write-Host "    Librosa, SHAP, and Numba) currently require Python 3.10, 3.11, or 3.12." -ForegroundColor Yellow
        Write-Host "    Pre-compiled C++/CUDA wheels do NOT exist for Python 3.13 or 3.14 yet." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "HOW TO RESOLVE ON WINDOWS (Takes 1-2 minutes):" -ForegroundColor Cyan
        Write-Host "  1. Open PowerShell or Command Prompt." -ForegroundColor Cyan
        Write-Host "  2. Install Python 3.12 via Windows Package Manager:" -ForegroundColor Cyan
        Write-Host "         winget install Python.Python.3.12" -ForegroundColor Green
        Write-Host "     (or download the official installer: https://www.python.org/downloads/release/python-3129/)" -ForegroundColor Gray
        Write-Host "  3. If an incompatible .venv exists, delete it:" -ForegroundColor Cyan
        Write-Host "         Remove-Item -Recurse -Force .venv" -ForegroundColor Green
        Write-Host "  4. Re-run this installer:" -ForegroundColor Cyan
        Write-Host "         .\install_dependencies.ps1" -ForegroundColor Green
        Write-Host ""
        exit 1
    }
}

$PyVersion = & $PythonBin -c "import sys; print(sys.version.split()[0])"
Write-Success "Active Python: $PythonBin (v$PyVersion)"

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
    $TorchCheck = & $PythonBin -c "import torch; print(str(torch.__version__) + '|' + str(torch.cuda.is_available()))" 2>$null
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

    $TorchInstalledSuccessfully = $false

    if ($HasNvidia -and -not $CpuOnly) {
        $CudaTargets = @($CudaVersion, "cu124", "cu118")
        foreach ($target in $CudaTargets) {
            Write-Host "[*] NVIDIA GPU detected! Attempting PyTorch with CUDA acceleration ($target)..." -ForegroundColor Green
            $TorchIndexUrl = "https://download.pytorch.org/whl/$target"
            & $PipBin install torch torchaudio --index-url $TorchIndexUrl
            if ($LASTEXITCODE -eq 0) {
                $TorchInstalledSuccessfully = $true
                break
            } else {
                Write-Warn "PyTorch install with index $target failed; trying next candidate..."
            }
        }
    }

    if (-not $TorchInstalledSuccessfully) {
        Write-Host "[*] Installing standard PyTorch wheels from PyPI..." -ForegroundColor Cyan
        & $PipBin install torch torchaudio
        if ($LASTEXITCODE -eq 0) {
            $TorchInstalledSuccessfully = $true
        }
    }

    if (-not $TorchInstalledSuccessfully) {
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
