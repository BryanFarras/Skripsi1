"""
Baseline AI Audio Classifier Configuration
Defines dataset locations, audio parameters, feature extraction specs, and model paths.
"""

import os
from pathlib import Path

# Base Paths
ML_BASELINE_DIR = Path(__file__).resolve().parent
APP_ROOT = ML_BASELINE_DIR.parent
WORKSPACE_ROOT = APP_ROOT.parent.parent

# Datasets Paths
DATASETS_DIR = WORKSPACE_ROOT / "Datasets"
SUNO_DIR = DATASETS_DIR / "suno" / "original"
SUNO_SPLITS_DIR = DATASETS_DIR / "suno" / "splits"
MUSDB18_DIR = DATASETS_DIR / "musdb18" / "train"

# Virtual Environment & Tools
VENV_PYTHON = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
FFMPEG_EXE = WORKSPACE_ROOT / ".venv" / "Scripts" / "ffmpeg.exe"

# Storage Directories
DATA_DIR = ML_BASELINE_DIR / "data"
MODELS_DIR = ML_BASELINE_DIR / "models"
REPORTS_DIR = ML_BASELINE_DIR / "reports"

# Audio Slicing Parameters
SAMPLE_RATE = 22050          # Consistent sample rate for forensic DSP extraction
SLICE_DURATION = 5.0         # 5 seconds per slice
SLICE_HOP = 2.5              # 50% overlap for comprehensive temporal coverage
MIN_ENERGY_THRESHOLD = 0.005 # Skip silent / empty lead-in or lead-out segments

# Feature Extraction Parameters
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_CONTRAST_BANDS = 6

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Label Definitions
LABEL_BONAFIDE = 0   # Human / Real studio music
LABEL_SPOOF = 1      # AI-Generated music
LABEL_NAMES = {0: "Bona-fide (Human)", 1: "Spoof (AI-Generated)"}
