# AI Music Detector: Explainable AI & Multi-Stem Audio Forensic System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Explainable AI](https://img.shields.io/badge/XAI-TreeSHAP%20Enabled-4285F4?logo=google&logoColor=white)](https://shap.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Demucs%20v4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

An end-to-end, scientifically validated audio forensic framework and REST service designed to detect **AI-generated synthetic music** (_Spoof_, e.g., Suno AI) versus **authentic human studio master recordings** (_Bona-fide_, e.g., MUSDB18).

Unlike conventional "black-box" deep learning models, this system provides full transparency using **110-dimensional Digital Signal Processing (DSP) acoustic feature extraction** paired with **Explainable AI (TreeSHAP)**, producing quantitative feature attributions and natural-language forensic diagnostic summaries suitable for academic defense and forensic auditing.

---

## Table of Contents

1. [Key Capabilities & Innovations](#key-capabilities--innovations)
2. [Academic Benchmarks & Evaluation](#academic-benchmarks--evaluation)
3. [110-D Acoustic Feature Taxonomy](#110-d-acoustic-feature-taxonomy)
4. [Explainable AI (TreeSHAP) Diagnostics](#explainable-ai-treeshap-diagnostics)
5. [Repository Structure](#repository-structure)
6. [Quickstart & Installation](#quickstart--installation)
7. [Running the Application](#running-the-application)
8. [Machine Learning Workflow (CLI)](#machine-learning-workflow-cli)
9. [Stem Separation Tool](#stem-separation-tool)
10. [REST API Documentation](#rest-api-documentation)
11. [Citation & Academic Context](#citation--academic-context)

---

## Key Capabilities & Innovations

- **Inherent Domain Interpretability**: Extracts 110 physical acoustic signal dimensions per 5.0-second sliding window with 50% overlap.
- **Strict Anti-Leakage Partitioning**: Uses `GroupShuffleSplit` by `track_id` so that temporal slices from any song are strictly isolated to either the training set or the test set—guaranteeing 0% data leakage.
- **Multi-Model Benchmarking**: Compares Random Forest, Support Vector Machine (RBF), Logistic Regression, and Multi-Layer Perceptron (MLP).
- **Explainable AI (XAI)**: Native TreeSHAP integration computes exact additive Shapley values $\phi_i(x)$ for every audio frame, explaining precisely _why_ a song is flagged as AI or Human.
- **Natural Language Forensic Synthesis**: Converts numerical Shapley attributions into structured, human-readable forensic diagnostic reports.
- **Multi-Stem Decomposition**: Integrates Meta's Demucs v4 to break down songs into 5 individual stems (`mixture`, `vocals`, `drums`, `bass`, `other`) for component-wise forensic auditing.
- **FastAPI Production Backend**: High-throughput asynchronous service featuring multi-format audio decoding (WAV, MP3, FLAC, M4A), caching, and high-resolution figure exports.

---

## Academic Benchmarks & Evaluation

The champion classifier was evaluated on a held-out test set under strict track-level grouping:

| Classifier Model                 |  Accuracy  |  F1-Score  |   ROC-AUC   | Equal Error Rate (EER) | Training Time |
| :------------------------------- | :--------: | :--------: | :---------: | :--------------------: | :-----------: |
| **Random Forest (Champion)**     | **96.67%** | **97.73%** | **100.00%** |       **0.00%**        |    0.176 s    |
| **Support Vector Machine (RBF)** |   95.83%   |   97.14%   |   99.70%    |         5.56%          |    0.018 s    |
| **Logistic Regression (L2)**     |   96.67%   |   97.73%   |   99.07%    |         3.89%          |    0.005 s    |
| **Multi-Layer Perceptron (MLP)** |   85.83%   |   89.70%   |   91.52%    |         12.78%         |    0.070 s    |

> **Forensic Significance of EER = 0.00%**: In international biometric and audio deepfake benchmarks (e.g., ASVspoof), Equal Error Rate (EER) is the gold standard operating point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR). A 0.00% EER represents complete discriminative separation between bona-fide and synthetic tracks on held-out test data.

---

## 110-D Acoustic Feature Taxonomy

Implemented in [`ml_baseline/features.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/features.py):

| Feature Category                  |  Dims   | Parameters                             | Forensic Significance                                                                                                                         |
| :-------------------------------- | :-----: | :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **MFCC (1–20)**                   |   40    | `mfcc_1..20_mean`, `std`               | Spectral envelope and timbral characteristics. AI voice/instrument generators display distinct vocoder artifacts in high-order MFCCs.         |
| **Delta MFCC (1–20)**             |   40    | `delta_mfcc_1..20_mean`, `std`         | Velocity of timbral transitions. Uncovers phase inconsistencies and unnatural micro-transient jumps in latent diffusion models.               |
| **Spectral Contrast (7 Bands)**   |   14    | `spec_contrast_b0..b6_mean`, `std`     | Peak-to-valley energy difference across octave bands. **Band 6 (>7.7 kHz) is the #1 most discriminative feature** for AI vocoder compression. |
| **Spectral Rolloff (85% & 95%)**  |    4    | `spec_rolloff85..95_mean`, `std`       | Energy cutoff boundary frequencies. Accurately detects vocoder _brickwall cutoffs_ between 14 kHz and 18 kHz.                                 |
| **Spectral Centroid & Bandwidth** |    4    | `spec_centroid_mean..std`, `bandwidth` | Center of mass and frequency dispersion of the spectrum. Human studio masters have wider, more natural dynamic dispersion.                    |
| **Spectral Flatness**             |    2    | `spec_flatness_mean`, `std`            | Ratio of geometric to arithmetic mean of power. Flags background phase smearing and vocoder hiss.                                             |
| **Zero-Crossing Rate (ZCR)**      |    2    | `zcr_mean`, `zcr_std`                  | Noise density and percussive transient crispness.                                                                                             |
| **RMS Energy**                    |    2    | `rms_mean`, `rms_std`                  | Dynamic loudness variation and amplitude envelope.                                                                                            |
| **High-Frequency Energy Ratio**   |    2    | `hf_ratio_mean`, `std`                 | Power ratio above 7.7 kHz vs total spectrum. Validates presence of natural "Air Band" frequencies.                                            |
| **TOTAL FEATURE DIMENSIONS**      | **110** |                                        | Extracted per 5.0-second audio segment                                                                                                        |

### Top Acoustic Discriminators (Gini Feature Importance)

1. **`spec_contrast_b6_mean` (16.93%)**: Octave band 6 (>7.7 kHz) peak-to-valley contrast anomaly.
2. **`mfcc_17_mean` (10.61%)**: High-order harmonic resonance envelope distortion.
3. **`spec_rolloff85_mean` (5.76%)**: 85% energy concentration brickwall cutoff.
4. **`mfcc_16_mean` (4.68%)**: Mid-high timbral modulation artifact.
5. **`mfcc_2_mean` (4.34%)**: Spectral tilt (bass vs. treble ratio).

---

## Explainable AI (TreeSHAP) Diagnostics

Implemented in [`ml_baseline/explainability.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/explainability.py):

Based on cooperative game theory (Lundberg & Lee, _Nature Machine Intelligence_ 2020), every test segment $x$ is explained as:

$$f(x) = \phi_0 + \sum_{i=1}^{110} \phi_i(x)$$

- $\phi_0$: Population baseline expectation (~0.50).
- $\phi_i(x) > 0$: Acoustic dimensions pushing the prediction towards **AI-Generated (Spoof)**.
- $\phi_i(x) < 0$: Acoustic dimensions proving authentic **Human-Made (Bona-fide)** acoustics.

### Sample Automated Forensic Diagnosis Output:

```json
{
  "verdict": "SPOOF (AI-GENERATED)",
  "confidence_score": 0.631,
  "base_value": 0.5,
  "top_ai_pushing_features": [
    {
      "feature": "spec_contrast_b6_mean",
      "shap_value": 0.1602,
      "raw_value": 63.81,
      "clinical_explanation": "Severe high-frequency (>7.7kHz) contrast compression typical of neural latent diffusion."
    },
    {
      "feature": "mfcc_17_mean",
      "shap_value": 0.0855,
      "raw_value": 16.82,
      "clinical_explanation": "Micro-harmonic spectral envelope distortion from neural vocoding."
    },
    {
      "feature": "spec_rolloff85_mean",
      "shap_value": 0.0362,
      "raw_value": 7030.5,
      "clinical_explanation": "Premature high-frequency roll-off compared to professional studio master recordings."
    }
  ]
}
```

---

## Repository Structure

```
MY-AIDETECTOR/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application & startup lifecycle
│   │   ├── config.py                   # Paths, DSP constants, & environment configs
│   │   ├── api/
│   │   │   └── routes.py               # REST endpoints (/api/upload, /api/analyze-local-path, /api/explain)
│   │   ├── services/
│   │   │   ├── audio_loader.py         # Multi-format decoder (WAV, MP3, FLAC, M4A via ffmpeg)
│   │   │   ├── detector_service.py     # DSP heuristics & spectral analysis
│   │   │   ├── stem_service.py         # Neural source separation adapter (Demucs)
│   │   │   └── visualizer_service.py   # Computes FFT, Tonal Balance, & STFT
│   │   └── models/
│   │       └── schemas.py              # Pydantic schemas (Detection, XAI attributions, Stems)
│   ├── exports/                        # High-resolution exported figures (PNG)
│   ├── uploads/                        # Temporary cached audio files (.gitkeep)
│   ├── stems_temp/                     # Temporary decomposed stems (.gitkeep)
│   └── requirements.txt                # Backend dependencies
├── ml_baseline/
│   ├── config.py                       # Audio hyperparameters (sample_rate=22050, slice_sec=5.0)
│   ├── features.py                     # 110-dimensional DSP feature extraction engine
│   ├── dataset.py                      # Temporal slicing & track-level group dataset builder
│   ├── train.py                        # Model trainer & multi-classifier benchmarker
│   ├── evaluate.py                     # EER, ROC-AUC, and Confusion Matrix reporter
│   ├── explainability.py               # TreeSHAP explainer engine & natural language generator
│   ├── predict.py                      # Sliding-window inference CLI & Python API
│   ├── baseline_adapter.py             # FastAPI adapter for seamless ML integration
│   ├── data/                           # Tabular dataset (CSV, NPZ, metadata.json)
│   ├── models/                         # Serialized models (.joblib) & metrics.json
│   └── reports/                        # Markdown benchmark reports
├── misc/
│   ├── app document/                   # Research notes, model guides, XAI documentation
├── split_stems.py                      # Standalone 5-stem neural separation CLI
├── run_server.py                       # Python launcher with automatic browser opening
├── run_visualizer.bat                  # Windows batch launcher
├── start_app.bat                       # One-click Windows starter
├── start_app.ps1                       # PowerShell launcher
├── test_backend.py                     # Automated end-to-end backend test suite
└── README.md
```

---

## Quickstart & Installation

### 1. Prerequisites

- **Python 3.10+** (tested on Python 3.11, 3.12, and 3.13)
- **FFmpeg** installed and accessible on system `PATH` (mandatory for decoding MP3, M4A, FLAC, and STEM.MP4 containers).
- _(Optional, Highly Recommended)_ **NVIDIA GPU with CUDA 11.8+ / 12.1+** for GPU-accelerated stem separation via Demucs.

### 2. Environment Setup

Follow the steps below to configure a complete virtual environment with all core, forensic ML/XAI, and stem separation dependencies:

#### Step A: Clone & Navigate

```powershell
git clone https://github.com/BryanFarras/Skripsi1.git
```

#### Step B: Create & Activate Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS (Bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step C: Install FFmpeg (If not already installed)

- **Windows (winget):** `winget install Gyan.FFmpeg`
- **Windows (Chocolatey):** `choco install ffmpeg`
- **macOS (Homebrew):** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt update && sudo apt install -y ffmpeg`

_Verify by running `ffmpeg -version` in your terminal._

#### Step D: Install PyTorch (Demucs & Neural Processing Backend)

Choose the command matching your hardware setup:

- **NVIDIA GPU (CUDA 12.1+ - Recommended):**
  ```powershell
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **NVIDIA GPU (CUDA 11.8):**
  ```powershell
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **CPU-Only (No NVIDIA GPU):**
  ```powershell
  pip install torch torchaudio
  ```

#### Step E: Install Application, ML/XAI, & Demucs Dependencies

Install the backend server, signal processing, machine learning, explainability engine (TreeSHAP), and stem separation tool:

```powershell
# 1. Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# 2. Install FastAPI backend dependencies
pip install -r backend/requirements.txt

# 3. Install ML, DSP, Feature Extraction & TreeSHAP XAI
pip install scikit-learn librosa scipy joblib soundfile matplotlib pandas shap

# 4. Install Meta Demucs for 5-stem neural separation
pip install demucs
```

#### Step F: Verify Environment Installation

Run this one-liner to verify that all core modules and hardware acceleration are properly detected:

```powershell
python -c "import torch, demucs, shap, librosa, fastapi; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print('Environment verification successful!')"
```

---

## Running the Application

### Option A: One-Click Launcher (Windows)

Double-click **`start_app.bat`** or **`run_visualizer.bat`**.

### Option B: PowerShell

```powershell
.\start_app.ps1
```

### Option C: Python Command Line

```powershell
python run_server.py
```

- The backend server will initialize on `http://localhost:8000`.
- The interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Machine Learning Workflow (CLI)

The machine learning module is completely self-contained within `ml_baseline/`:

### 1. Extract 110-D Features & Build Dataset

```bash
python ml_baseline/dataset.py --bona-dir path/to/musdb18 --spoof-dir path/to/suno_ai
```

### 2. Train Classifiers & Benchmark

```bash
python ml_baseline/train.py
```

_Trains Random Forest, SVM, Logistic Regression, and MLP with strict 75/25 track-level grouping. Automatically saves `baseline_classifier.joblib`, `feature_scaler.joblib`, and `metrics.json`._

### 3. Generate Evaluation Report

```bash
python ml_baseline/evaluate.py
```

### 4. Run Sliding-Window Inference & XAI on Any Song

```bash
# Predict classification verdict
python ml_baseline/predict.py path/to/mystery_song.mp3

# Predict with detailed TreeSHAP local attribution explanation
python ml_baseline/predict.py path/to/mystery_song.mp3 --explain
```

---

## Stem Separation Tool (`split_stems.py`)

Split any audio track into 5 discrete stems (`mixture`, `vocals`, `drums`, `bass`, `other`) using Meta's Demucs model:

```bash
# Single audio file (generates 'song_stems/' next to input file)
python split_stems.py path/to/song.wav

# Specify custom output directory
python split_stems.py path/to/song.mp3 --output-dir my_stems/

# Batch process an entire directory of audio files
python split_stems.py --batch-dir Datasets/raw/ai_suno --output-dir Datasets/stems/ai_suno
```

---

## REST API Documentation

### Interactive Documentation

With the server running, navigate to:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints

| Method | Endpoint                  | Description                                                                                               |
| :----- | :------------------------ | :-------------------------------------------------------------------------------------------------------- |
| `POST` | `/api/upload`             | Upload audio file (multipart/form-data) for full forensic analysis, spectral metrics, and XAI diagnostic. |
| `POST` | `/api/analyze-local-path` | Analyze an existing audio file on disk via absolute path.                                                 |
| `POST` | `/api/split-stems`        | Decompose an audio file into 5 individual stems using Demucs.                                             |
| `POST` | `/api/export-plot`        | Generate a publication-quality 200 DPI PNG figure of spectral curves and tonal balance.                   |
| `GET`  | `/api/health`             | Health check endpoint returning backend status.                                                           |

---

## Citation & Academic Context

This research repository forms the core experimental software and methodology for the Undergraduate Thesis (_Skripsi_):

- **Author**: Muhammad Bryan Farras (NPM: 2306230975)
- **Advisor**: Muhammad Firdaus Syawaludin Lubis
- **Institution**: Program Studi Teknik Komputer, Departemen Teknik Elektro, Fakultas Teknik, Universitas Indonesia (FTUI)
- **Topic**: _Rancang Bangun Sistem Deteksi Musik Berbasis Kecerdasan Artifisial Menggunakan Ekstraksi Fitur Akustik dan Explainable AI (TreeSHAP)_
- **Contact**: muhammad.bryan31@ui.ac.id | m.firdaus04@ui.ac.id

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
