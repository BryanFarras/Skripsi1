# Baseline AI Audio Detector: Spoof vs. Bona-fide Classifier

This directory contains the self-contained baseline machine learning system developed for the **AI Music Detection** thesis project (*Skripsi* FTUI). It classifies audio recordings into **Bona-fide (Human Authentic)** or **Spoof (AI-Generated)**.

---

## 1. Directory Structure

```
ml_baseline/
├── README.md               # This documentation
├── config.py               # Dataset directories, audio hyperparameters, model paths
├── features.py             # 110-dimensional forensic & acoustic DSP extractor
├── dataset.py              # Slices audio and builds balanced tabular datasets (CSV / NPZ)
├── train.py                # Trains & benchmarks 4 classifiers with track-level split
├── evaluate.py             # Generates markdown benchmark reports and confusion matrices
├── predict.py              # CLI & Python API for sliding-window inference on any audio file
├── baseline_adapter.py     # Fault-tolerant service adapter for MY-AIDETECTOR FastAPI backend
├── data/
│   ├── dataset.csv         # Human-readable tabular dataset with track metadata
│   ├── dataset.npz         # Fast compressed numpy arrays (X, y, groups, feature_names)
│   └── metadata.json       # Dataset extraction parameters and sample counts
├── models/
│   ├── baseline_classifier.joblib  # Serialized champion model (Random Forest)
│   ├── feature_scaler.joblib       # Fitted StandardScaler
│   └── metrics.json                # Test benchmark metrics and top features
└── reports/
    └── baseline_evaluation_report.md # Academic evaluation report with EER & ROC-AUC
```

---

## 2. Methodology & Feature Taxonomy

Rather than treating the audio as a black-box or relying purely on static heuristics, this baseline extracts **110 acoustic and forensic features** per 5-second slice (with 50% overlap / 2.5s hop):

| Feature Category | Dimension | Purpose in AI Audio Detection |
| :--- | :---: | :--- |
| **MFCCs (1–20)** Mean & Std | 40 | Models the spectral envelope and timbral characteristics of vocals and instruments. AI models exhibit distinct latent vocoder distributions. |
| **Delta MFCCs (1–20)** Mean & Std | 40 | Quantifies temporal dynamics and transition smoothness. Captures micro-transient inconsistencies in generative diffusion/transformers. |
| **Spectral Centroid & Bandwidth** Mean & Std | 4 | Brightness and spectral spread. Real music masters exhibit wider dynamic frequency dispersion. |
| **Spectral Rolloff (85% & 95%)** Mean & Std | 4 | High-frequency cutoff frequency. Detects the typical 16kHz–18kHz brickwall cutoff cliff common in Suno/Udio neural vocoders. |
| **Spectral Flatness** Mean & Std | 2 | Tonal vs. noisy character. Neural synthesis frequently introduces subtle harmonic phase smearing and mid-band background noise floors. |
| **Spectral Contrast (7 Bands)** Mean & Std | 14 | Measures peak-to-valley energy difference across octave bands. Quantifies differences in mastering compression and multi-band dynamic range. |
| **Zero-Crossing Rate (ZCR)** Mean & Std | 2 | Percussiveness and noise density. |
| **RMS Energy** Mean & Std | 2 | Signal dynamic range. |
| **High-Frequency Energy Ratio** Mean & Std | 2 | Ratio of energy above 7.7 kHz vs total spectrum. Flags artificial attenuation in generative audio. |
| **Total Feature Vector Dimension** | **110** | |

---

## 3. Data Integrity: Track-Level Partitioning

In audio machine learning, naive random slicing causes severe **data leakage** because 5-second slices from the same song share identical instrumentation, acoustics, and mastering.

To maintain strict scientific validity for the thesis:
- **Grouping:** All slices carry a `track_id`.
- **Partitioning:** We use `GroupShuffleSplit` (75% Train, 25% Test). Slices from any given song are **strictly confined to either the training set or the test set**, never both.

### Dataset Distribution:
- **Spoof (AI-Generated):** 180 slices extracted from Suno AI tracks.
- **Bona-fide (Human Authentic):** 240 slices extracted from MUSDB18 multi-track human studio masters.
- **Total:** 420 balanced slices across 14 distinct songs.

---

## 4. Benchmark Results on Held-Out Test Set

The baseline pipeline trained and cross-evaluated 4 candidate architectures:

| Classifier | Accuracy | F1-Score | ROC-AUC | Equal Error Rate (EER) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **96.67%** | **97.73%** | **100.00%** | **0.00%** | 0.088s |
| **Support Vector Machine (RBF)** | 95.83% | 97.14% | 99.70% | 5.56% | 0.016s |
| **Logistic Regression (L2)** | 96.67% | 97.73% | 99.07% | 3.89% | 0.016s |
| **Multi-Layer Perceptron (MLP)** | 85.83% | 89.70% | 91.52% | 12.78% | 0.172s |

> [!NOTE]
> **Equal Error Rate (EER)** is the standard academic benchmark metric used in ASVspoof and Audio Deepfake evaluations where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). Lower is better (0.00% represents perfect separation).

### Top 8 Discriminative Acoustic Features:
1. `spec_contrast_b6_mean` (16.93% importance) — High-frequency peak-to-valley contrast
2. `mfcc_17_mean` (10.61% importance) — High-order harmonic timbral resonance
3. `spec_rolloff85_mean` (5.76% importance) — 85% spectral energy rolloff boundary
4. `mfcc_16_mean` (4.68% importance)
5. `mfcc_2_mean` (4.34% importance) — Low-frequency spectral tilt
6. `spec_bandwidth_mean` (4.14% importance) — Spectral spread
7. `spec_contrast_b6_std` (3.43% importance)
8. `spec_centroid_mean` (2.68% importance) — Spectral center of mass

---

## 5. Usage & CLI Commands

Always run using the workspace virtual environment Python:
`c:\Users\asus\Documents\SKRIPSI\.venv\Scripts\python.exe`

### 1. Extract Features & Rebuild Dataset
```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.dataset --max-bonafide 10 --max-slices 30
```

### 2. Train and Benchmark Models
```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.train
```

### 3. Generate Academic Evaluation Report
```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.evaluate
```

### 4. Run Inference on Any Song or Stem
```powershell
# Inference on an AI song (MP3 / WAV)
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.predict "..\..\Datasets\suno\original\Hybrid Trap_Hybrid Trap Isoxo.mp3"

# Inference on a bona-fide human track (.stem.mp4 / WAV)
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.predict "..\..\Datasets\musdb18\train\A Classic Education - NightOwl.stem.mp4"

# Inference on an isolated stem
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.predict "..\..\Datasets\suno\splits\Hybrid Trap_Hybrid Trap Isoxo_stems\Hybrid Trap_Hybrid Trap Isoxo_vocals.wav"
```

---

## 6. Python API & Web Backend Integration

To query the baseline model in code:

```python
from ml_baseline.predict import BaselineDetectorPredictor

predictor = BaselineDetectorPredictor()

# Predict from audio file path
result = predictor.predict_file("path/to/song.mp3")

print(result["prediction"])                 # "SPOOF (AI-GENERATED)" or "BONA-FIDE (HUMAN)"
print(result["overall_ai_probability"])      # e.g., 0.864
print(result["confidence_percent"])         # e.g., 72.8%
print(result["timeline"])                   # Segment-by-segment P(AI) over time
```

Or via the fault-tolerant adapter:

```python
from ml_baseline.baseline_adapter import get_baseline_adapter

adapter = get_baseline_adapter()
if adapter.is_loaded:
    prediction = adapter.predict_audio(audio_np, sample_rate=22050)
```
