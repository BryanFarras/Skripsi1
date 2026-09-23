# Walkthrough: Audio Forensic Visualizer & Backend for AI Music Detector

We have built a modular, organized backend and localhost web visualizer inside [MY-AIDETECTOR](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR).

The system generates the exact 3 audio visual representations from your reference image (`visual analyzer1.png`) and equips you with acoustic forensic metrics specifically tailored for AI music detection research in your Skripsi.

---

## 🏗️ What Was Built

```
c:\Users\asus\Documents\SKRIPSI\APPS\MY-AIDETECTOR\
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application & route mounting
│   │   ├── config.py                   # Paths, directories, and audio settings
│   │   ├── api/
│   │   │   └── routes.py               # REST API (/api/upload, /api/analyze-local-path, /api/export-plot)
│   │   ├── services/
│   │   │   ├── audio_loader.py         # Multi-format decoder (WAV, MP3, FLAC, M4A via bundled ffmpeg)
│   │   │   ├── visualizer_service.py   # Computes FFT Spectrum, Tonal Balance, & STFT Spectrogram
│   │   │   └── detector_service.py     # AI brickwall cutoff & forensic metric extraction
│   │   ├── models/
│   │   │   └── schemas.py              # Pydantic data schemas
│   │   └── static/                     # Localhost dashboard
│   │       ├── index.html              # Modern dark-mode UI
│   │       ├── css/style.css           # DAW-style styling
│   │       └── js/app.js               # Canvas rendering & real-time playback sync
│   ├── uploads/                        # Cached audio files (sample_ai_test.wav ready)
│   ├── exports/                        # High-res publication plots (200 DPI PNG)
│   └── requirements.txt                # Dependencies specification
├── run_server.py                       # Python launch script with automatic browser opening
├── run_visualizer.bat                  # One-click Windows batch launcher
├── test_backend.py                     # Automated pipeline verification script
└── README.md                           # Documentation & guide
```

---

## 🔬 The 3 Visualizers Matching Your Reference

| Panel                | Implementation Details                                                                                                                                                                   | AI Forensic Clues Detected                                                                                                                          |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Equalizer**     | 256 logarithmic frequency bins (20 Hz - 20 kHz), dual-trace showing Peak Hold and Average Curve with cyan glow fill.                                                                     | Exposes unnatural resonant spikes and synthetic noise floor elevation.                                                                              |
| **2. Tonal Balance** | Smooth continuous spectrum curve with color fills across 5 bands: **Sub** (20-60Hz), **Bass** (60-250Hz), **Low-Mid** (250-2kHz), **High-Mid** (2-8kHz), and **Treble / Air** (8-20kHz). | Detects energy imbalances, such as hollow highs or exaggerated mid frequencies common in generated audio.                                           |
| **3. Spectrogram**   | High-resolution 2D STFT heatmap (Wave Candy / Inferno colormap) with real-time playback head syncing and interactive scrubbing.                                                          | **#1 Forensic Clue**: Automatically overlays a dashed cyan indicator line when an artificial brickwall cutoff (e.g., 16 kHz or 18 kHz) is detected. |

---

## 🧪 Verification & Results

We tested the entire pipeline using [test_backend.py](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/test_backend.py) on synthesized audio with an artificial 16 kHz frequency ceiling:

- **Audio Loading**: Successfully loaded and decoded 132,300 samples (3.00 seconds @ 44.1 kHz).
- **Forensic Detection**:
  - `Cutoff Detected`: `True`
  - `Estimated Cutoff`: `14,082.7 Hz`
  - `Cutoff Severity`: `Strong (<16.5kHz)`
  - `Spectral Rolloff (95%)`: `15,503.9 Hz`
- **Plot Export**: Generated high-res 200 DPI publication figure:
  - File: [`backend/exports/test_figure_export.png`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/backend/exports/test_figure_export.png)
  - Layout matches the 3 panels from `visual analyzer1.png`.

---

## 🚀 How to Run

### Method 1: Double-Click (Recommended)

Double-click [run_visualizer.bat](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/run_visualizer.bat).

### Method 2: Command Line

```powershell
python run_server.py
```

It will boot the server and automatically open **`http://localhost:8000`** in your default browser.
From there, you can:

1. Drag and drop any `.wav`, `.mp3`, or `.flac` audio file.
2. Or paste any local file path on your computer into the path input bar.
3. Play the audio with real-time synced visualization.
4. Click **"Export for Skripsi (PNG)"** to download publication-ready figures for your thesis.
