# AI Music Detector - Audio Forensic Visualizer

A modular audio forensics system and interactive visualizer designed for AI music detection research and thesis (Skripsi) analysis.

---

## 📁 Project Structure (Separated Frontend & Backend)

```
MY-AIDETECTOR/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application & route mounting
│   │   ├── config.py                   # Configuration, paths, & audio processing constants
│   │   ├── api/
│   │   │   └── routes.py               # REST API (/api/upload, /api/analyze-local-path, /api/export-plot)
│   │   ├── services/
│   │   │   ├── audio_loader.py         # Multi-format decoder (WAV, MP3, FLAC, M4A via bundled ffmpeg)
│   │   │   ├── visualizer_service.py   # Computes FFT spectrum, Tonal Balance, & STFT
│   │   │   └── detector_service.py     # AI cutoff and acoustic forensic metrics
│   │   └── models/
│   │       └── schemas.py              # Pydantic data schemas
│   ├── uploads/                        # Temporary cached audio files
│   ├── exports/                        # High-resolution exported figures (200 DPI PNG)
│   └── requirements.txt                # Python dependencies
├── frontend/
│   ├── index.html                      # 2-column split UI (inspired by FreeConvert)
│   ├── css/
│   │   └── style.css                   # Polished styling and responsive layout
│   └── js/
│       └── app.js                      # Canvas visualizers, playback synchronization, & controls
├── run_server.py                       # Python launch script with automatic browser opening
├── run_visualizer.bat                  # One-click Windows batch launcher
├── start_app.bat                       # Convenient quick launcher
├── start_app.ps1                       # PowerShell launcher
├── create_desktop_shortcut.bat         # Windows Desktop shortcut creator
├── test_backend.py                     # Automated pipeline verification script
└── README.md
```

---

## 🚀 How to Run

### Method 1: Double-Click (Recommended)
Double-click **`start_app.bat`** (or **`run_visualizer.bat`**).

### Method 2: PowerShell
Right-click **`start_app.ps1`** and choose *"Run with PowerShell"*, or run:
```powershell
.\start_app.ps1
```

### Method 3: Command Line
```powershell
python run_server.py
```

The app will start on `http://localhost:8000` (or the next available port) and automatically open in your default browser.
Interactive API documentation is accessible at `http://localhost:8000/docs`.
