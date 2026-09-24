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
│   │   │   ├── detector_service.py     # AI cutoff and acoustic forensic metrics
│   │   │   └── stem_service.py         # 5-stem neural separation (Demucs / MUSDB)
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
│       ├── main.js                     # Modular application entry point & bootstrap
│       ├── state.js                    # Central state registry & cached DOM elements
│       ├── audioPlayer.js              # Web Audio API AnalyserNode & 60 FPS render loop
│       └── components/
├── split_stems.py                      # Multi-stem separation CLI (Full Mix, Vocals, Drums, Bass, Other)
├── run_server.py                       # Python launch script with automatic browser opening
├── run_visualizer.bat                  # One-click Windows batch launcher
├── start_app.bat                       # Convenient quick launcher
├── start_app.ps1                       # PowerShell launcher
├── create_desktop_shortcut.bat         # Windows Desktop shortcut creator
├── test_backend.py                     # Automated pipeline verification script
└── README.md
```

---

## 🎛️ Stem Separation Tool (`split_stems.py`)

Split any audio track into **5 individual forensic components** (`mixture`, `vocals`, `drums`, `bass`, `other`):

```bash
# 1. Single audio file (automatically creates 'song1_stems/' next to the file)
python split_stems.py path/to/song1.wav

# 2. Specify custom parent directory (creates 'my_stems/song1_stems/')
python split_stems.py path/to/song1.mp3 --output-dir my_stems/

# 3. Batch processing an entire folder of songs (creates '<song>_stems/' for each track)
python split_stems.py --batch-dir Datasets/raw/ai_suno --output-dir Datasets/stems/ai_suno
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
