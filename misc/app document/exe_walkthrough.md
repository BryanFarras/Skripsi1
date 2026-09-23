# Walkthrough: Audio Forensic Visualizer & Backend for AI Music Detector

We have built a modular, organized backend and web visualizer in [MY-AIDETECTOR](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR) with executable scripts for one-click launching.

---

## 🚀 Executable Launch Scripts Created

You have three convenient ways to run the app:

1. **[start_app.bat](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/start_app.bat)** / **[run_visualizer.bat](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/run_visualizer.bat)** (Windows Batch Launcher)
   - Double-click to run.
   - Automatically detects Python (system PATH, `py -3`, or local embedded Python).
   - Dynamically searches for an open port (starting at 8000).
   - Starts the server and automatically opens `http://localhost:8000` in your default browser.
   - Keeps the window open if any error occurs so you can easily inspect logs.

2. **[start_app.ps1](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/start_app.ps1)** (PowerShell Launcher)
   - Right-click and choose _"Run with PowerShell"_, or execute `.\start_app.ps1` from the terminal.

3. **[create_desktop_shortcut.bat](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/create_desktop_shortcut.bat)** (Desktop Shortcut Generator)
   - Double-click to place a desktop icon named **"AI Music Detector Visualizer"** directly onto your Windows Desktop.

---

## 🎨 Frontend Layout (Matching Reference)

- **2-Column Split Layout**:
  - **Left**: Drag-and-drop viewport & media player with Wave Candy spectrogram, centered play button overlay (`▶`), time scrubber, and quick buttons (`Trim Start`, `Trim End`, `Reset Selection`).
  - **Right**: Forensic options sidebar with `hh:mm:ss.ms` inputs, purple `[Copy Player Time]` buttons, FFT size & colormap selectors, live AI cutoff diagnostics, and the large bottom button `Export Skripsi Publication Plot (PNG) →`.
  - **Floating Action Button (`+`)** in the bottom corner for quick file selection.
