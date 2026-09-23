# Walkthrough: Audio Forensic Visualizer & Backend for AI Music Detector

We have updated the frontend layout of [MY-AIDETECTOR](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR) to match the 2-column design inspired by [frontend inspo1.png](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/misc/inspo/frontend%20inspo1.png).

---

## 🎨 New 2-Column Split Layout

```
+-----------------------------------------------------------------------------------+
|  [Logo] AI Forensics   Visualizer ▾   AI Detector ▾   Tools ▾   [Backend: Ready]  |
+-----------------------------------------------------------------------------------+
|  Home › Audio Forensic Visualizer & Spectrogram Analyzer                          |
+-----------------------------------------------------------------------------------+
|  [AI FORENSICS] Spectral Forensics for AI Music Detection      [Export Skripsi]   |
+----------------------------------------------------+------------------------------+
|  LEFT COLUMN: Media Viewport & Visualizer          | RIGHT COLUMN: Options        |
|                                                    | ⚙️ Forensic Options          |
|  +----------------------------------------------+  |                              |
|  |  [Spectrogram / Equalizer / Tonal / Stacked] |  | Trim start (?)               |
|  |  +----------------------------------------+  |  | [ 00 : 00 : 00 . 00 ]       |
|  |  | Wave Candy STFT Spectrogram Heatmap    |  |  | [ Copy Player Time ]         |
|  |  | (With Playhead Cursor & Cutoff Marker) |  |                              |
|  |  |             ( ▶ ) Big Play Overlay     |  | Trim end (?)                 |
|  |  +----------------------------------------+  |  | [ 00 : 00 : 00 . 00 ]       |
|  |  [▶] [======== Progress Scrubber =======]   |  | [ Copy Player Time ]         |
|  +----------------------------------------------+  |                              |
|  Use current position as: [Trim Start] [Trim End]  | FFT Window Size: [ 2048 ▾ ]  |
|                                                    | Colormap: [ Wave Candy ▾ ]   |
|  [ Analyze direct local file path: __________ ]    | [x] AI Cutoff Detection      |
|                                                    | [ AI Forensic Summary Box ]  |
|  (Optional Stacked View: Equalizer & Tonal Cards)  |                              |
|                                                    | +--------------------------+ |
|                                                    | | Export Publication Plot →| |
|                                                    | +--------------------------+ |
+----------------------------------------------------+------------------------------+
|                                                                 ( + ) Floating FAB|
+-----------------------------------------------------------------------------------+
```

---

## 🔬 Component Mapping to Inspiration

| Reference Element (`frontend inspo1.png`)     | Implemented Feature in `MY-AIDETECTOR`                                                                                                                                                                              |
| :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Top Header & Breadcrumbs**                  | Clean navbar with logo, dropdown menus, backend status LED, and breadcrumb path `Home › Audio Forensic Visualizer`.                                                                                                 |
| **Top Announcement Banner**                   | Violet-indigo gradient banner highlighting thesis AI forensic capabilities and an instant export button.                                                                                                            |
| **Left Media Player Screen**                  | Dark bezel viewport hosting the **Wave Candy Spectrogram**, with a big centered Play overlay (`▶`), real-time playhead cursor, and view switcher tabs (`Spectrogram`, `Equalizer`, `Tonal Balance`, `All Stacked`). |
| **"Use current position as:"**                | Directly under the player: `[Trim Start]`, `[Trim End]`, and `[Reset Selection]` buttons that read the live playhead timestamp.                                                                                     |
| **Right Sidebar: `⚙️ GIF Options`**           | Styled identically with soft card borders, `(?)` help circles, and description text below each field.                                                                                                               |
| **"Copy Player Time" Buttons**                | Full-width purple/indigo buttons (`#6366f1`) beneath the `Trim start` and `Trim end` timecode inputs.                                                                                                               |
| **Bottom Action Button ("Convert to GIF →")** | Full-width gradient button: `Export Skripsi Publication Plot (PNG) →` triggering server-side 200 DPI figure generation.                                                                                             |
| **Floating `(+)` Button**                     | Fixed floating action button in the bottom right corner for one-click file browsing.                                                                                                                                |

---

## 🧪 Verification

All endpoints and assets were tested and verified:

- `GET /`: `200 OK` (serves the 2-column layout).
- `GET /static/css/style.css`: `200 OK`.
- `GET /static/js/app.js`: `200 OK`.
- `POST /api/analyze-local-path` with `sample_ai_test.wav`: `200 OK`, returning full STFT spectrogram, Equalizer FFT, Tonal Balance, and AI brickwall cutoff detection at 14,082.7 Hz.

---

## 🚀 How to Launch and View It

1. Double-click [run_visualizer.bat](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/run_visualizer.bat) or run `python run_server.py`.
2. The browser will open `http://localhost:8000`.
3. Click **"Load Sample Audio"** in the top-right header to test the layout immediately, or drag and drop any audio file directly into the left screen!
