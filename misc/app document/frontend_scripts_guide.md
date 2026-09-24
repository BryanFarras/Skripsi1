# Frontend Scripts Architecture & Technical Reference Guide
## `APPS/MY-AIDETECTOR/frontend/js/`

> **Dokumentasi & Penjelasan Komprehensif Arsitektur Frontend JavaScript [MY-AIDETECTOR](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR).**  
> *Mencakup seluruh alur kerja modul, Web Audio API, perenderan kanvas 60 FPS, algoritma 3D Waterfall, visualisasi Wave Candy & Fruity Parametric EQ 2, hingga integrasi pemisah stem audio.*

---

## 📑 Daftar Isi File
1. [Arsitektur Umum & Hubungan Antar Modul](#1-arsitektur-umum--hubungan-antar-modul)
2. [Modul Inti (Root Scripts)](#2-modul-inti-root-scripts)
   - [2.1. main.js — Bootstrap & Koordinator Aplikasi](#21-mainjs--bootstrap--koordinator-aplikasi)
   - [2.2. state.js — Central State Registry & Colormap Engine](#22-statejs--central-state-registry--colormap-engine)
   - [2.3. audioPlayer.js — Web Audio API & 60 FPS Render Loop](#23-audioplayerjs--web-audio-api--60-fps-render-loop)
   - [2.4. uploader.js — Ingest Audio, Path Analyzer & Pipeline Dispatcher](#24-uploaderjs--ingest-audio-path-analyzer--pipeline-dispatcher)
3. [Modul Komponen Forensik (components/)](#3-modul-komponen-forensik-components)
   - [3.1. spectrogram.js — 2D STFT Heatmap (Wave Candy & Edison Modes)](#31-spectrogramjs--2d-stft-heatmap-wave-candy--edison-modes)
   - [3.2. waterfall3d.js — 3D Spectrogram Waterfall & Orbit Controls](#32-waterfall3djs--3d-spectrogram-waterfall--orbit-controls)
   - [3.3. equalizer.js — Fruity Parametric EQ 2 Flame FFT Analyzer](#33-equalizerjs--fruity-parametric-eq-2-flame-fft-analyzer)
   - [3.4. tonalBalance.js — Ozone-Style Dynamic Tonal Balance Corridor](#34-tonalbalancejs--ozone-style-dynamic-tonal-balance-corridor)
   - [3.5. controls.js — View Switcher, Trim Tools & Smoothing Sliders](#35-controlsjs--view-switcher-trim-tools--smoothing-sliders)
   - [3.6. forensics.js — AI Forensic Diagnostics & Skripsi Plot Exporter](#36-forensicsjs--ai-forensic-diagnostics--skripsi-plot-exporter)
   - [3.7. stems.js — Multi-Stem Separation, Isolated Preview & Inspector](#37-stemsjs--multi-stem-separation-isolated-preview--inspector)
4. [Ringkasan Alur Data & State Lifecycle](#4-ringkasan-alur-data--state-lifecycle)

---

## 1. Arsitektur Umum & Hubungan Antar Modul

Frontend dirancang dengan pola **Vanilla ES6 Modules** tanpa dependensi framework eksternal (React/Vue/Tailwind), menjamin performa perenderan grafis tanpa overhead virtual DOM dan kompatibilitas penuh dengan sistem kanvas audio WebGL/2D.

```
                                  [ index.html ]
                                         │
                                         ▼
                                   [ main.js ] (Bootstrap)
                                         │
       ┌─────────────────┬───────────────┴───────────────┬────────────────┐
       ▼                 ▼                               ▼                ▼
   [ state.js ]    [ uploader.js ]              [ audioPlayer.js ]  [ stems.js ]
 (Registry & DOM) (Upload & Parse)              (Web Audio & Loop)  (5-Stem Demucs)
       │                 │                               │                │
       └─────────────────┼───────────────────────────────┼────────────────┘
                         ▼                               ▼
                 [ REST API Backend ]           [ Canvas Renderers ]
               - /api/upload                     - spectrogram.js
               - /api/analyze-local-path         - waterfall3d.js
               - /api/split-stems                - equalizer.js
               - /api/load-stem                  - tonalBalance.js
               - /api/export-plot                - forensics.js
                                                 - controls.js
```

---

## 2. Modul Inti (Root Scripts)

### 2.1. `main.js` — Bootstrap & Koordinator Aplikasi
- **Lokasi**: [frontend/js/main.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/main.js)
- **Tanggung Jawab Utama**:
  1. Menginisialisasi pemetaan elemen DOM (`initDOMElements()`) sebelum modul lain berjalan.
  2. Mendaftarkan pendengar acara (*event listeners*) untuk seluruh komponen aplikasi.
  3. Memastikan bootstrap berjalan aman baik saat dokumen berstatus `loading` maupun sudah `interactive`.

#### Cuplikan Kode Kunci:
```javascript
export function init() {
  initDOMElements();                // 1. Cache semua ID elemen HTML ke memory
  setupUploader();                  // 2. Pasang drag-and-drop & uploader file
  setupAudioPlayer();               // 3. Hubungkan pemutar audio & scrub bar
  setupControls();                  // 4. Pasang tab navigasi & slider smoothing
  setupSpectrogramControls(...);    // 5. Pasang toolbar Wave Candy vs Edison
  setupWaterfallOrbitControls(...); // 6. Pasang kontrol rotasi 3D Waterfall
  initStemSplitter();               // 7. Pasang event listener tombol pemisah stem
}
```

---

### 2.2. `state.js` — Central State Registry & Colormap Engine
- **Lokasi**: [frontend/js/state.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/state.js)
- **Tanggung Jawab Utama**:
  1. **Single Source of Truth**: Menyimpan status reaktif aplikasi (file aktif, durasi audio, posisi trim, matriks STFT, status rotasi 3D).
  2. **Cache Elemen DOM (`elements`)**: Mengindeks semua elemen HTML sekali saja pada awal sesi, mencegah pemanggilan berulang `document.getElementById()`.
  3. **Mesin Interpolasi Palet Warna**: Mengimplementasikan palet ilmiah (**Inferno / Wave Candy**, **Magma**, **Plasma**, dan **Cyan Glow**) untuk memetakan nilai dB (-80 dB s/d 0 dB) ke warna RGB secara cepat.
  4. **Utilitas Waktu**: Konversi format waktu desimal detik ke kode waktu presisi tinggi `hh:mm:ss.ms` (`formatTimecode` dan `parseTimecode`).

#### Struktur State Kunci:
```javascript
export const state = {
  currentFileId: null,          // ID unik sesi audio dari backend
  currentDuration: 0,           // Panjang audio dalam detik
  analysisData: null,           // Payload JSON lengkap (STFT, FFT, Forensics)
  currentView: 'spectrogram',   // 'spectrogram' | 'waterfall' | 'equalizer' | 'tonal' | 'stacked'

  // Kamera 3D Waterfall
  camYaw: -0.68, camPitch: 0.42, camZoom: 1.0,
  isDragging3D: false, dragVelX: 0, dragVelY: 0,

  // Mode Spektrogram
  specViewMode: 'scroll',       // 'scroll' (Wave Candy) atau 'full' (Edison)
  specWindowSec: 6.0,           // Jendela waktu (3s, 6s, 12s)

  // Web Audio Real-Time
  audioCtx: null, analyserNode: null, liveMagnitudesDb: null,
  averagingMs: 400,             // Jendela smoothing temporal (3ms - 400ms)

  // Stem Separation State
  currentStemsData: null,
  isSplittingStems: false
};
```

---

### 2.3. `audioPlayer.js` — Web Audio API & 60 FPS Render Loop
- **Lokasi**: [frontend/js/audioPlayer.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/audioPlayer.js)
- **Tanggung Jawab Utama**:
  1. **Web Audio Graph Pipeline**:
     ```
     HTMLMediaElement (<audio>)
             │
             ▼
     AudioContext.createMediaElementSource()
             │
             ▼
     AnalyserNode (FFT 2048, minDecibels: -90, maxDecibels: -10)
             │
             ├──► [ Uint8Array Live Frequency Data ] ──► (Equalizer & Tonal Canvas)
             ▼
     AudioContext.destination (Speakers / Headphones)
     ```
  2. **60 FPS Live Render Loop (`renderLiveVisualizers`)**:
     Menggunakan `requestAnimationFrame` untuk mengekstrak spektrum frekuensi saat lagu diputar, menerapkan smoothing temporal (`averagingMs`), dan merender kanvas Equalizer / Tonal Balance tanpa lag.
  3. **Presisi Scrubbing & Seeking**:
     Mendukung klik pada progress bar dan scrubbing kursor waktu tanpa mereset pemutaran ke awal via HTTP 206 Partial Range request.
  4. **Sinkronisasi Resolusi Kanvas (`syncCanvasSizes`)**:
     Secara otomatis menyesuaikan `canvas.width` dan `canvas.height` dengan rasio piksel layar (`devicePixelRatio`) agar gambar selalu tajam pada layar Retina / 4K.

---

### 2.4. `uploader.js` — Ingest Audio, Path Analyzer & Pipeline Dispatcher
- **Lokasi**: [frontend/js/uploader.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/uploader.js)
- **Tanggung Jawab Utama**:
  1. **Multi-Strategy Audio Upload (`uploadAudio`)**:
     Mencoba upload `multipart/form-data` terlebih dahulu; jika terbentur batasan payload jaringan, beralih otomatis ke *direct binary stream* (`application/octet-stream` dengan header `x-filename`).
  2. **Local Path Ingest (`analyzeLocalPath`)**:
     Memungkinkan pengguna menganalisis file audio di komputer lokal secara instan tanpa perlu mengunggah ulang via HTTP (`/api/analyze-local-path`).
  3. **Central Dispatcher (`handleAnalysisLoaded`)**:
     Fungsi pusat yang dipanggil setiap kali file baru selesai dianalisis:
     - Menyimpan data JSON backend ke `state.analysisData`.
     - Memperbarui label metadata (sample rate, channel, durasi).
     - Mengarahkan `<audio>` element ke `/api/audio/{file_id}`.
     - Memperbarui kartu diagnostik forensik AI (`updateDiagnostics`).
     - Mengaktifkan tombol pemisah stem (`onAudioLoadedForStems`).
     - Membuka tab spektrogram dengan playhead siap putar.

---

## 3. Modul Komponen Forensik (`components/`)

### 3.1. `spectrogram.js` — 2D STFT Heatmap (Wave Candy & Edison Modes)
- **Lokasi**: [frontend/js/components/spectrogram.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/spectrogram.js)
- **Fungsi Forensik**:
  Memetakan matriks Short-Time Fourier Transform (waktu vs frekuensi vs desibel).
- **Fitur Unggulan**:
  1. **Mode "Moving (Wave Candy)"**: Menampilkan jendela bergerak dinamis (3 detik, 6 detik, atau 12 detik) yang bergulir mulus seiring lagu berputar seperti visualizer Wave Candy di FL Studio.
  2. **Mode "↔ Full Track (Edison)"**: Menampilkan ikhtisar seluruh lagu dengan kursor playhead yang melintas dari kiri ke kanan.
  3. **Overlay Garis Cutoff AI**: Secara otomatis menggambar garis horizontal putus-putus berwarna cyan terang pada frekuensi cutoff (misal 14.082 Hz) dengan label badge, memudahkan deteksi cepat batasan frekuensi model neural audio.
  4. **Offscreen Canvas Caching**: Merender seluruh peta warna ke kanvas tak terlihat di memori agar proses scrubbing dan repainting sangat ringan di GPU/CPU.

---

### 3.2. `waterfall3d.js` — 3D Spectrogram Waterfall & Orbit Controls
- **Lokasi**: [frontend/js/components/waterfall3d.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/waterfall3d.js)
- **Fungsi Forensik**:
  Membangun visualisasi 3 dimensi di atas kanvas 2D standar menggunakan aljabar linear proyeksi perspektif (*isometric/perspective rotation matrix*).
- **Fitur Unggulan**:
  1. **Painter's Depth-Sort Algorithm**: Mengurutkan irisan frame spektrum dari jarak terjauh (*depth Z*) ke jarak terdekat agar pita frekuensi yang tinggi tidak tertutup oleh frame di depannya secara keliru.
  2. **Rotasi 360° & Inersia Momentum**: Pengguna dapat mengklik dan menggeser mouse untuk memutar visualizer 3D dari sudut mana pun, dilengkapi inersia pelambatan fisik saat mouse dilepas.
  3. **Preset Sudut Kamera Cepat**:
     - *Sketch Angle*: Sudut miring diagonal khas DAW reference.
     - *Isometric*: Proyeksi teknis 45°.
     - *Front (EQ)*: Tampilan tegak lurus depan mirip equalizer visual.
  4. **Pengatur Transparansi Frame Latar Belakang (`wfInactiveAlpha`)**: Slider yang mengatur transparansi irisan spektrum non-aktif (default 15%), sehingga gelombang aktif saat ini dapat terlihat menonjol tanpa terganggu derau visual.

---

### 3.3. `equalizer.js` — Fruity Parametric EQ 2 Flame FFT Analyzer
- **Lokasi**: [frontend/js/components/equalizer.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/equalizer.js)
- **Fungsi Forensik**:
  Meniru penganalisis spektrum *Fruity Parametric EQ 2* untuk memeriksa lonjakan resonansi harmonik dan profil noise floor audio buatan AI.
- **Fitur Unggulan**:
  1. **Dual-Trace System**:
     - *Trace 1*: Kurva FFT instan dengan isian gradien cyan bercahaya (*glow fill*).
     - *Trace 2*: Garis *Peak Hold* yang melayang di puncak tertinggi frekuensi dan meluruh perlahan (*decay*), mengungkap frekuensi resonansi tersembunyi.
  2. **Skala Frekuensi Logaritmik**: 256 bin yang dipetakan secara logaritmik dari 20 Hz hingga 20.000 Hz sesuai kurva pendengaran telinga manusia.
  3. **7 Titik Pembagian Frekuensi**: Sub, Bass, Low-Mid, Mid, High-Mid, Presence, Treble.

---

### 3.4. `tonalBalance.js` — Ozone-Style Dynamic Tonal Balance Corridor
- **Lokasi**: [frontend/js/components/tonalBalance.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/tonalBalance.js)
- **Fungsi Forensik**:
  Mengukur distribusi energi akustik relatif lintas 5 pita frekuensi standar industri mastering (iZotope Ozone Tonal Balance style):
  - **Sub**: 20 Hz – 60 Hz
  - **Bass**: 60 Hz – 250 Hz
  - **Low-Mid**: 250 Hz – 2 kHz
  - **High-Mid**: 2 kHz – 8 kHz
  - **Air / Treble**: 8 kHz – 20 kHz
- **Fitur Unggulan**:
  1. **Target Crest Corridor**: Area koridor abu-abu yang menunjukkan toleransi distribusi energi ideal rekaman studio manusia. Jika kurva audio AI jatuh di bawah koridor pada pita Air/Treble, visualizer langsung menunjukkan indikator defisit energi.
  2. **Persentase Energi Real-Time**: Tampilan angka persentase energi live untuk setiap pita instrumen saat lagu diputar.

---

### 3.5. `controls.js` — View Switcher, Trim Tools & Smoothing Sliders
- **Lokasi**: [frontend/js/components/controls.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/controls.js)
- **Tanggung Jawab Utama**:
  1. **Pengalih Tampilan (`switchView`)**:
     Mengelola transisi aktif antara kanvas Spektrogram, 3D Waterfall, Equalizer, Tonal Balance, dan mode All-Stacked.
  2. **Pengatur Pemotongan Titik Trim**:
     Menyalin posisi kursor pemutar ke input `Trim Start` dan `Trim End` (`copyStartBtn`, `copyEndBtn`, `useTrimStartBtn`, `useTrimEndBtn`).
  3. **Slider Smoothing Temporal (`averagingSlider`)**:
     Menyediakan pengaturan jendela perata-rataan spektral dari **3 ms** (sangat cepat untuk mendeteksi transien perkusif) hingga **400 ms** (standar Ozone Tonal Balance untuk inspeksi kurva makro). Dilengkapi tombol chip preset cepat (3ms, 50ms, 150ms, 400ms).
  4. **Pemilih Palet Warna**: Mengubah palet visual secara dinamis saat dropdown diubah.

---

### 3.6. `forensics.js` — AI Forensic Diagnostics & Skripsi Plot Exporter
- **Lokasi**: [frontend/js/components/forensics.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/forensics.js)
- **Tanggung Jawab Utama**:
  1. **Visualisasi Kartu Diagnostik (`updateDiagnostics`)**:
     Menampilkan ringkasan metrik forensik dari backend:
     - *Cutoff Status Badge*: Hijau (*Authentic*), Kuning (*Mild/Moderate Cutoff*), Merah (*Strong AI Cutoff < 16.5 kHz*).
     - *Spectral Rolloff (95%) & Centroid*: Indikator kecerahan dan batas energi suara.
     - *Air Power Ratio*: Persentase energi di atas 16 kHz.
     - *Forensic Notes*: Catatan intepretasi akustik otomatis.
  2. **Ekspor Gambar Publikasi Skripsi (`exportPlot`)**:
     Mengirim permintaan ke `POST /api/export-plot/{file_id}`, menerima file gambar PNG 200 DPI berkualitas tinggi dari backend, dan memicu unduhan otomatis ke komputer pengguna untuk disisipkan langsung ke dokumen tugas akhir.

---

### 3.7. `stems.js` — Multi-Stem Separation, Isolated Preview & Inspector
- **Lokasi**: [frontend/js/components/stems.js](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/frontend/js/components/stems.js)
- **Tanggung Jawab Utama**:
  1. **Manajemen Pemisahan 5 Stem**:
     Mengirim permintaan pemisahan ke endpoint `POST /api/split-stems` dan menampilkan status pemrosesan dengan spinner interaktif.
  2. **Penyajian 5 Komponen Terisolasi**:
     Merender kartu untuk setiap komponen yang dihasilkan:
     - 🎵 **Full Mix** (Lagu asli tanpa separasi)
     - 🎙️ **Vocals** (Vokal terisolasi — bukti #1 artefak fase neural vocoder)
     - 🥁 **Drums** (Perkusi — bukti penumpulan transien kick/snare)
     - 🎸 **Bass** (Garis bass & frekuensi sub)
     - 🎹 **Other** (Gitar, piano, synths, reverb tail)
  3. **Pemutar Preview Mandiri (`toggleStemAudioPreview`)**:
     Pengguna dapat langsung mendengarkan vokal atau drum yang terpisah hanya dengan mengklik tombol *Preview*, tanpa mengganggu atau memuat ulang halaman.
  4. **Inspeksi Langsung ke Visualizer (`loadStemToVisualizer`)**:
     Mengirim `POST /api/load-stem` ke backend dan langsung memuat komponen yang dipilih ke seluruh visualizer (3D Waterfall, Spektrogram, EQ, Tonal Balance, Cutoff) untuk analisis mendalam per instrumen.
  5. **Indikator File Sementara (Auto-Delete 1 Jam)**:
     Menampilkan penanda visual bahwa file stem disimpan secara temporer dan otomatis dihapus setelah 60 menit.

---

## 4. Ringkasan Alur Data & State Lifecycle

```
[ Pengguna Memasukkan File Audio ]
       │
       ▼ (Drag & Drop / Browse / Local Path)
[ uploader.js: uploadAudio() / analyzeLocalPath() ]
       │
       ▼ (HTTP POST ke Backend FastAPI)
[ API Response: Metadata, STFT Matrix, FFT Spectrum, AI Forensics ]
       │
       ▼
[ uploader.js: handleAnalysisLoaded() ]
       ├─► Simpan ke state.js (analysisData, currentFileId, duration)
       ├─► Set <audio>.src = /api/audio/{file_id}
       ├─► Update forensics.js (Badges, Rolloff, Cutoff Notes)
       ├─► Aktifkan stems.js (Tampilkan kartu pemisah stem)
       └─► switchView('spectrogram') (Render canvas STFT)
             │
             ▼ (Saat Pengguna Menekan Play ▶)
[ audioPlayer.js: requestAnimationFrame Loop (60 FPS) ]
       ├─► Baca frekuensi live dari AnalyserNode
       ├─► Terapkan perata-rataan temporal (averagingMs)
       ├─► Render equalizer.js (Live FFT + Peak Hold Curve)
       ├─► Render tonalBalance.js (5-Band Corridor & Gauges)
       ├─► Render waterfall3d.js (Rotasi 3D & Active Frame Slice)
       └─► Perbarui posisi Playhead & Timecode
             │
             ▼ (Saat Pengguna Menekan "Split Audio Stems")
[ stems.js: handleSplitStemsClick() ]
       ├─► HTTP POST /api/split-stems
       ├─► Tampilkan status Demucs
       └─► Render 5 kartu stem (Preview, Inspect in Visualizer, Download WAV)
```
