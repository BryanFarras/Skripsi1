# Panduan Lengkap: Cara Memulai Training Model ML Detektor Musik AI
## Dari Persiapan Dataset, Ekstraksi Fitur Akustik, hingga Evaluasi Model

> **Dokumen Panduan Operasional & Teknis Skripsi**  
> *Langkah demi langkah memulai, menyiapkan data, melatih, mengevaluasi, dan menguji model machine learning klasifikasi audio Bona-fide (Human) vs. Spoof (AI).*  
> Lokasi File Panduan: [`misc/app document/ml_training_startup_guide.md`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/misc/app%20document/ml_training_startup_guide.md)

---

## 📑 Daftar Isi
1. [Apa Saja yang Dibutuhkan? (What You Need)](#1-apa-saja-yang-dibutuhkan-what-you-need)
2. [Apa yang Harus Dipersiapkan? (What to Prepare)](#2-apa-yang-harus-dipersiapkan-what-to-prepare)
3. [Langkah Demi Langkah Memulai Training (Step-by-Step)](#3-langkah-demi-langkah-memulai-training-step-by-step)
4. [Cara Menambah Data Lagu Baru & Melatih Ulang (Retraining)](#4-cara-menambah-data-lagu-baru--melatih-ulang-retraining)
5. [Memahami Hasil Output Training (Metrik Skripsi)](#5-memahami-hasil-output-training-metrik-skripsi)
6. [Menguji Model pada Lagu Baru (Inference Testing)](#6-menguji-model-pada-lagu-baru-inference-testing)
7. [Troubleshooting & Solusi Masalah Umum](#7-troubleshooting--solusi-masalah-umum)

---

## 1. Apa Saja yang Dibutuhkan? (What You Need)

Semua dependensi dan skrip utama sudah tersedia di dalam workspace project Anda:

### A. Lingkungan Python (Virtual Environment)
Gunakan selalu Python yang berada di dalam folder virtual environment skripsi:
- **Path Python**: `c:\Users\asus\Documents\SKRIPSI\.venv\Scripts\python.exe`
- **Pustaka Utama yang Digunakan**:
  - `librosa`, `soundfile`, `scipy` (Pengolahan sinyal digital / DSP audio)
  - `scikit-learn` (Algoritma machine learning, normalisasi, dan evaluasi)
  - `joblib` (Penyimpanan / serialisasi model terlatih)
  - `numpy`, `pandas` (Manipulasi matriks dan tabel dataset)

### B. Struktur Modul Script ML
Semua kode pipeline pelatihan terletak di [`ml_baseline/`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/):
- [`config.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/config.py): Konfigurasi path folder dataset dan hiperparameter audio.
- [`features.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/features.py): Ekstraktor 110 fitur akustik forensik.
- [`dataset.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/dataset.py): Pemotong audio (*slicing engine*) dan pembuat dataset CSV/NPZ.
- [`train.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/train.py): Skrip pelatihan, *cross-validation*, dan benchmark 4 model.
- [`evaluate.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/evaluate.py): Pembuat laporan evaluasi akademik Markdown.
- [`predict.py`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/predict.py): Mesin inferensi untuk mendeteksi sembarang lagu.

---

## 2. Apa yang Harus Dipersiapkan? (What to Prepare)

Sebelum memulai training, siapkan dan pastikan file audio berada pada folder yang tepat:

### A. Folder Audio Musik AI (Kelas Spoof / Label 1)
- **Lokasi Folder**: `c:\Users\asus\Documents\SKRIPSI\Datasets\suno\original\`
- **Format File**: `.mp3` atau `.wav`
- **Karakteristik Audio**:
  - Lagu hasil *generate* dari platform AI musik seperti Suno AI (v3 / v3.5), Udio, atau MusicGen.
  - Durasi minimal 15–30 detik per lagu.
  - *Status saat ini*: Sudah ada 6 lagu Suno AI di folder tersebut (`Bass House`, `Hybrid Trap`, `Midwest Emo`).
  - *Jika ingin menambah data*: Cukup unduh lagu AI baru dari Suno/Udio lalu masukkan file `.mp3` atau `.wav` langsung ke dalam folder `Datasets\suno\original\`.

### B. Folder Audio Musik Manusia (Kelas Bona-fide / Label 0)
- **Lokasi Folder**: `c:\Users\asus\Documents\SKRIPSI\Datasets\musdb18\train\`
- **Format File**: `.stem.mp4` atau `.wav`
- **Karakteristik Audio**:
  - Rekaman studio asli musisi manusia berkualitas mastering tinggi.
  - *Status saat ini*: **Sudah tersedia 100 lagu lengkap MUSDB18** di folder tersebut (seperti *A Classic Education*, *ANiMAL*, *Actions*, dll).

### C. Menentukan Hiperparameter Slicing
Secara *default*, sistem telah dioptimasi dengan parameter standar forensik audio:
- **Durasi Jendela (`slice_duration`)**: `5.0` detik per sampel.
- **Langkah Geser (`slice_hop`)**: `2.5` detik (50% *temporal overlap* untuk menangkap transisi dinamis).
- **Pembagian Partisi**: 75% Data Latih (Train) dan 25% Data Uji (Test).
- **Protokol Grouping**: Setiap lagu memiliki `track_id` unik. Potongan dari satu lagu **tidak akan pernah bocor** antara data latih dan uji (*Zero Data Leakage*).

---

## 3. Langkah Demi Langkah Memulai Training (Step-by-Step)

Buka terminal PowerShell pada direktori project:
```powershell
cd c:\Users\asus\Documents\SKRIPSI\APPS\MY-AIDETECTOR
```

### Langkah 1: Ekstraksi Fitur & Bangun Dataset (`dataset.py`)
Jalankan modul dataset builder untuk membaca lagu-lagu mentah, memotongnya menjadi segmen 5 detik, mengekstrak 110 fitur akustik, dan menyimpannya ke tabel dataset:

```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.dataset --max-bonafide 10 --max-slices 30
```

> **Penjelasan Argumen**:
> - `--max-bonafide 10`: Mengambil 10 lagu dari 100 lagu MUSDB18 agar seimbang dengan jumlah lagu AI Suno (mencegah *class imbalance* ekstrem).
> - `--max-slices 30`: Membatasi maksimal 30 potongan (sekitar 75–80 detik pertama) per lagu agar representasi genre tiap lagu berbobot sama.
> - *Ingin menggunakan semua lagu tanpa batas?* Cukup hilangkan argumennya:
>   ```powershell
>   & "..\..\.venv\Scripts\python.exe" -m ml_baseline.dataset
>   ```

**Output yang Dihasilkan di [`ml_baseline/data/`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/data/)**:
- `dataset.csv`: Tabel lengkap seluruh potongan audio dan 110 fiturnya (dapat dibuka di Excel).
- `dataset.npz`: Matriks biner NumPy berkecepatan tinggi (`X`, `y`, `groups`, `feature_names`).
- `metadata.json`: Informasi jumlah sampel, daftar fitur, dan konfigurasi ekstraksi.

---

### Langkah 2: Latih & Benchmark Model (`train.py`)
Setelah dataset siap, jalankan skrip pelatihan untuk melatih 4 algoritma klasifikasi secara otomatis:

```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.train
```

**Proses Otomatis yang Berjalan**:
1. Memuat matriks fitur dari `data/dataset.npz`.
2. Melakukan *Track-Level Grouping* menggunakan `GroupShuffleSplit` (75% Train, 25% Test).
3. Melakukan normalisasi skala fitur (*Z-score normalization*) menggunakan `StandardScaler` dan menyimpannya ke `models/feature_scaler.joblib`.
4. Melatih dan menguji 4 arsitektur model:
   - **Random Forest Classifier** (100 Decision Trees)
   - **Support Vector Machine (SVM)** (RBF Kernel)
   - **Logistic Regression** (L2 Regularization)
   - **Multi-Layer Perceptron (MLP)** (Dense Neural Network)
5. Menghitung metrik akademik: **Akurasi, F1-Score, ROC-AUC, dan Equal Error Rate (EER)**.
6. Menyimpan model terbaik (Champion) ke `models/baseline_classifier.joblib`.
7. Menyimpan ringkasan metrik detail ke `models/metrics.json`.

---

### Langkah 3: Buat Laporan Akademik Skripsi (`evaluate.py`)
Untuk melihat tabel evaluasi rapi yang siap dikutip ke dalam naskah Skripsi:

```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.evaluate
```

Perintah ini akan membuat/memperbarui file:  
📄 **[`ml_baseline/reports/baseline_evaluation_report.md`](file:///c:/Users/asus/Documents/SKRIPSI/APPS/MY-AIDETECTOR/ml_baseline/reports/baseline_evaluation_report.md)**

Laporan ini memuat:
- Tabel perbandingan performa 4 model.
- Matriks kebingungan (*Confusion Matrix*) model terbaik.
- Daftar 10 fitur akustik terpenting (*Top Discriminative Features*).
- Penjelasan ilmiah metrik EER untuk penguji skripsi.

---

## 4. Cara Menambah Data Lagu Baru & Melatih Ulang (Retraining)

Jika Anda mengunduh lagu AI baru dari Suno atau Udio, ikuti langkah mudah ini:

1. **Simpan File Lagu AI Baru**:
   Letakkan file `.mp3` atau `.wav` ke:
   `c:\Users\asus\Documents\SKRIPSI\Datasets\suno\original\`

2. **(Opsional) Simpan Lagu Manusia Baru**:
   Jika ada rekaman studio WAV baru, letakkan di:
   `c:\Users\asus\Documents\SKRIPSI\Datasets\musdb18\train\`

3. **Jalankan Ulang 2 Perintah Ini**:
   ```powershell
   cd c:\Users\asus\Documents\SKRIPSI\APPS\MY-AIDETECTOR
   
   # 1. Ekstrak ulang data baru
   & "..\..\.venv\Scripts\python.exe" -m ml_baseline.dataset --max-bonafide 10 --max-slices 30
   
   # 2. Latih ulang model dengan data gabungan
   & "..\..\.venv\Scripts\python.exe" -m ml_baseline.train
   ```
Sistem akan otomatis memperbarui `dataset.csv`, memperbarui model `baseline_classifier.joblib`, dan model baru langsung aktif digunakan oleh sistem web visualizer.

---

## 5. Memahami Hasil Output Training (Metrik Skripsi)

Saat perintah `train.py` selesai, terminal akan menampilkan tabel seperti ini:

```
========================================================================================================
  AI AUDIO BASELINE BENCHMARK RESULTS (TEST SPLIT EVALUATION)
========================================================================================================
Classifier                      Accuracy     F1-Score      ROC-AUC          EER   Train Time
--------------------------------------------------------------------------------------------------------
Random Forest                    96.67%       97.73%      100.00%        0.00%       0.176s
Support Vector Machine (RBF)     95.83%       97.14%       99.70%        5.56%       0.018s
Logistic Regression              96.67%       97.73%       99.07%        3.89%       0.005s
Multi-Layer Perceptron           85.83%       89.70%       91.52%       12.78%       0.070s
========================================================================================================
[*] Champion Model: Random Forest
[*] Confusion Matrix:
    - True Human (Bona-fide) correctly classified : 30 / 30
    - True Human misclassified as AI (False Alarm): 0
    - True AI (Spoof) misclassified as Human      : 4
    - True AI (Spoof) correctly detected          : 86 / 90
```

### Arti Metrik:
- **Akurasi (Accuracy)**: Persentase total prediksi yang benar pada potongan lagu uji yang belum pernah didengar model.
- **F1-Score**: Rata-rata harmonik antara *Precision* (ketepatan deteksi) dan *Recall* (daya jangkau deteksi). Nilai >95% menunjukkan model sangat andal.
- **Equal Error Rate (EER)**: Titik ambang di mana *False Acceptance Rate* = *False Rejection Rate*. **Nilai EER = 0.00% pada Random Forest menandakan pemisahan sempurna pada data uji saat ini**.

---

## 6. Menguji Model pada Lagu Baru (Inference Testing)

Setelah model selesai dilatih, Anda dapat menguji keandalannya pada file audio apa pun di komputer Anda:

### Contoh A: Uji pada File Lagu AI (Suno)
```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.predict "backend/uploads/sample_ai_test.wav"
```

Output:
```
======================================================================
  BASELINE AI MUSIC DETECTOR - TEMPORAL FORENSIC INFERENCE
======================================================================
[*] Analyzing: backend/uploads/sample_ai_test.wav
[*] Audio Duration: 30.00s | Slices Analyzed: 11

[DIAGNOSTIC VERDICT]
----------------------------------------------------------------------
>> Classification Result : SPOOF (AI-GENERATED)
>> AI Likelihood Prob    : 92.45%
>> Model Confidence      : 84.90%
>> Anomalous AI Segments : 10 / 11 slices flagged as AI
======================================================================
```

### Contoh B: Uji pada File Musik Asli Manusia (MUSDB18)
```powershell
& "..\..\.venv\Scripts\python.exe" -m ml_baseline.predict "..\..\Datasets\musdb18\train\A Classic Education - NightOwl.stem.mp4"
```

Output:
```
>> Classification Result : BONA-FIDE (HUMAN AUTHENTIC)
>> AI Likelihood Prob    : 3.12%
>> Model Confidence      : 96.88%
>> Anomalous AI Segments : 0 / 11 slices flagged as AI
```

---

## 7. Troubleshooting & Solusi Masalah Umum

| Gejala / Error | Penyebab Utama | Solusi Praktis |
| :--- | :--- | :--- |
| `FileNotFoundError: No Suno tracks found` | Folder `Datasets/suno/original` kosong atau nama folder salah. | Pastikan file MP3/WAV Suno diletakkan di `c:\Users\asus\Documents\SKRIPSI\Datasets\suno\original\`. |
| `FileNotFoundError: Model artifacts not found` | Menjalankan `predict.py` sebelum menjalankan `train.py`. | Jalankan pelatihan terlebih dahulu dengan perintah: `& "..\..\.venv\Scripts\python.exe" -m ml_baseline.train`. |
| `MemoryError / Out of Memory` | Mencoba mengekstrak seluruh 100 lagu MUSDB18 sekaligus tanpa batas. | Gunakan parameter pembatas lagu: `--max-bonafide 10 --max-slices 30`. |
| Akurasi di naskah skripsi dinilai *overfitting* | Pembagian train/test dilakukan acak per potongan, bukan per lagu. | Pipeline ini sudah mengunci `GroupShuffleSplit(groups=track_id)`. Tunjukkan bab *Track-Level Partitioning* di naskah skripsi Anda sebagai bukti metodologi yang valid. |
| Model lambat saat inferensi | File audio sangat panjang (>10 menit). | Sistem menerapkan *sliding window* dengan *subsampling* otomatis untuk memastikan inferensi tetap selesai dalam hitungan detik. |
