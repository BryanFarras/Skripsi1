# Audio Machine Learning: Foundations, Architectures, and End-to-End Engineering Guide

> **Panduan Komprehensif Skripsi: Fondasi Teori, Sinyal Digital, Arsitektur Deep Learning, dan Alur Pelatihan Model dari Awal hingga Selesai**  
> *Disusun berdasarkan tinjauan literatur ilmiah (ASVspoof, CLAD, Procedia Computer Science, IEEE, Elsevier), standar pemrosesan sinyal digital (DSP), dan acuan Tugas Akhir Departemen Teknik Elektro FTUI.*

---

## Daftar Isi
1. [Fondasi Matematis & Pemrosesan Sinyal Audio Digital (DSP)](#1-fondasi-matematis--pemrosesan-sinyal-audio-digital-dsp)
2. [Taksonomi Representasi & Ekstraksi Fitur Akustik](#2-taksonomi-representasi--ekstraksi-fitur-akustik)
3. [Arsitektur Deep Learning untuk Audio & Deteksi Deepfake](#3-arsitektur-deep-learning-untuk-audio--deteksi-deepfake)
4. [Alur Rekayasa End-to-End: Membangun, Melatih, dan Menguji Model](#4-alur-rekayasa-end-to-end-membangun-melatih-dan-menguji-model)
5. [Protokol Evaluasi, Metrik Biometrik & Analisis Forensik](#5-protokol-evaluasi-metrik-biometrik--analisis-forensik)
6. [Ketahanan Model, Serangan Manipulasi Audio & Pertahanan](#6-ketahanan-model-serangan-manipulasi-audio--pertahanan)
7. [Arsitektur Produksi, MLOps & Integrasi Aplikasi Forensik](#7-arsitektur-produksi-mlops--integrasi-aplikasi-forensik)
8. [Blueprint Kode PyTorch Lengkap (Siap Pakai)](#8-blueprint-kode-pytorch-lengkap-siap-pakai)

---

## 1. Fondasi Matematis & Pemrosesan Sinyal Audio Digital (DSP)

Pemrosesan audio dalam machine learning adalah jembatan antara fisika gelombang mekanik dan aljabar linier komputasional. Berbeda dengan citra visual yang memiliki piksel spasial teratur, sinyal audio adalah gelombang tekanan kontinu 1 dimensi yang struktur informasinya terdistribusi secara simultan di domain **waktu (time)**, **frekuensi (frequency)**, dan **fase (phase)**.

```
Gelombang Tekanan Akustik Kontinu x(t)
                │
                ▼  [Sampling: fs ≥ 2·fmax] (Teorema Nyquist-Shannon)
                ▼  [Kuantisasi: Bit Depth 16/24/32-bit float]
Sinyal Audio Digital Diskrit: x[n] (Domain Waktu, 1D Array)
                │
     ┌──────────┴─────────────────────────────────────────┐
     ▼                                                    ▼
[Pemrosesan Raw Waveform Langsung]          [Transformasi Waktu-Frekuensi (DSP)]
(SincNet, RawNet2, AASIST)                  ├── Short-Time Fourier Transform (STFT)
                                            ├── Mel-Spektrogram (Skala Auditori)
                                            ├── Constant-Q Transform (CQT, Skala Musikal)
                                            └── Koefisien Cepstral (MFCC, LFCC, CQCC)
```

### 1.1. Digitalisasi Sinyal Suara
1. **Teorema Sampling Nyquist-Shannon**:
   Sinyal analog kontinu $x(t)$ dengan frekuensi tertinggi $f_{\max}$ dapat direkonstruksi secara sempurna tanpa kehilangan informasi jika disampling pada frekuensi $f_s$:
   $$f_s \ge 2 f_{\max}$$
   Jika $f_s < 2 f_{\max}$, komponen frekuensi tinggi akan melipat ke frekuensi rendah (disebut **aliasing**), merusak integritas akustik.
   - Standar suara manusia/ucapan: $f_s = 16.000\text{ Hz}$ ($f_{\max} = 8\text{ kHz}$, mencakup sebagian besar formant vokal).
   - Standar musik profesional: $f_s = 44.100\text{ Hz}$ atau $48.000\text{ Hz}$ ($f_{\max} = 22.05\text{ kHz}$, mencakup seluruh ambang dengar manusia).

2. **Kuantisasi & Bit Depth**:
   Amplitudo kontinu dipetakan ke dalam $2^B$ tingkat diskrit ($B$ = bit depth):
   - 16-bit integer: $65.536$ level kuantisasi, dynamic range $\approx 96.3\text{ dB}$.
   - 32-bit floating point: Dynamic range $> 1500\text{ dB}$, mencegah terjadinya digital clipping saat normalisasi dan augmentasi.
   - Rasio Sinyal terhadap Derau Kuantisasi:
     $$\text{SQNR} \approx 6.02 B + 1.76\text{ dB}$$

### 1.2. Transformasi Fourier Diskrit (DFT) & FFT
Sinyal domain waktu didekomposisi menjadi deretan gelombang sinusoidal menggunakan Discrete Fourier Transform:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$
Di mana $X[k]$ adalah bilangan kompleks dengan magnitudo $|X[k]|$ (amplitudo frekuensi) dan sudut fase $\angle X[k] = \arctan\left(\frac{\text{Im}(X[k])}{\text{Re}(X[k])}\right)$.

### 1.3. Short-Time Fourier Transform (STFT) & Spektrogram
Karena sinyal suara bersifat non-stasioner (frekuensi berubah setiap saat), STFT membagi sinyal menjadi segmen-segmen pendek menggunakan window geser $w[n]$:
$$X(m, k) = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$
Di mana:
- $m$: Indeks frame waktu (sumbu horizontal).
- $k$: Indeks bin frekuensi (sumbu vertikal).
- $N$: Ukuran window FFT (misal $1024$ atau $2048$ sampel).
- $H$: Hop length / jarak geser window (misal $256$ atau $512$ sampel).
- $w[n]$: Fungsi jendela peredam kebocoran spektral (Hann window):
  $$w_{\text{Hann}}[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$

**Spektrogram Daya & Konversi Desibel**:
$$P(m, k) = |X(m, k)|^2$$
$$S_{\text{dB}}(m, k) = 10 \log_{10} \left( \frac{P(m, k)}{\max(P) + \epsilon} \right)$$

---

## 2. Taksonomi Representasi & Ekstraksi Fitur Akustik

| Kelas Fitur | Bentuk Representasi | Penggunaan Optimal | Keunggulan Utama | Limitasi |
|---|---|---|---|---|
| **Spektral Linier** | STFT, Linear Spectrogram | Analisis Cutoff AI, Deteksi Artefak Vocoder | Resolusi frekuensi merata hingga batas Nyquist | Dimensi besar; tidak selaras dengan telinga manusia |
| **Psikoakustik** | Mel-Spectrogram, MFCC | Speech Recognition, Klasifikasi Suara Umum | Meniru respon filter koklea telinga manusia | Resolusi frekuensi tinggi terkompresi |
| **Harmonik/Musikal**| Constant-Q Transform (CQT), Chroma | Deteksi Musik AI, Analisis Nada/Instrumen | Spacing frekuensi logaritmik, konstan per oktaf | Komputasi variabel window lebih lambat |
| **Self-Supervised** | WavLM, Wav2Vec 2.0, HuBERT, MERT | Deteksi Deepfake Lintas Domain | Fitur kontekstual semantik kaya dari jutaan data | Komputasi berat; sulit diinterpretasikan (black-box) |
| **End-to-End Raw**  | Raw Waveform ($x[n]$) + SincNet | Anti-spoofing Speech (AASIST, RawNet2) | Mempertahankan fase murni dan interaksi waktu-frekuensi | Membutuhkan arsitektur khusus & data pelatihan besar |

---

### 2.1. Mel-Scale & Mel-Frequency Cepstral Coefficients (MFCC)
Telinga manusia merespon frekuensi secara linier di bawah $1\text{ kHz}$ dan logaritmik di atas $1\text{ kHz}$. Skala Mel merumuskan fenomena ini:
$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

1. **Mel-Filterbank**: Mengalikan spektrogram linier dengan $M$ filter segitiga tumpang-tindih ($M = 40$ s.d. $128$).
2. **Discrete Cosine Transform (DCT)**: Menerapkan DCT-II pada log Mel energy untuk mendekorelasikan energi antar-filter dan menghasilkan $C$ koefisien cepstral ($C = 13$ s.d. $20$).
3. **Turunan Dinamis ($\Delta$ dan $\Delta\Delta$)**:
   - Koefisien Statis: Karakteristik bentuk saluran vokal / instrumen.
   - Delta ($\Delta$): Kecepatan transisi spektral terhadap waktu.
   - Delta-Delta ($\Delta\Delta$): Akselerasi dinamika akustik.

---

### 2.2. Constant-Q Transform (CQT) untuk Sinyal Musik
Berbeda dengan STFT yang memiliki lebar filter konstan, CQT memiliki rasio kualitas konstan $Q$:
$$Q = \frac{f_k}{\Delta f_k} = \text{konstan}, \quad N_k = Q \cdot \frac{f_s}{f_k}$$
- **Frekuensi Rendah (Bass/Kick)**: Jendela waktu panjang $\rightarrow$ **resolusi frekuensi sangat tajam** (dapat membedakan nada $C1$ dan $C\#1$).
- **Frekuensi Tinggi (Hi-Hat/Air)**: Jendela waktu pendek $\rightarrow$ **resolusi waktu sangat cepat** (menangkap ketukan perkusif tajam).
- **Bins per Octave ($B$)**: Dengan $B=12$ atau $24$, setiap bin frekuensi berkorespondensi tepat dengan semitone nada musik.

---

### 2.3. Parameter Akustik Forensik yang Dapat Dijelaskan (Explainable Descriptors)
1. **Spectral Centroid**: Titik pusat massa spektrum (menunjukkan kecerahan suara):
   $$\mu_{\text{Centroid}} = \frac{\sum_{k} f_k |X[k]|}{\sum_{k} |X[k]|}$$
2. **Spectral Bandwidth**: Lebar sebaran frekuensi di sekitar centroid:
   $$\mu_{\text{Bandwidth}} = \sqrt{\frac{\sum_{k} (f_k - \mu_{\text{Centroid}})^2 |X[k]|}{\sum_{k} |X[k]|}}$$
3. **Spectral Rolloff ($R_{85}, R_{95}$)**: Titik frekuensi di mana $85\%$ atau $95\%$ total energi spektral berada:
   $$\sum_{k=0}^{K_{\text{rolloff}}} |X[k]|^2 = 0.85 \sum_{k=0}^{N/2} |X[k]|^2$$
   *Indikator Kunci AI*: Model sintesis AI sering meninggalkan cutoff buatan di $16\text{ kHz}$ atau $18\text{ kHz}$ akibat vocoder bandlimiting, menyebabkan nilai $R_{95}$ drop drastis.
4. **Spectral Flatness (Wiener Entropy)**: Rasio mean geometrik terhadap mean aritmatik:
   $$\text{SF} = \frac{\exp\left(\frac{1}{K}\sum_{k=1}^K \ln |X[k]|^2\right)}{\frac{1}{K}\sum_{k=1}^K |X[k]|^2}$$
   Bernilai mendekati $0$ untuk instrumen harmonis bernada murni, dan mendekati $1$ untuk white noise.
5. **Zero-Crossing Rate (ZCR)**: Frekuensi pemotongan sumbu nol oleh sinyal waktu.

---

## 3. Arsitektur Deep Learning untuk Audio & Deteksi Deepfake

### 3.1. Pendekatan Computer Vision (2D CNN pada CQT / Mel-Spektrogram)
Dengan mengubah audio menjadi gambar 2D ($C \times F \times T$):
- **ResNet-18 / ResNet-50**: Menggunakan skip connections $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ yang memungkinkan pelatihan jaringan dalam tanpa gradien lenyap (vanishing gradients).
- **EfficientNet**: Melakukan compound scaling pada kedalaman (depth), lebar kanal (width), dan resolusi spektrogram secara harmonis.
- **Karakteristik Deteksi**: AI Generator (Suno, Udio, ElevenLabs) meninggalkan artefak mikro berupa pola kotak-kotak (checkerboard patterns) dari dekonvolusi vocoder, blurring frekuensi tinggi, dan diskontinuitas fase yang sangat mudah dideteksi oleh kernel konvolusi 2D.

### 3.2. SincNet & RawNet2
Alih-alih mengizinkan konvolusi 1D belajar filter acak, SincNet membatasi filter lapis pertama pada fungsi sinc band-pass:
$$g[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n)$$
Model hanya mempelajari batas bawah $f_1$ dan batas atas $f_2$ untuk setiap filter. RawNet2 menggabungkan SincNet dengan blok residual dan feature map scaling untuk langsung mengklasifikasikan raw waveform.

### 3.3. AASIST (Integrated Spectro-Temporal Graph Attention Networks)
AASIST adalah model state-of-the-art untuk mendeteksi audio spoofing:
1. **Front-End**: SincNet mengekstraksi peta fitur representasi spektro-temporal awal dari raw waveform.
2. **Konstruksi Graf Spektro-Temporal**:
   - Node Spektral ($G_s$): Dibentuk dari max-pooling terhadap domain waktu.
   - Node Temporal ($G_t$): Dibentuk dari max-pooling terhadap domain frekuensi.
   - Graf Heterogen: Menggabungkan node waktu dan frekuensi menjadi satu struktur graf utuh.
3. **Heterogeneous Stacking Graph Attention Layers (HS-GAL)**: Mekanisme graph attention mempelajari bobot hubungan antar-node waktu dan node frekuensi yang mencurigakan.
4. **Readout Layer & Output**: Menghasilkan skor probabilitas bonafide vs spoof.

### 3.4. Contrastive Learning & Representation Clustering (CLAD)
Pada data audio nyata yang mengalami noise atau manipulasi volume, model supervised biasa sering gagal. Metode CLAD menggunakan:
- **Momentum Contrast (MoCo)**: Memelihara memory queue untuk sampel negatif ($K=6144$) dan melatih encoder agar vektor representasi audio asli berkumpul dekat dan audio manipulasi/spoof saling berjauhan.
- **Length Loss**: Memaksa vektor audio asli memiliki panjang vektor $\|q\|_2$ yang pendek (berkumpul dekat origin) dan audio spoof memiliki panjang vektor yang jauh di luar margin:
  $$\mathcal{L}_{\text{len}} = \frac{1}{N}\sum_{i=1}^N \left[ y_i \cdot w \cdot \|q_i\|_2 + (1 - y_i) \cdot \max(0, \text{margin} - \|q_i\|_2) \right]$$

---

## 4. Alur Rekayasa End-to-End: Membangun, Melatih, dan Menguji Model

### Tahap 1: Kurasi Dataset & Higiene Data
1. **Keseimbangan Kelas**: 50% Bonafide (Musik Asli Manusia, misal MUSDB18 / FMA / LibriSpeech) dan 50% Spoof (AI Generator, misal Suno / Udio / ElevenLabs).
2. **Penyelarasan Transkrip / Konten**: Pada speech, kalimat harus identik; pada musik, genre dan tempo harus seimbang agar model tidak belajar shortcut bias.
3. **Pemisahan Partisi Disjoint (Wajib Mencegah Data Leakage)**:
   - **Speaker/Artist-Disjoint**: Artis atau penutur yang ada di Training Set **TIDAK BOLEH** muncul di Validation Set atau Test Set.
   - Standar pembagian: 60% Train, 20% Validation, 20% Test (atau 70/15/15).

### Tahap 2: Standardisasi Sinyal
1. **Resampling**: Diseragamkan ke 16.000 Hz atau 44.100 Hz menggunakan filter polifase.
2. **Konversi Mono**:
   $$x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$$
3. **Normalisasi Amplitudo Puncak**:
   $$x_{\text{norm}}[n] = \frac{x[n]}{\max(|x[n]|) + 10^{-7}}$$
4. **Penyesuaian Durasi Tetap ($N_{\text{target}} = f_s \times T_{\text{target}}$)**:
   - Jika audio lebih panjang: Truncation (potong awal hingga target).
   - Jika audio lebih pendek: **Repeat-Padding** (ulang waveform hingga memenuhi target, lebih unggul daripada zero padding karena mempertahankan pola frekuensi).

### Tahap 3: Augmentasi & Injeksi Gangguan
Untuk melatih model yang tangguh (robust) terhadap kondisi dunia nyata:
- **Additive White Gaussian Noise (AWGN)**:
  1. Hitung daya sinyal: $P_{\text{signal}} = \frac{1}{N}\sum x[n]^2$
  2. Hitung daya noise target: $P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}}/10}}$
  3. Bangkitkan noise: $w[n] \sim \mathcal{N}(0, \sqrt{P_{\text{noise}}})$
  4. Tambahkan ke sinyal: $y[n] = x[n] + w[n]$
- **Volume Scaling**: Pengurangan amplitudo sebesar faktor $0.1 - 0.9$.
- **Fading**: Fade-in dan fade-out half-sinusoidal.

### Tahap 4: Protokol Pelatihan & Optimasi
- **Fungsi Loss**: Cross-Entropy Loss dengan class weight.
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Learning Rate**: $1 \times 10^{-4}$ dengan Cosine Annealing scheduler.
- **Early Stopping & Pemilihan Checkpoint**:
  - Pantau nilai **Validation Equal Error Rate (EER)**, bukan training loss!
  - Simpan checkpoint saat Validation EER mencapai rekor terendah.
  - Hentikan pelatihan jika dalam 10 epoch tidak ada perbaikan.

---

## 5. Protokol Evaluasi, Metrik Biometrik & Analisis Forensik

### 5.1. Matriks Kesalahan (Confusion Matrix)
- **True Positive (TP)**: Audio asli diprediksi benar sebagai Asli.
- **True Negative (TN)**: Audio AI diprediksi benar sebagai AI/Spoof.
- **False Positive (FP)**: Audio AI salah diterima sebagai Asli (**Pelanggaran Keamanan Kritis / False Accept**).
- **False Negative (FN)**: Audio asli salah ditolak sebagai AI (**False Rejection**).

### 5.2. Metrik Standar Internasional
1. **False Acceptance Rate (FAR)**:
   $$\text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
2. **False Rejection Rate (FRR)**:
   $$\text{FRR} = \frac{\text{FN}}{\text{FN} + \text{TP}}$$
3. **Equal Error Rate (EER)**:
   Titik ekuilibrium di mana False Acceptance Rate tepat sama dengan False Rejection Rate:
   $$\text{EER} = \text{FAR}(\theta^*) = \text{FRR}(\theta^*)$$
   *Semakin kecil nilai EER, semakin unggul kemampuan model dalam memisahkan kedua kelas.*
4. **Kurva ROC & Area Under Curve (AUC)**:
   Menampilkan performa separasi skor pada seluruh kemungkinan nilai threshold. Nilai $\text{AUC} \to 1.0$ menunjukkan separasi sempurna.

---

## 6. Ketahanan Model, Serangan Manipulasi Audio & Pertahanan

Berdasarkan studi empiris terbaru (FTUI 2026, Procedia Computer Science):
1. **Model Pretrained Mengalami Kerentanan pada Data Noisy**:
   Model yang hanya dilatih pada kondisi clean mengalami kenaikan EER drastis hingga $> 40\%$ saat diuji pada audio dengan noise AWGN atau fading.
2. **Strategi Fine-Tuning Multi-Level AWGN**:
   - Model yang di-fine-tune pada noise ringan (20 dB, 15 dB) belum cukup kuat menghadapi noise berat.
   - Model yang di-fine-tune pada noise terlalu ekstrem (-5 dB) mengalami kesulitan belajar karena informasi akustik tertutup noise.
   - **Kondisi Optimal**: Fine-tuning pada **AWGN 0 dB (Daya Noise = Daya Sinyal)** memberikan titik keseimbangan terbaik (Pareto optimal) — mempertahankan EER $0.00\%$ pada audio clean dan mencapai EER terendah pada kondisi noisy ekstrem.

---

## 7. Arsitektur Produksi, MLOps & Integrasi Aplikasi Forensik

```
[File Audio Pengguna (.wav/.mp3)]
               │
               ▼
   [FastAPI Backend Server]
   ├── Audio Loader (torchaudio / librosa)
   ├── DSP Engine (STFT / Mel / CQT / Centroid / Cutoff Hz)
   └── Torch / ONNX Runtime Inference Engine
               │
               ├──► Prediksi Kelas: Bonafide vs AI-Generated (Probabilitas & EER Threshold)
               ├──► Deteksi Cutoff Frekuensi (Brickwall Cutoff Finder)
               └──► Payload 3D Waterfall, Spektrogram 2D, & Parametric EQ
               │
               ▼
   [Frontend Visualizer Dashboard (HTML5 / WebGL / Canvas)]
   ├── 2D Spektrogram dengan Garis Deteksi AI Cutoff
   ├── 3D Waterfall 360° Drag & Zoom
   └── Real-Time Parametric EQ 2 Spectrum Analyzer
```

---

## 8. Blueprint Kode PyTorch Lengkap (Siap Pakai)

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import roc_curve

# -------------------------------------------------------------
# 1. Dataset & Preprocessing Pipeline
# -------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_sr=16000, duration_sec=4.0, snr_db=None):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration_sec)
        self.snr_db = snr_db
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=20.0,
            f_max=target_sr // 2
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0)

    def __len__(self):
        return len(self.file_paths)

    def _add_awgn(self, waveform, snr_target):
        sig_pwr = torch.mean(waveform ** 2)
        if sig_pwr <= 0:
            return waveform
        noise_pwr = sig_pwr / (10.0 ** (snr_target / 10.0))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_pwr)
        return waveform + noise

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load & downmix mono
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # Resample
        if sr != self.target_sr:
            wav = T.Resample(sr, self.target_sr)(wav)
            
        # Normalisasi amplitudo
        max_amp = torch.max(torch.abs(wav))
        if max_amp > 0:
            wav = wav / max_amp
            
        # Penyesuaian durasi (Repeat-padding atau Truncation)
        n_samples = wav.shape[-1]
        if n_samples < self.target_samples:
            repeats = math.ceil(self.target_samples / n_samples)
            wav = wav.repeat(1, repeats)[:, :self.target_samples]
        else:
            wav = wav[:, :self.target_samples]
            
        # Injeksi AWGN
        if self.snr_db is not None:
            wav = self._add_awgn(wav, self.snr_db)
            
        # Konversi ke Mel-Spektrogram
        spec = self.amp_to_db(self.mel_transform(wav))
        return spec, torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------
# 2. Arsitektur Model: 2D Residual Spectrogram Classifier
# -------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + res
        return F.relu(out)

class AudioClassifierModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)
        return logits, feat

# -------------------------------------------------------------
# 3. Metrik Evaluasi: Perhitungan EER
# -------------------------------------------------------------
def calculate_eer(bona_scores, spoof_scores):
    labels = np.array([1] * len(bona_scores) + [0] * len(spoof_scores))
    scores = np.concatenate([bona_scores, spoof_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100.0, thresholds[idx]
```
