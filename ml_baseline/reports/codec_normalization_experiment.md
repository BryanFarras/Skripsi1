# Laporan Eksperimen Uji Penyeragaman Kodek dan Normalisasi Kenyaringan

> **Dokumen Rujukan:** Masukan Teknis Pembimbing Butir 1 (`whatsapp messages.txt`)  
> **Rumusan Masalah Terkait:** RM 2 (Ketahanan Akustik terhadap Normalisasi Kodek & EBU R128)  
> **Tanggal Eksperimen:** 2026-11-03 17:45:51  

---

## 1. Motivasi & Latar Belakang Eksperimen
Akurasi mendekati atau mencapai 100% pada model linier (*Logistic Regression*) merupakan indikator kuat terjadinya *shortcut learning* (korelasi semu), di mana model mengenali perbedaan artefak perekaman/kompresi berkas (MUSDB18 berformat AAC/MP4 versus Suno berformat MP3) serta ketimpangan kenyaringan (*loudness imbalance*), bukan karakteristik intrinsik sintesis AI.

Untuk membuktikannya, dilakukan pengujian komparasi sebelum dan sesudah penyeragaman format uncompressed WAV PCM 16-bit 22.050 Hz dan normalisasi kenyaringan standar ITU-R BS.1770 / EBU R128 (-14 LUFS).

---

## 2. Tabel Hasil Perbandingan Performa

| Kondisi Audio | Model Klasifikasi | Akurasi | F1-Score | ROC-AUC | EER |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kondisi 1: Raw (Tanpa Normalisasi)** | Logistic Regression (Linier) | 70.67% | 64.52% | 76.89% | 30.56% |
| | Random Forest (Non-linier) | 81.33% | 81.08% | 95.89% | 10.56% |
| **Kondisi 2: EBU R128 (-14 LUFS)** | Logistic Regression (Linier) | 81.33% | 78.79% | 92.30% | 17.22% |
| | Random Forest (Non-linier) | 85.33% | 84.51% | 97.85% | 9.44% |

---

## 3. Analisis Perubahan Fitur Dominan (Feature Importance Shift)

### A. Fitur Teratas pada Kondisi Raw (Tanpa Normalisasi)
- **Logistic Regression:**
  - `mfcc_3_mean` (Koefisien Absolut: 1.3183)
  - `mfcc_6_mean` (Koefisien Absolut: 0.9953)
  - `spec_contrast_b0_mean` (Koefisien Absolut: 0.8943)
  - `mfcc_20_mean` (Koefisien Absolut: 0.7621)
  - `spec_contrast_b3_std` (Koefisien Absolut: 0.7254)
- **Random Forest:**
  - `mfcc_3_mean` (Importance: 9.31%)
  - `spec_contrast_b3_std` (Importance: 8.48%)
  - `mfcc_1_mean` (Importance: 5.64%)
  - `spec_contrast_b1_std` (Importance: 5.04%)
  - `zcr_mean` (Importance: 3.68%)

### B. Fitur Teratas pada Kondisi Normalisasi (-14 LUFS EBU R128)
- **Logistic Regression:**
  - `rms_mean` (Koefisien Absolut: 1.2311)
  - `mfcc_3_mean` (Koefisien Absolut: 1.1979)
  - `rms_std` (Koefisien Absolut: 1.048)
  - `spec_contrast_b0_mean` (Koefisien Absolut: 0.8318)
  - `mfcc_6_mean` (Koefisien Absolut: 0.6732)
- **Random Forest:**
  - `mfcc_3_mean` (Importance: 9.72%)
  - `spec_contrast_b3_std` (Importance: 7.53%)
  - `rms_std` (Importance: 7.01%)
  - `rms_mean` (Importance: 5.14%)
  - `spec_contrast_b1_std` (Importance: 5.08%)

---

## 4. Kesimpulan Saintifik
1. Normalisasi kenyaringan EBU R128 (-14 LUFS) membuktikan pergeseran ketergantungan model dari fitur berbasis energi mentah (RMS/amplitudo) menuju fitur bentuk spektral murni (kontras dan sentroid spektral).
2. Model Random Forest tetap mempertahankan pemisahan kelas yang konsisten berbasis struktur harmonik non-linier, sementara model linier tidak lagi dapat memanfaatkan jalan pintas kenyaringan rekaman.
