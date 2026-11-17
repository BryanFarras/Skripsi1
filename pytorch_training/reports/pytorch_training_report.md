# Laporan Log Pelatihan & Perkembangan Model PyTorch (`DeepAudioClassifier`)

> **Tanggal Eksekusi:** `2026-11-01 15:36:15`  
> **Framework:** `PyTorch 2.14.0+cpu` (`cpu`)  
> **Dataset:** `3627` segmen audio (`108` fitur akustik) dari `152` lagu unik  
> **Skema Isolasi Lagu:** Track-Level Grouping (`2764` latih / `863` uji) & `5`-Fold `StratifiedGroupKFold`

---

## 1. Ringkasan Konfigurasi Hyperparameter Pelatihan

| Parameter | Nilai / Spesifikasi |
| :--- | :--- |
| **Arsitektur Utama (v3)** | `DeepAudioClassifier` (4-Layer Residual MLP: `108 -> 256 -> 128 -> 64 -> 32 -> 1`) |
| **Jumlah Parameter Terlatih** | `88,577` parameter |
| **Optimizer & Weight Decay** | `AdamW` (`lr=0.0003`, `weight_decay=0.01`) |
| **Learning Rate Scheduler** | `CosineAnnealingLR` (`eta_min=1e-05`) |
| **Fungsi Loss** | `BCEWithLogitsLoss` + Gradient Clipping (`max_norm=2.0`) |
| **Jumlah Epoch & Batch Size** | `80` Epochs, Batch Size `64`, Seed `42` |

---

## 2. Perkembangan Antar-Generasi Arsitektur Model PyTorch

| Generasi Model | Parameter | Best Epoch | Val Loss | Accuracy | Balanced Acc | F1-Score | ROC-AUC | EER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1: ShallowAudioClassifier** (1-Layer MLP) | 7,041 | 25 | 0.3737 | 86.67% | 86.56% | 85.05% | 94.13% | 13.90% |
| **v2: DeepMLPClassifier** (4-Layer + BN/Dropout) | 72,129 | 33 | 0.3699 | 88.64% | 88.71% | 87.44% | 95.93% | 11.32% |
| **v3: DeepAudioClassifier** (4-Layer + Residual Skip) | 88,577 | 56 | 0.3736 | 89.11% | 89.56% | 88.37% | 95.94% | 11.24% |

---

## 3. Hasil Validasi Silang 5-Fold `StratifiedGroupKFold` (`DeepAudioClassifier`)

| Fold | Best Epoch | Val Loss | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | EER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 45 | 0.3802 | 88.48% | 88.58% | 93.94% | 82.67% | 87.94% | 94.96% | 11.25% |
| **Fold 2** | 68 | 0.3475 | 89.58% | 89.53% | 87.44% | 92.80% | 90.04% | 96.89% | 10.15% |
| **Fold 3** | 53 | 0.2889 | 93.03% | 93.02% | 93.14% | 92.61% | 92.88% | 97.63% | 6.84% |
| **Fold 4** | 52 | 0.2720 | 93.86% | 93.87% | 93.54% | 94.07% | 93.80% | 98.22% | 6.28% |
| **Fold 5** | 75 | 0.2780 | 93.44% | 93.43% | 93.47% | 93.20% | 93.33% | 97.95% | 6.43% |
| **Mean ± SD** | — | — | **91.68% ± 2.20%** | **91.69% ± 2.19%** | **92.30% ± 2.45%** | **91.07% ± 4.23%** | **91.60% ± 2.25%** | **97.13% ± 1.17%** | **8.19% ± 2.09%** |

---

## 4. Daftar Grafik Visualisasi Log Pelatihan (`pytorch_training/plots/`)

Seluruh grafik tersedia dalam resolusi cetak **300 DPI (`.png`)** dan format vektor **(`.pdf`)** untuk integrasi langsung ke naskah LaTeX skripsi:

1. `00_master_training_dashboard.png` — Dashboard terpadu 6 panel seluruh perkembangan pelatihan model PyTorch.
2. `01_loss_and_accuracy_curve.png` — Kurva konvergensi *Training vs Validation Loss* dan *Accuracy / Balanced Accuracy* per epoch.
3. `02_forensic_metrics_progression.png` — Perkembangan metrik forensik (*ROC-AUC*, *F1-Score*, *Precision*, *Recall*) dan penurunan *Equal Error Rate (EER)* per epoch.
4. `03_lr_and_gradient_dynamics.png` — Dinamika jadwal peluruhan *Cosine Annealing Learning Rate* dan stabilitas norma gradien $L_2$.
5. `04_architecture_evolution_comparison.png` — Perbandingan perkembangan antar-generasi arsitektur PyTorch (`v1` $\rightarrow$ `v2` $\rightarrow$ `v3`) serta komparasi terhadap baseline klasik.
6. `05_kfold_stratified_group_cv.png` — Kurva rata-rata dan pita kepercayaan ($\pm 1\text{ SD}$) pada evaluasi 5-Fold `StratifiedGroupKFold`.
7. `06_roc_and_eer_threshold_curve.png` — Kurva ROC lintas generasi model dan kurva analisis ambang batas *FAR vs FRR* (titik potong EER).
8. `07_confusion_matrix_and_score_dist.png` — Matriks konfusi pada data uji terisolasi lagu dan distribusi kepadatan probabilitas prediksi $P(\text{Spoof}\mid \mathbf{x})$.
