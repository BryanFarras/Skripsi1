# PyTorch Training & Model Progression Pipeline (`pytorch_training`)

Modul ini menyediakan alur pelatihan (*training pipeline*), pencatatan log metrik per epoch, komparasi evolusi arsitektur jaringan saraf tiruan, validasi silang *5-Fold StratifiedGroupKFold*, serta generator grafik visualisasi beresolusi tinggi (300 DPI `.png` & vektor `.pdf`) untuk kebutuhan naskah skripsi.

---

## 📁 Struktur Direktori

```text
pytorch_training/
├── __init__.py
├── config.py                     # Konfigurasi path, seed, dan hyperparameter pelatihan
├── dataset_loader.py             # PyTorch Dataset & DataLoader dengan isolasi grup lagu (Track-Level Split)
├── train.py                      # Skrip pelatihan utama (v1, v2, v3 + 5-Fold StratifiedGroupKFold)
├── plot_training_logs.py         # Generator grafik perkembangan model (300 DPI PNG & Vektor PDF)
├── README.md                     # Dokumentasi modul PyTorch Training
├── models/
│   ├── __init__.py
│   └── deep_audio_classifier.py  # Definisi arsitektur v1 (ShallowMLP), v2 (DeepMLP), v3 (DeepAudioClassifier)
├── checkpoints/                  # Bobot model PyTorch terlatih (.pt)
│   ├── best_deep_audio_classifier.pt
│   └── last_checkpoint.pt
├── logs/                         # Log numerik hasil pelatihan (CSV, JSON, NPZ)
│   ├── training_history.csv      # Log per epoch model utama (Loss, Acc, Bal-Acc, F1, ROC-AUC, EER, LR, Grad Norm)
│   ├── training_history.json     # Format JSON dari riwayat epoch model utama
│   ├── architecture_progression.csv  # Riwayat epoch gabungan v1, v2, dan v3
│   ├── architecture_progression.json # Ringkasan dan riwayat komparasi antar-generasi arsitektur
│   ├── kfold_training_history.csv    # Log per epoch untuk seluruh 5 fold (StratifiedGroupKFold)
│   ├── kfold_summary.json            # Statistik agregat Mean ± SD dari 5-Fold CV
│   ├── roc_eer_curves.npz            # Array kurva FPR, TPR, Threshold, dan probabilitas prediksi
│   └── training_summary.json         # Ringkasan eksekutif seluruh eksperimen pelatihan
├── plots/                        # Grafik visualisasi perkembangan model (PNG 300 DPI & subfolder PDF)
│   ├── 00_master_training_dashboard.png
│   ├── 01_loss_and_accuracy_curve.png
│   ├── 02_forensic_metrics_progression.png
│   ├── 03_lr_and_gradient_dynamics.png
│   ├── 04_architecture_evolution_comparison.png
│   ├── 05_kfold_stratified_group_cv.png
│   ├── 06_roc_and_eer_threshold_curve.png
│   ├── 07_confusion_matrix_and_score_dist.png
│   └── pdf/                      # Versi vektor (.pdf) untuk penyematan langsung di LaTeX
└── reports/
    └── pytorch_training_report.md    # Laporan lengkap hasil pelatihan dan analisis tabel metrik
```

---

## 🚀 Cara Menjalankan Ulang

Gunakan lingkungan virtual proyek (`.venv`):

```powershell
# 1. Jalankan pelatihan model dan pencatatan log
C:\Users\asus\Documents\SKRIPSI\.venv\Scripts\python.exe pytorch_training/train.py

# 2. Bangkitkan seluruh grafik visualisasi dan laporan Markdown
C:\Users\asus\Documents\SKRIPSI\.venv\Scripts\python.exe pytorch_training/plot_training_logs.py
```

