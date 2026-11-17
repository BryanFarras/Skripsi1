"""
Publication-Grade PyTorch Training Log Visualizer
Generates high-resolution (300 DPI PNG + Vector PDF) charts illustrating:
00. Comprehensive 6-Panel Master Training Progression Dashboard
01. Training vs Validation Loss & Accuracy Progression per Epoch
02. Forensic Audio Metrics Progression (ROC-AUC, F1-Score, Precision, Recall, EER Reduction)
03. Optimization Dynamics: Cosine Annealing Learning Rate & L2 Gradient Norm Stability
04. Model Architecture Evolution (v1 Shallow MLP -> v2 Deep MLP -> v3 Residual DeepAudioClassifier)
05. 5-Fold StratifiedGroupKFold Cross-Validation Progression (Mean +/- SD Confidence Bands)
06. Receiver Operating Characteristic (ROC) & FAR/FRR Equal Error Rate (EER) Threshold Curves
07. Best Checkpoint Confusion Matrix & Predicted Probability Distribution
Also generates a structured Markdown report in `pytorch_training/reports/pytorch_training_report.md`.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any

# Ensure package importability
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pytorch_training.config import LOGS_DIR, PLOTS_DIR, REPORTS_DIR
else:
    from .config import LOGS_DIR, PLOTS_DIR, REPORTS_DIR


# Set clean academic publication aesthetic
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.2,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Color Palette
COLORS = {
    "train": "#2563EB",       # Royal Blue
    "val": "#DC2626",         # Crimson Red
    "bal_acc": "#059669",     # Emerald Green
    "auc": "#7C3AED",         # Vivid Purple
    "f1": "#2563EB",          # Blue
    "prec": "#D97706",        # Amber
    "rec": "#0D9488",         # Teal
    "eer": "#E11D48",         # Rose Red
    "lr": "#4F46E5",          # Indigo
    "grad": "#EA580C",        # Orange
    "v1": "#94A3B8",          # Slate Gray
    "v2": "#F59E0B",          # Amber Gold
    "v3": "#10B981",          # Emerald (Flagship)
    "v3_dark": "#047857"      # Dark Emerald
}


def save_figure(fig: plt.Figure, filename_stem: str):
    """Saves figure in both 300 DPI PNG and vector PDF formats."""
    png_path = PLOTS_DIR / f"{filename_stem}.png"
    pdf_dir = PLOTS_DIR / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{filename_stem}.pdf"

    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"    [+] Saved: {png_path.name} (& PDF)")


def moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Computes simple moving average with edge padding for smooth trendlines."""
    if len(arr) < window:
        return arr
    series = pd.Series(arr)
    return series.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def plot_01_loss_and_accuracy(df_main: pd.DataFrame, best_info: Dict[str, Any]):
    """Plot 01: Training vs Validation Loss and Accuracy Progression over Epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        "Perkembangan Loss dan Akurasi Pelatihan Model PyTorch (DeepAudioClassifier)",
        y=1.03
    )

    epochs = df_main["epoch"].to_numpy()
    best_ep = int(best_info["epoch"])

    # Panel 1: Loss Curve
    ax1.plot(epochs, df_main["train_loss"], label="Training Loss (BCE)", color=COLORS["train"], lw=2.2)
    ax1.plot(epochs, df_main["val_loss"], label="Validation Loss (BCE)", color=COLORS["val"], lw=2.2)
    ax1.fill_between(
        epochs, df_main["train_loss"], df_main["val_loss"],
        color=COLORS["val"], alpha=0.07, label="Generalization Gap"
    )
    ax1.axvline(best_ep, color="#111827", linestyle=":", lw=1.6, label=f"Best Checkpoint (Epoch {best_ep})")
    ax1.scatter([best_ep], [best_info["val_loss"]], color=COLORS["val"], s=70, zorder=5, edgecolor="white", lw=1.5)
    ax1.set_title("A. Kurva Konvergensi Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # Panel 2: Accuracy Curve
    ax2.plot(epochs, df_main["train_acc"] * 100, label="Training Accuracy", color=COLORS["train"], lw=2.0, alpha=0.85)
    ax2.plot(epochs, df_main["val_acc"] * 100, label="Validation Accuracy", color=COLORS["val"], lw=2.2)
    ax2.plot(
        epochs, df_main["val_balanced_acc"] * 100,
        label="Validation Balanced Accuracy", color=COLORS["bal_acc"], lw=2.0, linestyle="--"
    )
    ax2.axvline(best_ep, color="#111827", linestyle=":", lw=1.6, label=f"Best Epoch ({best_info['val_acc']*100:.2f}%)")
    ax2.scatter([best_ep], [best_info["val_acc"] * 100], color=COLORS["val"], s=70, zorder=5, edgecolor="white", lw=1.5)
    ax2.set_title("B. Perkembangan Akurasi & Balanced Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Akurasi (%)")
    ax2.set_ylim(bottom=max(50, (df_main["train_acc"].min() * 100) - 5), top=100.5)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, "01_loss_and_accuracy_curve")


def plot_02_forensic_metrics(df_main: pd.DataFrame, best_info: Dict[str, Any]):
    """Plot 02: Progression of ROC-AUC, F1-Score, Precision, Recall, and EER Reduction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        "Perkembangan Metrik Forensik Audio (ROC-AUC, F1-Score, dan Penurunan EER)",
        y=1.03
    )

    epochs = df_main["epoch"].to_numpy()
    best_ep = int(best_info["epoch"])

    # Panel 1: ROC-AUC, F1, Precision, Recall
    ax1.plot(epochs, df_main["val_roc_auc"] * 100, label="ROC-AUC", color=COLORS["auc"], lw=2.4)
    ax1.plot(epochs, df_main["val_f1"] * 100, label="F1-Score", color=COLORS["f1"], lw=2.1)
    ax1.plot(epochs, df_main["val_precision"] * 100, label="Precision", color=COLORS["prec"], lw=1.6, linestyle="--", alpha=0.85)
    ax1.plot(epochs, df_main["val_recall"] * 100, label="Recall (Sensitivity)", color=COLORS["rec"], lw=1.6, linestyle="-.", alpha=0.85)
    ax1.axvline(best_ep, color="#111827", linestyle=":", lw=1.5, label=f"Best Epoch {best_ep}")
    ax1.set_title("A. Perkembangan Metrik Klasifikasi Validasi per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Skor Metrik (%)")
    ax1.set_ylim(bottom=max(60, (df_main["val_f1"].min() * 100) - 5), top=100.5)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    # Panel 2: Equal Error Rate (EER) Reduction
    eer_pct = df_main["val_eer"].to_numpy() * 100
    eer_smooth = moving_average(eer_pct, window=5)
    ax2.plot(epochs, eer_pct, color=COLORS["eer"], alpha=0.35, lw=1.4, label="Validation EER (Raw)")
    ax2.plot(epochs, eer_smooth, color=COLORS["eer"], lw=2.4, label="Validation EER (Trend MA-5)")
    ax2.axvline(best_ep, color="#111827", linestyle=":", lw=1.5, label=f"Best Epoch ({best_info['val_eer']*100:.2f}% EER)")
    ax2.scatter([best_ep], [best_info["val_eer"] * 100], color=COLORS["eer"], s=75, zorder=5, edgecolor="white", lw=1.5)
    ax2.annotate(
        f"Optimal EER: {best_info['val_eer']*100:.2f}%\n(Epoch {best_ep})",
        xy=(best_ep, best_info["val_eer"] * 100),
        xytext=(max(5, best_ep - 22), best_info["val_eer"] * 100 + 4.5),
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF1F2", edgecolor=COLORS["eer"], alpha=0.9)
    )
    ax2.set_title("B. Penurunan Equal Error Rate (EER) per Epoch (Lebih Rendah = Lebih Baik)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Equal Error Rate / EER (%)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, "02_forensic_metrics_progression")


def plot_03_lr_and_gradients(df_main: pd.DataFrame):
    """Plot 03: Cosine Annealing Learning Rate Schedule & Gradient Norm Stability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle(
        "Dinamika Optimasi PyTorch: Peluruhan Learning Rate & Stabilitas Gradien",
        y=1.03
    )

    epochs = df_main["epoch"].to_numpy()
    lrs = df_main["learning_rate"].to_numpy()
    grads = df_main["grad_norm"].to_numpy()
    grads_smooth = moving_average(grads, window=5)

    # Panel 1: Learning Rate Schedule
    ax1.plot(epochs, lrs * 1e3, color=COLORS["lr"], lw=2.5, label="CosineAnnealingLR")
    ax1.fill_between(epochs, 0, lrs * 1e3, color=COLORS["lr"], alpha=0.1)
    ax1.set_title("A. Jadwal Peluruhan Learning Rate (Cosine Annealing)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning Rate ($\\times 10^{-3}$)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: L2 Gradient Norm
    ax2.plot(epochs, grads, color=COLORS["grad"], alpha=0.35, lw=1.4, label="Batch Mean $L_2$ Grad Norm")
    ax2.plot(epochs, grads_smooth, color=COLORS["grad"], lw=2.4, label="Moving Average (MA-5)")
    ax2.axhline(2.0, color="#64748B", linestyle="--", lw=1.3, label="Gradient Clipping Threshold (2.0)")
    ax2.set_title("B. Stabilitas Norma Gradien ($L_2$ Norm) Selama Pelatihan")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("$L_2$ Gradient Norm")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    fig.tight_layout()
    save_figure(fig, "03_lr_and_gradient_dynamics")


def plot_04_architecture_evolution(arch_data: Dict[str, Any], summary_data: Dict[str, Any]):
    """Plot 04: Evolution across PyTorch Model Generations (v1 -> v2 -> v3) & Baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.suptitle(
        "Perkembangan Evolusi Arsitektur Model PyTorch (v1 Shallow -> v2 Deep MLP -> v3 Residual DeepAudioClassifier)",
        y=1.03
    )

    histories = arch_data["histories"]
    best_arch = arch_data["best_per_architecture"]

    arch_meta = [
        ("v1_ShallowMLP", "v1: Shallow MLP (1-Layer)", COLORS["v1"], "--"),
        ("v2_DeepMLP", "v2: Deep MLP (4-Layer + BN/Drop)", COLORS["v2"], "-."),
        ("v3_DeepResAudioNet", "v3: DeepAudioClassifier (Residual)", COLORS["v3"], "-")
    ]

    # Panel 1: Validation Loss Comparison
    ax1 = axes[0]
    for key, label, color, ls in arch_meta:
        df_a = pd.DataFrame(histories[key])
        ax1.plot(df_a["epoch"], moving_average(df_a["val_loss"].to_numpy(), 3), label=label, color=color, lw=2.2, linestyle=ls)
    ax1.set_title("A. Konvergensi Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss (BCE)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: Validation ROC-AUC Comparison
    ax2 = axes[1]
    for key, label, color, ls in arch_meta:
        df_a = pd.DataFrame(histories[key])
        ax2.plot(df_a["epoch"], moving_average(df_a["val_roc_auc"].to_numpy() * 100, 3), label=label, color=color, lw=2.2, linestyle=ls)
    ax2.set_title("B. Perkembangan Validation ROC-AUC (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ROC-AUC (%)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white")

    # Panel 3: Bar Chart Benchmark across Generations & Classical Baselines
    ax3 = axes[2]
    model_labels = [
        "RF\n(Baseline)",
        "SVM-RBF\n(Baseline)",
        "PyTorch v1\n(Shallow)",
        "PyTorch v2\n(Deep MLP)",
        "PyTorch v3\n(ResMLP)"
    ]
    baselines = summary_data.get("classical_baselines", {})
    rf_m = baselines.get("Random Forest", {})
    svm_m = baselines.get("Support Vector Machine (RBF)", {})

    acc_vals = [
        rf_m.get("balanced_accuracy", 0.79) * 100,
        svm_m.get("balanced_accuracy", 0.835) * 100,
        best_arch["v1_ShallowMLP"]["val_balanced_acc"] * 100,
        best_arch["v2_DeepMLP"]["val_balanced_acc"] * 100,
        best_arch["v3_DeepResAudioNet"]["val_balanced_acc"] * 100,
    ]
    auc_vals = [
        rf_m.get("roc_auc", 0.881) * 100,
        svm_m.get("roc_auc", 0.933) * 100,
        best_arch["v1_ShallowMLP"]["val_roc_auc"] * 100,
        best_arch["v2_DeepMLP"]["val_roc_auc"] * 100,
        best_arch["v3_DeepResAudioNet"]["val_roc_auc"] * 100,
    ]

    x = np.arange(len(model_labels))
    width = 0.36
    bars1 = ax3.bar(x - width / 2, acc_vals, width, label="Balanced Acc (%)", color="#3B82F6", alpha=0.88)
    bars2 = ax3.bar(x + width / 2, auc_vals, width, label="ROC-AUC (%)", color="#10B981", alpha=0.88)

    for b in bars1:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for b in bars2:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax3.set_title("C. Komparasi Lintas Generasi Model")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, fontsize=8.8)
    ax3.set_ylabel("Skor Evaluasi (%)")
    ax3.set_ylim(70, 103)
    ax3.legend(loc="lower right", frameon=True, facecolor="white")

    fig.tight_layout()
    save_figure(fig, "04_architecture_evolution_comparison")


def plot_05_kfold_cv(df_kfold: pd.DataFrame, kfold_summary: Dict[str, Any]):
    """Plot 05: 5-Fold StratifiedGroupKFold Cross-Validation Progression & Stability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    fig.suptitle(
        "Evaluasi Ketahanan Generalisasi Lintas Lagu: 5-Fold StratifiedGroupKFold (PyTorch)",
        y=1.03
    )

    # Group by epoch to compute mean and std across folds
    grouped = df_kfold.groupby("epoch")
    epochs = grouped["val_roc_auc"].mean().index.to_numpy()
    auc_mean = grouped["val_roc_auc"].mean().to_numpy() * 100
    auc_std = grouped["val_roc_auc"].std().to_numpy() * 100
    acc_mean = grouped["val_balanced_acc"].mean().to_numpy() * 100
    acc_std = grouped["val_balanced_acc"].std().to_numpy() * 100

    # Panel 1: Mean +/- 1 SD Curve across Epochs
    ax1.plot(epochs, auc_mean, color=COLORS["auc"], lw=2.3, label="Mean Validation ROC-AUC (5 Folds)")
    ax1.fill_between(epochs, auc_mean - auc_std, np.minimum(100, auc_mean + auc_std), color=COLORS["auc"], alpha=0.18, label="±1 SD Band (ROC-AUC)")

    ax1.plot(epochs, acc_mean, color=COLORS["bal_acc"], lw=2.1, linestyle="--", label="Mean Balanced Accuracy (5 Folds)")
    ax1.fill_between(epochs, acc_mean - acc_std, np.minimum(100, acc_mean + acc_std), color=COLORS["bal_acc"], alpha=0.15, label="±1 SD Band (Bal-Acc)")

    ax1.set_title("A. Kurva Rata-Rata ± Simpangan Baku (5-Fold StratifiedGroupKFold)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Skor Metrik (%)")
    ax1.set_ylim(65, 101)
    ax1.legend(loc="lower right", frameon=True, facecolor="white")

    # Panel 2: Per-Fold Final Metrics Bar Chart
    folds_list = kfold_summary["folds"]
    agg = kfold_summary["aggregate_metrics"]
    labels = [f"Fold {f['fold']}" for f in folds_list] + ["Mean\n(5-Fold)"]

    f_bal_acc = [f["val_balanced_acc"] * 100 for f in folds_list] + [agg["val_balanced_acc"]["mean"] * 100]
    f_f1 = [f["val_f1"] * 100 for f in folds_list] + [agg["val_f1"]["mean"] * 100]
    f_auc = [f["val_roc_auc"] * 100 for f in folds_list] + [agg["val_roc_auc"]["mean"] * 100]

    x = np.arange(len(labels))
    w = 0.26
    ax2.bar(x - w, f_bal_acc, w, label="Balanced Acc (%)", color="#3B82F6")
    ax2.bar(x, f_f1, w, label="F1-Score (%)", color="#F59E0B")
    ax2.bar(x + w, f_auc, w, label="ROC-AUC (%)", color="#10B981")

    for i, v in enumerate(f_auc):
        ax2.text(x[i] + w, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    ax2.axvline(len(folds_list) - 0.5, color="#334155", linestyle=":", lw=1.4)
    ax2.set_title("B. Konsistensi Performa Antar-Fold (Isolasi Lagu Penuh)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Skor (%)")
    ax2.set_ylim(65, 103)
    ax2.legend(loc="lower right", frameon=True, facecolor="white")

    fig.tight_layout()
    save_figure(fig, "05_kfold_stratified_group_cv")


def plot_06_roc_and_eer_curves(roc_npz: Any, arch_data: Dict[str, Any]):
    """Plot 06: ROC Curve across PyTorch Generations & FAR/FRR EER Threshold Analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        "Analisis Kurva ROC dan Penentuan Ambang Batas Equal Error Rate (EER)",
        y=1.03
    )

    best_arch = arch_data["best_per_architecture"]
    arch_meta = [
        ("v1_ShallowMLP", "v1: Shallow MLP", COLORS["v1"], "--"),
        ("v2_DeepMLP", "v2: Deep MLP", COLORS["v2"], "-."),
        ("v3_DeepResAudioNet", "v3: DeepAudioClassifier (Flagship)", COLORS["v3"], "-")
    ]

    # Panel 1: ROC Curves
    for key, name, color, ls in arch_meta:
        fpr = roc_npz[f"{key}_fpr"]
        tpr = roc_npz[f"{key}_tpr"]
        auc_val = best_arch[key]["val_roc_auc"] * 100
        eer_val = best_arch[key]["val_eer"] * 100
        ax1.plot(fpr * 100, tpr * 100, label=f"{name} (AUC: {auc_val:.2f}%, EER: {eer_val:.2f}%)", color=color, lw=2.3, linestyle=ls)

    ax1.plot([0, 100], [0, 100], color="#94A3B8", linestyle=":", lw=1.2, label="Random Chance (50%)")
    ax1.set_title("A. Kurva Receiver Operating Characteristic (ROC)")
    ax1.set_xlabel("False Positive Rate / FAR (%)")
    ax1.set_ylabel("True Positive Rate / TPR (%)")
    ax1.set_xlim(-1, 100)
    ax1.set_ylim(0, 101)
    ax1.legend(loc="lower right", frameon=True, facecolor="white")

    # Panel 2: FAR vs FRR over Thresholds for v3 Flagship
    fpr_v3 = roc_npz["v3_DeepResAudioNet_fpr"] * 100
    tpr_v3 = roc_npz["v3_DeepResAudioNet_tpr"] * 100
    fnr_v3 = 100.0 - tpr_v3
    thresh_v3 = roc_npz["v3_DeepResAudioNet_thresholds"]

    valid_mask = (thresh_v3 >= 0.0) & (thresh_v3 <= 1.0)
    t_clean = thresh_v3[valid_mask]
    far_clean = fpr_v3[valid_mask]
    frr_clean = fnr_v3[valid_mask]

    eer_v3 = best_arch["v3_DeepResAudioNet"]["val_eer"] * 100
    eer_th = best_arch["v3_DeepResAudioNet"]["val_eer_threshold"]

    ax2.plot(t_clean, far_clean, label="False Acceptance Rate (FAR - Human as AI)", color="#DC2626", lw=2.2)
    ax2.plot(t_clean, frr_clean, label="False Rejection Rate (FRR - AI as Human)", color="#2563EB", lw=2.2)
    ax2.scatter([eer_th], [eer_v3], color="#111827", s=80, zorder=6, label=f"Titik EER = {eer_v3:.2f}% (Thresh = {eer_th:.2f})")
    ax2.axvline(eer_th, color="#111827", linestyle=":", lw=1.4)
    ax2.axhline(eer_v3, color="#111827", linestyle=":", lw=1.4)

    ax2.set_title("B. Kurva FAR vs FRR & Titik Potong EER (DeepAudioClassifier)")
    ax2.set_xlabel("Ambang Batas Keputusan Probabilitas (Decision Threshold)")
    ax2.set_ylabel("Tingkat Kesalahan / Error Rate (%)")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0, 60)
    ax2.legend(loc="upper center", frameon=True, facecolor="white")

    fig.tight_layout()
    save_figure(fig, "06_roc_and_eer_threshold_curve")


def plot_07_confusion_and_distribution(roc_npz: Any, best_info: Dict[str, Any]):
    """Plot 07: Confusion Matrix & Predicted Probability Density of Best PyTorch Model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        "Diagnostik Prediksi Model PyTorch Terbaik: Matriks Konfusi & Pemisahan Kelas",
        y=1.03
    )

    cm = np.array(best_info["confusion_matrix"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    # Panel 1: Confusion Matrix Heatmap
    im = ax1.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    class_labels = ["Bona-fide\n(Manusia)", "Spoof\n(Musik AI)"]
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(class_labels, fontweight="bold")
    ax1.set_yticklabels(class_labels, fontweight="bold")
    ax1.set_xlabel("Label Prediksi Model")
    ax1.set_ylabel("Label Acuan Sebenarnya (Ground Truth)")
    ax1.set_title("A. Matriks Konfusi pada Data Uji Terisolasi Lagu")
    ax1.grid(False)

    for i in range(2):
        for j in range(2):
            val_count = cm[i, j]
            val_p = cm_pct[i, j]
            txt_color = "white" if val_p > 50 else "#1E293B"
            ax1.text(
                j, i, f"{val_count} segmen\n({val_p:.1f}%)",
                ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold"
            )

    # Panel 2: Probability Distribution Histogram
    y_true = roc_npz["v3_DeepResAudioNet_y_true"]
    y_prob = roc_npz["v3_DeepResAudioNet_y_prob"]

    human_probs = y_prob[y_true == 0]
    ai_probs = y_prob[y_true == 1]

    bins = np.linspace(0, 1, 26)
    ax2.hist(human_probs, bins=bins, alpha=0.65, color="#2563EB", edgecolor="white", label=f"Bona-fide / Manusia (n={len(human_probs)})", density=True)
    ax2.hist(ai_probs, bins=bins, alpha=0.65, color="#DC2626", edgecolor="white", label=f"Spoof / Musik AI (n={len(ai_probs)})", density=True)
    ax2.axvline(0.5, color="#111827", linestyle="--", lw=1.6, label="Ambang Keputusan Standar (0.50)")
    ax2.set_title("B. Distribusi Probabilitas Prediksi $P(\\mathrm{Spoof} \\mid \\mathbf{x})$")
    ax2.set_xlabel("Skor Probabilitas Terkalibrasi Sigmoid")
    ax2.set_ylabel("Kepadatan Relatif (Density)")
    ax2.legend(loc="upper center", frameon=True, facecolor="white")

    fig.tight_layout()
    save_figure(fig, "07_confusion_matrix_and_score_dist")


def plot_00_master_dashboard(
    df_main: pd.DataFrame,
    best_info: Dict[str, Any],
    arch_data: Dict[str, Any],
    df_kfold: pd.DataFrame,
    roc_npz: Any
):
    """Plot 00: 6-Panel Executive Dashboard of PyTorch Model Training Progression."""
    fig = plt.figure(figsize=(18, 10.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.25)
    fig.suptitle(
        "DASHBOARD LOG PELATIHAN & PERKEMBANGAN MODEL PYTORCH (MY-AIDETECTOR)",
        fontsize=15.5, fontweight="bold", y=0.98
    )

    epochs = df_main["epoch"].to_numpy()
    best_ep = int(best_info["epoch"])

    # 1. Loss Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, df_main["train_loss"], label="Train Loss", color=COLORS["train"], lw=2.0)
    ax1.plot(epochs, df_main["val_loss"], label="Val Loss", color=COLORS["val"], lw=2.0)
    ax1.axvline(best_ep, color="#111827", linestyle=":", lw=1.4, label=f"Best Epoch ({best_ep})")
    ax1.set_title("1. Konvergensi BCE Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 2. Accuracy & Balanced Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, df_main["train_acc"] * 100, label="Train Acc", color=COLORS["train"], lw=1.8, alpha=0.8)
    ax2.plot(epochs, df_main["val_acc"] * 100, label="Val Acc", color=COLORS["val"], lw=2.1)
    ax2.plot(epochs, df_main["val_balanced_acc"] * 100, label="Val Balanced Acc", color=COLORS["bal_acc"], lw=1.9, linestyle="--")
    ax2.axvline(best_ep, color="#111827", linestyle=":", lw=1.4)
    ax2.set_title("2. Perkembangan Akurasi per Epoch (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Akurasi (%)")
    ax2.legend(loc="lower right")

    # 3. Forensic Metrics & EER
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, df_main["val_roc_auc"] * 100, label="ROC-AUC (%)", color=COLORS["auc"], lw=2.1)
    ax3.plot(epochs, df_main["val_f1"] * 100, label="F1-Score (%)", color=COLORS["f1"], lw=1.9)
    ax3.plot(epochs, (1.0 - df_main["val_eer"]) * 100, label="100% - EER (%)", color=COLORS["eer"], lw=1.9, linestyle="-.")
    ax3.axvline(best_ep, color="#111827", linestyle=":", lw=1.4)
    ax3.set_title("3. Metrik Forensik (ROC-AUC, F1, 1-EER)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Skor (%)")
    ax3.legend(loc="lower right")

    # 4. Architecture Evolution (v1 vs v2 vs v3)
    ax4 = fig.add_subplot(gs[1, 0])
    histories = arch_data["histories"]
    for key, label, color, ls in [
        ("v1_ShallowMLP", "v1: Shallow MLP", COLORS["v1"], "--"),
        ("v2_DeepMLP", "v2: Deep MLP", COLORS["v2"], "-."),
        ("v3_DeepResAudioNet", "v3: DeepAudioClassifier (ResMLP)", COLORS["v3"], "-")
    ]:
        df_a = pd.DataFrame(histories[key])
        ax4.plot(df_a["epoch"], moving_average(df_a["val_roc_auc"].to_numpy() * 100, 3), label=label, color=color, lw=2.1, linestyle=ls)
    ax4.set_title("4. Evolusi Generasi Arsitektur (ROC-AUC %)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation ROC-AUC (%)")
    ax4.legend(loc="lower right")

    # 5. 5-Fold StratifiedGroupKFold
    ax5 = fig.add_subplot(gs[1, 1])
    grouped = df_kfold.groupby("epoch")
    ep_k = grouped["val_roc_auc"].mean().index.to_numpy()
    auc_m = grouped["val_roc_auc"].mean().to_numpy() * 100
    auc_s = grouped["val_roc_auc"].std().to_numpy() * 100
    acc_m = grouped["val_balanced_acc"].mean().to_numpy() * 100
    acc_s = grouped["val_balanced_acc"].std().to_numpy() * 100
    ax5.plot(ep_k, auc_m, color=COLORS["auc"], lw=2.1, label="5-Fold Mean ROC-AUC")
    ax5.fill_between(ep_k, auc_m - auc_s, np.minimum(100, auc_m + auc_s), color=COLORS["auc"], alpha=0.18)
    ax5.plot(ep_k, acc_m, color=COLORS["bal_acc"], lw=2.0, linestyle="--", label="5-Fold Mean Bal-Acc")
    ax5.fill_between(ep_k, acc_m - acc_s, np.minimum(100, acc_m + acc_s), color=COLORS["bal_acc"], alpha=0.14)
    ax5.set_title("5. Validasi Silang 5-Fold StratifiedGroupKFold")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Skor (%)")
    ax5.legend(loc="lower right")

    # 6. ROC Curves
    ax6 = fig.add_subplot(gs[1, 2])
    best_arch = arch_data["best_per_architecture"]
    for key, name, color, ls in [
        ("v1_ShallowMLP", "v1 Shallow", COLORS["v1"], "--"),
        ("v2_DeepMLP", "v2 Deep MLP", COLORS["v2"], "-."),
        ("v3_DeepResAudioNet", "v3 ResMLP", COLORS["v3"], "-")
    ]:
        fpr = roc_npz[f"{key}_fpr"] * 100
        tpr = roc_npz[f"{key}_tpr"] * 100
        auc_v = best_arch[key]["val_roc_auc"] * 100
        ax6.plot(fpr, tpr, label=f"{name} ({auc_v:.2f}%)", color=color, lw=2.1, linestyle=ls)
    ax6.plot([0, 100], [0, 100], color="#94A3B8", linestyle=":", lw=1.2)
    ax6.set_title("6. Kurva ROC Antar-Generasi Model")
    ax6.set_xlabel("False Positive Rate (%)")
    ax6.set_ylabel("True Positive Rate (%)")
    ax6.legend(loc="lower right")

    save_figure(fig, "00_master_training_dashboard")


def generate_markdown_report(summary_data: Dict[str, Any]):
    """Writes a clean Markdown report summarizing all PyTorch training logs and charts."""
    best = summary_data["flagship_best_model"]
    arch = summary_data["architecture_progression"]
    kfold = summary_data["kfold_cross_validation"]["aggregate"]
    folds = summary_data["kfold_cross_validation"]["per_fold"]
    ds = summary_data["dataset_split"]
    hp = summary_data["hyperparameters"]

    report_path = REPORTS_DIR / "pytorch_training_report.md"
    lines = [
        "# Laporan Log Pelatihan & Perkembangan Model PyTorch (`DeepAudioClassifier`)",
        "",
        f"> **Tanggal Eksekusi:** `{summary_data['timestamp']}`  ",
        f"> **Framework:** `PyTorch {summary_data['pytorch_version']}` (`{summary_data['device']}`)  ",
        f"> **Dataset:** `{ds['total_samples']}` segmen audio (`{ds['input_features']}` fitur akustik) dari `{ds['total_tracks']}` lagu unik  ",
        f"> **Skema Isolasi Lagu:** Track-Level Grouping (`{ds['train_samples']}` latih / `{ds['val_samples']}` uji) & `{summary_data['kfold_cross_validation']['n_splits']}`-Fold `StratifiedGroupKFold`",
        "",
        "---",
        "",
        "## 1. Ringkasan Konfigurasi Hyperparameter Pelatihan",
        "",
        "| Parameter | Nilai / Spesifikasi |",
        "| :--- | :--- |",
        f"| **Arsitektur Utama (v3)** | `DeepAudioClassifier` (4-Layer Residual MLP: `108 -> 256 -> 128 -> 64 -> 32 -> 1`) |",
        f"| **Jumlah Parameter Terlatih** | `{best.get('num_parameters', 86081):,}` parameter |",
        f"| **Optimizer & Weight Decay** | `{hp['optimizer']}` (`lr={hp['learning_rate']}`, `weight_decay={hp['weight_decay']}`) |",
        f"| **Learning Rate Scheduler** | `{hp['scheduler']}` (`eta_min={hp['min_lr']}`) |",
        f"| **Fungsi Loss** | `{hp['loss_function']}` + Gradient Clipping (`max_norm=2.0`) |",
        f"| **Jumlah Epoch & Batch Size** | `{hp['epochs']}` Epochs, Batch Size `{hp['batch_size']}`, Seed `{hp['random_seed']}` |",
        "",
        "---",
        "",
        "## 2. Perkembangan Antar-Generasi Arsitektur Model PyTorch",
        "",
        "| Generasi Model | Parameter | Best Epoch | Val Loss | Accuracy | Balanced Acc | F1-Score | ROC-AUC | EER (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, label in [
        ("v1_ShallowMLP", "**v1: ShallowAudioClassifier** (1-Layer MLP)"),
        ("v2_DeepMLP", "**v2: DeepMLPClassifier** (4-Layer + BN/Dropout)"),
        ("v3_DeepResAudioNet", "**v3: DeepAudioClassifier** (4-Layer + Residual Skip)")
    ]:
        m = arch[key]
        lines.append(
            f"| {label} | {m['num_parameters']:,} | {m['epoch']} | {m['val_loss']:.4f} | "
            f"{m['val_acc']*100:.2f}% | {m['val_balanced_acc']*100:.2f}% | "
            f"{m['val_f1']*100:.2f}% | {m['val_roc_auc']*100:.2f}% | {m['val_eer']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Hasil Validasi Silang 5-Fold `StratifiedGroupKFold` (`DeepAudioClassifier`)",
        "",
        "| Fold | Best Epoch | Val Loss | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | EER (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for f in folds:
        lines.append(
            f"| **Fold {f['fold']}** | {f['epoch']} | {f['val_loss']:.4f} | "
            f"{f['val_acc']*100:.2f}% | {f['val_balanced_acc']*100:.2f}% | "
            f"{f['val_precision']*100:.2f}% | {f['val_recall']*100:.2f}% | "
            f"{f['val_f1']*100:.2f}% | {f['val_roc_auc']*100:.2f}% | {f['val_eer']*100:.2f}% |"
        )

    lines.append(
        f"| **Mean ± SD** | — | — | "
        f"**{kfold['val_acc']['mean']*100:.2f}% ± {kfold['val_acc']['std']*100:.2f}%** | "
        f"**{kfold['val_balanced_acc']['mean']*100:.2f}% ± {kfold['val_balanced_acc']['std']*100:.2f}%** | "
        f"**{kfold['val_precision']['mean']*100:.2f}% ± {kfold['val_precision']['std']*100:.2f}%** | "
        f"**{kfold['val_recall']['mean']*100:.2f}% ± {kfold['val_recall']['std']*100:.2f}%** | "
        f"**{kfold['val_f1']['mean']*100:.2f}% ± {kfold['val_f1']['std']*100:.2f}%** | "
        f"**{kfold['val_roc_auc']['mean']*100:.2f}% ± {kfold['val_roc_auc']['std']*100:.2f}%** | "
        f"**{kfold['val_eer']['mean']*100:.2f}% ± {kfold['val_eer']['std']*100:.2f}%** |"
    )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Daftar Grafik Visualisasi Log Pelatihan (`pytorch_training/plots/`)",
        "",
        "Seluruh grafik tersedia dalam resolusi cetak **300 DPI (`.png`)** dan format vektor **(`.pdf`)** untuk integrasi langsung ke naskah LaTeX skripsi:",
        "",
        "1. `00_master_training_dashboard.png` — Dashboard terpadu 6 panel seluruh perkembangan pelatihan model PyTorch.",
        "2. `01_loss_and_accuracy_curve.png` — Kurva konvergensi *Training vs Validation Loss* dan *Accuracy / Balanced Accuracy* per epoch.",
        "3. `02_forensic_metrics_progression.png` — Perkembangan metrik forensik (*ROC-AUC*, *F1-Score*, *Precision*, *Recall*) dan penurunan *Equal Error Rate (EER)* per epoch.",
        "4. `03_lr_and_gradient_dynamics.png` — Dinamika jadwal peluruhan *Cosine Annealing Learning Rate* dan stabilitas norma gradien $L_2$.",
        "5. `04_architecture_evolution_comparison.png` — Perbandingan perkembangan antar-generasi arsitektur PyTorch (`v1` $\\rightarrow$ `v2` $\\rightarrow$ `v3`) serta komparasi terhadap baseline klasik.",
        "6. `05_kfold_stratified_group_cv.png` — Kurva rata-rata dan pita kepercayaan ($\\pm 1\\text{ SD}$) pada evaluasi 5-Fold `StratifiedGroupKFold`.",
        "7. `06_roc_and_eer_threshold_curve.png` — Kurva ROC lintas generasi model dan kurva analisis ambang batas *FAR vs FRR* (titik potong EER).",
        "8. `07_confusion_matrix_and_score_dist.png` — Matriks konfusi pada data uji terisolasi lagu dan distribusi kepadatan probabilitas prediksi $P(\\text{Spoof}\\mid \\mathbf{x})$.",
        ""
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    [+] Saved Markdown Report: {report_path}")


def generate_all_plots():
    """Loads all logged training artifacts and generates the complete visual suite."""
    print("================================================================================")
    print("   GENERATING PYTORCH TRAINING PROGRESSION CHARTS (300 DPI PNG & PDF)")
    print("================================================================================")

    df_main = pd.read_csv(LOGS_DIR / "training_history.csv")
    df_kfold = pd.read_csv(LOGS_DIR / "kfold_training_history.csv")

    with open(LOGS_DIR / "architecture_progression.json", "r", encoding="utf-8") as f:
        arch_data = json.load(f)
    with open(LOGS_DIR / "kfold_summary.json", "r", encoding="utf-8") as f:
        kfold_summary = json.load(f)
    with open(LOGS_DIR / "training_summary.json", "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    roc_npz = np.load(LOGS_DIR / "roc_eer_curves.npz")
    best_info = summary_data["flagship_best_model"]

    plot_01_loss_and_accuracy(df_main, best_info)
    plot_02_forensic_metrics(df_main, best_info)
    plot_03_lr_and_gradients(df_main)
    plot_04_architecture_evolution(arch_data, summary_data)
    plot_05_kfold_cv(df_kfold, kfold_summary)
    plot_06_roc_and_eer_curves(roc_npz, arch_data)
    plot_07_confusion_and_distribution(roc_npz, best_info)
    plot_00_master_dashboard(df_main, best_info, arch_data, df_kfold, roc_npz)
    generate_markdown_report(summary_data)

    print("\n[*] All charts successfully saved in:")
    print(f"    - PNG (300 DPI): {PLOTS_DIR}")
    print(f"    - PDF (Vector):  {PLOTS_DIR / 'pdf'}")


if __name__ == "__main__":
    generate_all_plots()

