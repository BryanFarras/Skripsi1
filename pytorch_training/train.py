"""
PyTorch Training & Model Development Logging Pipeline
Executes:
1. Primary Training of DeepAudioClassifier (v3 ResMLP) with full epoch-by-epoch logging.
2. Architectural Evolution Benchmark:
   - v1: ShallowAudioClassifier (Basic 1-Layer MLP)
   - v2: DeepMLPClassifier (4-Layer Deep MLP with BatchNorm & Dropout)
   - v3: DeepAudioClassifier (4-Layer Residual Deep MLP with Skip Projection)
3. 5-Fold StratifiedGroupKFold Cross-Validation to verify track-isolated generalization.
Saves structured logs (CSV, JSON, NPZ) and model checkpoints (.pt).
"""

import os
import sys
import csv
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# Ensure package importability
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pytorch_training.config import (
        CHECKPOINTS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        BASELINE_METRICS_PATH,
        RANDOM_SEED,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        MIN_LR,
        WEIGHT_DECAY,
        LABEL_SMOOTHING,
        N_SPLITS
    )
    from pytorch_training.dataset_loader import load_raw_dataset, create_dataloaders
    from pytorch_training.models import (
        ShallowAudioClassifier,
        DeepMLPClassifier,
        DeepAudioClassifier
    )
else:
    from .config import (
        CHECKPOINTS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        BASELINE_METRICS_PATH,
        RANDOM_SEED,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        MIN_LR,
        WEIGHT_DECAY,
        LABEL_SMOOTHING,
        N_SPLITS
    )
    from .dataset_loader import load_raw_dataset, create_dataloaders
    from .models import (
        ShallowAudioClassifier,
        DeepMLPClassifier,
        DeepAudioClassifier
    )


def set_seed(seed: int = RANDOM_SEED):
    """Sets deterministic random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER), optimal threshold, and ROC/DET curves.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh, fpr, tpr, thresholds


def evaluate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluates a PyTorch model on validation DataLoader and returns comprehensive forensic metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            by_smooth = by * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = criterion(logits, by_smooth)
            probs = torch.sigmoid(logits)

            batch_n = bx.size(0)
            total_loss += loss.item() * batch_n
            total_samples += batch_n

            all_probs.append(probs.cpu().numpy())
            all_targets.append(by.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    val_loss = total_loss / max(total_samples, 1)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    eer, eer_thresh, _, _, _ = calculate_eer(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "val_loss": float(val_loss),
        "val_acc": float(acc),
        "val_balanced_acc": float(bal_acc),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_f1": float(f1),
        "val_roc_auc": float(auc),
        "val_eer": float(eer),
        "val_eer_threshold": float(eer_thresh),
        "confusion_matrix": cm
    }, y_true, y_prob


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device = torch.device("cpu"),
    save_checkpoints: bool = False,
    scaler: Any = None,
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Trains a PyTorch model across `epochs` and logs all training/validation progression metrics.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LR)

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_epoch_info: Dict[str, Any] = {}
    best_y_true = np.array([])
    best_y_prob = np.array([])
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        grad_norms = []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            logits = model(bx)
            by_smooth = by * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = criterion(logits, by_smooth)
            loss.backward()

            # Compute total L2 gradient norm before optimizer step
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            grad_norms.append(total_norm_sq ** 0.5)

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            batch_n = bx.size(0)
            train_loss_sum += loss.item() * batch_n
            train_correct += (preds == by).sum().item()
            train_total += batch_n

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        # Validation evaluation
        val_metrics, y_true, y_prob = evaluate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        # Composite selection score (prioritizing Balanced Accuracy, ROC-AUC, F1, and low EER)
        composite_score = (
            0.35 * val_metrics["val_balanced_acc"]
            + 0.30 * val_metrics["val_roc_auc"]
            + 0.20 * val_metrics["val_f1"]
            + 0.15 * (1.0 - val_metrics["val_eer"])
        )

        record = {
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_metrics["val_acc"], 6),
            "val_balanced_acc": round(val_metrics["val_balanced_acc"], 6),
            "val_precision": round(val_metrics["val_precision"], 6),
            "val_recall": round(val_metrics["val_recall"], 6),
            "val_f1": round(val_metrics["val_f1"], 6),
            "val_roc_auc": round(val_metrics["val_roc_auc"], 6),
            "val_eer": round(val_metrics["val_eer"], 6),
            "val_eer_threshold": round(val_metrics["val_eer_threshold"], 6),
            "learning_rate": round(current_lr, 8),
            "grad_norm": round(mean_grad_norm, 6),
            "epoch_time_sec": round(epoch_time, 4),
            "confusion_matrix": val_metrics["confusion_matrix"]
        }
        history.append(record)

        if composite_score > best_score:
            best_score = composite_score
            best_epoch_info = dict(record)
            best_y_true = y_true.copy()
            best_y_prob = y_prob.copy()
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if save_checkpoints:
                ckpt_path = CHECKPOINTS_DIR / "best_deep_audio_classifier.pt"
                torch.save({
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": best_state_dict,
                    "metrics": best_epoch_info,
                    "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
                    "scaler_scale": scaler.scale_.tolist() if scaler is not None else None
                }, ckpt_path)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"    Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']*100:6.2f}% | "
                f"ROC-AUC: {val_metrics['val_roc_auc']*100:6.2f}% | "
                f"EER: {val_metrics['val_eer']*100:5.2f}% | "
                f"LR: {current_lr:.6f}"
            )

    if save_checkpoints:
        last_ckpt_path = CHECKPOINTS_DIR / "last_checkpoint.pt"
        torch.save({
            "epoch": epochs,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "metrics": history[-1]
        }, last_ckpt_path)

    return history, best_epoch_info, best_y_true, best_y_prob


def save_csv_log(records: List[Dict[str, Any]], csv_path: Path):
    """Writes a list of epoch dictionary records to a clean CSV file."""
    if not records:
        return
    headers = [k for k in records[0].keys() if k != "confusion_matrix"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run_all_training_experiments() -> Dict[str, Any]:
    """
    Executes the complete PyTorch training and progression logging suite.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================================================================================")
    print("   PYTORCH DEEP AUDIO CLASSIFIER - TRAINING & PROGRESSION LOGGING PIPELINE")
    print("================================================================================")
    print(f"[*] Device: {device} | PyTorch Version: {torch.__version__} | Seed: {RANDOM_SEED}")

    raw_data = load_raw_dataset()
    X = raw_data["X"]
    y = raw_data["y"]
    groups = raw_data["groups"]
    input_dim = X.shape[1]

    print(f"[*] Loaded Dataset: {len(X)} slices, {input_dim} acoustic features across {len(set(groups))} tracks.")

    # -------------------------------------------------------------------------
    # EXPERIMENT 1 & 2: Track-Grouped Holdout Split (75% Train / 25% Val)
    # Matches ml_baseline GroupShuffleSplit for fair comparison across v1, v2, v3
    # -------------------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    train_loader, val_loader, scaler, _, _ = create_dataloaders(X, y, train_idx, val_idx, batch_size=BATCH_SIZE)

    print(f"[*] Track-Level Split: {len(train_idx)} train slices ({len(set(groups[train_idx]))} tracks) | "
          f"{len(val_idx)} val slices ({len(set(groups[val_idx]))} tracks)")

    model_builders = {
        "v1_ShallowMLP": lambda: ShallowAudioClassifier(input_dim=input_dim, hidden_dim=64),
        "v2_DeepMLP": lambda: DeepMLPClassifier(input_dim=input_dim, hidden_dims=[256, 128, 64, 32]),
        "v3_DeepResAudioNet": lambda: DeepAudioClassifier(input_dim=input_dim, hidden_dims=[256, 128, 64, 32])
    }

    arch_histories: Dict[str, List[Dict[str, Any]]] = {}
    arch_best_summary: Dict[str, Dict[str, Any]] = {}
    all_arch_rows: List[Dict[str, Any]] = []
    roc_curves_payload: Dict[str, np.ndarray] = {}

    for arch_key, builder_fn in model_builders.items():
        print(f"\n[*] Training PyTorch Model Generation: {arch_key} ({EPOCHS} Epochs)...")
        set_seed(RANDOM_SEED)
        # Recreate dataloaders with deterministic generator per run
        train_loader, val_loader, scaler, _, _ = create_dataloaders(X, y, train_idx, val_idx, batch_size=BATCH_SIZE)
        model = builder_fn()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable Parameters: {num_params:,}")

        is_flagship = (arch_key == "v3_DeepResAudioNet")
        history, best_info, best_y_true, best_y_prob = train_single_model(
            model=model,
            model_name=arch_key,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            device=device,
            save_checkpoints=is_flagship,
            scaler=scaler,
            verbose=True
        )

        best_info["num_parameters"] = int(num_params)
        arch_histories[arch_key] = history
        arch_best_summary[arch_key] = best_info
        all_arch_rows.extend(history)

        eer, eer_thresh, fpr, tpr, thresholds = calculate_eer(best_y_true, best_y_prob)
        roc_curves_payload[f"{arch_key}_y_true"] = best_y_true
        roc_curves_payload[f"{arch_key}_y_prob"] = best_y_prob
        roc_curves_payload[f"{arch_key}_fpr"] = fpr
        roc_curves_payload[f"{arch_key}_tpr"] = tpr
        roc_curves_payload[f"{arch_key}_thresholds"] = thresholds

        print(f"    -> Best Epoch for {arch_key}: Epoch {best_info['epoch']} | "
              f"Acc: {best_info['val_acc']*100:.2f}% | Bal-Acc: {best_info['val_balanced_acc']*100:.2f}% | "
              f"F1: {best_info['val_f1']*100:.2f}% | ROC-AUC: {best_info['val_roc_auc']*100:.2f}% | "
              f"EER: {best_info['val_eer']*100:.2f}%")

    # Save Flagship v3 primary logs
    flagship_history = arch_histories["v3_DeepResAudioNet"]
    save_csv_log(flagship_history, LOGS_DIR / "training_history.csv")
    with open(LOGS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(flagship_history, f, indent=2)

    # Save Architectural Progression logs
    save_csv_log(all_arch_rows, LOGS_DIR / "architecture_progression.csv")
    with open(LOGS_DIR / "architecture_progression.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_per_architecture": arch_best_summary,
            "histories": arch_histories
        }, f, indent=2)

    # Save ROC & prediction arrays
    np.savez_compressed(LOGS_DIR / "roc_eer_curves.npz", **roc_curves_payload)

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: 5-Fold StratifiedGroupKFold Cross-Validation on v3 Flagship
    # -------------------------------------------------------------------------
    print(f"\n[*] Running {N_SPLITS}-Fold StratifiedGroupKFold Cross-Validation on DeepAudioClassifier...")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)
    kfold_rows: List[Dict[str, Any]] = []
    kfold_best_per_fold: List[Dict[str, Any]] = []

    for fold_idx, (f_train_idx, f_val_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        set_seed(RANDOM_SEED + fold_idx)
        f_train_loader, f_val_loader, _, _, _ = create_dataloaders(
            X, y, f_train_idx, f_val_idx, batch_size=BATCH_SIZE
        )
        fold_model = DeepAudioClassifier(input_dim=input_dim, hidden_dims=[256, 128, 64, 32])
        f_history, f_best, _, _ = train_single_model(
            model=fold_model,
            model_name=f"Fold_{fold_idx}",
            train_loader=f_train_loader,
            val_loader=f_val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            device=device,
            save_checkpoints=False,
            verbose=False
        )
        for row in f_history:
            row["fold"] = fold_idx
        kfold_rows.extend(f_history)
        f_best["fold"] = fold_idx
        kfold_best_per_fold.append(f_best)

        print(f"    Fold {fold_idx}/{N_SPLITS} -> Best Epoch {f_best['epoch']:02d} | "
              f"Acc: {f_best['val_acc']*100:6.2f}% | Bal-Acc: {f_best['val_balanced_acc']*100:6.2f}% | "
              f"F1: {f_best['val_f1']*100:6.2f}% | ROC-AUC: {f_best['val_roc_auc']*100:6.2f}% | "
              f"EER: {f_best['val_eer']*100:5.2f}%")

    save_csv_log(kfold_rows, LOGS_DIR / "kfold_training_history.csv")

    kfold_aggregate = {}
    for metric_key in ["val_acc", "val_balanced_acc", "val_precision", "val_recall", "val_f1", "val_roc_auc", "val_eer"]:
        vals = [fb[metric_key] for fb in kfold_best_per_fold]
        kfold_aggregate[metric_key] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6)
        }

    with open(LOGS_DIR / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_splits": N_SPLITS,
            "aggregate_metrics": kfold_aggregate,
            "folds": kfold_best_per_fold
        }, f, indent=2)

    # Load classical baseline metrics if available for unified summary
    baseline_results = {}
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_results = baseline_data.get("results", {})

    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "min_lr": MIN_LR,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss_function": "BCEWithLogitsLoss",
            "random_seed": RANDOM_SEED
        },
        "dataset_split": {
            "total_samples": int(len(X)),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "total_tracks": int(len(set(groups))),
            "train_tracks": int(len(set(groups[train_idx]))),
            "val_tracks": int(len(set(groups[val_idx]))),
            "input_features": int(input_dim)
        },
        "flagship_best_model": arch_best_summary["v3_DeepResAudioNet"],
        "architecture_progression": arch_best_summary,
        "kfold_cross_validation": {
            "n_splits": N_SPLITS,
            "aggregate": kfold_aggregate,
            "per_fold": kfold_best_per_fold
        },
        "classical_baselines": baseline_results
    }

    with open(LOGS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n[*] All training logs and checkpoints saved to:")
    print(f"    - Logs Dir:        {LOGS_DIR}")
    print(f"    - Checkpoints Dir: {CHECKPOINTS_DIR}")
    return summary_payload


if __name__ == "__main__":
    run_all_training_experiments()
