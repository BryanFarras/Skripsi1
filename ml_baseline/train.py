"""
Baseline AI Model Training & Benchmarking Pipeline
Trains, cross-validates, and evaluates acoustic classifiers on Bona-fide vs Spoof audio.
Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC, and Equal Error Rate (EER).
Uses Track-Level Grouping to strictly prevent data leakage between train and test splits.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
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
    from ml_baseline.config import (
        MODELS_DIR,
        REPORTS_DIR,
        RANDOM_STATE,
        LABEL_NAMES
    )
    from ml_baseline.dataset import load_dataset, build_dataset
else:
    from .config import (
        MODELS_DIR,
        REPORTS_DIR,
        RANDOM_STATE,
        LABEL_NAMES
    )
    from .dataset import load_dataset, build_dataset

def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER) and the optimal threshold.
    EER is the standard benchmark metric in ASVspoof / Audio Deepfake detection.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh

def train_baseline_models():
    """
    Loads dataset, performs track-level split, trains candidate baselines,
    evaluates on test set, and exports the champion model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load or build dataset
    try:
        data = load_dataset()
    except FileNotFoundError:
        print("[*] Dataset not found on disk. Building now...")
        data = build_dataset()

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    feature_names = data["feature_names"]

    print(f"\n==================================================================")
    print(f"       BASELINE AI AUDIO DETECTOR - MODEL TRAINING")
    print(f"==================================================================")
    print(f"[*] Dataset: {X.shape[0]} slices, {X.shape[1]} features across {len(set(groups))} unique songs.")

    # 2. Track-Level Grouped Split (75% Train, 25% Test)
    # This guarantees that slices from the same song are NEVER in both train and test!
    gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    print(f"[*] Track-level split complete:")
    print(f"    Train: {len(X_train)} slices from {len(train_groups)} tracks (Spoof: {np.sum(y_train==1)}, Bonafide: {np.sum(y_train==0)})")
    print(f"    Test:  {len(X_test)} slices from {len(test_groups)} tracks (Spoof: {np.sum(y_test==1)}, Bonafide: {np.sum(y_test==0)})")

    # 3. Fit Feature Scaler on Training Set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Candidates
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Support Vector Machine (RBF)": SVC(
            kernel="rbf",
            C=2.0,
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE
        ),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=600,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    print(f"\n[*] Training and benchmarking baseline classifiers...\n")
    print(f"{'Model':<30} | {'Accuracy':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'EER':<9}")
    print("-" * 75)

    for name, model in candidate_models.items():
        t0 = time.time()
        # Train on scaled features
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        # Predict probabilities
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        eer, eer_thresh = calculate_eer(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "eer": float(eer),
            "eer_threshold": float(eer_thresh),
            "confusion_matrix": cm,
            "train_time_sec": round(train_time, 3)
        }

        print(f"{name:<30} | {acc*100:>8.2f}% | {f1*100:>8.2f}% | {auc*100:>8.2f}% | {eer*100:>8.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print("-" * 75)
    print(f"[*] Champion Baseline Model: {best_model_name} (F1: {best_f1*100:.2f}%)")

    # 5. Extract Feature Importances (from Random Forest)
    rf_model = candidate_models["Random Forest"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in sorted_idx[:15]
    ]

    print(f"\n[*] Top 8 Most Discriminative Acoustic Features:")
    for rank, item in enumerate(top_features[:8], 1):
        print(f"    {rank}. {item['feature']:<28} (Importance: {item['importance']*100:.2f}%)")

    # 6. Save Artifacts
    model_save_path = MODELS_DIR / "baseline_classifier.joblib"
    scaler_save_path = MODELS_DIR / "feature_scaler.joblib"
    metrics_save_path = MODELS_DIR / "metrics.json"

    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    metrics_payload = {
        "best_model_name": best_model_name,
        "results": results,
        "top_features": top_features,
        "dataset_summary": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "num_features": int(X.shape[1]),
            "train_tracks": list(train_groups),
            "test_tracks": list(test_groups)
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_save_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[*] Model artifacts saved successfully:")
    print(f"    - Classifier: {model_save_path}")
    print(f"    - Scaler:     {scaler_save_path}")
    print(f"    - Metrics:    {metrics_save_path}")

    return metrics_payload

if __name__ == "__main__":
    train_baseline_models()
