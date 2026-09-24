"""
Baseline Model Evaluation & Reporting Tool
Generates formatted markdown evaluation reports, confusion matrices, and benchmark summaries.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Ensure package importability
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml_baseline.config import MODELS_DIR, REPORTS_DIR
else:
    from .config import MODELS_DIR, REPORTS_DIR

def generate_report():
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        print("[!] No metrics.json found in models directory. Please run train.py first.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "baseline_evaluation_report.md"

    best_name = data.get("best_model_name", "N/A")
    results = data.get("results", {})
    top_features = data.get("top_features", [])
    summary = data.get("dataset_summary", {})

    lines = [
        "# AI Music Detector: Baseline Acoustic Model Evaluation",
        "",
        f"**Date:** {data.get('timestamp', 'N/A')}",
        f"**Champion Baseline Architecture:** `{best_name}`",
        "",
        "## 1. Dataset & Partitioning Summary",
        f"- **Total Slices:** {summary.get('total_samples', 'N/A')} (5.0s window, 2.5s hop)",
        f"- **Train Slices:** {summary.get('train_samples', 'N/A')}",
        f"- **Test Slices:** {summary.get('test_samples', 'N/A')}",
        f"- **Acoustic Feature Dimension:** {summary.get('num_features', 'N/A')} forensic features",
        f"- **Validation Scheme:** Grouped Track-Level Split (75% train / 25% test, 0% song-chunk leakage)",
        "",
        "## 2. Model Benchmark Comparison",
        "",
        "| Classifier | Accuracy | F1-Score | ROC-AUC | Equal Error Rate (EER) | Train Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for model_name, m in results.items():
        lines.append(
            f"| **{model_name}** | {m['accuracy']*100:.2f}% | {m['f1_score']*100:.2f}% | {m['roc_auc']*100:.2f}% | {m['eer']*100:.2f}% | {m['train_time_sec']}s |"
        )

    lines.extend([
        "",
        "## 3. Best Model Confusion Matrix",
        ""
    ])

    best_metrics = results.get(best_name, {})
    cm = best_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    lines.extend([
        f"- **True Bona-fide (Human) predicted as Human:** {cm[0][0]}",
        f"- **True Bona-fide (Human) misclassified as AI (False Alarm):** {cm[0][1]}",
        f"- **True Spoof (AI) misclassified as Human (Miss):** {cm[1][0]}",
        f"- **True Spoof (AI) predicted as AI (Detection):** {cm[1][1]}",
        "",
        "## 4. Top Discriminative Acoustic Forensic Features",
        "The baseline Random Forest classifier identified the following top acoustic features distinguishing AI (Suno) from Human Studio (MUSDB18) audio:",
        ""
    ])

    for rank, f_info in enumerate(top_features[:10], 1):
        lines.append(f"{rank}. **`{f_info['feature']}`**: {f_info['importance']*100:.2f}% importance")

    lines.extend([
        "",
        "## 5. Metric Explanations for Thesis",
        "- **Equal Error Rate (EER):** The threshold point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). Lower is better (0.00% is perfect). This is the golden standard metric in ASVspoof and audio deepfake benchmarks.",
        "- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures the model's discriminative ability across all possible classification thresholds.",
        "- **Track-Level Grouping:** Ensures segments from the same song are restricted to either train or test, avoiding overly optimistic evaluation caused by identical audio backgrounds."
    ])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[*] Evaluation report generated at: {report_file}")

if __name__ == "__main__":
    generate_report()
