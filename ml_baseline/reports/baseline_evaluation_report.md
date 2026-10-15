# AI Music Detector: Baseline Acoustic Model Evaluation

**Date:** 2026-10-15 09:34:52
**Champion Baseline Architecture:** `Support Vector Machine (RBF)`

## 1. Dataset & Partitioning Summary
- **Total Slices:** 8297 (5.0s window, 2.5s hop)
- **Train Slices:** 6196
- **Test Slices:** 2101
- **Acoustic Feature Dimension:** 110 forensic features
- **Validation Scheme:** Grouped Track-Level Split (75% train / 25% test, 0% song-chunk leakage)

## 2. Model Benchmark Comparison

| Classifier | Accuracy | F1-Score | ROC-AUC | Equal Error Rate (EER) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 99.76% | 99.87% | 99.99% | 1.00% | 0.443s |
| **Support Vector Machine (RBF)** | 100.00% | 100.00% | 100.00% | 0.00% | 1.261s |
| **Logistic Regression** | 100.00% | 100.00% | 100.00% | 0.00% | 0.025s |
| **Multi-Layer Perceptron (MLP)** | 99.19% | 99.54% | 99.70% | 2.10% | 0.764s |

## 3. Best Model Confusion Matrix

- **True Bona-fide (Human) predicted as Human:** 245
- **True Bona-fide (Human) misclassified as AI (False Alarm):** 0
- **True Spoof (AI) misclassified as Human (Miss):** 0
- **True Spoof (AI) predicted as AI (Detection):** 1856

## 4. Top Discriminative Acoustic Forensic Features
The baseline Random Forest classifier identified the following top acoustic features distinguishing AI (Suno) from Human Studio (MUSDB18) audio:

1. **`spec_contrast_b6_mean`**: 39.11% importance
2. **`spec_contrast_b6_std`**: 15.02% importance
3. **`spec_contrast_b3_std`**: 2.62% importance
4. **`spec_contrast_b3_mean`**: 2.32% importance
5. **`spec_contrast_b0_mean`**: 1.98% importance
6. **`mfcc_1_mean`**: 1.71% importance
7. **`mfcc_3_mean`**: 1.61% importance
8. **`spec_contrast_b2_std`**: 1.61% importance
9. **`spec_contrast_b4_mean`**: 1.38% importance
10. **`mfcc_6_mean`**: 1.17% importance

## 5. Metric Explanations for Thesis
- **Equal Error Rate (EER):** The threshold point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). Lower is better (0.00% is perfect). This is the golden standard metric in ASVspoof and audio deepfake benchmarks.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures the model's discriminative ability across all possible classification thresholds.
- **Track-Level Grouping:** Ensures segments from the same song are restricted to either train or test, avoiding overly optimistic evaluation caused by identical audio backgrounds.