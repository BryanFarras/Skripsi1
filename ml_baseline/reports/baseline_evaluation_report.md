# AI Music Detector: Baseline Acoustic Model Evaluation

**Date:** 2026-09-25 00:21:26
**Champion Baseline Architecture:** `Random Forest`

## 1. Dataset & Partitioning Summary
- **Total Slices:** 420 (5.0s window, 2.5s hop)
- **Train Slices:** 300
- **Test Slices:** 120
- **Acoustic Feature Dimension:** 110 forensic features
- **Validation Scheme:** Grouped Track-Level Split (75% train / 25% test, 0% song-chunk leakage)

## 2. Model Benchmark Comparison

| Classifier | Accuracy | F1-Score | ROC-AUC | Equal Error Rate (EER) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 96.67% | 97.73% | 100.00% | 0.00% | 0.176s |
| **Support Vector Machine (RBF)** | 95.83% | 97.14% | 99.70% | 5.56% | 0.018s |
| **Logistic Regression** | 96.67% | 97.73% | 99.07% | 3.89% | 0.005s |
| **Multi-Layer Perceptron (MLP)** | 85.83% | 89.70% | 91.52% | 12.78% | 0.07s |

## 3. Best Model Confusion Matrix

- **True Bona-fide (Human) predicted as Human:** 30
- **True Bona-fide (Human) misclassified as AI (False Alarm):** 0
- **True Spoof (AI) misclassified as Human (Miss):** 4
- **True Spoof (AI) predicted as AI (Detection):** 86

## 4. Top Discriminative Acoustic Forensic Features
The baseline Random Forest classifier identified the following top acoustic features distinguishing AI (Suno) from Human Studio (MUSDB18) audio:

1. **`spec_contrast_b6_mean`**: 16.93% importance
2. **`mfcc_17_mean`**: 10.61% importance
3. **`spec_rolloff85_mean`**: 5.76% importance
4. **`mfcc_16_mean`**: 4.68% importance
5. **`mfcc_2_mean`**: 4.34% importance
6. **`spec_bandwidth_mean`**: 4.14% importance
7. **`spec_contrast_b6_std`**: 3.43% importance
8. **`spec_centroid_mean`**: 2.68% importance
9. **`mfcc_12_mean`**: 2.61% importance
10. **`spec_rolloff95_mean`**: 2.31% importance

## 5. Metric Explanations for Thesis
- **Equal Error Rate (EER):** The threshold point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). Lower is better (0.00% is perfect). This is the golden standard metric in ASVspoof and audio deepfake benchmarks.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures the model's discriminative ability across all possible classification thresholds.
- **Track-Level Grouping:** Ensures segments from the same song are restricted to either train or test, avoiding overly optimistic evaluation caused by identical audio backgrounds.