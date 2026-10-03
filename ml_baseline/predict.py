"""
Baseline AI Detector Inference Engine
Performs temporal segment-by-segment inference on any audio file,
producing Spoof vs Bona-fide probabilities, confidence, and forensic anomaly details.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure package importability
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml_baseline.config import (
        MODELS_DIR,
        SLICE_DURATION,
        SLICE_HOP,
        LABEL_NAMES
    )
    from ml_baseline.features import AudioFeatureExtractor
else:
    from .config import (
        MODELS_DIR,
        SLICE_DURATION,
        SLICE_HOP,
        LABEL_NAMES
    )
    from .features import AudioFeatureExtractor

class BaselineDetectorPredictor:
    """
    Inference class for the baseline AI music detector.
    Loads trained classifier and standard scaler to evaluate arbitrary audio clips.
    """

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / "baseline_classifier.joblib")
        self.scaler_path = scaler_path or (MODELS_DIR / "feature_scaler.joblib")
        self.metrics_path = MODELS_DIR / "metrics.json"

        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {MODELS_DIR}. Please run train.py first!"
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.extractor = AudioFeatureExtractor()
        self.feature_names = self.extractor.get_feature_names()

        # Initialize XAI Explainer
        try:
            if __package__ is None or __package__ == "":
                from ml_baseline.explainability import AudioForensicExplainer
            else:
                from .explainability import AudioForensicExplainer
            self.explainer = AudioForensicExplainer(self.model, self.scaler, self.feature_names)
        except Exception as e:
            print(f"[XAI Warning] Explainer initialization note: {e}")
            self.explainer = None

        # Load metrics if available
        self.metadata = {}
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                pass

    def predict_waveform(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Runs fast sliding-window inference over a raw audio waveform.
        For longer audio (>5s), samples representative slices and extracts features
        in parallel with ThreadPoolExecutor for low latency.
        """
        from concurrent.futures import ThreadPoolExecutor

        if len(y) == 0:
            raise ValueError("Audio waveform is empty.")

        dur_sec = len(y) / sr
        slice_samples = int(SLICE_DURATION * sr)

        # 1. Short audio (<= 5.0 seconds): direct single extraction
        if dur_sec <= SLICE_DURATION or len(y) <= slice_samples:
            feat = self.extractor.extract_from_waveform(y, sr=sr)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            prob_ai = float(self.model.predict_proba(feat_scaled)[0, 1])
            xai_res = None
            if self.explainer is not None:
                xai_res = self.explainer.explain_vector(feat, feat_scaled[0])
            return {
                "prediction": "SPOOF (AI-GENERATED)" if prob_ai >= 0.5 else "BONA-FIDE (HUMAN)",
                "overall_ai_probability": round(prob_ai, 4),
                "overall_human_probability": round(1.0 - prob_ai, 4),
                "peak_ai_probability": round(prob_ai, 4),
                "flagged_slices_ratio": 1.0 if prob_ai >= 0.5 else 0.0,
                "confidence_percent": round(abs(prob_ai - 0.5) * 200, 1),
                "total_slices_analyzed": 1,
                "timeline": [{"slice_idx": 0, "start_sec": 0.0, "ai_prob": round(prob_ai, 3), "is_ai_flagged": prob_ai >= 0.5}],
                "xai": xai_res
            }

        # 2. Longer audio: sample up to 8 representative 5-second slices across the track
        max_slices = 8
        n_slices = min(max_slices, max(2, int(dur_sec // SLICE_DURATION)))
        step_sec = (dur_sec - SLICE_DURATION) / max(1, n_slices - 1)

        sample_slices = []
        for i in range(n_slices):
            st = i * step_sec
            start_sample = int(st * sr)
            end_sample = min(len(y), start_sample + slice_samples)
            chunk = y[start_sample:end_sample]
            if len(chunk) < slice_samples:
                chunk = np.pad(chunk, (0, slice_samples - len(chunk)))
            sample_slices.append((i, round(st, 2), chunk))

        # Parallel extraction across slices
        def _extract_task(item):
            idx, st, ch = item
            f = self.extractor.extract_from_waveform(ch, sr=sr)
            return idx, st, f

        max_workers = min(4, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            extracted_results = list(executor.map(_extract_task, sample_slices))

        # Sort back in timeline order
        extracted_results.sort(key=lambda r: r[0])

        slice_features = [r[2] for r in extracted_results]
        slice_meta = [{"slice_idx": r[0], "start_sec": r[1]} for r in extracted_results]

        X_slices = np.array(slice_features, dtype=np.float32)
        X_scaled = self.scaler.transform(X_slices)
        probs_ai = self.model.predict_proba(X_scaled)[:, 1]

        timeline = []
        for meta, p in zip(slice_meta, probs_ai):
            timeline.append({
                "slice_idx": meta["slice_idx"],
                "start_sec": meta["start_sec"],
                "ai_prob": round(float(p), 4),
                "is_ai_flagged": bool(p >= 0.5)
            })

        mean_ai_prob = float(np.mean(probs_ai))
        max_ai_prob = float(np.max(probs_ai))
        ai_slice_ratio = float(np.mean(probs_ai >= 0.5))

        prediction_label = "SPOOF (AI-GENERATED)" if mean_ai_prob >= 0.5 else "BONA-FIDE (HUMAN)"
        confidence = float(abs(mean_ai_prob - 0.5) * 200)

        # Compute track-level explainability (SHAP & Forensic Synthesis)
        xai_explanation = None
        if self.explainer is not None and len(slice_features) > 0:
            feat_mean = np.mean(slice_features, axis=0)
            feat_scaled_mean = self.scaler.transform(feat_mean.reshape(1, -1))[0]
            xai_explanation = self.explainer.explain_vector(feat_mean, feat_scaled_mean)

        return {
            "prediction": prediction_label,
            "overall_ai_probability": round(mean_ai_prob, 4),
            "overall_human_probability": round(1.0 - mean_ai_prob, 4),
            "peak_ai_probability": round(max_ai_prob, 4),
            "flagged_slices_ratio": round(ai_slice_ratio, 3),
            "confidence_percent": round(confidence, 1),
            "total_slices_analyzed": len(sample_slices),
            "timeline": timeline,
            "xai": xai_explanation
        }

    def predict_file(self, audio_path: Path) -> Dict[str, Any]:
        """
        Loads an audio file (standard or .stem.mp4) and performs temporal inference.
        """
        audio_path = Path(audio_path)
        y, sr = self.extractor.load_audio_file(audio_path)
        duration_sec = len(y) / sr

        result = self.predict_waveform(y, sr)
        result["file_name"] = audio_path.name
        result["duration_seconds"] = round(duration_sec, 2)
        return result

def print_prediction_report(result: Dict[str, Any]):
    """Pretty prints the prediction report to console."""
    print("\n" + "=" * 65)
    print(f"       AI AUDIO DETECTOR - INFERENCE REPORT")
    print("=" * 65)
    print(f" File:                 {result.get('file_name', 'Waveform')}")
    print(f" Duration:             {result.get('duration_seconds', 0.0)} seconds")
    print(f" Slices Analyzed:      {result.get('total_slices_analyzed', 0)} (5s window, 50% hop)")
    print("-" * 65)
    
    badge = "[!] SPOOF (AI GENERATED)" if result["prediction"].startswith("SPOOF") else "[OK] BONA-FIDE (HUMAN)"
    print(f" Final Verdict:        {badge}")
    print(f" Confidence:           {result['confidence_percent']}%")
    print(f" AI Probability:       {result['overall_ai_probability']*100:.1f}%")
    print(f" Human Probability:    {result['overall_human_probability']*100:.1f}%")
    print(f" Peak AI Spike:        {result['peak_ai_probability']*100:.1f}%")
    print(f" Slices Flagged as AI: {result['flagged_slices_ratio']*100:.1f}%")
    print("-" * 65)
    
    # Mini visual timeline
    print(" Temporal Slice Timeline (P(AI)):")
    timeline = result.get("timeline", [])
    bar_display = []
    for item in timeline[:36]: # show first 36 slices
        p = item["ai_prob"]
        if p >= 0.8:
            bar_display.append("#")
        elif p >= 0.5:
            bar_display.append("=")
        elif p >= 0.2:
            bar_display.append("-")
        else:
            bar_display.append(".")
    print(f"  [{''.join(bar_display)}]")
    print("  Legend: # (High AI >80%) | = (Probable AI >50%) | - (Low AI >20%) | . (Clean)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline AI detector inference on an audio file")
    parser.add_argument("audio_path", type=str, help="Path to audio file (MP3, WAV, FLAC, STEM.MP4)")
    args = parser.parse_args()

    predictor = BaselineDetectorPredictor()
    res = predictor.predict_file(Path(args.audio_path))
    print_prediction_report(res)
