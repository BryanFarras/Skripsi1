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
import librosa
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

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        model: Optional[Any] = None,
        scaler: Optional[Any] = None,
        explainer: Optional[Any] = None
    ):
        self.model_path = model_path or (MODELS_DIR / "baseline_classifier.joblib")
        self.scaler_path = scaler_path or (MODELS_DIR / "feature_scaler.joblib")
        self.metrics_path = MODELS_DIR / "metrics.json"

        # Model resolution
        if model is not None:
            self.model = model
        elif self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(
                f"Model artifacts not found in {MODELS_DIR}. Please run train.py first!"
            )

        # Scaler resolution
        if scaler is not None:
            self.scaler = scaler
        elif self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError(
                f"Scaler artifact not found at {self.scaler_path}. Please run train.py first!"
            )

        self.extractor = AudioFeatureExtractor()
        self.feature_names = self.extractor.get_feature_names()

        # Explainer resolution
        if explainer is not None:
            self.explainer = explainer
        else:
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

    def compute_slice_shap(self, X_scaled: np.ndarray) -> np.ndarray:
        r"""
        Computes TreeSHAP attributions for each slice in a (K, D) scaled feature matrix.
        Returns a (K, D) numpy array of SHAP values corresponding to the AI (Spoof) class (\phi_{k, i}).
        """
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)

        K, D = X_scaled.shape
        shap_matrix = None

        raw_expl = getattr(self.explainer, "explainer", None) if self.explainer is not None else None
        if raw_expl is None and self.explainer is not None and hasattr(self.explainer, "shap_values"):
            raw_expl = self.explainer

        if raw_expl is not None:
            try:
                if hasattr(raw_expl, "shap_values"):
                    raw_shap = raw_expl.shap_values(X_scaled)
                    if isinstance(raw_shap, list) and len(raw_shap) > 1:
                        shap_matrix = np.array(raw_shap[1], dtype=np.float32)
                    elif isinstance(raw_shap, np.ndarray):
                        if raw_shap.ndim == 3 and raw_shap.shape[2] == 2:
                            shap_matrix = np.array(raw_shap[:, :, 1], dtype=np.float32)
                        elif raw_shap.ndim == 2:
                            shap_matrix = np.array(raw_shap, dtype=np.float32)
                elif callable(raw_expl):
                    res_obj = raw_expl(X_scaled)
                    vals = getattr(res_obj, "values", res_obj)
                    if isinstance(vals, np.ndarray):
                        if vals.ndim == 3 and vals.shape[2] == 2:
                            shap_matrix = np.array(vals[:, :, 1], dtype=np.float32)
                        elif vals.ndim == 2:
                            shap_matrix = np.array(vals, dtype=np.float32)
            except Exception as e:
                pass

        if shap_matrix is None or shap_matrix.shape != (K, D):
            if self.explainer is not None and hasattr(self.explainer, "_fallback_local_attribution"):
                shap_matrix = np.array([
                    self.explainer._fallback_local_attribution(s)
                    for s in X_scaled
                ], dtype=np.float32)
            else:
                shap_matrix = np.zeros((K, D), dtype=np.float32)

        return shap_matrix

    def predict_waveform(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        r"""
        Runs fast sliding-window inference over a raw audio waveform.
        For longer audio (>5s), samples representative slices, extracts features
        in parallel with ThreadPoolExecutor, computes TreeSHAP per segment slice
        (\phi_{k, i}), and averages them across all K slices (\bar{\phi}_i = 1/K \sum \phi_{k, i}).
        """
        from concurrent.futures import ThreadPoolExecutor

        if len(y) == 0:
            raise ValueError("Audio waveform is empty.")

        # Ensure audio is 1D mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)

        # Standardize sample rate to model training sample rate (22050 Hz)
        target_sr = self.extractor.sample_rate
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        dur_sec = len(y) / sr
        slice_samples = int(SLICE_DURATION * sr)

        # 1. Short audio (<= 5.0 seconds): direct single extraction
        if dur_sec <= SLICE_DURATION or len(y) <= slice_samples:
            feat = self.extractor.extract_from_waveform(y, sr=sr)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            prob_ai = float(self.model.predict_proba(feat_scaled)[0, 1])

            slice_shap_matrix = self.compute_slice_shap(feat_scaled)  # shape (1, D)
            mean_shap_values = slice_shap_matrix[0]  # shape (D,)

            xai_res = None
            if self.explainer is not None:
                if hasattr(self.explainer, "_format_attribution_result"):
                    xai_res = self.explainer._format_attribution_result(
                        prob_ai=prob_ai,
                        shap_values_raw=mean_shap_values,
                        feat_raw=feat,
                        total_slices=1
                    )
                else:
                    xai_res = {
                        "prediction": "SPOOF (AI-GENERATED)" if prob_ai >= 0.5 else "BONA-FIDE (HUMAN)",
                        "ai_probability": round(prob_ai, 4),
                        "human_probability": round(1.0 - prob_ai, 4),
                        "all_shap_values": {self.feature_names[i]: round(float(mean_shap_values[i]), 4) for i in range(len(self.feature_names))}
                    }

                xai_res["shap_mean"] = [round(float(v), 4) for v in mean_shap_values]
                xai_res["slice_shap_matrix"] = [
                    [round(float(v), 4) for v in row] for row in slice_shap_matrix
                ]
                xai_res["slice_shap_values"] = xai_res["slice_shap_matrix"]

            top_pos_idx = int(np.argmax(slice_shap_matrix[0]))
            timeline_item = {
                "slice_idx": 0,
                "start_sec": 0.0,
                "end_sec": round(dur_sec, 2),
                "ai_prob": round(prob_ai, 4),
                "human_prob": round(1.0 - prob_ai, 4),
                "is_ai_flagged": bool(prob_ai >= 0.5),
                "top_positive_feature": self.feature_names[top_pos_idx] if top_pos_idx < len(self.feature_names) else "",
                "top_shap_value": round(float(slice_shap_matrix[0, top_pos_idx]), 4),
                "slice_shap": [round(float(v), 4) for v in slice_shap_matrix[0]]
            }

            return {
                "prediction": "SPOOF (AI-GENERATED)" if prob_ai >= 0.5 else "BONA-FIDE (HUMAN)",
                "overall_ai_probability": round(prob_ai, 4),
                "overall_human_probability": round(1.0 - prob_ai, 4),
                "peak_ai_probability": round(prob_ai, 4),
                "flagged_slices_ratio": 1.0 if prob_ai >= 0.5 else 0.0,
                "confidence_percent": round(abs(prob_ai - 0.5) * 200, 1),
                "total_slices_analyzed": 1,
                "timeline": [timeline_item],
                "xai": xai_res,
                "feat_mean": [round(float(v), 4) for v in feat],
                "prob_ai_mean": round(prob_ai, 4),
                "shap_mean": [round(float(v), 4) for v in mean_shap_values],
                "slice_shap_values": [
                    [round(float(v), 4) for v in row] for row in slice_shap_matrix
                ]
            }

        # 2. Longer audio: contiguous 5-second slices across the full audio (0-5s, 5-10s, 10-15s, etc.)
        max_slices = 36  # Support up to 3 minutes of continuous 5s slices
        n_slices = min(max_slices, max(1, int(np.ceil(dur_sec / SLICE_DURATION))))

        sample_slices = []
        for i in range(n_slices):
            st = i * SLICE_DURATION
            end_t = min(dur_sec, (i + 1) * SLICE_DURATION)
            start_sample = int(st * sr)
            end_sample = min(len(y), start_sample + slice_samples)
            chunk = y[start_sample:end_sample]
            if len(chunk) < slice_samples:
                chunk = np.pad(chunk, (0, slice_samples - len(chunk)))
            sample_slices.append((i, round(st, 2), round(end_t, 2), chunk))

        # Parallel extraction across slices
        def _extract_task(item):
            idx, st, end_t, ch = item
            f = self.extractor.extract_from_waveform(ch, sr=sr)
            return idx, st, end_t, f

        max_workers = min(4, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            extracted_results = list(executor.map(_extract_task, sample_slices))

        # Sort back in timeline order
        extracted_results.sort(key=lambda r: r[0])

        slice_features = [r[3] for r in extracted_results]
        slice_meta = [{"slice_idx": r[0], "start_sec": r[1], "end_sec": r[2]} for r in extracted_results]

        X_slices = np.array(slice_features, dtype=np.float32)
        X_scaled = self.scaler.transform(X_slices)
        probs_ai = self.model.predict_proba(X_scaled)[:, 1]

        mean_ai_prob = float(np.mean(probs_ai))
        max_ai_prob = float(np.max(probs_ai))
        ai_slice_ratio = float(np.mean(probs_ai >= 0.5))

        prediction_label = "SPOOF (AI-GENERATED)" if mean_ai_prob >= 0.5 else "BONA-FIDE (HUMAN)"
        confidence = float(abs(mean_ai_prob - 0.5) * 200)

        # 1. Evaluate TreeSHAP per segment slice (\phi_{k, i}) on shape (K, D)
        slice_shap_matrix = self.compute_slice_shap(X_scaled)  # shape (K, D)

        # 2. Average SHAP values across all K slices: \bar{\phi}_i = 1/K \sum \phi_{k, i}
        mean_shap_values = np.mean(slice_shap_matrix, axis=0)  # shape (D,)
        feat_mean = np.mean(X_slices, axis=0)  # shape (D,)

        # Build timeline with slice-level SHAP attributions
        timeline = []
        for i, (meta, p) in enumerate(zip(slice_meta, probs_ai)):
            p_ai = float(p)
            top_pos_idx = int(np.argmax(slice_shap_matrix[i])) if i < len(slice_shap_matrix) else 0
            top_pos_feat = self.feature_names[top_pos_idx] if top_pos_idx < len(self.feature_names) else ""
            top_pos_val = round(float(slice_shap_matrix[i, top_pos_idx]), 4) if i < len(slice_shap_matrix) else 0.0

            timeline.append({
                "slice_idx": meta["slice_idx"],
                "start_sec": meta["start_sec"],
                "end_sec": meta["end_sec"],
                "ai_prob": round(p_ai, 4),
                "human_prob": round(1.0 - p_ai, 4),
                "is_ai_flagged": bool(p_ai >= 0.5),
                "top_positive_feature": top_pos_feat,
                "top_shap_value": top_pos_val,
                "slice_shap": [round(float(v), 4) for v in slice_shap_matrix[i]] if i < len(slice_shap_matrix) else []
            })

        # Compute track-level explainability (SHAP & Forensic Synthesis)
        xai_explanation = None
        if self.explainer is not None and len(slice_features) > 0:
            if hasattr(self.explainer, "_format_attribution_result"):
                xai_explanation = self.explainer._format_attribution_result(
                    prob_ai=mean_ai_prob,
                    shap_values_raw=mean_shap_values,
                    feat_raw=feat_mean,
                    total_slices=len(slice_features)
                )
            else:
                xai_explanation = {
                    "prediction": prediction_label,
                    "ai_probability": round(mean_ai_prob, 4),
                    "human_probability": round(1.0 - mean_ai_prob, 4),
                    "all_shap_values": {self.feature_names[i]: round(float(mean_shap_values[i]), 4) for i in range(len(self.feature_names))}
                }

            xai_explanation["shap_mean"] = [round(float(v), 4) for v in mean_shap_values]
            xai_explanation["slice_shap_matrix"] = [
                [round(float(v), 4) for v in row] for row in slice_shap_matrix
            ]
            xai_explanation["slice_shap_values"] = xai_explanation["slice_shap_matrix"]

        return {
            "prediction": prediction_label,
            "overall_ai_probability": round(mean_ai_prob, 4),
            "overall_human_probability": round(1.0 - mean_ai_prob, 4),
            "peak_ai_probability": round(max_ai_prob, 4),
            "flagged_slices_ratio": round(ai_slice_ratio, 3),
            "confidence_percent": round(confidence, 1),
            "total_slices_analyzed": len(sample_slices),
            "timeline": timeline,
            "xai": xai_explanation,
            "feat_mean": [round(float(v), 4) for v in feat_mean],
            "prob_ai_mean": round(mean_ai_prob, 4),
            "shap_mean": [round(float(v), 4) for v in mean_shap_values],
            "slice_shap_values": [
                [round(float(v), 4) for v in row] for row in slice_shap_matrix
            ]
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

def predict_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    explainer: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    r"""
    Module-level function fulfilling PROJECT.md Interface Contract:
    1. Extracts features per segment slice (K x D).
    2. Evaluates TreeSHAP per segment slice (\phi_{k, i}).
    3. Aggregates: shap_mean = np.mean(shap_values, axis=0) (\bar{\phi}_i = 1/K \sum \phi_{k, i}).
    4. Returns feat_mean, prob_ai_mean, shap_mean along with timeline and detailed explanations.
    """
    predictor = BaselineDetectorPredictor(model=model, scaler=scaler, explainer=explainer)
    return predictor.predict_waveform(waveform, sample_rate)

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
