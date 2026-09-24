"""
ML Baseline Package for AI Music Detection
Contains feature extraction, dataset generation, model training, evaluation, and inference.
"""

from .config import (
    SUNO_DIR,
    MUSDB18_DIR,
    DATA_DIR,
    MODELS_DIR,
    LABEL_BONAFIDE,
    LABEL_SPOOF,
    LABEL_NAMES
)
from .features import AudioFeatureExtractor
from .baseline_adapter import get_baseline_adapter

__all__ = [
    "AudioFeatureExtractor",
    "get_baseline_adapter",
    "SUNO_DIR",
    "MUSDB18_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LABEL_BONAFIDE",
    "LABEL_SPOOF",
    "LABEL_NAMES"
]
