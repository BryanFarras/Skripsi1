"""
ML Baseline Package for AI Music Detection
Contains feature extraction, dataset generation, model training, evaluation, and inference.
(Configured for 100% Open-Source AI Music Generators: ACE-Step, HeartMuLa, DiffRhythm, Muse; Suno deprecated).
"""

from .config import (
    BONAFIDE_DIR,
    MUSDB18_DIR,
    OPEN_SOURCE_GENERATOR_DIRS,
    ACESTEP_DIR,
    HEARTMULA_DIR,
    DIFFRHYTHM_DIR,
    MUSE_DIR,
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
    "BONAFIDE_DIR",
    "MUSDB18_DIR",
    "OPEN_SOURCE_GENERATOR_DIRS",
    "ACESTEP_DIR",
    "HEARTMULA_DIR",
    "DIFFRHYTHM_DIR",
    "MUSE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LABEL_BONAFIDE",
    "LABEL_SPOOF",
    "LABEL_NAMES"
]
