"""
Baseline Model Adapter for MY-AIDETECTOR Backend Integration
Provides a seamless, fault-tolerant interface allowing FastAPI or background services
to query the trained baseline AI model alongside heuristic DSP analysis.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger("baseline_adapter")

class BaselineModelAdapter:
    """Singleton adapter to load and query baseline classifier."""
    _instance = None

    def __init__(self):
        self.predictor = None
        self.is_loaded = False
        self._initialize()

    def _initialize(self):
        try:
            from .predict import BaselineDetectorPredictor
            self.predictor = BaselineDetectorPredictor()
            self.is_loaded = True
            logger.info("Baseline AI model successfully loaded.")
        except Exception as e:
            logger.warning(f"Baseline AI model not loaded (Model may not be trained yet): {e}")
            self.is_loaded = False

    def predict_audio(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        Runs baseline prediction on raw audio array. Returns None if model is unavailable.
        """
        if not self.is_loaded or self.predictor is None:
            return None

        try:
            return self.predictor.predict_waveform(audio, sample_rate)
        except Exception as e:
            logger.error(f"Error during baseline inference: {e}")
            return None

    def predict_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Runs baseline prediction on audio file path."""
        if not self.is_loaded or self.predictor is None:
            return None

        try:
            return self.predictor.predict_file(file_path)
        except Exception as e:
            logger.error(f"Error during baseline file inference: {e}")
            return None

_ADAPTER_INSTANCE = None

def get_baseline_adapter() -> BaselineModelAdapter:
    """Returns global singleton adapter."""
    global _ADAPTER_INSTANCE
    if _ADAPTER_INSTANCE is None:
        _ADAPTER_INSTANCE = BaselineModelAdapter()
    return _ADAPTER_INSTANCE
