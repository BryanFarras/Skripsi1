"""
Audio Codec & Loudness Normalization Module
Standardizes audio files to uncompressed WAV PCM 16-bit, 22.050 Hz mono,
and normalizes integrated loudness to -14.0 LUFS (EBU R128 / ITU-R BS.1770-4)
to eliminate recording-origin shortcuts and confounding codec bias.
"""

import sys
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Ensure package importability
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml_baseline.config import SAMPLE_RATE, FFMPEG_EXE
else:
    from .config import SAMPLE_RATE, FFMPEG_EXE


def measure_loudness(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Measures integrated loudness in LUFS using ITU-R BS.1770-4 K-weighting filter.
    Returns -70.0 for near-silent audio.
    """
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        if np.isnan(loudness) or np.isinf(loudness):
            return -70.0
        return float(loudness)
    except Exception:
        return -70.0


def normalize_audio_loudness(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_lufs: float = -14.0,
    max_true_peak: float = 0.95
) -> Tuple[np.ndarray, float, float]:
    """
    Normalizes a mono float32 waveform to the target integrated loudness (LUFS).
    Applies true-peak limiting to prevent clipping.

    Returns:
        (normalized_waveform, original_lufs, final_lufs)
    """
    meter = pyln.Meter(sr)
    orig_lufs = measure_loudness(y, sr)

    # Avoid amplifying pure silence or extremely quiet noise
    if orig_lufs <= -65.0:
        return y.astype(np.float32), orig_lufs, orig_lufs

    try:
        y_norm = pyln.normalize.loudness(y, orig_lufs, target_lufs)
    except Exception:
        y_norm = y.copy()

    # Peak safety limiter
    peak = np.max(np.abs(y_norm))
    if peak > max_true_peak and peak > 1e-6:
        y_norm = y_norm * (max_true_peak / peak)

    final_lufs = measure_loudness(y_norm, sr)
    return y_norm.astype(np.float32), orig_lufs, final_lufs


def load_raw_waveform(file_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Decodes any audio file into a 32-bit float mono numpy array at target_sr.
    Handles .stem.mp4, .mp3, .wav, .m4a, and others via ffmpeg or soundfile.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # For Native Instruments STEM.MP4 files, extract stream 0 (full mix)
    ffmpeg_bin = str(FFMPEG_EXE) if FFMPEG_EXE.exists() else "ffmpeg"
    if file_path.name.lower().endswith(".stem.mp4"):
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(file_path),
            "-map", "0:a:0",
            "-ar", str(target_sr),
            "-ac", "1",
            "-f", "f32le",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_bytes = proc.stdout.read()
        proc.wait()
        if len(raw_bytes) > 0:
            return np.frombuffer(raw_bytes, dtype=np.float32)

    # Standard audio decode via ffmpeg pipe
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(file_path),
        "-ar", str(target_sr),
        "-ac", "1",
        "-f", "f32le",
        "pipe:1"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_bytes = proc.stdout.read()
        proc.wait()
        if len(raw_bytes) > 0:
            return np.frombuffer(raw_bytes, dtype=np.float32)
    except Exception:
        pass

    # Fallback to soundfile or librosa
    try:
        import librosa
        y, _ = librosa.load(str(file_path), sr=target_sr, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Could not decode audio {file_path.name}: {e}")


def standardize_audio_file(
    input_path: Path,
    output_wav_path: Path,
    target_sr: int = SAMPLE_RATE,
    target_lufs: float = -14.0
) -> Dict[str, Any]:
    """
    Converts any audio file to standardized uncompressed WAV PCM 16-bit,
    22.050 Hz mono, normalized to target LUFS.
    """
    input_path = Path(input_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    y = load_raw_waveform(input_path, target_sr=target_sr)
    y_norm, orig_lufs, final_lufs = normalize_audio_loudness(
        y, sr=target_sr, target_lufs=target_lufs
    )

    sf.write(str(output_wav_path), y_norm, target_sr, subtype="PCM_16")

    return {
        "original_file": input_path.name,
        "standardized_wav": output_wav_path.name,
        "sample_rate": target_sr,
        "channels": 1,
        "format": "WAV PCM 16-bit",
        "original_lufs": round(orig_lufs, 2),
        "normalized_lufs": round(final_lufs, 2),
        "duration_sec": round(len(y_norm) / target_sr, 2)
    }

