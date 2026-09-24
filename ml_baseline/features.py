"""
Acoustic & Forensic Feature Extraction Pipeline
Extracts 110-dimensional acoustic feature vectors from audio waveforms:
- MFCCs (1-20): Mean & Std (40 dims)
- Delta-MFCCs (1-20): Mean & Std (40 dims)
- Spectral Centroid: Mean & Std (2 dims)
- Spectral Bandwidth: Mean & Std (2 dims)
- Spectral Rolloff (85%): Mean & Std (2 dims)
- Spectral Rolloff (95%): Mean & Std (2 dims)
- Spectral Flatness: Mean & Std (2 dims)
- Spectral Contrast (6 bands + 1): Mean & Std (14 dims)
- Zero-Crossing Rate: Mean & Std (2 dims)
- RMS Energy: Mean & Std (2 dims)
- High-Frequency Energy Ratio (>14kHz): Mean & Std (2 dims)
"""

import os
import sys
import subprocess
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator

from .config import (
    SAMPLE_RATE,
    SLICE_DURATION,
    SLICE_HOP,
    MIN_ENERGY_THRESHOLD,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    FFMPEG_EXE
)

class AudioFeatureExtractor:
    """Extracts forensic acoustic features tailored for AI vs Human audio classification."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns the ordered list of all 110 feature names."""
        names = []
        # MFCC
        for i in range(1, N_MFCC + 1):
            names.append(f"mfcc_{i}_mean")
            names.append(f"mfcc_{i}_std")
        # Delta MFCC
        for i in range(1, N_MFCC + 1):
            names.append(f"delta_mfcc_{i}_mean")
            names.append(f"delta_mfcc_{i}_std")
        # Spectral Centroid & Bandwidth
        names.extend(["spec_centroid_mean", "spec_centroid_std"])
        names.extend(["spec_bandwidth_mean", "spec_bandwidth_std"])
        # Spectral Rolloff
        names.extend(["spec_rolloff85_mean", "spec_rolloff85_std"])
        names.extend(["spec_rolloff95_mean", "spec_rolloff95_std"])
        # Spectral Flatness
        names.extend(["spec_flatness_mean", "spec_flatness_std"])
        # Spectral Contrast (7 bands: 6 octave sub-bands + 1)
        for i in range(7):
            names.append(f"spec_contrast_b{i}_mean")
            names.append(f"spec_contrast_b{i}_std")
        # Temporal & Dynamic Features
        names.extend(["zcr_mean", "zcr_std"])
        names.extend(["rms_mean", "rms_std"])
        names.extend(["hf_ratio_mean", "hf_ratio_std"])
        return names

    def extract_from_waveform(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Extracts a 110-dim feature vector from an audio waveform array.
        """
        if sr is None:
            sr = self.sample_rate
        if len(y.shape) > 1:
            y = np.mean(y, axis=0) # Convert to mono for feature extraction

        features = []

        # 1. MFCCs (1-20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for coef in mfcc:
            features.append(float(np.mean(coef)))
            features.append(float(np.std(coef)))

        # 2. Delta MFCCs
        delta_mfcc = librosa.feature.delta(mfcc)
        for d_coef in delta_mfcc:
            features.append(float(np.mean(d_coef)))
            features.append(float(np.std(d_coef)))

        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        # 4. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        # 5. Spectral Rolloff (85% and 95%)
        rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(float(np.mean(rolloff_85)))
        features.append(float(np.std(rolloff_85)))

        rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(float(np.mean(rolloff_95)))
        features.append(float(np.std(rolloff_95)))

        # 6. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(float(np.mean(flatness)))
        features.append(float(np.std(flatness)))

        # 7. Spectral Contrast (6 bands + 1 = 7)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=100.0)
        for band in contrast:
            features.append(float(np.mean(band)))
            features.append(float(np.std(band)))

        # 8. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # 9. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # 10. High-Frequency Energy Ratio (Frequency > 7000Hz relative to Nyquist in 22050Hz)
        # For sr=22050, Nyquist is 11025. We measure energy in top 30% of spectrum vs total
        stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        hf_idx = np.where(freqs >= (sr * 0.35))[0] # > 7.7 kHz
        if len(hf_idx) > 0:
            total_spec_energy = np.sum(stft ** 2, axis=0) + 1e-12
            hf_spec_energy = np.sum(stft[hf_idx, :] ** 2, axis=0)
            hf_ratio_frames = hf_spec_energy / total_spec_energy
            features.append(float(np.mean(hf_ratio_frames)))
            features.append(float(np.std(hf_ratio_frames)))
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def load_audio_file(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file into a mono float32 array at self.sample_rate.
        Supports standard formats and MUSDB18 .stem.mp4 files.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Check for .stem.mp4 (Extract mixture stream 0 via ffmpeg)
        if file_path.name.lower().endswith(".stem.mp4"):
            return self._load_musdb_mixture(file_path)

        # Standard formats (WAV, MP3, FLAC, M4A)
        try:
            y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            return y, sr
        except Exception:
            # Fallback to ffmpeg decode if librosa fails
            return self._load_via_ffmpeg(file_path)

    def _load_musdb_mixture(self, stem_path: Path) -> Tuple[np.ndarray, int]:
        """Extracts Stream 0 (Mixture) from a Native Instruments STEM.MP4 file."""
        ffmpeg_bin = str(FFMPEG_EXE) if FFMPEG_EXE.exists() else "ffmpeg"
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(stem_path),
            "-map", "0:a:0",           # Stream 0: Mixture
            "-ar", str(self.sample_rate),
            "-ac", "1",                 # Mono
            "-f", "f32le",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_bytes = proc.stdout.read()
        proc.wait()
        if len(raw_bytes) == 0:
            raise RuntimeError(f"Failed to extract audio from {stem_path.name} via ffmpeg")
        y = np.frombuffer(raw_bytes, dtype=np.float32)
        return y, self.sample_rate

    def _load_via_ffmpeg(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Fallback ffmpeg decoder for any media container."""
        ffmpeg_bin = str(FFMPEG_EXE) if FFMPEG_EXE.exists() else "ffmpeg"
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(file_path),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "f32le",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_bytes = proc.stdout.read()
        proc.wait()
        y = np.frombuffer(raw_bytes, dtype=np.float32)
        return y, self.sample_rate

    def slice_waveform(
        self,
        y: np.ndarray,
        duration: float = SLICE_DURATION,
        hop: float = SLICE_HOP
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (slice_index, start_time_sec, audio_chunk) for active slices.
        Rejects near-silent segments to avoid corrupting training data.
        """
        slice_samples = int(duration * self.sample_rate)
        hop_samples = int(hop * self.sample_rate)
        total_samples = len(y)

        slice_idx = 0
        for start_idx in range(0, total_samples - slice_samples + 1, hop_samples):
            chunk = y[start_idx : start_idx + slice_samples]
            # Skip silent chunks
            energy = np.mean(chunk ** 2)
            if energy < (MIN_ENERGY_THRESHOLD ** 2):
                continue
            start_sec = start_idx / self.sample_rate
            yield slice_idx, start_sec, chunk
            slice_idx += 1
