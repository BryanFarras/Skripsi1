import os
import subprocess
from pathlib import Path
from typing import Tuple
import numpy as np

def load_audio_via_ffmpeg(file_path: Path, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Decodes any audio file (mp3, wav, flac, m4a, ogg) to float32 mono waveform using bundled ffmpeg.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-v", "error",
        "-i", str(file_path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(target_sr),
        "-ac", "1",  # Downmix to mono
        "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg decoding failed: {err.decode('utf-8', errors='ignore')}")
        
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        if len(audio) == 0:
            raise ValueError("Decoded audio stream is empty.")
        return audio, target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio file via ffmpeg: {e}")

def load_audio(file_path: Path, target_sr: int = 44100) -> Tuple[np.ndarray, int, float]:
    """
    Loads audio from file_path, returning (waveform_mono_float32, sample_rate, duration_seconds).
    Tries soundfile first, then falls back to ffmpeg.
    """
    audio = None
    sr = target_sr

    try:
        import soundfile as sf
        data, file_sr = sf.read(str(file_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Mono downmix
        
        if file_sr != target_sr:
            # Resample using linear interpolation if scipy/librosa not yet ready
            duration = len(data) / file_sr
            num_target_samples = int(duration * target_sr)
            data = np.interp(
                np.linspace(0, len(data), num_target_samples, endpoint=False),
                np.arange(len(data)),
                data
            ).astype(np.float32)
            sr = target_sr
        else:
            sr = file_sr
        audio = data
    except Exception:
        # Fall back to FFmpeg decoder
        audio, sr = load_audio_via_ffmpeg(file_path, target_sr=target_sr)

    # Normalize if peak exceeds 1.0
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    duration_seconds = float(len(audio) / sr)
    return audio, sr, duration_seconds
