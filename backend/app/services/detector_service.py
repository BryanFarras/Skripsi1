import numpy as np
from typing import Dict, Any, List, Optional
from ..models.schemas import AIForensicMetrics

def calculate_spectral_centroid(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    total_mag = np.sum(magnitudes)
    if total_mag < 1e-12:
        return 0.0
    return float(np.sum(magnitudes * freqs) / total_mag)

def calculate_spectral_rolloff(magnitudes: np.ndarray, freqs: np.ndarray, percentile: float = 0.85) -> float:
    total_energy = np.sum(magnitudes ** 2)
    if total_energy < 1e-12:
        return 0.0
    cumulative_energy = np.cumsum(magnitudes ** 2)
    threshold = percentile * total_energy
    idx = np.where(cumulative_energy >= threshold)[0]
    if len(idx) > 0:
        return float(freqs[idx[0]])
    return float(freqs[-1])

def calculate_spectral_flatness(magnitudes: np.ndarray) -> float:
    pwr = magnitudes ** 2 + 1e-12
    geometric_mean = np.exp(np.mean(np.log(pwr)))
    arithmetic_mean = np.mean(pwr)
    if arithmetic_mean < 1e-12:
        return 0.0
    return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))

def detect_brickwall_cutoff(freqs: np.ndarray, avg_db: np.ndarray, sample_rate: int) -> Optional[float]:
    """
    Detects if there is a sharp brickwall cutoff (common in AI models like Suno, Udio, MusicGen,
    or MP3/AAC compression codecs) typically occurring between 14kHz and 22kHz.
    """
    nyquist = sample_rate / 2.0
    # Analyze region from 12kHz to Nyquist
    valid_idx = np.where((freqs >= 12000) & (freqs <= nyquist))[0]
    if len(valid_idx) < 10:
        return None

    region_freqs = freqs[valid_idx]
    region_db = avg_db[valid_idx]

    # Reference mid-band energy (1kHz - 8kHz)
    mid_idx = np.where((freqs >= 1000) & (freqs <= 8000))[0]
    mid_mean_db = np.mean(avg_db[mid_idx]) if len(mid_idx) > 0 else -30.0

    # Look for sharp slope or energy dropping below noise floor (-70 dB or -45 dB relative to mids)
    drop_threshold_db = max(mid_mean_db - 45.0, -80.0)
    
    # Calculate negative gradient (steep drops)
    gradients = np.diff(region_db)
    
    # Check for steep drop cliff
    steep_drop_idx = np.where(gradients < -8.0)[0]
    if len(steep_drop_idx) > 0:
        cutoff_candidate = region_freqs[steep_drop_idx[0]]
        # Verify subsequent bins stay low
        subsequent = region_db[steep_drop_idx[0]:]
        if np.mean(subsequent) < drop_threshold_db:
            return round(float(cutoff_candidate), 1)

    # Check for sustained floor
    sub_floor_idx = np.where(region_db < drop_threshold_db)[0]
    if len(sub_floor_idx) > len(region_db) * 0.4:
        return round(float(region_freqs[sub_floor_idx[0]]), 1)

    return None

def extract_forensic_metrics(audio: np.ndarray, sample_rate: int) -> AIForensicMetrics:
    """
    Extracts forensic features from the audio waveform.
    """
    # Compute overall FFT magnitude spectrum
    n_fft = min(4096, len(audio))
    # Apply Hanning window
    window = np.hanning(n_fft)
    
    # Average across segments
    hop = n_fft // 2
    num_frames = max(1, (len(audio) - n_fft) // hop)
    # Subsample if audio is very long to maintain high speed
    if num_frames > 200:
        step = num_frames // 200
        frame_indices = range(0, num_frames, step)
    else:
        frame_indices = range(num_frames)

    fft_accum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0
    for i in frame_indices:
        start = i * hop
        segment = audio[start:start + n_fft]
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))
        spectrum = np.abs(np.fft.rfft(segment * window))
        fft_accum += spectrum
        count += 1

    avg_spectrum = fft_accum / max(count, 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # Convert to dB
    ref = np.max(avg_spectrum) + 1e-12
    avg_db = 20 * np.log10(np.clip(avg_spectrum / ref, 1e-5, 1.0))

    # Calculate metrics
    centroid = calculate_spectral_centroid(avg_spectrum, freqs)
    rolloff_85 = calculate_spectral_rolloff(avg_spectrum, freqs, 0.85)
    rolloff_95 = calculate_spectral_rolloff(avg_spectrum, freqs, 0.95)
    flatness = calculate_spectral_flatness(avg_spectrum)

    # High frequency energy ratio (> 16kHz)
    high_idx = np.where(freqs >= 16000)[0]
    total_pwr = np.sum(avg_spectrum ** 2) + 1e-12
    high_pwr = np.sum(avg_spectrum[high_idx] ** 2) if len(high_idx) > 0 else 0.0
    hf_ratio = float(high_pwr / total_pwr)

    # Detect cutoff
    cutoff_hz = detect_brickwall_cutoff(freqs, avg_db, sample_rate)
    
    notes = []
    severity = "None"
    cutoff_detected = False

    if cutoff_hz is not None:
        cutoff_detected = True
        if cutoff_hz <= 16500:
            severity = "Strong (<16.5kHz)"
            notes.append(f"Hard frequency cutoff detected at ~{cutoff_hz:.0f} Hz. Highly characteristic of neural audio models (e.g., 32kHz/24kHz vocoders, early Suno/MusicLM).")
        elif cutoff_hz <= 19000:
            severity = "Moderate (16.5-19kHz)"
            notes.append(f"Steep cutoff detected at ~{cutoff_hz:.0f} Hz. Often associated with compressed neural audio or MP3 128kbps encoding.")
        else:
            severity = "Mild (19-20.5kHz)"
            notes.append(f"High-frequency ceiling at ~{cutoff_hz:.0f} Hz. Consistent with lossy audio codecs or standard sample-rate conversions.")
    else:
        notes.append("No sharp artificial brickwall cutoff detected in the high-frequency spectrum.")

    if hf_ratio < 0.005:
        notes.append("Very low energy above 16kHz (<0.5% of total spectral power).")
    else:
        notes.append(f"Air band (>16kHz) retains {hf_ratio * 100:.2f}% of spectral energy.")

    return AIForensicMetrics(
        estimated_cutoff_hz=cutoff_hz,
        cutoff_detected=cutoff_detected,
        cutoff_severity=severity,
        spectral_rolloff_85=round(rolloff_85, 1),
        spectral_rolloff_95=round(rolloff_95, 1),
        spectral_centroid_hz=round(centroid, 1),
        spectral_flatness=round(flatness, 4),
        high_freq_energy_ratio=round(hf_ratio, 5),
        forensic_notes=notes
    )
