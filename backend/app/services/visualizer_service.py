import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..models.schemas import (
    FFTSpectrumData,
    TonalBalanceBand,
    TonalBalanceData,
    SpectrogramData,
    AIForensicMetrics
)
from .detector_service import detect_brickwall_cutoff

def compute_fft_spectrum(audio: np.ndarray, sample_rate: int, num_log_bins: int = 256) -> FFTSpectrumData:
    """
    Computes an Equalizer-style frequency response curve across log-spaced frequencies (20Hz - 20kHz).
    Returns averaged magnitude dB and peak hold dB.
    """
    n_fft = 4096
    hop = 1024
    window = np.hanning(n_fft)
    
    # Chunk audio
    num_frames = max(1, (len(audio) - n_fft) // hop)
    if num_frames > 300:
        step = num_frames // 300
        indices = range(0, num_frames, step)
    else:
        indices = range(num_frames)

    specs = []
    for idx in indices:
        seg = audio[idx * hop: idx * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = np.abs(np.fft.rfft(seg * window))
        specs.append(spec)

    specs = np.array(specs)  # (frames, freq_bins)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    avg_mag = np.mean(specs, axis=0)
    peak_mag = np.max(specs, axis=0)

    # Reference peak for dB conversion
    max_ref = np.max(peak_mag) + 1e-12
    avg_db_raw = 20 * np.log10(np.clip(avg_mag / max_ref, 1e-5, 1.0))
    peak_db_raw = 20 * np.log10(np.clip(peak_mag / max_ref, 1e-5, 1.0))

    # Logarithmic frequency bins from 20 Hz to min(20000, Nyquist)
    max_hz = min(20000.0, sample_rate / 2.0 - 100)
    log_freqs = np.geomspace(20.0, max_hz, num_log_bins)

    # Interpolate to log scale
    avg_db_log = np.interp(log_freqs, freqs, avg_db_raw)
    peak_db_log = np.interp(log_freqs, freqs, peak_db_raw)

    # Smooth curve
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    avg_smooth = np.convolve(avg_db_log, kernel, mode='same')
    peak_smooth = np.convolve(peak_db_log, kernel, mode='same')

    return FFTSpectrumData(
        frequencies=[round(float(f), 1) for f in log_freqs],
        magnitudes_db=[round(float(v), 2) for v in avg_smooth],
        peaks_db=[round(float(v), 2) for v in peak_smooth],
        min_db=-90.0,
        max_db=0.0
    )

def compute_tonal_balance(audio: np.ndarray, sample_rate: int) -> TonalBalanceData:
    """
    Computes Tonal Balance curve and 5 standard frequency bands (Sub, Bass, Low-Mid, High-Mid, Treble/Air).
    """
    band_defs = [
        {"name": "Sub", "range": [20.0, 60.0], "color": "#E53935"},
        {"name": "Bass", "range": [60.0, 250.0], "color": "#FB8C00"},
        {"name": "Low Mid", "range": [250.0, 2000.0], "color": "#43A047"},
        {"name": "High Mid", "range": [2000.0, 8000.0], "color": "#00ACC1"},
        {"name": "Treble / Air", "range": [8000.0, 20000.0], "color": "#8E24AA"}
    ]

    # Overall FFT
    n_fft = 4096
    window = np.hanning(n_fft)
    hop = 2048
    num_frames = max(1, (len(audio) - n_fft) // hop)
    if num_frames > 200:
        step = num_frames // 200
        indices = range(0, num_frames, step)
    else:
        indices = range(num_frames)

    fft_accum = np.zeros(n_fft // 2 + 1)
    for idx in indices:
        seg = audio[idx * hop: idx * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        fft_accum += np.abs(np.fft.rfft(seg * window))

    avg_mag = fft_accum / max(len(indices), 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    pwr = avg_mag ** 2
    total_energy = np.sum(pwr) + 1e-12

    bands: List[TonalBalanceBand] = []
    for b in band_defs:
        f_min, f_max = b["range"]
        idx = np.where((freqs >= f_min) & (freqs < f_max))[0]
        if len(idx) > 0:
            band_energy = np.sum(pwr[idx])
            band_pct = (band_energy / total_energy) * 100.0
            avg_db = float(20 * np.log10(np.clip(np.mean(avg_mag[idx]) / (np.max(avg_mag) + 1e-12), 1e-5, 1.0)))
        else:
            band_pct = 0.0
            avg_db = -80.0
        
        bands.append(TonalBalanceBand(
            name=b["name"],
            freq_range=[float(f_min), float(f_max)],
            energy_percent=round(float(band_pct), 2),
            avg_db=round(avg_db, 2),
            color_hex=b["color"]
        ))

    # Compute a continuous tonal balance curve across 128 points
    curve_freqs = np.geomspace(20.0, min(20000.0, sample_rate / 2.0 - 50), 128)
    ref = np.max(avg_mag) + 1e-12
    raw_levels = 20 * np.log10(np.clip(np.interp(curve_freqs, freqs, avg_mag) / ref, 1e-4, 1.0))
    # Smooth with rolling average
    smooth_levels = np.convolve(raw_levels, np.ones(7)/7, mode='same')

    return TonalBalanceData(
        curve_freqs=[round(float(f), 1) for f in curve_freqs],
        curve_levels=[round(float(l), 2) for l in smooth_levels],
        bands=bands
    )

def compute_spectrogram(audio: np.ndarray, sample_rate: int, max_time_bins: int = 400, max_freq_bins: int = 128) -> SpectrogramData:
    """
    Computes downsampled 2D STFT Spectrogram matrix in dB for smooth web rendering.
    """
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)

    # Frame processing
    num_frames = (len(audio) - n_fft) // hop
    if num_frames < 2:
        num_frames = 2
        audio = np.pad(audio, (0, n_fft * 2))

    # Downsample time if audio is long
    time_step = max(1, num_frames // max_time_bins)
    frame_indices = range(0, num_frames, time_step)

    stft_matrix = []
    time_points = []
    for f_idx in frame_indices:
        start = f_idx * hop
        seg = audio[start: start + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        mag = np.abs(np.fft.rfft(seg * window))
        stft_matrix.append(mag)
        time_points.append(start / sample_rate)

    stft_matrix = np.array(stft_matrix).T  # Shape: (freq_bins, time_frames)
    full_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # Target frequency bins up to Nyquist (or 22.05kHz)
    max_hz = min(22050.0, sample_rate / 2.0)
    target_freqs = np.linspace(20.0, max_hz, max_freq_bins)

    # Resample frequency axis
    downsampled_matrix = np.zeros((max_freq_bins, len(time_points)), dtype=np.float32)
    for t in range(len(time_points)):
        downsampled_matrix[:, t] = np.interp(target_freqs, full_freqs, stft_matrix[:, t])

    # Convert to dB
    ref = np.max(downsampled_matrix) + 1e-12
    db_matrix = 20 * np.log10(np.clip(downsampled_matrix / ref, 1e-4, 1.0))  # Scale: -80 dB to 0 dB

    # Detect cutoff
    avg_spectrum = np.mean(downsampled_matrix, axis=1)
    avg_db = 20 * np.log10(np.clip(avg_spectrum / (np.max(avg_spectrum) + 1e-12), 1e-4, 1.0))
    detected_cutoff = detect_brickwall_cutoff(target_freqs, avg_db, sample_rate)

    # Clean 2D list
    matrix_list = [[round(float(val), 1) for val in row] for row in db_matrix]

    return SpectrogramData(
        time_points=[round(float(t), 2) for t in time_points],
        frequencies=[round(float(f), 1) for f in target_freqs],
        matrix_db=matrix_list,
        min_db=-80.0,
        max_db=0.0,
        detected_cutoff_hz=detected_cutoff
    )

def render_publication_plot(
    audio: np.ndarray,
    sample_rate: int,
    filename: str,
    output_path: Path,
    forensics: AIForensicMetrics
) -> Path:
    """
    Renders a 3-panel figure replicating visual analyzer1.png and saves to output_path.
    """
    fig, (ax_eq, ax_tonal, ax_spec) = plt.subplots(3, 1, figsize=(12, 10), facecolor="#12141a")
    
    # 1. Equalizer Panel
    fft_data = compute_fft_spectrum(audio, sample_rate)
    ax_eq.set_facecolor("#181b22")
    ax_eq.plot(fft_data.frequencies, fft_data.peaks_db, color="#5c6b84", linewidth=1.0, label="Peak Hold")
    ax_eq.plot(fft_data.frequencies, fft_data.magnitudes_db, color="#00e5ff", linewidth=1.5, label="Average Curve")
    ax_eq.set_xscale("log")
    ax_eq.set_xlim(20, 20000)
    ax_eq.set_ylim(-80, 5)
    ax_eq.set_title("Equalizer (FFT Frequency Spectrum)", color="#ffffff", fontsize=12, pad=10, loc="left", fontweight="bold")
    ax_eq.set_xlabel("Frequency (Hz)", color="#a0aec0", fontsize=9)
    ax_eq.set_ylabel("Magnitude (dB)", color="#a0aec0", fontsize=9)
    ax_eq.grid(True, which="both", color="#2a303c", linestyle="--", linewidth=0.5)
    ax_eq.tick_params(colors="#a0aec0")
    ax_eq.legend(loc="upper right", facecolor="#181b22", edgecolor="#2a303c", labelcolor="#e2e8f0")

    # 2. Tonal Balance Panel
    tb_data = compute_tonal_balance(audio, sample_rate)
    ax_tonal.set_facecolor("#181b22")
    curve_f = np.array(tb_data.curve_freqs)
    curve_l = np.array(tb_data.curve_levels)
    ax_tonal.plot(curve_f, curve_l, color="#ffffff", linewidth=1.8)
    
    # Fill bands with distinct colors
    for band in tb_data.bands:
        f_min, f_max = band.freq_range
        mask = (curve_f >= f_min) & (curve_f <= f_max)
        if np.any(mask):
            sub_f = curve_f[mask]
            sub_l = curve_l[mask]
            ax_tonal.fill_between(sub_f, sub_l, -80, color=band.color_hex, alpha=0.6, label=f"{band.name} ({band.energy_percent:.1f}%)")

    ax_tonal.set_xscale("log")
    ax_tonal.set_xlim(20, 20000)
    ax_tonal.set_ylim(-70, 0)
    ax_tonal.set_title("Tonal Balance Control", color="#ffffff", fontsize=12, pad=10, loc="left", fontweight="bold")
    ax_tonal.set_xlabel("Frequency (Hz)", color="#a0aec0", fontsize=9)
    ax_tonal.set_ylabel("Relative Level (dB)", color="#a0aec0", fontsize=9)
    ax_tonal.grid(True, which="both", color="#2a303c", linestyle="--", linewidth=0.5)
    ax_tonal.tick_params(colors="#a0aec0")
    ax_tonal.legend(loc="lower left", facecolor="#181b22", edgecolor="#2a303c", labelcolor="#e2e8f0", fontsize=8)

    # 3. Spectrogram Panel
    spec_data = compute_spectrogram(audio, sample_rate)
    ax_spec.set_facecolor("#181b22")
    t_extent = [spec_data.time_points[0], spec_data.time_points[-1], spec_data.frequencies[0], spec_data.frequencies[-1]]
    matrix_arr = np.array(spec_data.matrix_db)
    im = ax_spec.imshow(
        matrix_arr,
        aspect="auto",
        origin="lower",
        extent=t_extent,
        cmap="inferno",
        vmin=-75,
        vmax=0
    )
    
    # Draw cutoff marker if detected
    if forensics.cutoff_detected and forensics.estimated_cutoff_hz:
        ax_spec.axhline(forensics.estimated_cutoff_hz, color="#00ffcc", linestyle="--", linewidth=1.5,
                        label=f"Detected AI Cutoff: {forensics.estimated_cutoff_hz:.0f} Hz")
        ax_spec.legend(loc="upper right", facecolor="#181b22", edgecolor="#00ffcc", labelcolor="#00ffcc")

    ax_spec.set_title(f"Spectrogram (Time vs Frequency Heatmap) - {filename}", color="#ffffff", fontsize=12, pad=10, loc="left", fontweight="bold")
    ax_spec.set_xlabel("Time (seconds)", color="#a0aec0", fontsize=9)
    ax_spec.set_ylabel("Frequency (Hz)", color="#a0aec0", fontsize=9)
    ax_spec.tick_params(colors="#a0aec0")
    cbar = fig.colorbar(im, ax=ax_spec, pad=0.015, aspect=15)
    cbar.set_label("Magnitude (dB)", color="#a0aec0", fontsize=8)
    cbar.ax.tick_params(colors="#a0aec0")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return output_path
