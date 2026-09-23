import numpy as np
import wave
import struct
from pathlib import Path

# Synthesize a test audio file with an artificial brickwall cutoff at 16kHz
test_file = Path(__file__).resolve().parent / "test_synth.wav"
sr = 44100
duration = 3.0
num_samples = int(sr * duration)
t = np.linspace(0, duration, num_samples, endpoint=False)

# Mix fundamental frequencies up to 16kHz
freqs = [220, 440, 880, 1760, 3520, 7040, 14080, 15500]
signal = np.zeros(num_samples)
for f in freqs:
    signal += 0.12 * np.sin(2 * np.pi * f * t)

# Normalize to 16-bit PCM
signal = np.clip(signal, -0.95, 0.95)
int_data = (signal * 32767).astype(np.int16)

with wave.open(str(test_file), "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(int_data.tobytes())

print(f"[OK] Generated test audio at: {test_file}")

# Test Audio Loader
from backend.app.services.audio_loader import load_audio
audio, sample_rate, dur = load_audio(test_file, target_sr=sr)
print(f"[OK] Loaded audio: duration={dur:.2f}s, sample_rate={sample_rate}, samples={len(audio)}")

# Test Detector Service
from backend.app.services.detector_service import extract_forensic_metrics
forensics = extract_forensic_metrics(audio, sample_rate)
print(f"[OK] Forensic analysis complete:")
print(f"     - Cutoff Detected: {forensics.cutoff_detected}")
print(f"     - Estimated Cutoff: {forensics.estimated_cutoff_hz} Hz")
print(f"     - Severity: {forensics.cutoff_severity}")
print(f"     - Rolloff (95%): {forensics.spectral_rolloff_95} Hz")
print(f"     - Notes: {forensics.forensic_notes}")

# Test Visualizer Service
from backend.app.services.visualizer_service import (
    compute_fft_spectrum,
    compute_tonal_balance,
    compute_spectrogram,
    render_publication_plot
)
fft_spec = compute_fft_spectrum(audio, sample_rate)
print(f"[OK] FFT Spectrum computed: {len(fft_spec.frequencies)} bins")

tonal_bal = compute_tonal_balance(audio, sample_rate)
print(f"[OK] Tonal Balance computed: {len(tonal_bal.bands)} bands")
for b in tonal_bal.bands:
    print(f"     • {b.name}: {b.energy_percent:.1f}%")

spectro = compute_spectrogram(audio, sample_rate)
print(f"[OK] Spectrogram computed: matrix shape {len(spectro.matrix_db)} x {len(spectro.matrix_db[0])}")

# Test Render Publication Plot
from backend.app.config import EXPORTS_DIR
export_path = EXPORTS_DIR / "test_figure_export.png"
render_publication_plot(audio, sample_rate, "test_synth.wav", export_path, forensics)
print(f"[OK] Publication plot successfully saved at: {export_path}")
print(f"     File size: {export_path.stat().st_size} bytes")

print("\nALL BACKEND SERVICES TESTED & OPERATIONAL!")
