from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AudioMetadata(BaseModel):
    file_id: str
    original_filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    total_samples: int

class FFTSpectrumData(BaseModel):
    frequencies: List[float]       # Hz values (e.g. 20 to 22050 Hz downsampled)
    magnitudes_db: List[float]     # Average dB level
    peaks_db: List[float]          # Peak hold dB level
    min_db: float
    max_db: float

class TonalBalanceBand(BaseModel):
    name: str                      # "Sub", "Bass", "Low Mid", "High Mid", "Treble / Air"
    freq_range: List[float]        # [min_hz, max_hz]
    energy_percent: float          # Percentage of total spectral energy
    avg_db: float
    color_hex: str                 # Visual color representation

class TonalBalanceData(BaseModel):
    curve_freqs: List[float]
    curve_levels: List[float]
    bands: List[TonalBalanceBand]

class SpectrogramData(BaseModel):
    time_points: List[float]       # Timestamps in seconds
    frequencies: List[float]       # Frequency bins in Hz
    # 2D grid downsampled for fast WebGL/Canvas rendering
    matrix_db: List[List[float]]   # matrix_db[freq_bin][time_bin]
    min_db: float
    max_db: float
    detected_cutoff_hz: Optional[float] = None

class AIForensicMetrics(BaseModel):
    estimated_cutoff_hz: Optional[float] = None
    cutoff_detected: bool = False
    cutoff_severity: str = "None"           # "None", "Mild (18-20kHz)", "Strong (<16kHz)"
    spectral_rolloff_85: float
    spectral_rolloff_95: float
    spectral_centroid_hz: float
    spectral_flatness: float
    high_freq_energy_ratio: float           # Energy > 16kHz / Total Energy
    forensic_notes: List[str]

class AnalysisResponse(BaseModel):
    metadata: AudioMetadata
    fft_spectrum: FFTSpectrumData
    tonal_balance: TonalBalanceData
    spectrogram: SpectrogramData
    forensics: AIForensicMetrics
