# Services package
from .audio_loader import load_audio
from .stem_service import StemService, ensure_ffmpeg_in_path

try:
    from .detector_service import extract_forensic_metrics
    from .visualizer_service import (
        compute_fft_spectrum,
        compute_tonal_balance,
        compute_spectrogram,
        render_publication_plot
    )
except ImportError:
    extract_forensic_metrics = None
    compute_fft_spectrum = None
    compute_tonal_balance = None
    compute_spectrogram = None
    render_publication_plot = None

__all__ = [
    "load_audio",
    "extract_forensic_metrics",
    "compute_fft_spectrum",
    "compute_tonal_balance",
    "compute_spectrogram",
    "render_publication_plot",
    "StemService",
    "ensure_ffmpeg_in_path"
]
