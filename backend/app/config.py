from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Processing Defaults
TARGET_SAMPLE_RATE = 44100
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
MAX_UPLOAD_SIZE_MB = 100

# Supported Extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Server Settings
HOST = "127.0.0.1"
PORT = 8000
