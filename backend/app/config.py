from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
APP_DIR = BACKEND_DIR / "app"

# Frontend directory (separated at root level)
FRONTEND_DIR = ROOT_DIR / "frontend"

# Runtime directories inside backend
UPLOADS_DIR = BACKEND_DIR / "uploads"
EXPORTS_DIR = BACKEND_DIR / "exports"
STEMS_TEMP_DIR = BACKEND_DIR / "stems_temp"
STEMS_EXPIRE_SECONDS = 3600  # Stems automatically expire and delete after 1 hour

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
STEMS_TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
