"""
PyTorch Training Pipeline Configuration
Defines directory paths, training hyperparameters, and reproducible random seeds.
"""

from pathlib import Path

# Base Directories
PYTORCH_DIR = Path(__file__).resolve().parent
APP_ROOT = PYTORCH_DIR.parent

# Input Dataset Paths (from ml_baseline)
ML_BASELINE_DIR = APP_ROOT / "ml_baseline"
DATASET_NPZ_PATH = ML_BASELINE_DIR / "data" / "dataset.npz"
METADATA_JSON_PATH = ML_BASELINE_DIR / "data" / "metadata.json"
BASELINE_METRICS_PATH = ML_BASELINE_DIR / "models" / "metrics.json"

# Output Artifact Directories (Neat, Structured Organization)
CHECKPOINTS_DIR = PYTORCH_DIR / "checkpoints"
LOGS_DIR = PYTORCH_DIR / "logs"
PLOTS_DIR = PYTORCH_DIR / "plots"
REPORTS_DIR = PYTORCH_DIR / "reports"

# Ensure all output folders exist
for d in [CHECKPOINTS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Reproducibility & Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.06
N_SPLITS = 5  # For StratifiedGroupKFold cross-validation
