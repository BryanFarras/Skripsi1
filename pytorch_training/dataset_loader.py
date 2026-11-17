"""
PyTorch Dataset & Track-Grouped DataLoader Builder
Ensures strict song-level isolation (no slices from the same track in both train and validation).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple

from .config import DATASET_NPZ_PATH, METADATA_JSON_PATH, BATCH_SIZE, RANDOM_SEED


class AcousticTabularDataset(Dataset):
    """PyTorch Dataset wrapping standardized 108-dimensional acoustic feature vectors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_raw_dataset() -> Dict[str, Any]:
    """Loads cached acoustic feature dataset (.npz) and metadata (.json)."""
    if not DATASET_NPZ_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_NPZ_PATH}")

    data = np.load(DATASET_NPZ_PATH, allow_pickle=True)
    metadata = {}
    if METADATA_JSON_PATH.exists():
        with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "groups": data["groups"],
        "feature_names": list(data["feature_names"]),
        "metadata": metadata
    }


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, StandardScaler, np.ndarray, np.ndarray]:
    """
    Fits StandardScaler strictly on X[train_idx], transforms train and validation sets,
    and returns PyTorch DataLoaders.
    """
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = AcousticTabularDataset(X_train_scaled, y_train)
    val_dataset = AcousticTabularDataset(X_val_scaled, y_val)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, scaler, X_val_scaled, y_val

