"""
Deep Audio Classifier Architectures in PyTorch
Designed for acoustic tabular feature representations in AI music vs Bona-fide audio detection.

Includes three progressive model generations to track architectural development:
- v1: ShallowAudioClassifier (Basic 1-layer MLP without BatchNorm or skip connections)
- v2: DeepMLPClassifier (4-layer Deep MLP with BatchNorm, Dropout, and LeakyReLU)
- v3: DeepAudioClassifier (Flagship Residual Deep MLP with skip projection, BatchNorm, Dropout, LeakyReLU)
"""

import torch
import torch.nn as nn
from typing import List


class ShallowAudioClassifier(nn.Module):
    """
    Model Generation v1: Shallow Multi-Layer Perceptron.
    Serves as the initial neural baseline without normalization or residual connections.
    """
    def __init__(self, input_dim: int = 108, hidden_dim: int = 64, num_classes: int = 1):
        super(ShallowAudioClassifier, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class DeepMLPClassifier(nn.Module):
    """
    Model Generation v2: Deep Feed-Forward Network with BatchNorm & Dropout.
    Introduces multi-layer hierarchical representation learning and regularization,
    without residual shortcut connections.
    """
    def __init__(
        self,
        input_dim: int = 108,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rates: List[float] = [0.3, 0.25, 0.2, 0.1],
        num_classes: int = 1
    ):
        super(DeepMLPClassifier, self).__init__()
        self.input_dim = input_dim

        layers = []
        in_d = input_dim
        for h_d, drop_p in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.BatchNorm1d(h_d),
                nn.LeakyReLU(0.1),
                nn.Dropout(drop_p)
            ])
            in_d = h_d

        self.feature_extractor = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.out(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class DeepAudioClassifier(nn.Module):
    """
    Model Generation v3 (Flagship): Deep Residual Audio Classifier (ResMLP).
    Combines multi-layer feed-forward blocks with a linear residual skip projection
    from Block 1 to Block 3 for smoother gradient flow and superior generalization.
    """
    def __init__(
        self,
        input_dim: int = 108,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rates: List[float] = [0.3, 0.25, 0.2, 0.1],
        num_classes: int = 1
    ):
        super(DeepAudioClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        # Block 1: Input projection
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.act1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(dropout_rates[0])
        
        # Block 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.act2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(dropout_rates[1])
        
        # Block 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.act3 = nn.LeakyReLU(0.1)
        self.drop3 = nn.Dropout(dropout_rates[2])
        
        # Block 4
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])
        self.act4 = nn.LeakyReLU(0.1)
        self.drop4 = nn.Dropout(dropout_rates[3])
        
        # Residual projection layer if dimensions match or need downsampling
        self.res_proj = nn.Linear(hidden_dims[0], hidden_dims[2])
        
        # Classification head (outputs single raw logit)
        self.out = nn.Linear(hidden_dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        h1 = self.drop1(self.act1(self.bn1(self.fc1(x))))
        
        # Block 2
        h2 = self.drop2(self.act2(self.bn2(self.fc2(h1))))
        
        # Block 3 with residual shortcut from Block 1
        h3_direct = self.drop3(self.act3(self.bn3(self.fc3(h2))))
        h1_res = self.res_proj(h1)
        h3 = h3_direct + h1_res
        
        # Block 4
        h4 = self.drop4(self.act4(self.bn4(self.fc4(h3))))
        
        # Output logit
        logits = self.out(h4)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get calibrated probabilities via sigmoid."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
