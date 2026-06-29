"""LSTM-based action classifier operating on keypoint sequences.

Privacy-preserving: uses only skeleton data, no raw video pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learns which timesteps matter most for classification.

    Args:
        hidden_size: Size of the LSTM hidden state (doubled if bidirectional).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted sum over timesteps.

        Args:
            lstm_output: Tensor of shape ``(B, T, H)``.

        Returns:
            Tensor of shape ``(B, H)`` — weighted sum over time.
        """
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        return (lstm_output * weights).sum(dim=1)


class KeypointLSTM(nn.Module):
    """Bi-LSTM action classifier with attention pooling.

    Architecture::

        Input (B, T, 51)
          → LayerNorm(51)
          → Bi-LSTM(51→hidden, num_layers, dropout)
          → AttentionPooling
          → Linear(hidden*2 → 64) + ReLU + Dropout
          → Linear(64 → num_classes)

    Args:
        input_size: Flattened keypoint dimension (17 × 3 = 51).
        hidden_size: LSTM hidden units per direction.
        num_layers: Stacked LSTM layers.
        num_classes: Number of output action labels.
        dropout: Dropout probability (applied between LSTM layers and before FC).
        bidirectional: Use bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int = 51,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 15,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = AttentionPooling(lstm_out_size)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, input_size)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        pooled = self.attention(lstm_out)
        return self.classifier(pooled)

    def get_model_info(self) -> dict:
        """Return a dict of model hyperparameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "bidirectional": self.bidirectional,
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
