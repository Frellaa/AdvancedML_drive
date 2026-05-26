import torch
import torch.nn as nn


class ForecastHead(nn.Module):
  """Predict next-horizon envelope (max, min) from macro latent."""

  def __init__(self, d_latent: int = 64, hidden: int = 64, dropout: float = 0.2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_latent, hidden),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, 2),
    )

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    """Returns (B, 2) with columns [week_max, week_min]."""
    return self.net(z)
