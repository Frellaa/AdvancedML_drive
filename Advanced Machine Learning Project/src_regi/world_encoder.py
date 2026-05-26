import torch
import torch.nn as nn


class WorldEncoder(nn.Module):
  """Transformer encoder mapping multivariate windows to macro latent z."""

  def __init__(
    self,
    num_features: int,
    seq_len: int,
    d_model: int = 128,
    d_latent: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
  ):
    super().__init__()
    self.seq_len = seq_len
    self.d_latent = d_latent
    self.feature_projection = nn.Linear(num_features, d_model)
    self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dropout=dropout,
      batch_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.to_latent = nn.Sequential(
      nn.Linear(d_model, d_latent),
      nn.LayerNorm(d_latent),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, F) -> z: (B, d_latent)"""
    h = self.feature_projection(x) + self.pos_encoder
    h = self.transformer(h)
    h = h.transpose(1, 2)
    h = self.pool(h).squeeze(-1)
    return self.to_latent(h)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward(x)
