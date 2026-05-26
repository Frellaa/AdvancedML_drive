import torch
import torch.nn as nn


class LatentMapper(nn.Module):
  def __init__(self, d_latent: int, hidden: int = 128):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_latent, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Linear(hidden, d_latent),
    )

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.net(z)


class LatentCycleGAN(nn.Module):
  """Cycle-consistent translators between regime latents."""

  def __init__(self, d_latent: int, hidden: int = 128):
    super().__init__()
    self.g_ab = LatentMapper(d_latent, hidden)
    self.g_ba = LatentMapper(d_latent, hidden)
    self.d_a = nn.Sequential(
      nn.Linear(d_latent, hidden),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden, 1),
    )
    self.d_b = nn.Sequential(
      nn.Linear(d_latent, hidden),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden, 1),
    )

  def forward(self, z: torch.Tensor, regime: torch.Tensor) -> dict[str, torch.Tensor]:
    mask_a = regime == 0
    mask_b = regime == 1
    z_ab = torch.zeros_like(z)
    z_ba = torch.zeros_like(z)
    if mask_a.any():
      z_ab[mask_a] = self.g_ab(z[mask_a])
    if mask_b.any():
      z_ba[mask_b] = self.g_ba(z[mask_b])
    return {"z_ab": z_ab, "z_ba": z_ba}

  def cycle(self, z: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
    out = self.forward(z, regime)
    z_cycle = torch.zeros_like(z)
    mask_a = regime == 0
    mask_b = regime == 1
    if mask_a.any():
      z_cycle[mask_a] = self.g_ba(out["z_ab"][mask_a])
    if mask_b.any():
      z_cycle[mask_b] = self.g_ba(self.g_ab(z[mask_b]))
    return z_cycle

  def discriminate_a(self, z: torch.Tensor) -> torch.Tensor:
    return self.d_a(z).squeeze(-1)

  def discriminate_b(self, z: torch.Tensor) -> torch.Tensor:
    return self.d_b(z).squeeze(-1)
