import torch
import torch.nn as nn


class _GRUBlock(nn.Module):
  def __init__(self, in_dim: int, hidden: int, out_dim: int | None = None):
    super().__init__()
    self.gru = nn.GRU(in_dim, hidden, batch_first=True)
    self.out = nn.Linear(hidden, out_dim or hidden)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out, _ = self.gru(x)
    return self.out(out)


class TimeGAN(nn.Module):
  """TimeGAN-style embedder, recovery, generator, supervisor, discriminator."""

  def __init__(
    self,
    num_features: int,
    seq_len: int,
    hidden: int = 64,
    z_dim: int | None = None,
  ):
    super().__init__()
    self.seq_len = seq_len
    self.hidden = hidden
    self.z_dim = z_dim or hidden

    self.embedder = _GRUBlock(num_features, hidden, hidden)
    self.recovery = _GRUBlock(hidden, hidden, num_features)
    self.generator = _GRUBlock(self.z_dim, hidden, hidden)
    self.supervisor = _GRUBlock(hidden, hidden, hidden)
    self.discriminator = nn.Sequential(
      nn.GRU(hidden, hidden, batch_first=True),
      nn.Flatten(),
    )
    self.disc_head = nn.Linear(hidden * seq_len, 1)

  def embed(self, x: torch.Tensor) -> torch.Tensor:
    return self.embedder(x)

  def recover(self, h: torch.Tensor) -> torch.Tensor:
    return self.recovery(h)

  def generate(self, z: torch.Tensor) -> torch.Tensor:
    return self.generator(z)

  def supervise(self, h: torch.Tensor) -> torch.Tensor:
    return self.supervisor(h)

  def discriminate(self, h: torch.Tensor) -> torch.Tensor:
    out, _ = self.discriminator[0](h)
    flat = out.reshape(h.size(0), -1)
    return self.disc_head(flat).squeeze(-1)

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    h = self.embed(x)
    x_hat = self.recover(h)
    h_sup = self.supervise(h)
    return {"h": h, "x_hat": x_hat, "h_sup": h_sup}

  @staticmethod
  def pool_hidden(h: torch.Tensor) -> torch.Tensor:
    return h.mean(dim=1)

  def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, self.seq_len, self.z_dim, device=device)
