import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
  return F.mse_loss(x_hat, x)


def supervised_loss(h: torch.Tensor, h_sup: torch.Tensor) -> torch.Tensor:
  return F.mse_loss(h_sup[:, :-1, :], h[:, 1:, :])


def alignment_loss(z_world: torch.Tensor, z_timegan: torch.Tensor) -> torch.Tensor:
  return F.mse_loss(z_world, z_timegan)


def adversarial_loss(logits: torch.Tensor, target_ones: bool) -> torch.Tensor:
  target = torch.full_like(logits, 1.0 if target_ones else 0.0)
  return F.binary_cross_entropy_with_logits(logits, target)


def cycle_loss(z: torch.Tensor, z_cycle: torch.Tensor) -> torch.Tensor:
  return F.l1_loss(z_cycle, z)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
  losses = []
  for i, q in enumerate(quantiles):
    err = target[:, i] - pred[:, i]
    losses.append(torch.max(q * err, (q - 1) * err).mean())
  return sum(losses) / len(losses)


def envelope_loss(
  pred: torch.Tensor,
  week_max: torch.Tensor,
  week_min: torch.Tensor,
  loss_type: str = "pinball",
) -> torch.Tensor:
  target = torch.stack([week_max, week_min], dim=1)
  if loss_type == "mse":
    return F.mse_loss(pred, target)
  return pinball_loss(pred, target, quantiles=(0.9, 0.1))


class Phase1LossBundle(nn.Module):
  def __init__(self, weights: dict[str, float]):
    super().__init__()
    self.w = weights

  def forward(
    self,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    h: torch.Tensor,
    h_sup: torch.Tensor,
    z_world: torch.Tensor,
    z_pooled: torch.Tensor,
    d_real: torch.Tensor,
    d_fake: torch.Tensor,
    z_cycle: torch.Tensor,
    d_regime_real: torch.Tensor | None = None,
    d_regime_fake: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    l_recon = reconstruction_loss(x, x_hat)
    l_sup = supervised_loss(h, h_sup)
    l_align = alignment_loss(z_world, z_pooled)
    l_adv_g = adversarial_loss(d_fake, True)
    l_cycle = cycle_loss(z_world, z_cycle)
    total = (
      self.w.get("recon", 10.0) * l_recon
      + self.w.get("supervised", 5.0) * l_sup
      + self.w.get("align", 1.0) * l_align
      + self.w.get("adv", 1.0) * l_adv_g
      + self.w.get("cycle", 5.0) * l_cycle
    )
    metrics = {
      "recon": l_recon.item(),
      "supervised": l_sup.item(),
      "align": l_align.item(),
      "adv_g": l_adv_g.item(),
      "cycle": l_cycle.item(),
    }
    if d_regime_fake is not None:
      l_reg = adversarial_loss(d_regime_fake, True)
      total = total + self.w.get("adv", 1.0) * l_reg
      metrics["adv_regime"] = l_reg.item()
    return total, metrics
