"""Evaluation metrics for envelope forecasting."""

import numpy as np
import torch

from src_regi.losses import envelope_loss, pinball_loss


def inverse_envelope_scale(
  pred_max: np.ndarray,
  pred_min: np.ndarray,
  envelope_mean: list[float],
  envelope_scale: list[float],
) -> tuple[np.ndarray, np.ndarray]:
  mean = np.array(envelope_mean)
  scale = np.array(envelope_scale)
  stacked = np.stack([pred_max, pred_min], axis=1) * scale + mean
  return stacked[:, 0], stacked[:, 1]


@torch.no_grad()
def predict_envelopes(
  world_encoder,
  head,
  loader,
  device,
  envelope_mean: list[float] | None = None,
  envelope_scale: list[float] | None = None,
):
  world_encoder.eval()
  head.eval()
  preds_max, preds_min = [], []
  true_max, true_min = [], []
  for batch in loader:
    x = batch["x"].to(device)
    z = world_encoder(x)
    pred = head(z)
    pm = pred[:, 0].cpu().numpy()
    pn = pred[:, 1].cpu().numpy()
    tm = batch["week_max"].numpy()
    tn = batch["week_min"].numpy()
    if envelope_mean is not None and envelope_scale is not None:
      pm, pn = inverse_envelope_scale(pm, pn, envelope_mean, envelope_scale)
      tm, tn = inverse_envelope_scale(tm, tn, envelope_mean, envelope_scale)
    preds_max.append(pm)
    preds_min.append(pn)
    true_max.append(tm)
    true_min.append(tn)
  return (
    np.concatenate(preds_max),
    np.concatenate(preds_min),
    np.concatenate(true_max),
    np.concatenate(true_min),
  )


def mae(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.mean(np.abs(a - b)))


def interval_coverage(
  pred_max: np.ndarray,
  pred_min: np.ndarray,
  true_max: np.ndarray,
  true_min: np.ndarray,
) -> float:
  """Fraction of samples where true envelope is inside predicted band."""
  inside = (true_max <= pred_max) & (true_min >= pred_min)
  return float(np.mean(inside))


def naive_baseline_mae(
  features_cumsum_target: np.ndarray,
  true_max: np.ndarray,
  true_min: np.ndarray,
  horizon: int = 5,
) -> dict[str, float]:
  """Use last observed horizon-day range as prediction."""
  n = len(true_max)
  pred_max = np.zeros(n)
  pred_min = np.zeros(n)
  for i in range(n):
    start = max(0, i - horizon)
    window = features_cumsum_target[start:i] if i > 0 else features_cumsum_target[:1]
    if len(window) == 0:
      pred_max[i] = true_max[i]
      pred_min[i] = true_min[i]
    else:
      pred_max[i] = window.max()
      pred_min[i] = window.min()
  return {
    "mae_max": mae(pred_max, true_max),
    "mae_min": mae(pred_min, true_min),
  }


def evaluate_predictions(
  pred_max: np.ndarray,
  pred_min: np.ndarray,
  true_max: np.ndarray,
  true_min: np.ndarray,
  loss_type: str = "pinball",
) -> dict[str, float]:
  pred_t = torch.tensor(np.stack([pred_max, pred_min], axis=1), dtype=torch.float32)
  t_max = torch.tensor(true_max, dtype=torch.float32)
  t_min = torch.tensor(true_min, dtype=torch.float32)
  loss = envelope_loss(pred_t, t_max, t_min, loss_type=loss_type).item()
  return {
    "loss": loss,
    "mae_max": mae(pred_max, true_max),
    "mae_min": mae(pred_min, true_min),
    "coverage": interval_coverage(pred_max, pred_min, true_max, true_min),
    "mean_pred_range": float(np.mean(pred_max - pred_min)),
    "mean_true_range": float(np.mean(true_max - true_min)),
  }
