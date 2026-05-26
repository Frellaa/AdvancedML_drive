"""Test-set trading simulation from envelope forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src_regi.config import artifacts_dir, load_config
from src_regi.data import PreparedData, load_log_returns, prepare_data, split_arrays
from src_regi.datasets import WindowDataset
from src_regi.evaluate import inverse_envelope_scale
from src_regi.forecast_head import ForecastHead
from src_regi.train_phase2 import build_encoder_from_checkpoint
from src_regi.utils import get_device, load_checkpoint


@dataclass
class BacktestResult:
  equity_curve: pd.Series
  trades: pd.DataFrame
  summary: dict[str, float]
  decisions: pd.DataFrame


def test_decision_indices(prepared: PreparedData, seq_len: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
  sl = split_arrays(prepared)["test"]
  n = sl.stop - sl.start - seq_len
  indices = np.array([sl.start + idx + seq_len - 1 for idx in range(n)])
  dates = prepared.features.index[indices]
  return indices, dates


def build_test_predictions(
  cfg: dict,
  phase1_path=None,
  phase2_path=None,
  device=None,
) -> pd.DataFrame:
  """One row per test decision day: predictions, realized forward stats, dates."""
  device = device or get_device()
  art = artifacts_dir(cfg)
  ckpt1 = load_checkpoint(phase1_path or art / "phase1_best.pt", map_location=device)
  ckpt2 = load_checkpoint(phase2_path or art / "phase2_head.pt", map_location=device)

  prepared = prepare_data(cfg)
  seq_len = cfg["data"]["sequence_length"]
  horizon = cfg["data"]["forecast_horizon"]
  target = cfg["data"]["target_ticker"]
  raw = load_log_returns(cfg)
  if target not in raw.columns:
    raise ValueError(f"{target} not in log returns")

  cum_log = raw[target].cumsum()
  indices, dates = test_decision_indices(prepared, seq_len)

  sl = split_arrays(prepared)["test"]
  ds = WindowDataset(
    prepared.features.values.astype(np.float32),
    prepared.regimes.values.astype(np.int64),
    seq_len,
    envelope_max=prepared.envelopes["week_max"].values.astype(np.float32),
    envelope_min=prepared.envelopes["week_min"].values.astype(np.float32),
    start_idx=sl.start,
    end_idx=sl.stop,
  )
  loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

  encoder = build_encoder_from_checkpoint(cfg, ckpt1, len(prepared.feature_columns), seq_len, device)
  head = ForecastHead(d_latent=cfg["model"]["d_latent"], dropout=cfg["model"]["dropout"]).to(device)
  head.load_state_dict(ckpt2["forecast_head"])
  encoder.eval()
  head.eval()

  env_mean = ckpt1.get("envelope_mean")
  env_scale = ckpt1.get("envelope_scale")

  pred_max_list, pred_min_list = [], []
  with torch.no_grad():
    for batch in loader:
      z = encoder(batch["x"].to(device))
      pred = head(z)
      pm = pred[:, 0].cpu().numpy()
      pn = pred[:, 1].cpu().numpy()
      if env_mean is not None and env_scale is not None:
        pm, pn = inverse_envelope_scale(pm, pn, env_mean, env_scale)
      pred_max_list.append(pm)
      pred_min_list.append(pn)

  pred_max = np.concatenate(pred_max_list)
  pred_min = np.concatenate(pred_min_list)

  rows = []
  for k, t in enumerate(indices):
    fwd = raw[target].iloc[t + 1 : t + 1 + horizon]
    fwd_log = float(fwd.sum())
    fwd_simple = float(np.exp(fwd_log) - 1.0)
    p_t = float(cum_log.iloc[t])
    true_max = float(cum_log.iloc[t + 1 : t + 1 + horizon].max())
    true_min = float(cum_log.iloc[t + 1 : t + 1 + horizon].min())
    rows.append(
      {
        "date": dates[k],
        "pred_max": pred_max[k],
        "pred_min": pred_min[k],
        "current_cum": p_t,
        "true_max": true_max,
        "true_min": true_min,
        "fwd_log_return": fwd_log,
        "fwd_simple_return": fwd_simple,
      }
    )
  df = pd.DataFrame(rows).set_index("date")
  df["pred_mid"] = (df["pred_max"] + df["pred_min"]) / 2.0
  df["pred_upside"] = df["pred_max"] - df["current_cum"]
  df["pred_downside"] = df["current_cum"] - df["pred_min"]
  df["pred_range"] = df["pred_max"] - df["pred_min"]
  df["pred_edge"] = df["pred_mid"] - df["current_cum"]
  return df


def _prepare_signals(decisions: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
  """Add rank columns used by relative entry rules."""
  out = decisions.copy()
  out["edge_rank"] = out["pred_edge"].rank(pct=True)
  out["upside_rank"] = out["pred_upside"].rank(pct=True)
  out["range_rank"] = out["pred_range"].rank(pct=True)
  out["top_frac_threshold"] = 1.0 - top_fraction
  return out


def _entry_signal(row: pd.Series, rule: str, min_upside: float) -> bool:
  if rule == "always_long":
    return True
  if rule == "always_flat":
    return False
  if rule == "mid_above_current":
    return row["pred_mid"] > row["current_cum"]
  if rule == "pred_edge_positive":
    return row["pred_edge"] > 0
  if rule == "upside_skew":
    return row["pred_upside"] > row["pred_downside"]
  if rule == "min_upside":
    return row["pred_upside"] > min_upside
  if rule == "combined":
    return (row["pred_mid"] > row["current_cum"]) and (row["pred_upside"] > row["pred_downside"])
  if rule == "top_quintile":
    return row["edge_rank"] >= row["top_frac_threshold"]
  if rule == "top_upside_quintile":
    return row["upside_rank"] >= row["top_frac_threshold"]
  if rule == "wide_band":
    return row["range_rank"] >= row["top_frac_threshold"]
  if rule == "oracle":
    return row["fwd_simple_return"] > 0
  raise ValueError(f"Unknown entry rule: {rule}")


def simulate_strategy(
  decisions: pd.DataFrame,
  initial_capital: float = 10_000.0,
  entry_rule: str = "upside_skew",
  min_upside: float = 0.0,
  position_fraction: float = 1.0,
  cost_bps: float = 0.0,
  non_overlap: bool = True,
  hold_steps: int = 5,
  top_fraction: float = 0.2,
) -> BacktestResult:
  """
  Long-only: enter at close on signal day, earn the realized 5-day forward return.

  If non_overlap=True, after a trade wait `hold_steps` decision rows before re-entering.
  """
  cost = cost_bps / 10_000.0
  cash = initial_capital
  equity_points: dict[pd.Timestamp, float] = {}
  trade_rows = []
  blocked_until_loc = 0
  decisions = _prepare_signals(decisions, top_fraction)

  for loc, (date, row) in enumerate(decisions.iterrows()):
    if non_overlap and loc < blocked_until_loc:
      equity_points[date] = cash
      continue

    if not _entry_signal(row, entry_rule, min_upside):
      equity_points[date] = cash
      continue

    invest = cash * position_fraction
    gross_ret = float(row["fwd_simple_return"])
    fees = invest * cost * 2
    net_pnl = invest * gross_ret - fees
    cash += net_pnl

    trade_rows.append(
      {
        "entry_date": date,
        "invested": invest,
        "fwd_return": gross_ret,
        "pnl": net_pnl,
        "equity_after": cash,
        "pred_upside": row["pred_upside"],
        "pred_downside": row["pred_downside"],
      }
    )
    if non_overlap:
      blocked_until_loc = loc + hold_steps
    equity_points[date] = cash

  for date in decisions.index:
    if date not in equity_points:
      equity_points[date] = cash

  equity_curve = pd.Series(equity_points, name="equity").sort_index()
  trades = pd.DataFrame(trade_rows)
  total_return = (cash / initial_capital) - 1.0

  summary = {
    "initial_capital": initial_capital,
    "final_equity": cash,
    "total_return_pct": 100.0 * total_return,
    "num_trades": len(trades),
    "win_rate_pct": 100.0 * (trades["pnl"] > 0).mean() if len(trades) else 0.0,
    "avg_trade_return_pct": 100.0 * trades["fwd_return"].mean() if len(trades) else 0.0,
  }
  if len(trades):
    summary["total_pnl"] = float(trades["pnl"].sum())

  return BacktestResult(
    equity_curve=equity_curve,
    trades=trades,
    summary=summary,
    decisions=decisions,
  )


def buy_and_hold_curve(
  decisions: pd.DataFrame,
  raw_returns: pd.Series,
  initial_capital: float = 10_000.0,
) -> pd.Series:
  """Buy and hold target over the calendar span covered by test decisions."""
  start, end = decisions.index.min(), decisions.index.max()
  daily = raw_returns.loc[start:end]
  cum = (1.0 + daily.apply(np.expm1)).cumprod() * initial_capital
  cum.name = "buy_hold"
  return cum
