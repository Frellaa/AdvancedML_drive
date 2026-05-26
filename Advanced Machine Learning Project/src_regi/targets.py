import numpy as np
import pandas as pd


def cumulative_log_price(log_returns: pd.Series) -> pd.Series:
    """Cumulative log-price proxy from daily log returns."""
    return log_returns.cumsum()


def compute_envelope_targets(
    log_returns: pd.DataFrame,
    target_ticker: str,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    For each day t, label max/min of cumulative log-price over t+1..t+horizon.
    Rows without a full forward window are dropped.
    """
    if target_ticker not in log_returns.columns:
        raise ValueError(f"Target ticker {target_ticker} not in log_returns columns")

    cum_path = cumulative_log_price(log_returns[target_ticker])
    n = len(cum_path)
    week_max = np.full(n, np.nan)
    week_min = np.full(n, np.nan)

    values = cum_path.values
    for t in range(n - horizon):
        forward = values[t + 1 : t + horizon + 1]
        week_max[t] = forward.max()
        week_min[t] = forward.min()

    out = pd.DataFrame(
        {
            "week_max": week_max,
            "week_min": week_min,
        },
        index=log_returns.index,
    )
    out["week_range"] = out["week_max"] - out["week_min"]
    return out.dropna()
