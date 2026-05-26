from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src_regi.config import resolve_path
from src_regi.targets import compute_envelope_targets


@dataclass
class PreparedData:
    features: pd.DataFrame
    envelopes: pd.DataFrame
    regimes: pd.Series
    scaler: StandardScaler
    envelope_scaler: StandardScaler
    feature_columns: list[str]
    train_end: int
    val_end: int


def load_log_returns(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg, cfg["data"]["log_returns_path"])
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.sort_index()


def compute_regime_labels(
    features: pd.DataFrame,
    train_mask: np.ndarray,
    vol_window: int = 20,
) -> pd.Series:
    """Equal-weight basket vol; median threshold fit on train only."""
    basket = features.mean(axis=1)
    realized_vol = basket.rolling(vol_window, min_periods=vol_window).std()
    train_vol = realized_vol[train_mask].dropna()
    threshold = train_vol.median()
    regimes = (realized_vol > threshold).astype(int)
    regimes.name = "regime"
    return regimes.fillna(0).astype(int)


def chronological_split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return train_end, val_end


def prepare_data(cfg: dict) -> PreparedData:
    raw = load_log_returns(cfg)
    data_cfg = cfg["data"]
    horizon = data_cfg["forecast_horizon"]
    target = data_cfg["target_ticker"]

    envelopes = compute_envelope_targets(raw, target, horizon=horizon)
    common_idx = raw.index.intersection(envelopes.index)
    features = raw.loc[common_idx]
    envelopes = envelopes.loc[common_idx]

    n = len(features)
    train_end, val_end = chronological_split_indices(
        n, data_cfg["train_ratio"], data_cfg["val_ratio"]
    )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True

    regimes = compute_regime_labels(
        features, train_mask, vol_window=data_cfg.get("regime_vol_window", 20)
    )

    scaler = StandardScaler()
    scaled = features.copy()
    scaled.iloc[:train_end] = scaler.fit_transform(features.iloc[:train_end])
    scaled.iloc[train_end:val_end] = scaler.transform(features.iloc[train_end:val_end])
    scaled.iloc[val_end:] = scaler.transform(features.iloc[val_end:])

    env_cols = ["week_max", "week_min"]
    envelope_scaler = StandardScaler()
    env_scaled = envelopes.copy()
    env_scaled.loc[:, env_cols] = np.nan
    env_scaled.loc[env_scaled.index[:train_end], env_cols] = envelope_scaler.fit_transform(
        envelopes[env_cols].iloc[:train_end]
    )
    env_scaled.loc[env_scaled.index[train_end:val_end], env_cols] = envelope_scaler.transform(
        envelopes[env_cols].iloc[train_end:val_end]
    )
    env_scaled.loc[env_scaled.index[val_end:], env_cols] = envelope_scaler.transform(
        envelopes[env_cols].iloc[val_end:]
    )

    return PreparedData(
        features=scaled,
        envelopes=env_scaled,
        regimes=regimes,
        scaler=scaler,
        envelope_scaler=envelope_scaler,
        feature_columns=list(features.columns),
        train_end=train_end,
        val_end=val_end,
    )


def split_arrays(prepared: PreparedData) -> dict[str, slice]:
    return {
        "train": slice(0, prepared.train_end),
        "val": slice(prepared.train_end, prepared.val_end),
        "test": slice(prepared.val_end, len(prepared.features)),
    }
