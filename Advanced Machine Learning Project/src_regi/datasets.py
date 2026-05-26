import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src_regi.data import PreparedData, split_arrays


class WindowDataset(Dataset):
    """Sliding windows of multivariate returns with optional envelope targets."""

    def __init__(
        self,
        features: np.ndarray,
        regimes: np.ndarray,
        seq_len: int,
        envelope_max: np.ndarray | None = None,
        envelope_min: np.ndarray | None = None,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        self.features = features
        self.regimes = regimes
        self.seq_len = seq_len
        self.envelope_max = envelope_max
        self.envelope_min = envelope_min
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(features)
        self._max_start = self.end_idx - seq_len
        if self._max_start <= start_idx:
            raise ValueError("Not enough rows for sequence length in this split")

    def __len__(self) -> int:
        return self._max_start - self.start_idx

    def __getitem__(self, idx: int):
        i = self.start_idx + idx
        x = self.features[i : i + self.seq_len]
        regime = int(self.regimes[i + self.seq_len - 1])
        x_t = torch.tensor(x, dtype=torch.float32)
        out = {"x": x_t, "regime": torch.tensor(regime, dtype=torch.long)}
        if self.envelope_max is not None:
            label_idx = i + self.seq_len - 1
            out["week_max"] = torch.tensor(self.envelope_max[label_idx], dtype=torch.float32)
            out["week_min"] = torch.tensor(self.envelope_min[label_idx], dtype=torch.float32)
        return out


def _slice_numpy(arr: np.ndarray, sl: slice) -> np.ndarray:
    return arr[sl]


def build_dataloaders(
    prepared: PreparedData,
    seq_len: int,
    batch_size: int,
    phase: str = "phase1",
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    feats = prepared.features.values.astype(np.float32)
    regimes = prepared.regimes.values.astype(np.int64)
    env_max = prepared.envelopes["week_max"].values.astype(np.float32)
    env_min = prepared.envelopes["week_min"].values.astype(np.float32)

    splits = split_arrays(prepared)
    loaders = {}
    for name, sl in splits.items():
        start = sl.start
        end = sl.stop
        use_envelope = phase == "phase2"
        ds = WindowDataset(
            feats,
            regimes,
            seq_len,
            envelope_max=env_max if use_envelope else None,
            envelope_min=env_min if use_envelope else None,
            start_idx=start,
            end_idx=end,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle_train and name == "train"),
            drop_last=(name == "train"),
        )
    return loaders
