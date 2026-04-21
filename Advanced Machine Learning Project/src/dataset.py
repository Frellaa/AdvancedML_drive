import torch
from torch.utils.data import Dataset

class FinancialSequenceDataset(Dataset):
    def __init__(self, features_df, target_series, sequence_length):
        """
        features_df: Pandas DataFrame of log returns for all stocks
        target_series: Pandas Series of binary targets (1 or 0)
        sequence_length: Lookback window (e.g., 30 days)
        """
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.targets = torch.tensor(target_series.values, dtype=torch.float32)
        self.seq_len = sequence_length
        
    def __len__(self):
        # Total valid windows
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        # X shape: (Sequence_Length, Num_Stocks)
        x = self.features[idx : idx + self.seq_len]
        # y shape: (1) -> The target for the day *after* the sequence
        y = self.targets[idx + self.seq_len - 1]
        
        return x, y