import yfinance as yf
import pandas as pd
import numpy as np

def download_and_prepare_data(tickers, start_date, end_date, target_ticker):
    """
    Downloads daily Close prices, calculates log returns, and prepares the target.
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Forward fill missing data (e.g., trading halts), then drop columns with too many NaNs
    data = data.ffill().dropna(axis=1, thresh=int(len(data)*0.9))
    data = data.dropna() # Drop remaining rows with NaNs
    
    # Calculate daily log returns: ln(P_t / P_{t-1})
    log_returns = np.log(data / data.shift(1)).dropna()
    
    # Create the binary target (1 if target stock goes up tomorrow, 0 if down)
    # We shift the target return backwards by 1 day to align "today's features" with "tomorrow's target"
    target_returns = log_returns[target_ticker].shift(-1)
    binary_target = (target_returns > 0).astype(int)
    
    # Drop the last row because its target will be NaN (we don't know tomorrow yet)
    log_returns = log_returns.iloc[:-1]
    binary_target = binary_target.iloc[:-1]
    
    print(f"Data prepared. Shape: {log_returns.shape}")
    return log_returns, binary_target