import os
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_eurusd_data(ticker="EURUSD=X", start_date="2010-01-01", end_date="2026-09-19", save_path=None):
    """
    Downloads historical EUR/USD OHLCV data using yfinance.
    """
    print(f"Fetching {ticker} data from {start_date} to {end_date} via yfinance...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Handle potential MultiIndex columns returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.dropna(how="all")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Data saved successfully to {save_path}")
        
    return df

def load_data(file_path=None, ticker="EURUSD=X", start_date="2010-01-01", end_date="2026-09-19"):
    """
    Loads EUR/USD market data from CSV or downloads it if file is missing.
    Handles both Case 1 (Date is index) and Case 2 (Date is a normal column).
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading dataset from file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Case 1 & Case 2 Handling
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df = df.set_index("Date")
        else:
            # Date might already be index or index_col was 0
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
    else:
        print("CSV file not found or not provided. Fetching fresh market data...")
        df = fetch_eurusd_data(ticker=ticker, start_date=start_date, end_date=end_date, save_path=file_path)
        
    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Strictly sort chronologically
    df = df.sort_index()
    
    print("\nDataset Summary:")
    print(f"- Shape: {df.shape}")
    print(f"- Start Date: {df.index.min().strftime('%Y-%m-%d')}")
    print(f"- End Date: {df.index.max().strftime('%Y-%m-%d')}")
    print(f"- Chronological Monotonicity: {df.index.is_monotonic_increasing}")
    
    return df
