import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates point-in-time financial features strictly using information available
    prior to or at the end of trading day t.
    """
    data = df.copy()
    
    # 1. Daily Returns (Log returns for additivity and financial symmetry)
    # R_t = ln(P_t / P_{t-1})
    data["Daily_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    
    # 2. Lag Features
    data["Close_Lag1"] = data["Close"].shift(1)
    data["Close_Lag2"] = data["Close"].shift(2)
    data["Close_Lag3"] = data["Close"].shift(3)
    data["Close_Lag5"] = data["Close"].shift(5)
    data["Close_Lag10"] = data["Close"].shift(10)
    data["Close_Lag20"] = data["Close"].shift(20)
    
    # 3. Simple Moving Averages (SMA)
    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_100"] = data["Close"].rolling(window=100).mean()
    
    # 4. Exponential Moving Averages (EMA)
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    
    # 5. Volatility (Rolling standard deviation of log returns)
    data["Volatility_5"] = data["Daily_Return"].rolling(window=5).std()
    data["Volatility_10"] = data["Daily_Return"].rolling(window=10).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
    
    # 6. Price-based Range Features
    data["High_Low_Range"] = data["High"] - data["Low"]
    data["Open_Close_Range"] = data["Close"] - data["Open"]
    
    return data

def create_target(df):
    """
    Creates the target variable: Next trading day's Closing Price.
    Target_t = Close_{t+1}
    """
    data = df.copy()
    data["Target"] = data["Close"].shift(-1)
    return data

def get_feature_lists():
    """
    Returns the list of feature column names used for training.
    """
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Close_Lag1",
        "Close_Lag2",
        "Close_Lag3",
        "Close_Lag5",
        "Close_Lag10",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_10",
        "EMA_20",
        "Volatility_10",
        "Volatility_20",
        "High_Low_Range",
        "Open_Close_Range"
    ]
    target_col = "Target"
    return feature_cols, target_col

def prepare_dataset(df):
    """
    Applies feature engineering, target generation, and drops NaN rows.
    """
    df_feat = create_features(df)
    df_target = create_target(df_feat)
    
    feature_cols, target_col = get_feature_lists()
    
    # Drop NaNs caused by rolling/lag windows (first 100 rows) and final row (missing target)
    cleaned_df = df_target.dropna(subset=feature_cols + [target_col]).copy()
    
    print(f"Feature engineering completed:")
    print(f"- Total rows before cleaning: {len(df)}")
    print(f"- Total rows after cleaning NaNs: {len(cleaned_df)}")
    print(f"- Number of features: {len(feature_cols)}")
    
    return cleaned_df, feature_cols, target_col
