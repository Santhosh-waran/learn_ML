import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data_chronological(df, feature_cols, target_col, train_pct=0.80):
    """
    Splits the time-series dataset chronologically into 80% Training and 20% Testing.
    Strictly preserves temporal order (no random shuffling).
    """
    split_idx = int(len(df) * train_pct)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print("\nChronological Train-Test Split Summary:")
    print(f"- Total Samples: {len(df)}")
    print(f"- Training Samples: {len(train_df)} ({train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')})")
    print(f"- Testing Samples:  {len(test_df)} ({test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})")
    
    return X_train, y_train, X_test, y_test, train_df, test_df

def train_raw_linear_regression(X_train, y_train):
    """
    Trains an unscaled Multiple Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_standardized_linear_regression(X_train, y_train):
    """
    Trains a standardized Multiple Linear Regression pipeline.
    StandardScaler is fitted ONLY on X_train to prevent data leakage.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_simple_linear_regression(X_train, y_train):
    """
    Trains a Simple Linear Regression model using only today's Close price to predict tomorrow's Close.
    Y_{t+1} = beta_0 + beta_1 * Close_t
    """
    simple_model = LinearRegression()
    simple_model.fit(X_train[["Close"]], y_train)
    slope = simple_model.coef_[0]
    intercept = simple_model.intercept_
    return simple_model, slope, intercept

def run_walk_forward_validation(df, feature_cols, target_col, n_splits=5, min_train_pct=0.50):
    """
    Executes expanding window walk-forward validation for financial time-series.
    """
    total_len = len(df)
    min_train_size = int(total_len * min_train_pct)
    test_chunk_size = (total_len - min_train_size) // n_splits
    
    wf_results = []
    
    for fold in range(n_splits):
        train_end_idx = min_train_size + fold * test_chunk_size
        test_end_idx = train_end_idx + test_chunk_size if fold < n_splits - 1 else total_len
        
        train_fold = df.iloc[:train_end_idx]
        test_fold = df.iloc[train_end_idx:test_end_idx]
        
        X_tr = train_fold[feature_cols]
        y_tr = train_fold[target_col]
        X_te = test_fold[feature_cols]
        y_te = test_fold[target_col]
        
        # Fit pipeline
        pipe = train_standardized_linear_regression(X_tr, y_tr)
        preds = pipe.predict(X_te)
        
        # Evaluate
        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2 = r2_score(y_te, preds)
        
        actual_dir = np.sign(y_te.values - X_te["Close"].values)
        pred_dir = np.sign(preds - X_te["Close"].values)
        dir_acc = (actual_dir == pred_dir).mean() * 100
        
        wf_results.append({
            "Fold": fold + 1,
            "Train_Start": train_fold.index.min().strftime('%Y-%m-%d'),
            "Train_End": train_fold.index.max().strftime('%Y-%m-%d'),
            "Test_Start": test_fold.index.min().strftime('%Y-%m-%d'),
            "Test_End": test_fold.index.max().strftime('%Y-%m-%d'),
            "Train_Size": len(train_fold),
            "Test_Size": len(test_fold),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Directional_Accuracy_%": dir_acc
        })
        
    wf_df = pd.DataFrame(wf_results)
    return wf_df
