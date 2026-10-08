import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE) in percentage.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_predictions(y_true, y_pred, price_today=None):
    """
    Calculates statistical evaluation metrics:
    - MAE
    - MSE
    - RMSE
    - R²
    - MAPE
    - Directional Accuracy (if price_today is provided)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_arr, y_pred_arr)
    mape = calculate_mape(y_true_arr, y_pred_arr)
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_%": mape
    }
    
    if price_today is not None:
        price_today_arr = np.array(price_today)
        actual_dir = np.sign(y_true_arr - price_today_arr)
        pred_dir = np.sign(y_pred_arr - price_today_arr)
        dir_acc = (actual_dir == pred_dir).mean() * 100
        metrics["Directional_Accuracy_%"] = dir_acc
        
    return metrics

def evaluate_naive_baseline(X_test, y_test):
    """
    Evaluates the Naive Baseline: Tomorrow's predicted price = Today's closing price.
    Y_pred_naive = Close_t
    """
    price_today = X_test["Close"]
    y_pred_naive = price_today.values
    
    metrics = evaluate_predictions(y_test, y_pred_naive, price_today=price_today)
    metrics["Model"] = "Naive Baseline (Close_t)"
    return metrics

def extract_coefficients(feature_cols, raw_model, pipeline_model=None):
    """
    Extracts raw coefficients and standardized coefficients into a clean DataFrame.
    """
    raw_coefs = raw_model.coef_
    
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Raw_Coefficient": raw_coefs,
        "Abs_Raw_Coef": np.abs(raw_coefs)
    })
    
    if pipeline_model is not None:
        # Standardized model inside sklearn Pipeline
        std_model = pipeline_model.named_steps["model"]
        std_coefs = std_model.coef_
        coef_df["Standardized_Coefficient"] = std_coefs
        coef_df["Abs_Std_Coef"] = np.abs(std_coefs)
        coef_df = coef_df.sort_values(by="Abs_Std_Coef", ascending=False)
    else:
        coef_df = coef_df.sort_values(by="Abs_Raw_Coef", ascending=False)
        
    return coef_df

def print_model_report(train_metrics, test_metrics, baseline_metrics, raw_intercept, coef_df, train_dates, test_dates, n_train, n_test):
    """
    Prints a structured text report of all key model findings.
    """
    print("\n" + "=" * 70)
    print("      FINAL MACHINE LEARNING MODEL PERFORMANCE REPORT")
    print("=" * 70)
    
    print(f"Dataset Period:        2010 - 2026")
    print(f"Total Observations:    {n_train + n_test}")
    print(f"Training Observations: {n_train} ({train_dates[0]} -> {train_dates[1]})")
    print(f"Testing Observations:  {n_test} ({test_dates[0]} -> {test_dates[1]})")
    print(f"Model Algorithm:       Multiple Linear Regression")
    print(f"Target Variable:       Next-Day EUR/USD Close")
    print(f"Model Intercept:       {raw_intercept:.6f}")
    
    print("\n" + "-" * 70)
    print("TOP POSITIVE & NEGATIVE STANDARDIZED COEFFICIENTS:")
    print("-" * 70)
    std_sorted = coef_df.sort_values(by="Standardized_Coefficient", ascending=False)
    print("Top Positive Drivers:")
    for _, row in std_sorted.head(3).iterrows():
        print(f"  * {row['Feature']:<20}: Std Coef = {row['Standardized_Coefficient']:+.6f} (Raw = {row['Raw_Coefficient']:+.6f})")
        
    print("Top Negative Drivers:")
    for _, row in std_sorted.tail(3).iloc[::-1].iterrows():
        print(f"  * {row['Feature']:<20}: Std Coef = {row['Standardized_Coefficient']:+.6f} (Raw = {row['Raw_Coefficient']:+.6f})")
        
    print("\n" + "-" * 70)
    print("MODEL METRICS COMPARISON:")
    print("-" * 70)
    print(f"{'Metric':<25} | {'Train Model':<15} | {'Test Model':<15} | {'Naive Baseline':<15}")
    print("-" * 70)
    print(f"{'MAE':<25} | {train_metrics['MAE']:<15.6f} | {test_metrics['MAE']:<15.6f} | {baseline_metrics['MAE']:<15.6f}")
    print(f"{'RMSE':<25} | {train_metrics['RMSE']:<15.6f} | {test_metrics['RMSE']:<15.6f} | {baseline_metrics['RMSE']:<15.6f}")
    print(f"{'R² Score':<25} | {train_metrics['R2']:<15.6f} | {test_metrics['R2']:<15.6f} | {baseline_metrics['R2']:<15.6f}")
    print(f"{'MAPE (%)':<25} | {train_metrics['MAPE_%']:<15.4f}% | {test_metrics['MAPE_%']:<15.4f}% | {baseline_metrics['MAPE_%']:<15.4f}%")
    print(f"{'Directional Acc (%)':<25} | {train_metrics['Directional_Accuracy_%']:<15.2f}% | {test_metrics['Directional_Accuracy_%']:<15.2f}% | {baseline_metrics['Directional_Accuracy_%']:<15.2f}%")
    print("=" * 70)
