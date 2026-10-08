import os
import pandas as pd
import numpy as np

from src.data_loader import load_data
from src.preprocessing import validate_data
from src.features import prepare_dataset
from src.model import (
    split_data_chronological,
    train_raw_linear_regression,
    train_standardized_linear_regression,
    train_simple_linear_regression,
    run_walk_forward_validation
)
from src.evaluation import (
    evaluate_predictions,
    evaluate_naive_baseline,
    extract_coefficients,
    print_model_report
)
from src.visualization import (
    plot_historical_close,
    plot_train_test_split,
    plot_actual_vs_predicted_timeline,
    plot_actual_vs_predicted_scatter,
    plot_simple_regression,
    plot_raw_coefficients,
    plot_standardized_coefficients,
    plot_residuals,
    plot_residual_distribution,
    plot_model_vs_baseline
)

def run_pipeline():
    print("Starting EUR/USD Next-Day Price Prediction ML Pipeline...")
    
    data_dir = "data"
    output_dir = "outputs"
    plots_dir = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(output_dir, "metrics")
    preds_dir = os.path.join(output_dir, "predictions")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    
    csv_file = os.path.join(data_dir, "eurusd.csv")
    
    # 1. Load Data
    df = load_data(file_path=csv_file, ticker="EURUSD=X", start_date="2010-01-01", end_date="2026-09-19")
    
    # 2. Data Validation
    validation_res = validate_data(df)
    
    # 3. Feature Engineering & Target Creation
    df_clean, feature_cols, target_col = prepare_dataset(df)
    
    # 4. Chronological 80/20 Train-Test Split
    X_train, y_train, X_test, y_test, train_df, test_df = split_data_chronological(
        df_clean, feature_cols, target_col, train_pct=0.80
    )
    
    # 5. Model Training
    # A. Simple Linear Regression (Close_t -> Close_{t+1})
    simple_model, simple_slope, simple_intercept = train_simple_linear_regression(X_train, y_train)
    simple_train_preds = simple_model.predict(X_train[["Close"]])
    simple_r2 = simple_model.score(X_train[["Close"]], y_train)
    
    # B. Multiple Linear Regression (Raw)
    raw_model = train_raw_linear_regression(X_train, y_train)
    
    # C. Multiple Linear Regression (Standardized Pipeline)
    std_pipeline = train_standardized_linear_regression(X_train, y_train)
    
    # 6. Model Predictions
    train_preds = std_pipeline.predict(X_train)
    test_preds = std_pipeline.predict(X_test)
    
    # Results DataFrame
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": test_preds,
        "Close_Today": X_test["Close"],
        "Error": y_test - test_preds,
        "Absolute_Error": np.abs(y_test - test_preds)
    }, index=y_test.index)
    results_df.to_csv(os.path.join(preds_dir, "test_predictions.csv"))
    
    # 7. Model Evaluation
    train_metrics = evaluate_predictions(y_train, train_preds, price_today=X_train["Close"])
    test_metrics = evaluate_predictions(y_test, test_preds, price_today=X_test["Close"])
    baseline_metrics = evaluate_naive_baseline(X_test, y_test)
    
    # Extract Coefficients
    coef_df = extract_coefficients(feature_cols, raw_model, std_pipeline)
    coef_df.to_csv(os.path.join(metrics_dir, "coefficients.csv"), index=False)
    
    # Save Metrics Comparison
    metrics_summary = pd.DataFrame([
        {"Model": "Training Set Linear Regression", **train_metrics},
        {"Model": "Testing Set Linear Regression", **test_metrics},
        {"Model": "Naive Baseline (Close_t)", **baseline_metrics}
    ])
    metrics_summary.to_csv(os.path.join(metrics_dir, "model_metrics.csv"), index=False)
    
    # 8. Walk-Forward Cross Validation
    wf_df = run_walk_forward_validation(df_clean, feature_cols, target_col, n_splits=5, min_train_pct=0.50)
    wf_df.to_csv(os.path.join(metrics_dir, "walk_forward_metrics.csv"), index=False)
    
    # 9. Generate Visualizations (11 Plots)
    plot_historical_close(df, output_dir=plots_dir)
    plot_train_test_split(train_df, test_df, output_dir=plots_dir)
    plot_actual_vs_predicted_timeline(results_df, output_dir=plots_dir)
    plot_actual_vs_predicted_scatter(results_df, test_metrics, output_dir=plots_dir)
    plot_simple_regression(X_train, y_train, simple_slope, simple_intercept, simple_r2, output_dir=plots_dir)
    plot_raw_coefficients(coef_df, output_dir=plots_dir)
    plot_standardized_coefficients(coef_df, output_dir=plots_dir)
    plot_residuals(results_df, output_dir=plots_dir)
    plot_residual_distribution(results_df, output_dir=plots_dir)
    plot_model_vs_baseline(test_metrics, baseline_metrics, output_dir=plots_dir)
    
    # 10. Print Structured Executive Report
    train_dates = (train_df.index.min().strftime('%Y-%m-%d'), train_df.index.max().strftime('%Y-%m-%d'))
    test_dates = (test_df.index.min().strftime('%Y-%m-%d'), test_df.index.max().strftime('%Y-%m-%d'))
    print_model_report(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        raw_intercept=raw_model.intercept_,
        coef_df=coef_df,
        train_dates=train_dates,
        test_dates=test_dates,
        n_train=len(train_df),
        n_test=len(test_df)
    )

if __name__ == "__main__":
    run_pipeline()
