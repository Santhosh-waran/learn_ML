import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set clean visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def save_fig(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {filepath}")

def plot_historical_close(df, output_dir="outputs/plots"):
    """
    Visualization 1: EUR/USD Historical Closing Price (2010-2026) with Rolling Mean.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="EUR/USD Close")
    if "SMA_50" in df.columns:
        ax1.plot(df.index, df["SMA_50"], color="#ff7f0e", linewidth=1.5, label="50-Day SMA", linestyle="--")
    if "SMA_200" in df.columns or "SMA_100" in df.columns:
        sma_col = "SMA_100" if "SMA_100" in df.columns else "SMA_200"
        ax1.plot(df.index, df[sma_col], color="#2ca02c", linewidth=1.5, label=f"{sma_col}", linestyle=":")
        
    ax1.set_title("EUR/USD Historical Closing Price (2010 – 2026)", fontsize=16, fontweight="bold", pad=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("EUR/USD Exchange Rate", fontsize=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "01_historical_close.png")

def plot_train_test_split(train_df, test_df, output_dir="outputs/plots"):
    """
    Visualization 2: Train / Test Split Timeline with demarcation line.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(train_df.index, train_df["Close"], color="#1f77b4", linewidth=1.2, label="Training Data (80%)")
    ax.plot(test_df.index, test_df["Close"], color="#e377c2", linewidth=1.2, label="Testing Data (20%)")
    
    split_date = test_df.index.min()
    ax.axvline(x=split_date, color="black", linestyle="--", linewidth=2, label=f"Split Boundary ({split_date.strftime('%Y-%m-%d')})")
    
    ax.set_title("Chronological Train / Test Split Timeline (80% Train, 20% Test)", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("EUR/USD Close Price", fontsize=12)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "02_train_test_split.png")

def plot_actual_vs_predicted_timeline(results_df, output_dir="outputs/plots"):
    """
    Visualization 3: Actual vs Predicted Price Timeline on Test Data.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(results_df.index, results_df["Actual"], color="#1f77b4", linewidth=1.5, label="Actual Next-Day Close", alpha=0.85)
    ax.plot(results_df.index, results_df["Predicted"], color="#d62728", linewidth=1.2, label="Predicted Next-Day Close", linestyle="--", alpha=0.85)
    
    ax.set_title("EUR/USD Next-Day Price: Actual vs Predicted (Test Period)", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("EUR/USD Close Price", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "03_actual_vs_predicted_timeline.png")

def plot_actual_vs_predicted_scatter(results_df, metrics, output_dir="outputs/plots"):
    """
    Visualization 4: Actual vs Predicted Scatter plot with y = x line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(results_df["Actual"], results_df["Predicted"], alpha=0.4, color="#2b5c8f", edgecolors="none", s=25)
    
    # y = x reference line
    min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
    max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal (y = x)")
    
    # Annotate metrics
    stats_text = f"R² = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.5f}\nMAE = {metrics['MAE']:.5f}\nMAPE = {metrics['MAPE_%']:.3f}%"
    ax.text(0.05, 0.82, stats_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    
    ax.set_title("Actual vs Predicted Next-Day Close Price", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Next-Day Close Price ($y$)", fontsize=12)
    ax.set_ylabel(r"Predicted Next-Day Close Price ($\hat{y}$)", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "04_actual_vs_predicted_scatter.png")

def plot_simple_regression(X_train, y_train, slope, intercept, r2, output_dir="outputs/plots"):
    """
    Visualization 5 & 6: Simple Linear Regression Today's Close vs Tomorrow's Close & Equation.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x_train_close = X_train["Close"].values
    ax.scatter(x_train_close, y_train, alpha=0.3, color="#4c72b0", s=20, label="Train Observations")
    
    # Generate exact calculated regression line
    x_line = np.linspace(x_train_close.min(), x_train_close.max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color="#d62728", linewidth=2.5, label=rf"Fitted Line ($\hat{{Y}} = {intercept:.4f} + {slope:.4f} X$)")
    
    eq_text = f"Simple Linear Regression Equation:\nTomorrow's Close = {intercept:.6f} + ({slope:.6f} × Today's Close)\nSlope (β₁) = {slope:.6f}\nIntercept (β₀) = {intercept:.6f}\nR² = {r2:.4f}"
    ax.text(0.05, 0.75, eq_text, transform=ax.transAxes, fontsize=11, fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff9e6", edgecolor="#ffe0b2", alpha=0.95))
    
    ax.set_title("Simple Linear Regression: Today's Close vs Tomorrow's Close", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Today's Close Price ($X_t$)", fontsize=12)
    ax.set_ylabel("Tomorrow's Close Price ($Y_{t+1}$)", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "05_simple_regression_scatter.png")
    
    # Dedicated Equation Card (Visualization 6)
    fig_card, ax_card = plt.subplots(figsize=(8, 4))
    ax_card.axis("off")
    card_str = (
        "==========================================================\n"
        "           SIMPLE LINEAR REGRESSION FORMULA CARD          \n"
        "==========================================================\n\n"
        f"  Equation:  Ŷ_(t+1) = {intercept:.6f} + {slope:.6f} · Close_t\n\n"
        f"  * Slope (β₁):      {slope:.6f}\n"
        f"  * Intercept (β₀):  {intercept:.6f}\n"
        f"  * Train R² Score:  {r2:.4f}\n\n"
        "  Interpretation:\n"
        f"  For every $1.00 increase in Today's EUR/USD Close,\n"
        f"  Tomorrow's Close is predicted to change by ${slope:.6f}.\n"
        "=========================================================="
    )
    ax_card.text(0.05, 0.15, card_str, fontfamily="monospace", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f5f5", edgecolor="#333333"))
    save_fig(fig_card, output_dir, "06_simple_regression_equation.png")

def plot_raw_coefficients(coef_df, output_dir="outputs/plots"):
    """
    Visualization 7: Raw Linear Regression Coefficients Bar Chart.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df_sorted = coef_df.sort_values(by="Raw_Coefficient", ascending=True)
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in df_sorted["Raw_Coefficient"]]
    
    ax.barh(df_sorted["Feature"], df_sorted["Raw_Coefficient"], color=colors, height=0.6)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    
    ax.set_title("Multiple Linear Regression Raw Coefficients", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Raw Coefficient Value (Unscaled Features)", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "07_raw_coefficients.png")

def plot_standardized_coefficients(coef_df, output_dir="outputs/plots"):
    """
    Visualization 8: Standardized Linear Regression Coefficients Bar Chart.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df_sorted = coef_df.sort_values(by="Standardized_Coefficient", ascending=True)
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in df_sorted["Standardized_Coefficient"]]
    
    ax.barh(df_sorted["Feature"], df_sorted["Standardized_Coefficient"], color=colors, height=0.6)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    
    ax.set_title("Standardized Linear Regression Coefficients (Comparable Scale)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Standardized Coefficient Value (Std Dev Impact)", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "08_standardized_coefficients.png")

def plot_residuals(results_df, output_dir="outputs/plots"):
    """
    Visualization 9: Residuals vs Predicted Price Scatter Plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = results_df["Error"]  # Actual - Predicted
    ax.scatter(results_df["Predicted"], residuals, alpha=0.4, color="#9467bd", s=20)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    
    ax.set_title("Residual Analysis: Residuals vs Predicted Price", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Next-Day Close Price", fontsize=12)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "09_residuals_vs_predicted.png")

def plot_residual_distribution(results_df, output_dir="outputs/plots"):
    """
    Visualization 10: Residual Distribution Histogram & KDE curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = results_df["Error"]
    sns.histplot(residuals, kde=True, ax=ax, color="#2b5c8f", bins=50, stat="density", edgecolor="white", alpha=0.6)
    
    mean_res = residuals.mean()
    std_res = residuals.std()
    skew_res = residuals.skew()
    
    stats_str = f"Mean Residual = {mean_res:.6f}\nStd Dev = {std_res:.6f}\nSkewness = {skew_res:.4f}"
    ax.text(0.05, 0.75, stats_str, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    
    ax.set_title("Distribution of Prediction Errors (Residuals)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Residual (Actual - Predicted Price)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    save_fig(fig, output_dir, "10_residual_distribution.png")

def plot_model_vs_baseline(model_metrics, baseline_metrics, output_dir="outputs/plots"):
    """
    Visualization 11: Linear Regression vs Naive Baseline Performance Comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error Metrics Comparison
    metrics_labels = ["MAE", "RMSE"]
    model_vals = [model_metrics["MAE"], model_metrics["RMSE"]]
    base_vals = [baseline_metrics["MAE"], baseline_metrics["RMSE"]]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    ax1.bar(x - width/2, model_vals, width, label="Linear Regression", color="#1f77b4")
    ax1.bar(x + width/2, base_vals, width, label="Naive Baseline (Close_t)", color="#ff7f0e")
    
    ax1.set_ylabel("Error Magnitude", fontsize=12)
    ax1.set_title("Forecast Error Comparison (Lower is Better)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_labels, fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    for i in range(len(metrics_labels)):
        ax1.text(i - width/2, model_vals[i]*1.02, f"{model_vals[i]:.5f}", ha="center", fontsize=9)
        ax1.text(i + width/2, base_vals[i]*1.02, f"{base_vals[i]:.5f}", ha="center", fontsize=9)
        
    # Directional Accuracy Comparison
    dir_labels = ["Directional Accuracy (%)"]
    model_dir = [model_metrics["Directional_Accuracy_%"]]
    base_dir = [baseline_metrics["Directional_Accuracy_%"]]
    
    x2 = np.arange(len(dir_labels))
    ax2.bar(x2 - width/2, model_dir, width, label="Linear Regression", color="#2ca02c")
    ax2.bar(x2 + width/2, base_dir, width, label="Naive Baseline (Close_t)", color="#7f7f7f")
    ax2.axhline(y=50, color="red", linestyle="--", label="Random Guess (50%)")
    
    ax2.set_ylabel("Accuracy Percentage (%)", fontsize=12)
    ax2.set_title("Directional Prediction Accuracy (Higher is Better)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(dir_labels, fontsize=11)
    ax2.set_ylim(40, 65)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    ax2.text(-width/2, model_dir[0]+0.5, f"{model_dir[0]:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.text(width/2, base_dir[0]+0.5, f"{base_dir[0]:.2f}%", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, output_dir, "11_model_vs_baseline.png")
