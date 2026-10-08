import pandas as pd
import numpy as np

def validate_data(df):
    """
    Performs data integrity checks:
    - Missing values
    - Duplicate timestamps
    - Duplicate rows
    - Chronological monotonicity
    - Logical OHLC consistency
    - Identifies extreme financial movement outliers
    """
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    # 1. Missing Values
    null_counts = df.isnull().sum()
    print(f"\n1. Missing Values per Column:\n{null_counts.to_string()}")
    
    # 2. Duplicate Timestamps
    dup_index = df.index.duplicated().sum()
    print(f"\n2. Duplicate Timestamps: {dup_index}")
    
    # 3. Duplicate Rows
    dup_rows = df.duplicated().sum()
    print(f"3. Duplicate Rows: {dup_rows}")
    
    # 4. Data Ordering
    is_monotonic = df.index.is_monotonic_increasing
    print(f"4. Data Ordering Monotonic Increasing: {is_monotonic}")
    
    # 5. OHLC Logical Consistency
    # High >= Open, High >= Close, Low <= Open, Low <= Close, High >= Low
    high_open_violations = (df["High"] < df["Open"]).sum()
    high_close_violations = (df["High"] < df["Close"]).sum()
    low_open_violations = (df["Low"] > df["Open"]).sum()
    low_close_violations = (df["Low"] > df["Close"]).sum()
    high_low_violations = (df["High"] < df["Low"]).sum()
    
    total_violations = (high_open_violations + high_close_violations + 
                        low_open_violations + low_close_violations + 
                        high_low_violations)
    
    print("\n5. OHLC Logical Consistency Checks:")
    print(f"   - High < Open Violations: {high_open_violations}")
    print(f"   - High < Close Violations: {high_close_violations}")
    print(f"   - Low > Open Violations: {low_open_violations}")
    print(f"   - Low > Close Violations: {low_close_violations}")
    print(f"   - High < Low Violations: {high_low_violations}")
    print(f"   - Total Logical Violations: {total_violations}")
    
    # 6. Outlier Analysis & Extreme Market Events
    daily_returns = df["Close"].pct_change()
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    extreme_mask = (daily_returns - mean_ret).abs() > (3 * std_ret)
    extreme_events = df[extreme_mask]
    
    print(f"\n6. Financial Outlier Analysis (Daily Return > 3 Std Dev):")
    print(f"   - Number of Extreme Days (>3 sigma): {len(extreme_events)}")
    print("   - Sample Extreme Volatility Dates:")
    for date, row in extreme_events.head(5).iterrows():
        ret_pct = daily_returns.loc[date] * 100
        print(f"     * {date.strftime('%Y-%m-%d')}: Close={row['Close']:.4f}, Return={ret_pct:+.2f}%")
        
    print("\nNote on Outliers:")
    print("Extreme EUR/USD price movements are NOT invalid data points or sensor errors.")
    print("They correspond to genuine macroeconomic regime shifts and systemic shocks such as:")
    print(" - Jan 2015: Swiss National Bank (SNB) unpegging the CHF/EUR rate.")
    print(" - Jun 2016: Brexit Referendum result vote.")
    print(" - Mar 2020: Global COVID-19 pandemic liquidity shock.")
    print(" - 2022: Federal Reserve aggressive interest rate hike cycle & Euro parity.")
    print("Removing these outliers would distort the true historical risk profile and model evaluation.")
    
    print("=" * 60)
    
    validation_passed = (null_counts.sum() == 0 and dup_index == 0 and 
                         is_monotonic and total_violations == 0)
    return {
        "null_counts": null_counts,
        "duplicate_timestamps": dup_index,
        "duplicate_rows": dup_rows,
        "is_monotonic": is_monotonic,
        "ohlc_violations": total_violations,
        "extreme_outlier_count": len(extreme_events),
        "validation_passed": validation_passed
    }
