import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

def calculate_timeripe_properties(df, target_col, period=5):
    # Ensure series is numeric and drop NaNs
    series = pd.to_numeric(df[target_col], errors='coerce').dropna().values
    
    # Check if data is constant to prevent ADF/STL crashes
    if len(np.unique(series)) <= 1:
        return {
            "Status": "Error: Data is constant (no variance to analyze)",
            "Stationarity": "N/A", "Trend": "N/A", "Seasonality": "N/A"
        }

    # 1. Trend & Seasonality Strength (Algorithm 1 & 2) [cite: 1630, 1635]
    try:
        stl = STL(series, period=period, robust=True).fit()
        trend_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))
        season_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
    except:
        trend_strength, season_strength = 0, 0
    
    # 2. Stationarity via ADF (Algorithm 3) [cite: 1657]
    try:
        p_val = adfuller(series)[1]
        stationarity = "High" if p_val <= 0.05 else "Low"
    except Exception as e:
        stationarity = f"Error: {e}"
    
    return {
        "Trend": "High" if trend_strength > 0.6 else "Low",
        "Seasonality": "High" if season_strength > 0.6 else "Low",
        "Stationarity": stationarity
    }

# Example usage for your files
btc_df = pd.read_csv('data/bitcoin.csv')
print(f"Unique BTC prices: {btc_df['Open'].nunique()}")

spy_df = pd.read_csv('data/spy.csv')
print(f"Unique SPY prices: {spy_df['SPY'].nunique()}")
print("BTC:", calculate_timeripe_properties(btc_df, target_col='Open', period=60))
print("SPY:", calculate_timeripe_properties(spy_df, target_col='SPY', period=60))