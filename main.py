import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

def calculate_timeripe_properties(df, target_col, period=60):
    series = pd.to_numeric(df[target_col], errors='coerce').dropna().values
    if len(np.unique(series)) <= 1:
        return {"Status": "Error: No variance"}

    # 1. Trend & Seasonality (Algorithm 1 & 2)
    stl = STL(series, period=period, robust=True).fit()
    trend_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))
    season_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))

    # 2. Stationarity (Algorithm 3)
    p_val = adfuller(series)[1]
    stationarity_val = 1 if p_val <= 0.05 else 0

    # 3. Shifting (Algorithm 4)
    z = (series - np.mean(series)) / np.std(series)
    thresholds = np.linspace(np.min(z), np.max(z), 10)
    medians = [np.median(np.where(z > s)[0]) for s in thresholds if len(np.where(z > s)[0]) > 0]
    norm_medians = (medians - np.min(medians)) / (np.max(medians) - np.min(medians))
    shifting_val = np.median(norm_medians)

    # 4. Transition (Algorithm 5 approximation)
    # Measures covariance of transition matrix across symbols
    diffs = np.sign(np.diff(series))
    matrix = np.zeros((3, 3))
    for i in range(len(diffs) - 1):
        matrix[int(diffs[i])+1, int(diffs[i+1])+1] += 1
    transition_val = np.trace(np.cov(matrix / len(diffs)))

    return {
        "Trend": "High" if trend_strength > 0.6 else "Low",
        "Seasonality": "High" if season_strength > 0.6 else "Low",
        "Stationarity": "High" if stationarity_val == 1 else "Low",
        "Shifting": "High" if shifting_val > 0.5 else "Low",
        "Transition": "High" if transition_val > 0.01 else "Low",
        "HL-Ratio": "High" if (len(series)*0.25) > 1 else "Low" # Contextual check
    }

# Execute for your data
btc_df = pd.read_csv('data/bitcoin.csv')
spy_df = pd.read_csv('data/spy.csv')
print("BTC Properties:", calculate_timeripe_properties(btc_df, 'Open'))
print("SPY Properties:", calculate_timeripe_properties(spy_df, 'SPY'))