import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    """
    Original metric function - kept for backward compatibility.
    Returns standard metrics tuple: (MAE, MSE, RMSE, MAPE, MSPE)
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# =============================================================================
# New Financial Metrics
# =============================================================================

def SMAPE(pred, true):
    """
    Symmetric Mean Absolute Percentage Error.
    More stable than MAPE when values are small or close to zero.
    
    Formula: 2 * |true - pred| / (|true| + |pred|)
    """
    numerator = np.abs(true - pred)
    denominator = np.abs(true) + np.abs(pred)
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(2.0 * numerator / denominator)


def MASE(pred, true, insample, seasonality=1):
    """
    Mean Absolute Scaled Error.
    Scale-free metric relative to a naive benchmark (in-sample naive forecast error).
    
    Args:
        pred: Predictions array
        true: Ground truth array
        insample: In-sample (training) data for computing naive forecast error
        seasonality: Seasonality period for naive forecast (default=1 for non-seasonal)
    
    Returns:
        MASE value
    """
    # Compute naive forecast error on in-sample data
    naive_errors = np.abs(insample[seasonality:] - insample[:-seasonality])
    scale = np.mean(naive_errors)
    
    # Avoid division by zero
    if scale < 1e-8:
        scale = 1e-8
    
    # Compute scaled error
    forecast_errors = np.abs(true - pred)
    return np.mean(forecast_errors) / scale


def directional_accuracy(pred, true, last_obs=None, target_type="return"):
    """
    Directional Accuracy / Hit Rate.
    Measures whether the model correctly predicts the direction/sign.
    
    Args:
        pred: Predictions array
        true: Ground truth array
        last_obs: Last observed values (required for price forecasting)
        target_type: "return" or "price"
            - "return": compare sign(pred) to sign(true)
            - "price": compare sign(pred - last_obs) to sign(true - last_obs)
    
    Returns:
        Directional accuracy (between 0 and 1)
    """
    if target_type == "return":
        # For returns: compare signs directly
        pred_direction = np.sign(pred)
        true_direction = np.sign(true)
    elif target_type == "price":
        # For prices: compare direction relative to last observed price
        if last_obs is None:
            raise ValueError("last_obs is required for price directional accuracy")
        pred_direction = np.sign(pred - last_obs)
        true_direction = np.sign(true - last_obs)
    else:
        raise ValueError(f"target_type must be 'return' or 'price', got {target_type}")
    
    # Calculate accuracy (fraction of correct direction predictions)
    correct = (pred_direction == true_direction)
    return np.mean(correct)


def oos_r2(pred, true, benchmark_pred):
    """
    Out-of-Sample R² (OOS R²).
    Measures whether the model beats a benchmark forecast.
    
    Formula: 1 - MSE_model / MSE_benchmark
    
    Args:
        pred: Model predictions
        true: Ground truth
        benchmark_pred: Benchmark predictions
    
    Returns:
        OOS R² value (positive = better than benchmark, negative = worse)
    """
    mse_model = np.mean((true - pred) ** 2)
    mse_benchmark = np.mean((true - benchmark_pred) ** 2)
    
    # Avoid division by zero
    if mse_benchmark < 1e-8:
        return np.nan
    
    return 1.0 - (mse_model / mse_benchmark)


def sharpe_from_signals(pred, true, annualization=252, target_type="return", last_obs=None):
    """
    Signal-based Sharpe Ratio.
    Constructs a simple long/short signal and computes annualized Sharpe.
    
    Signal logic:
        - Long (+1) if predicted value is positive
        - Short (-1) if predicted value is negative
    
    For price forecasts, uses predicted price change relative to last observed price.
    
    Args:
        pred: Predictions array
        true: Ground truth array (actual returns or prices)
        annualization: Annualization factor (252 for daily, 12 for monthly, etc.)
        target_type: "return" or "price"
        last_obs: Last observed values (required for price forecasting)
    
    Returns:
        Annualized Sharpe ratio
    """
    if target_type == "return":
        # Signal based on predicted return sign
        signal = np.sign(pred)
        actual_returns = true
    elif target_type == "price":
        # Signal based on predicted price change
        if last_obs is None:
            raise ValueError("last_obs is required for price signal Sharpe")
        signal = np.sign(pred - last_obs)
        # Actual returns from price changes
        actual_returns = (true - last_obs) / np.where(np.abs(last_obs) < 1e-8, 1e-8, last_obs)
    else:
        raise ValueError(f"target_type must be 'return' or 'price', got {target_type}")
    
    # Strategy returns = signal * actual returns
    strategy_returns = signal.flatten() * actual_returns.flatten()
    
    # Compute Sharpe ratio
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    if std_return < 1e-8:
        return np.nan
    
    sharpe = mean_return / std_return
    
    # Annualize
    annualized_sharpe = sharpe * np.sqrt(annualization)
    
    return annualized_sharpe


def financial_metrics(pred, true, target_type="return", insample=None, 
                      benchmark_pred=None, last_obs=None, annualization=252):
    """
    Main wrapper for computing financial evaluation metrics.
    
    Always includes:
        - mae, mse, rmse, mape, mspe, smape
    
    Conditionally includes:
        - mase: if insample is provided
        - directional_accuracy: if target_type is 'return' or 'price'
        - signal_sharpe: if target_type is 'return' or 'price'
        - oos_r2: if benchmark_pred is provided
    
    Args:
        pred: Predictions array
        true: Ground truth array
        target_type: "return", "price", or "volatility"
        insample: In-sample data for MASE calculation (optional)
        benchmark_pred: Benchmark predictions for OOS R² (optional)
        last_obs: Last observed values for price metrics (optional)
        annualization: Annualization factor for Sharpe (default 252)
    
    Returns:
        Dictionary of computed metrics
    """
    pred = np.array(pred)
    true = np.array(true)
    
    # Initialize results with standard metrics
    results = {
        'mae': float(MAE(pred, true)),
        'mse': float(MSE(pred, true)),
        'rmse': float(RMSE(pred, true)),
        'mape': float(MAPE(pred, true)),
        'mspe': float(MSPE(pred, true)),
        'smape': float(SMAPE(pred, true)),
    }
    
    # MASE (if insample provided)
    if insample is not None:
        insample = np.array(insample).flatten()
        try:
            results['mase'] = float(MASE(pred, true, insample, seasonality=1))
        except Exception as e:
            results['mase'] = np.nan
            results['mase_error'] = str(e)
    
    # Directional metrics (for return and price forecasting)
    if target_type in ["return", "price"]:
        try:
            results['directional_accuracy'] = float(
                directional_accuracy(pred, true, last_obs=last_obs, target_type=target_type)
            )
        except Exception as e:
            results['directional_accuracy'] = np.nan
            results['directional_accuracy_error'] = str(e)
        
        try:
            results['signal_sharpe'] = float(
                sharpe_from_signals(pred, true, annualization=annualization, 
                                   target_type=target_type, last_obs=last_obs)
            )
        except Exception as e:
            results['signal_sharpe'] = np.nan
            results['signal_sharpe_error'] = str(e)
    
    # OOS R² (if benchmark provided)
    if benchmark_pred is not None:
        benchmark_pred = np.array(benchmark_pred)
        try:
            results['oos_r2'] = float(oos_r2(pred, true, benchmark_pred))
        except Exception as e:
            results['oos_r2'] = np.nan
            results['oos_r2_error'] = str(e)
    
    # Add metadata
    results['target_type'] = target_type
    results['annualization'] = annualization
    
    return results