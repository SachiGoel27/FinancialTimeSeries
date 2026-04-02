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



def financial_metrics(pred, true, target_type="return", insample=None, 
                      benchmark_pred=None, last_obs=None, annualization=252):
    """
    Main wrapper for computing financial evaluation metrics.
    
    Always includes:
        - mae, mse, rmse, mape, mspe, smape
    
    Conditionally includes:
        - mase: if insample is provided
        - directional_accuracy: if target_type is 'return' or 'price'
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


# =============================================================================
# Macro Forecasting Metrics
# =============================================================================

def fit_ar1(series):
    """
    Fit AR(1) model: y_t = alpha + beta * y_{t-1}
    Uses OLS regression to estimate coefficients.
    
    Args:
        series: 1D array of time series values (training data)
    
    Returns:
        tuple: (alpha, beta) coefficients
    """
    series = np.array(series).flatten()
    
    if len(series) < 3:
        raise ValueError("Series must have at least 3 observations to fit AR(1)")
    
    # y_t = alpha + beta * y_{t-1}
    y = series[1:]      # y_t (from t=1 to T)
    x = series[:-1]     # y_{t-1} (from t=0 to T-1)
    
    # OLS estimation: beta = Cov(y, x) / Var(x), alpha = mean(y) - beta * mean(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    var_x = np.var(x)
    
    if var_x < 1e-10:
        # Constant series, return naive forecast
        beta = 0.0
        alpha = y_mean
    else:
        beta = cov_xy / var_x
        alpha = y_mean - beta * x_mean
    
    return alpha, beta


def ar1_forecast(last_value, alpha, beta, horizon):
    """
    Generate AR(1) forecasts iteratively for multiple steps.
    
    y_{t+1} = alpha + beta * y_t
    y_{t+2} = alpha + beta * y_{t+1}
    ...
    
    Args:
        last_value: Last observed value (y_t)
        alpha: AR(1) intercept
        beta: AR(1) coefficient
        horizon: Number of steps to forecast
    
    Returns:
        array: Forecasted values of length `horizon`
    """
    forecasts = np.zeros(horizon)
    current = last_value
    
    for h in range(horizon):
        current = alpha + beta * current
        forecasts[h] = current
    
    return forecasts


def ar1_forecast_batch(last_values, alpha, beta, horizon):
    """
    Generate AR(1) forecasts for a batch of starting values.
    
    Args:
        last_values: Array of last observed values (shape: [batch_size] or [batch_size, 1])
        alpha: AR(1) intercept
        beta: AR(1) coefficient  
        horizon: Number of steps to forecast
    
    Returns:
        array: Forecasted values of shape [batch_size, horizon]
    """
    last_values = np.array(last_values).flatten()
    batch_size = len(last_values)
    
    forecasts = np.zeros((batch_size, horizon))
    current = last_values.copy()
    
    for h in range(horizon):
        current = alpha + beta * current
        forecasts[:, h] = current
    
    return forecasts


def mse_ratio_ar1(pred, true, ar1_pred):
    """
    Compute ratio of model MSE to AR(1) MSE.
    
    Ratio < 1.0: Model beats AR(1)
    Ratio > 1.0: AR(1) beats model
    Ratio = 1.0: Equal performance
    
    Args:
        pred: Model predictions
        true: Ground truth
        ar1_pred: AR(1) predictions
    
    Returns:
        float: MSE_model / MSE_AR(1)
    """
    mse_model = np.mean((true - pred) ** 2)
    mse_ar1 = np.mean((true - ar1_pred) ** 2)
    
    if mse_ar1 < 1e-10:
        return np.nan
    
    return mse_model / mse_ar1


def rmse_ratio_ar1(pred, true, ar1_pred):
    """
    Compute ratio of model RMSE to AR(1) RMSE.
    This is sqrt(MSE_model) / sqrt(MSE_AR(1)) = sqrt(MSE_ratio).
    
    Args:
        pred: Model predictions
        true: Ground truth
        ar1_pred: AR(1) predictions
    
    Returns:
        float: RMSE_model / RMSE_AR(1)
    """
    rmse_model = np.sqrt(np.mean((true - pred) ** 2))
    rmse_ar1 = np.sqrt(np.mean((true - ar1_pred) ** 2))
    
    if rmse_ar1 < 1e-10:
        return np.nan
    
    return rmse_model / rmse_ar1


def rmsfe_by_horizon(pred, true, horizon_labels=None):
    """
    Compute Root Mean Square Forecast Error (RMSFE) at each forecast horizon.
    
    Args:
        pred: Predictions array of shape [n_samples, pred_len] or [n_samples, pred_len, n_features]
        true: Ground truth array of same shape
        horizon_labels: Optional labels for each horizon (e.g., ['h=1', 'h=2', ...])
    
    Returns:
        dict: RMSFE for each horizon {'h1': value, 'h2': value, ...}
    """
    pred = np.array(pred)
    true = np.array(true)
    
    # Handle different shapes
    if pred.ndim == 3:
        # Shape: [n_samples, pred_len, n_features] -> average over features
        pred = pred.mean(axis=-1)
        true = true.mean(axis=-1)
    
    if pred.ndim == 1:
        # Single horizon
        return {'h1': float(np.sqrt(np.mean((true - pred) ** 2)))}
    
    # Shape: [n_samples, pred_len]
    n_samples, pred_len = pred.shape
    
    rmsfe = {}
    for h in range(pred_len):
        errors = true[:, h] - pred[:, h]
        rmsfe_h = np.sqrt(np.mean(errors ** 2))
        label = horizon_labels[h] if horizon_labels else f'h{h+1}'
        rmsfe[label] = float(rmsfe_h)
    
    return rmsfe


def metrics_by_horizon(pred, true, ar1_pred=None):
    """
    Compute multiple metrics at each forecast horizon.
    
    Args:
        pred: Predictions array of shape [n_samples, pred_len] or [n_samples, pred_len, n_features]
        true: Ground truth array of same shape
        ar1_pred: AR(1) predictions of same shape (optional)
    
    Returns:
        dict: Metrics for each horizon
            {
                'h1': {'mse': x, 'rmse': x, 'mae': x, 'mse_ratio_ar1': x, ...},
                'h2': {...},
                ...
            }
    """
    pred = np.array(pred)
    true = np.array(true)
    
    # Handle different shapes
    if pred.ndim == 3:
        pred = pred.mean(axis=-1)
        true = true.mean(axis=-1)
        if ar1_pred is not None:
            ar1_pred = np.array(ar1_pred)
            if ar1_pred.ndim == 3:
                ar1_pred = ar1_pred.mean(axis=-1)
    
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
        true = true.reshape(-1, 1)
        if ar1_pred is not None:
            ar1_pred = ar1_pred.reshape(-1, 1)
    
    n_samples, pred_len = pred.shape
    
    results = {}
    for h in range(pred_len):
        pred_h = pred[:, h]
        true_h = true[:, h]
        
        horizon_metrics = {
            'mse': float(np.mean((true_h - pred_h) ** 2)),
            'rmse': float(np.sqrt(np.mean((true_h - pred_h) ** 2))),
            'mae': float(np.mean(np.abs(true_h - pred_h))),
        }
        
        if ar1_pred is not None:
            ar1_h = ar1_pred[:, h]
            mse_ar1 = np.mean((true_h - ar1_h) ** 2)
            
            if mse_ar1 > 1e-10:
                horizon_metrics['mse_ratio_ar1'] = float(horizon_metrics['mse'] / mse_ar1)
                horizon_metrics['rmse_ratio_ar1'] = float(horizon_metrics['rmse'] / np.sqrt(mse_ar1))
                horizon_metrics['oos_r2_ar1'] = float(1.0 - horizon_metrics['mse'] / mse_ar1)
            else:
                horizon_metrics['mse_ratio_ar1'] = np.nan
                horizon_metrics['rmse_ratio_ar1'] = np.nan
                horizon_metrics['oos_r2_ar1'] = np.nan
            
            horizon_metrics['mse_ar1'] = float(mse_ar1)
            horizon_metrics['rmse_ar1'] = float(np.sqrt(mse_ar1))
        
        results[f'h{h+1}'] = horizon_metrics
    
    return results


def theil_u_statistic(pred, true, naive_pred):
    """
    Compute Theil's U statistic.
    
    U = sqrt(MSE_model) / sqrt(MSE_naive)
    
    U < 1: Model beats naive forecast
    U = 1: Model equals naive forecast
    U > 1: Naive forecast beats model
    
    Args:
        pred: Model predictions
        true: Ground truth
        naive_pred: Naive forecast predictions (e.g., AR(1) or random walk)
    
    Returns:
        float: Theil's U statistic
    """
    rmse_model = np.sqrt(np.mean((true - pred) ** 2))
    rmse_naive = np.sqrt(np.mean((true - naive_pred) ** 2))
    
    if rmse_naive < 1e-10:
        return np.nan
    
    return rmse_model / rmse_naive


def macro_metrics(pred, true, insample, last_obs_all=None, ar1_coeffs=None):
    """
    Wrapper for macro forecasting evaluation metrics.
    
    Computes:
        - Standard metrics (mae, mse, rmse)
        - AR(1) benchmark comparison (mse_ratio_ar1, rmse_ratio_ar1, oos_r2_ar1)
        - Theil's U statistic
        - RMSFE by horizon
        - All metrics by horizon
    
    Args:
        pred: Predictions array of shape [n_samples, pred_len] or [n_samples, pred_len, n_features]
        true: Ground truth array of same shape
        insample: Training data for fitting AR(1) model
        last_obs_all: Last observed values before each prediction (shape: [n_samples] or [n_samples, 1])
        ar1_coeffs: Pre-computed AR(1) coefficients (alpha, beta). If None, will fit from insample.
    
    Returns:
        dict: Dictionary containing all macro metrics
    """
    pred = np.array(pred)
    true = np.array(true)
    insample = np.array(insample).flatten()
    
    # Get dimensions
    original_shape = pred.shape
    if pred.ndim == 3:
        n_samples, pred_len, n_features = pred.shape
        # For now, average over features for scalar metrics
        pred_flat = pred.mean(axis=-1)
        true_flat = true.mean(axis=-1)
    elif pred.ndim == 2:
        n_samples, pred_len = pred.shape
        n_features = 1
        pred_flat = pred
        true_flat = true
    else:
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        n_samples = len(pred_flat)
        pred_len = 1
        n_features = 1
    
    # Fit AR(1) model on training data
    if ar1_coeffs is None:
        alpha, beta = fit_ar1(insample)
    else:
        alpha, beta = ar1_coeffs
    
    # Generate AR(1) forecasts for each sample
    if last_obs_all is not None:
        last_obs_all = np.array(last_obs_all).flatten()
        if len(last_obs_all) != n_samples:
            # Try to broadcast or use last value of insample
            if len(last_obs_all) == 1:
                last_obs_all = np.full(n_samples, last_obs_all[0])
            else:
                raise ValueError(f"last_obs_all length {len(last_obs_all)} doesn't match n_samples {n_samples}")
        ar1_pred = ar1_forecast_batch(last_obs_all, alpha, beta, pred_len)
    else:
        # Use last value of insample for all predictions
        last_val = insample[-1]
        ar1_pred = ar1_forecast_batch(np.full(n_samples, last_val), alpha, beta, pred_len)
    
    # Initialize results
    results = {
        'ar1_alpha': float(alpha),
        'ar1_beta': float(beta),
    }
    
    # Overall metrics
    results['mae'] = float(MAE(pred_flat.flatten(), true_flat.flatten()))
    results['mse'] = float(MSE(pred_flat.flatten(), true_flat.flatten()))
    results['rmse'] = float(RMSE(pred_flat.flatten(), true_flat.flatten()))
    
    # AR(1) comparison metrics (overall)
    results['mse_ratio_ar1'] = float(mse_ratio_ar1(pred_flat, true_flat, ar1_pred))
    results['rmse_ratio_ar1'] = float(rmse_ratio_ar1(pred_flat, true_flat, ar1_pred))
    results['oos_r2_ar1'] = float(1.0 - results['mse_ratio_ar1']) if not np.isnan(results['mse_ratio_ar1']) else np.nan
    results['theil_u'] = float(theil_u_statistic(pred_flat, true_flat, ar1_pred))
    
    # AR(1) baseline metrics
    results['mse_ar1'] = float(MSE(ar1_pred.flatten(), true_flat.flatten()))
    results['rmse_ar1'] = float(RMSE(ar1_pred.flatten(), true_flat.flatten()))
    
    # RMSFE by horizon
    results['rmsfe_by_horizon'] = rmsfe_by_horizon(pred_flat, true_flat)
    
    # All metrics by horizon
    results['metrics_by_horizon'] = metrics_by_horizon(pred_flat, true_flat, ar1_pred)
    
    # Add metadata
    results['n_samples'] = n_samples
    results['pred_len'] = pred_len
    results['n_features'] = n_features
    
    return results