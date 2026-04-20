"""
Utility functions for Residual Embedding preprocessing.

Provides helper functions to:
1. Load market/factor data
2. Compute residual returns
3. Align asset and market data
4. Validate preprocessing results
"""

import torch
import numpy as np
import pandas as pd


def compute_log_returns(prices):
    """
    Compute log returns from price series.
    
    Args:
        prices: np.array or pd.Series of shape (T,) or (T, N)
                Price series (univariate or multivariate)
    
    Returns:
        returns: np.array of same shape as input
                 Log returns: log(price[t] / price[t-1])
                 First element is NaN (no return for first price)
    """
    if isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices.shape) == 1:
        prices = prices.reshape(-1, 1)
    
    # Compute log returns: log(p_t / p_{t-1})
    returns = np.log(prices[1:] / prices[:-1])
    
    # Prepend NaN for first element (no return available)
    nan_row = np.full((1, returns.shape[1]), np.nan)
    returns = np.vstack([nan_row, returns])
    
    return returns


def align_asset_market_data(asset_data, market_data):
    """
    Align asset and market time series data.
    
    Args:
        asset_data: np.array [T, N_assets] or pd.DataFrame
                    Asset price or return series
        market_data: np.array [T,] or [T, 1] or pd.Series
                     Market index price or return series
    
    Returns:
        asset_aligned: np.array [T, N_assets] aligned asset data
        market_aligned: np.array [T,] aligned market data
        valid_idx: np.array boolean mask of valid (non-NaN) entries
    """
    if isinstance(asset_data, pd.DataFrame):
        asset_values = asset_data.values
    else:
        asset_values = np.array(asset_data)
    
    if isinstance(market_data, pd.Series):
        market_values = market_data.values.reshape(-1, 1)
    elif isinstance(market_data, pd.DataFrame):
        market_values = market_data.values
    else:
        market_values = np.array(market_data)
        if len(market_values.shape) == 1:
            market_values = market_values.reshape(-1, 1)
    
    # Ensure same time dimension
    assert asset_values.shape[0] == market_values.shape[0], \
        f"Asset time steps {asset_values.shape[0]} != Market time steps {market_values.shape[0]}"
    
    # Create valid index (not NaN in either series)
    valid_idx = ~(np.isnan(asset_values).any(axis=1) | np.isnan(market_values).any(axis=1))
    
    return asset_values, market_values.squeeze(), valid_idx


def forward_fill_nans(data, limit=None):
    """
    Forward fill NaN values in time series data.
    
    Args:
        data: np.array [T, N] or [T,]
        limit: Maximum number of consecutive NaNs to fill
    
    Returns:
        data_filled: np.array with NaNs filled forward
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    data_filled = np.empty_like(data)
    
    for col in range(data.shape[1]):
        series = data[:, col].copy()
        mask = np.isnan(series)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        idx = np.maximum.accumulate(idx)
        data_filled[:, col] = series[idx]
    
    if squeeze:
        data_filled = data_filled.squeeze()
    
    return data_filled


def validate_residual_output(residuals, raw_returns, betas, market_returns, atol=1e-5):
    """
    Validate that residuals are computed correctly.
    
    Checks: residuals ≈ raw_returns - beta * market_returns
    
    Args:
        residuals: np.array [T, N]
        raw_returns: np.array [T, N]
        betas: np.array [T, N]
        market_returns: np.array [T,]
        atol: Absolute tolerance for comparison
    
    Returns:
        is_valid: bool, whether validation passes
        max_error: float, maximum reconstruction error
    """
    reconstructed = raw_returns - betas * market_returns.reshape(-1, 1)
    error = np.abs(residuals - reconstructed)
    max_error = np.nanmax(error)
    
    is_valid = max_error < atol
    
    return is_valid, max_error


def residuals_to_tensor(residuals, device='cpu', dtype=torch.float32):
    """
    Convert residuals (numpy or pandas) to PyTorch tensor.
    
    Args:
        residuals: np.array or pd.DataFrame [T, N]
        device: torch device
        dtype: torch data type
    
    Returns:
        tensor: torch.Tensor [1, T, N] (batch size 1)
    """
    if isinstance(residuals, pd.DataFrame):
        residuals = residuals.values
    
    residuals = np.array(residuals, dtype=np.float32)
    
    # Add batch dimension
    if len(residuals.shape) == 1:
        residuals = residuals.reshape(-1, 1)
    
    tensor = torch.from_numpy(residuals).to(device).to(dtype)
    tensor = tensor.unsqueeze(0)  # Add batch dimension [1, T, N]
    
    return tensor


def tensors_to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: torch.Tensor [B, T, N] or [T, N]
    
    Returns:
        array: np.array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


# Example usage documentation
RESIDUAL_EMBEDDING_EXAMPLE = """
# Example: Using Residual Embedding with TimeRecipe

from utils.residual_utils import compute_log_returns, align_asset_market_data
from module.norm import ResidualPreprocessor
import pandas as pd
import torch

# 1. Load asset and market data
asset_prices = pd.read_csv('asset_prices.csv', index_col='date')
market_prices = pd.read_csv('market_prices.csv', index_col='date')

# 2. Compute log returns
asset_returns = compute_log_returns(asset_prices)
market_returns = compute_log_returns(market_prices)

# 3. Align data
asset_aligned, market_aligned, valid_idx = align_asset_market_data(asset_returns, market_returns)

# 4. Compute residuals
preprocessor = ResidualPreprocessor(window_size=60)

asset_tensor = torch.from_numpy(asset_aligned).float().unsqueeze(0)  # [1, T, N]
market_tensor = torch.from_numpy(market_aligned).float().unsqueeze(0).unsqueeze(-1)  # [1, T, 1]

residuals, info = preprocessor(asset_tensor, [market_tensor])

# 5. Use residuals in TimeRecipe model
# Set config.use_residual_embedding = True
# Residuals will be automatically preprocessed in the model forward pass
"""
