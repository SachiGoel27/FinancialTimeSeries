import torch
import torch.nn as nn
import torch.nn.functional as F


class InstNorm(nn.Module):
    def __init__(self):
        super(InstNorm, self).__init__()

    def norm(self, x):
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / self.stdev
        return x
    
    def denorm(self, x):
        pred_len = x.shape[1]
        x = x * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
        x = x + (self.means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
        return x

class FracDiff(nn.Module):
    def __init__(self, d=0.4, window_size=500, threshold=1e-4):
        super(FracDiff, self).__init__()
        self.d = d
        self.window_size = window_size
        # Pre-compute weights using the binomial expansion of (1-L)^d
        weights = self._get_weights(d, threshold, window_size)
        # Register as a buffer so it moves to GPU with the model but isn't a "parameter"
        self.register_buffer('weights', weights.flip(0).view(1, 1, -1))

    def _get_weights(self, d, threshold, window_size):
        weights = [1.]
        for k in range(1, window_size):
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold: 
                break
            weights.append(w_k)
        return torch.tensor(weights, dtype=torch.float32)

    def norm(self, x):
        """
        Input x: [Batch, Lookback_L, Variables_D]
        We transpose to apply conv1d along the temporal axis (dim=1).
        """
        self.original_last_val = x[:, -1:, :].detach() # Store for potential 'denorm'
        
        # Reshape for conv1d: [B*D, 1, L]
        B, L, D = x.shape
        x = x.transpose(1, 2).reshape(B * D, 1, L)
        
        # Apply causal convolution (padding to maintain length)
        padding = self.weights.shape[-1] - 1
        x = F.pad(x, (padding, 0))
        x = F.conv1d(x, self.weights)
        
        # Reshape back to [B, L, D]
        return x.reshape(B, D, L).transpose(1, 2)

    def denorm(self, x):
        """
        In financial forecasting, you are typically predicting FD-returns.
        Converting back to price levels requires an 'Integration' (cumulative sum).
        For simplicity, we often just add the last known real price.
        """
        return x + self.original_last_val

class VolatilityNorm(nn.Module):
    """
    Volatility Normalization for financial time series.
    Scales returns by their estimated volatility to develop a signal relative to noise.
    """
    def __init__(self, method='ewma', window_size=21, alpha=0.1, eps=1e-7):
        super(VolatilityNorm, self).__init__()
        self.method = method
        self.window_size = window_size
        self.alpha = alpha
        self.eps = eps
        
        if self.method == 'ewma':
            # Pre-compute EWMA weights: w_i = alpha * (1-alpha)^i
            weights = [self.alpha * ((1 - self.alpha) ** i) for i in range(self.window_size)]
            weights = torch.tensor(weights, dtype=torch.float32)
            # Normalize weights to sum to 1 to prevent scaling drift
            weights = weights / weights.sum()
            self.register_buffer('weights', weights.view(1, 1, -1))
        elif self.method == 'rolling_std':
            # Pre-compute uniform weights for rolling mean/std
            weights = torch.ones(self.window_size, dtype=torch.float32) / self.window_size
            self.register_buffer('weights', weights.view(1, 1, -1))
            
    def forward(self, x):
        """
        Input x: [Batch, Sequence_Length, Variables]
        """
        self.last_volatility = None # Store for optional denorm if needed
        B, L, D = x.shape
        x_transpose = x.transpose(1, 2).reshape(B * D, 1, L)
        
        # Causal padding to ensure we only look at past data
        padding = self.window_size - 1
        x_padded = F.pad(x_transpose, (padding, 0))
        
        if self.method == 'ewma':
            # Approximate EWMA of squared returns (Assuming returns are zero-mean)
            x_sq = x_padded ** 2
            var = F.conv1d(x_sq, self.weights)
            volatility = torch.sqrt(var + self.eps)
        
        elif self.method == 'rolling_std':
            # Calculate rolling mean
            rolling_mean = F.conv1d(x_padded, self.weights)
            # Calculate rolling variance: E[x^2] - (E[x])^2
            x_sq_padded = x_padded ** 2
            rolling_mean_sq = F.conv1d(x_sq_padded, self.weights)
            var = rolling_mean_sq - (rolling_mean ** 2)
            # Clamp variance to prevent negative values due to floating point inaccuracies
            var = torch.clamp(var, min=0.0)
            volatility = torch.sqrt(var + self.eps)
            
        else:
            raise ValueError(f"Unknown Volatility Normalization method: {self.method}")
            
        # Reshape volatility back to [B, L, D]
        volatility = volatility.reshape(B, D, L).transpose(1, 2)
        self.last_volatility = volatility[:, -1:, :].detach()
        
        # Scale inputs by volatility
        return x / volatility

    def denorm(self, x):
        """
        Scale predictions back by the last known volatility.
        """
        pred_len = x.shape[1]
        return x * self.last_volatility.repeat(1, pred_len, 1)


class ResidualPreprocessor(nn.Module):
    """
    Residual Embedding Preprocessor for financial time series.
    
    Transforms raw asset returns into residual returns by removing the effect of 
    market/factor exposures. This allows the model to focus on stock-specific signals
    and idiosyncratic alpha.
    
    Formula:
        residual[t] = asset_return[t] - sum(beta_i[t] * factor_return_i[t])
        
    Where:
        - beta_i[t] is estimated using a rolling window regression (causal)
        - beta_i[t] uses only data up to time t-1 to avoid lookahead bias
        
    Args:
        window_size: Rolling window for beta estimation (default: 60)
        min_window: Minimum window size to start computing beta (default: 10)
        eps: Small epsilon to avoid division by zero (default: 1e-7)
    """
    
    def __init__(self, window_size=60, min_window=10, eps=1e-7):
        super(ResidualPreprocessor, self).__init__()
        self.window_size = window_size
        self.min_window = min_window
        self.eps = eps
        
        # Buffers to store intermediate values
        self.register_buffer('beta_values', None)
        self.register_buffer('factor_returns', None)
        
    def forward(self, asset_returns, factor_returns_list):
        """
        Compute residual returns by removing factor exposures.
        
        Args:
            asset_returns: Tensor of shape [Batch, TimeSteps, NumAssets]
                          Log returns of assets
            factor_returns_list: List of Tensors, each [Batch, TimeSteps, 1] or [Batch, TimeSteps, NumFactors]
                                Log returns of factors (market, sectors, macro factors, etc.)
        
        Returns:
            residual_returns: Tensor [Batch, TimeSteps, NumAssets]
                             Residual returns after removing factor exposures
            beta_dict: Dictionary containing:
                - 'betas': List of beta tensors [Batch, TimeSteps, NumAssets] for each factor
                - 'factor_returns': Stored factor returns for potential reconstruction
        """
        B, T, D = asset_returns.shape
        device = asset_returns.device
        
        # Handle single factor or list of factors
        if not isinstance(factor_returns_list, list):
            factor_returns_list = [factor_returns_list]
        
        num_factors = len(factor_returns_list)
        
        # Initialize outputs
        residual = asset_returns.clone()
        betas = []
        
        # Ensure all factors have the same time dimension
        for factor in factor_returns_list:
            assert factor.shape[1] == T, f"Factor time dimension {factor.shape[1]} != asset time dimension {T}"
        
        # Estimate beta for each factor and subtract its effect
        for factor_idx, factor_ret in enumerate(factor_returns_list):
            # factor_ret: [B, T, 1] or [B, T, F_dim]
            factor_ret = factor_ret.squeeze(-1) if factor_ret.shape[-1] == 1 else factor_ret.mean(dim=-1)  # [B, T]
            
            # Compute rolling beta with causal window (no lookahead bias)
            beta = self._compute_causal_rolling_beta(asset_returns, factor_ret, B, T, D, device)
            # beta: [B, T, D]
            
            betas.append(beta)
            
            # Subtract factor contribution from residuals
            # beta[t] * factor_ret[t] -> expand factor_ret to [B, T, D] for broadcasting
            factor_contribution = beta * factor_ret.unsqueeze(-1)  # [B, T, D]
            residual = residual - factor_contribution
        
        # Store for potential denormalization
        self.beta_values = betas
        self.factor_returns = factor_returns_list
        
        return residual, {'betas': betas, 'factor_returns': factor_returns_list}
    
    def _compute_causal_rolling_beta(self, asset_returns, factor_returns, B, T, D, device):
        """
        Compute rolling beta with causal window (only uses past data).
        
        beta[t] = cov(asset[t-window:t], factor[t-window:t]) / var(factor[t-window:t])
        
        Args:
            asset_returns: [B, T, D]
            factor_returns: [B, T]
            B, T, D: Batch, time, dimension
            device: Torch device
            
        Returns:
            beta: [B, T, D] rolling beta values
        """
        beta = torch.zeros(B, T, D, device=device)
        
        # Compute beta for each time step using causal window
        for t in range(T):
            # Use data from max(0, t - window_size) to t (inclusive)
            start_idx = max(0, t - self.window_size)
            end_idx = t + 1
            
            # Skip if window is too small
            if (end_idx - start_idx) < self.min_window:
                # Use a simple correlation-based estimate or previous beta
                if t > 0:
                    beta[:, t, :] = beta[:, t - 1, :]
                continue
            
            # Extract window
            asset_window = asset_returns[:, start_idx:end_idx, :]  # [B, window_len, D]
            factor_window = factor_returns[:, start_idx:end_idx]   # [B, window_len]
            
            # Center the data
            asset_mean = asset_window.mean(dim=1, keepdim=True)  # [B, 1, D]
            factor_mean = factor_window.mean(dim=1, keepdim=True)  # [B, 1]
            
            asset_centered = asset_window - asset_mean  # [B, window_len, D]
            factor_centered = factor_window.unsqueeze(-1) - factor_mean.unsqueeze(-1)  # [B, window_len, 1]
            
            # Compute covariance: cov(asset, factor) = mean((asset - mean_asset) * (factor - mean_factor))
            cov = (asset_centered * factor_centered).mean(dim=1)  # [B, D]
            
            # Compute variance of factor: var(factor) = mean((factor - mean_factor)^2)
            factor_var = (factor_centered ** 2).mean(dim=1).squeeze(-1)  # [B]
            
            # Compute beta = cov / var, avoiding division by zero
            # Reshape for broadcasting
            factor_var = torch.clamp(factor_var, min=self.eps).unsqueeze(-1)  # [B, 1]
            beta[:, t, :] = cov / factor_var
        
        return beta
    
    def denorm(self, residual_predictions, reconstruct_asset=True):
        """
        Reconstruct asset predictions from residuals.
        
        asset_pred[t] = residual_pred[t] + sum(beta_i[t] * factor_ret_i[t])
        
        Args:
            residual_predictions: [B, T_pred, D] residual predictions
            reconstruct_asset: If True, reconstruct asset predictions; else return residuals
            
        Returns:
            Reconstructed asset predictions or residuals as-is
        """
        if not reconstruct_asset or self.beta_values is None or self.factor_returns is None:
            return residual_predictions
        
        B, T_pred, D = residual_predictions.shape
        reconstructed = residual_predictions.clone()
        
        # Note: In practice, you'd need to forward-propagate beta values
        # For now, use the last known beta values
        for beta, factor_ret in zip(self.beta_values, self.factor_returns):
            # Take last beta value and extrapolate (or use as-is)
            last_beta = beta[:, -1:, :]  # [B, 1, D]
            factor_contribution = last_beta * factor_ret[:, -1:, :].squeeze(-1).unsqueeze(-1)  # [B, 1, D]
            reconstructed = reconstructed + factor_contribution.repeat(1, T_pred, 1)
        
        return reconstructed
    
class FourierSeasonalDemeaning(nn.Module):
    """
    Fourier Seasonal Demeaning for financial time series.
    
    Fits a sum of sinusoids at known financial frequencies to the input,
    subtracts the seasonal curve (pre-processing), and adds it back at
    the prediction horizon (post-processing).
    
    Frequencies assume daily-sampled data:
        5   = weekly
        21  = monthly  
        63  = quarterly
        126 = semi-annual
        252 = annual
    """
    def __init__(self, periods=[5, 21, 63, 126, 252], pred_len=1):
        super(FourierSeasonalDemeaning, self).__init__()
        self.periods = periods
        self.pred_len = pred_len

    def _build_fourier_basis(self, T, device):
        """
        Build sine/cosine basis matrix for time steps 0..T-1.
        Returns: [T, 2*num_periods]
        """
        t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)  # [T, 1]
        freqs = torch.tensor(self.periods, dtype=torch.float32, device=device).unsqueeze(0)  # [1, P]
        angles = 2 * torch.pi * t / freqs  # [T, P]
        basis = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [T, 2P]
        return basis

    def norm(self, x):
        """
        Input x: [B, L, D]
        Fit sinusoids on lookback window, subtract seasonal curve.
        """
        B, L, D = x.shape
        device = x.device

        # Build basis for lookback window [L, 2P]
        basis = self._build_fourier_basis(L, device)  # [L, 2P]

        # OLS: coefficients = (BtB)^-1 Bt x
        # basis: [L, 2P], x: [B, L, D]
        # Solve per batch and variable
        BtB = basis.T @ basis  # [2P, 2P]
        BtB_inv = torch.linalg.inv(BtB + 1e-6 * torch.eye(BtB.shape[0], device=device))  # regularize

        # x reshaped: [L, B*D]
        x_flat = x.permute(1, 0, 2).reshape(L, B * D)  # [L, B*D]
        
        # Coefficients: [2P, B*D]
        self.coeffs = BtB_inv @ basis.T @ x_flat  # [2P, B*D]

        # Fitted seasonal curve on lookback: [L, B*D]
        seasonal = basis @ self.coeffs  # [L, B*D]
        seasonal = seasonal.reshape(L, B, D).permute(1, 0, 2)  # [B, L, D]

        # Store L for post-processing
        self.L = L

        return x - seasonal

    def denorm(self, x):
        """
        Input x: [B, pred_len, D]
        Evaluate seasonal curve at future time steps and add back.
        """
        B, pred_len, D = x.shape
        device = x.device

        # Build basis for prediction horizon [pred_len, 2P]
        # Time indices continue from L
        t = torch.arange(self.L, self.L + pred_len, dtype=torch.float32, device=device).unsqueeze(1)
        freqs = torch.tensor(self.periods, dtype=torch.float32, device=device).unsqueeze(0)
        angles = 2 * torch.pi * t / freqs
        basis_future = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [pred_len, 2P]

        # Evaluate seasonal curve at future steps: [pred_len, B*D]
        seasonal_future = basis_future @ self.coeffs  # [pred_len, B*D]
        seasonal_future = seasonal_future.reshape(pred_len, B, D).permute(1, 0, 2)  # [B, pred_len, D]

        return x + seasonal_future