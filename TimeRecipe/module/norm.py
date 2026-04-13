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