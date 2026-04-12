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