import torch
import torch.nn as nn

class LogTransform(nn.Module):
    def __init__(self, eps=1e-8):
        super(LogTransform, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        Forward log transform
        x: [B, T, C]
        """
        x = torch.clamp(x, min=self.eps)
        return torch.log(x)

    def backward(self, x):
        """
        Inverse log transform
        x: [B, T, C]
        """
        return torch.exp(x)