import torch
import torch.nn as nn
import math


class NoEmbedding(nn.Module):
    def __init__(self):
        super(NoEmbedding, self).__init__()

    def forward(self, x):
        return x
    

class FreqEmbedding(nn.Module):
    def __init__(self):
        super(FreqEmbedding, self).__init__()

    def forward(self, x):
        return torch.fft.rfft(x, dim=1)
    
    def inverse(self, x):
        return torch.fft.irfft(x, dim=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(TokenEmbedding, self).__init__()

        self.value_embedding = ValueEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        x = x.permute(0,2,1)
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.permute(0,2,1)
        return self.dropout(x)
    

class InvertEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(InvertEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        x = x.permute(0,2,1)
        return self.dropout(x)


class ResidualEmbedding(nn.Module):
    """
    Residual Embedding for financial time series.
    
    Flexible embedding that can work with residual returns and optionally include:
    - Raw asset returns
    - Market/factor returns
    - Estimated beta values
    
    This allows the model to learn relationships between residual and raw components.
    
    Args:
        c_in: Input dimension (residual component)
        d_model: Output embedding dimension
        include_raw: If True, concatenate raw asset returns
        include_market: If True, concatenate market returns
        include_beta: If True, concatenate beta values
        dropout: Dropout rate
    """
    
    def __init__(self, c_in, d_model, include_raw=False, include_market=False, 
                 include_beta=False, dropout=0.1):
        super(ResidualEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.include_raw = include_raw
        self.include_market = include_market
        self.include_beta = include_beta
        
        # Calculate total input size after concatenation
        self.total_c_in = c_in
        if include_raw:
            self.total_c_in += c_in  # Add raw returns (same dim as residual)
        if include_market:
            self.total_c_in += 1  # Market returns (single factor)
        if include_beta:
            self.total_c_in += 1  # Beta values (single factor)
        
        # Value embedding layer
        self.value_embedding = ValueEmbedding(c_in=self.total_c_in, d_model=d_model)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x_residual, x_raw=None, x_market=None, x_beta=None):
        """
        Forward pass for Residual Embedding.
        
        Args:
            x_residual: [Batch, TimeSteps, NumAssets] residual returns (required)
            x_raw: [Batch, TimeSteps, NumAssets] raw returns (optional)
            x_market: [Batch, TimeSteps, 1] market returns (optional)
            x_beta: [Batch, TimeSteps, NumAssets] beta values (optional)
        
        Returns:
            Embedded tensor of shape [Batch, TimeSteps, d_model]
        """
        B, T, D = x_residual.shape
        
        # Start with residual returns
        x = x_residual
        
        # Concatenate additional components if provided
        if self.include_raw and x_raw is not None:
            x = torch.cat([x, x_raw], dim=-1)
        
        if self.include_market and x_market is not None:
            # Expand market returns to match asset dimension
            x_market_expanded = x_market.expand(B, T, D)
            x = torch.cat([x, x_market_expanded], dim=-1)
        
        if self.include_beta and x_beta is not None:
            x = torch.cat([x, x_beta], dim=-1)
        
        # Apply value embedding and positional encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x)