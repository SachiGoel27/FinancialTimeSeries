"""
Configuration template for TimeRecipe with Residual Embedding.

This file provides example configurations for different use cases.
Copy and modify as needed for your specific application.
"""

# ============================================================================
# CONFIGURATION 1: Minimal Setup (Single Market Factor)
# ============================================================================
class ResidualEmbeddingMinimal:
    """Minimal configuration with just market beta removal."""
    
    # Residual Embedding (NEW)
    use_residual_embedding = True
    residual_window = 60              # 3 months of trading days
    residual_mode = "replace"         # Use residuals directly
    residual_factors = ['market']     # Single market factor
    include_beta = False              # Don't include beta in embeddings
    include_market = False            # Don't include market returns
    
    # TimeRecipe Settings (Standard)
    use_norm = True
    use_decomp = False
    emb_type = 'token'
    ff_type = 'mlp'
    fusion = 'temporal'
    
    # Model Hyperparameters
    seq_len = 96
    pred_len = 24
    label_len = 12
    enc_in = 1  # Number of assets
    d_model = 64
    n_heads = 4
    e_layers = 2
    d_ff = 256
    dropout = 0.1
    activation = 'gelu'
    
    # Training Settings
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 10
    
    # Data Settings
    data_path = 'your_data.csv'
    features = 'S'  # 'S' for univariate
    scale = True
    freq = 'd'  # Daily


# ============================================================================
# CONFIGURATION 2: Standard Setup (Market Beta + Beta in Embeddings)
# ============================================================================
class ResidualEmbeddingStandard:
    """Standard configuration with market beta and beta inclusion."""
    
    # Residual Embedding
    use_residual_embedding = True
    residual_window = 60
    residual_mode = "replace"
    residual_factors = ['market']
    include_beta = True               # Include beta for interpretability
    include_market = False
    
    # Volatility Normalization (OPTIONAL)
    use_vol_norm = False
    vol_method = 'ewma'
    vol_window = 21
    
    # TimeRecipe Settings
    use_norm = True
    use_decomp = True
    emb_type = 'token'
    ff_type = 'transformer'
    fusion = 'temporal'
    
    # Model Hyperparameters
    seq_len = 96
    pred_len = 24
    label_len = 12
    enc_in = 5  # Multiple assets
    d_model = 128
    n_heads = 8
    e_layers = 3
    d_ff = 512
    dropout = 0.2
    activation = 'gelu'
    moving_avg = 25
    
    # Training Settings
    batch_size = 64
    learning_rate = 0.0001
    epochs = 200
    patience = 20
    
    # Data Settings
    features = 'M'  # Multivariate
    scale = True


# ============================================================================
# CONFIGURATION 3: Advanced Setup (Multi-Factor Residuals)
# ============================================================================
class ResidualEmbeddingAdvanced:
    """Advanced configuration with multiple factors and all components."""
    
    # Residual Embedding
    use_residual_embedding = True
    residual_window = 60
    residual_mode = "replace"
    residual_factors = ['market', 'sector', 'quality']  # Multiple factors
    include_beta = True               # Include all betas
    include_market = True             # Include market returns
    
    # Volatility Normalization
    use_vol_norm = True
    vol_method = 'ewma'
    vol_window = 21
    
    # TimeRecipe Settings
    use_norm = True
    use_decomp = True
    emb_type = 'patch'
    ff_type = 'transformer'
    fusion = 'temporal'
    
    # Model Hyperparameters
    seq_len = 252  # 1 year of trading days
    pred_len = 60  # 3 months forecast
    label_len = 126
    enc_in = 10    # Multiple assets
    d_model = 256
    n_heads = 16
    e_layers = 4
    d_ff = 1024
    dropout = 0.3
    activation = 'gelu'
    moving_avg = 25
    patch_len = 16
    
    # Training Settings
    batch_size = 128
    learning_rate = 0.00005
    epochs = 300
    patience = 30
    gradient_clip = 1.0
    
    # Data Settings
    features = 'M'
    scale = True


# ============================================================================
# CONFIGURATION 4: Fast Training (Minimal Features)
# ============================================================================
class ResidualEmbeddingFast:
    """Fast configuration for quick experimentation."""
    
    # Residual Embedding
    use_residual_embedding = True
    residual_window = 30             # Smaller window = faster
    residual_mode = "replace"
    residual_factors = ['market']
    include_beta = False
    include_market = False
    
    # TimeRecipe Settings (Minimal)
    use_norm = True
    use_decomp = False
    emb_type = 'token'
    ff_type = 'mlp'
    fusion = 'temporal'
    
    # Model Hyperparameters (Small)
    seq_len = 48
    pred_len = 12
    label_len = 6
    enc_in = 1
    d_model = 32
    n_heads = 2
    e_layers = 1
    d_ff = 128
    dropout = 0.1
    
    # Training Settings (Fast)
    batch_size = 16
    learning_rate = 0.001
    epochs = 50
    patience = 5


# ============================================================================
# CONFIGURATION 5: High Frequency Trading
# ============================================================================
class ResidualEmbeddingHighFreq:
    """Configuration for intraday/high-frequency data."""
    
    # Residual Embedding
    use_residual_embedding = True
    residual_window = 20             # Shorter window for intraday
    residual_mode = "replace"
    residual_factors = ['market']    # Hourly market index
    include_beta = True
    include_market = False
    
    # Volatility Normalization (Important for HF)
    use_vol_norm = True
    vol_method = 'ewma'
    vol_window = 20
    
    # TimeRecipe Settings
    use_norm = True
    use_decomp = True
    emb_type = 'token'
    ff_type = 'rnn'
    fusion = 'temporal'
    
    # Model Hyperparameters
    seq_len = 60   # 1 hour of 1-min bars
    pred_len = 15  # 15-min forecast
    label_len = 30
    enc_in = 1
    d_model = 64
    n_heads = 4
    e_layers = 2
    d_ff = 256
    dropout = 0.1
    
    # Training Settings
    batch_size = 256
    learning_rate = 0.001
    epochs = 100
    patience = 5


# ============================================================================
# CONFIGURATION 6: Pairs Trading Strategy
# ============================================================================
class ResidualEmbeddingPairsTrading:
    """Configuration optimized for pairs trading (mean reversion)."""
    
    # Residual Embedding
    use_residual_embedding = True
    residual_window = 120           # Longer window for stability
    residual_mode = "replace"
    residual_factors = ['market', 'sector']  # Remove market + sector
    include_beta = True             # Need beta for pairs analysis
    include_market = True           # Include market for reference
    
    # Volatility Normalization
    use_vol_norm = True
    vol_method = 'rolling_std'
    vol_window = 30
    
    # TimeRecipe Settings
    use_norm = True
    use_decomp = True               # Decompose trend/seasonality
    emb_type = 'token'
    ff_type = 'transformer'
    fusion = 'temporal'
    
    # Model Hyperparameters
    seq_len = 252  # 1 year history
    pred_len = 5   # 5-day mean reversion forecast
    label_len = 50
    enc_in = 2     # Pair of stocks
    d_model = 128
    n_heads = 8
    e_layers = 3
    d_ff = 512
    dropout = 0.2
    moving_avg = 25
    
    # Training Settings
    batch_size = 32
    learning_rate = 0.0001
    epochs = 200
    patience = 20


# ============================================================================
# CONFIGURATION 7: Portfolio Forecasting
# ============================================================================
class ResidualEmbeddingPortfolio:
    """Configuration for portfolio-level forecasting."""
    
    # Residual Embedding
    use_residual_embedding = True
    residual_window = 60
    residual_mode = "replace"
    residual_factors = ['market']   # Remove systematic risk
    include_beta = False            # Not needed for portfolio
    include_market = False
    
    # TimeRecipe Settings
    use_norm = True
    use_decomp = True
    emb_type = 'token'
    ff_type = 'transformer'
    fusion = 'feature'              # Feature fusion for multiple assets
    
    # Model Hyperparameters
    seq_len = 96
    pred_len = 24
    label_len = 12
    enc_in = 50   # Portfolio of 50 stocks
    d_model = 256
    n_heads = 16
    e_layers = 4
    d_ff = 1024
    dropout = 0.2
    moving_avg = 25
    
    # Training Settings
    batch_size = 128
    learning_rate = 0.00005
    epochs = 300
    patience = 30


# ============================================================================
# HELPER FUNCTION: Load Configuration
# ============================================================================
def load_config(config_name: str):
    """
    Load a configuration by name.
    
    Args:
        config_name: Name of the configuration class
        
    Returns:
        Configuration object
        
    Example:
        config = load_config('ResidualEmbeddingStandard')
        model = Model(config)
    """
    config_classes = {
        'minimal': ResidualEmbeddingMinimal,
        'standard': ResidualEmbeddingStandard,
        'advanced': ResidualEmbeddingAdvanced,
        'fast': ResidualEmbeddingFast,
        'high_freq': ResidualEmbeddingHighFreq,
        'pairs_trading': ResidualEmbeddingPairsTrading,
        'portfolio': ResidualEmbeddingPortfolio,
    }
    
    if config_name.lower() not in config_classes:
        raise ValueError(f"Unknown config: {config_name}")
    
    return config_classes[config_name.lower()]()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Load a configuration
    config = load_config('standard')
    
    # Print configuration
    print("Configuration: ResidualEmbeddingStandard")
    print("-" * 50)
    for key, value in vars(config).items():
        print(f"{key:30s}: {value}")
    
    # Or create a custom configuration
    class CustomConfig:
        """Custom configuration based on your specific needs."""
        
        # Copy from one of the above and modify
        use_residual_embedding = True
        residual_window = 90  # Custom window size
        
        # ... other settings ...
    
    custom = CustomConfig()
    print("\nCustom Configuration created successfully!")


# ============================================================================
# CONFIGURATION RECOMMENDATIONS
# ============================================================================
"""
Choose configuration based on your use case:

1. MINIMAL (ResidualEmbeddingMinimal)
   - For quick experiments
   - Single asset prediction
   - Market beta removal only
   
2. STANDARD (ResidualEmbeddingStandard)
   - For most financial applications
   - Multiple assets
   - Market beta removal + beta features
   
3. ADVANCED (ResidualEmbeddingAdvanced)
   - For professional/production systems
   - Multiple factors (market, sector, quality)
   - All components enabled
   
4. FAST (ResidualEmbeddingFast)
   - For rapid prototyping
   - Smaller models
   - Lower window sizes
   
5. HIGH_FREQ (ResidualEmbeddingHighFreq)
   - For intraday data
   - Shorter windows
   - Volatility normalization
   
6. PAIRS_TRADING (ResidualEmbeddingPairsTrading)
   - For mean reversion strategies
   - Longer windows
   - Multiple factors
   
7. PORTFOLIO (ResidualEmbeddingPortfolio)
   - For cross-sectional forecasting
   - Many assets (50+)
   - Feature fusion instead of temporal

FINE-TUNING TIPS:

• Window Size:
  - Daily: 60 (3 months)
  - Hourly: 20-30
  - Weekly: 120 (2 years)
  - Experiment with 0.5x, 1x, 2x default

• Model Size:
  - Start small, increase if needed
  - d_model: 32, 64, 128, 256
  - e_layers: 1, 2, 3, 4
  - n_heads: 2, 4, 8, 16

• Dropout:
  - Start with 0.1
  - Increase to 0.2-0.3 if overfitting
  - Decrease if underfitting

• Batch Size:
  - Start with 32-64
  - Increase if GPU memory allows
  - Adjust learning rate proportionally

• Learning Rate:
  - Start with 0.001
  - Reduce by 10x if unstable
  - Increase by 2x if training slow
"""
