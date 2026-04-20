# Example configurations for MAGN (Modality-Aware Gated Network)
# These examples show how to configure MAGN for different multimodal financial forecasting scenarios

# Example 1: Price + Macro factors
config_price_macro = {
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64,      # OHLCV data embedding dimension
        'macro': 64       # Macro economic indicators (inflation, rates, etc.)
    },
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'magn_shared_mlp': False,
    'd_model': 64,
    'd_ff': 256,
    'dropout': 0.1,
    'activation': 'gelu'
}

# Example 2: Price + Sentiment + Macro
config_multimodal_full = {
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64,       # OHLCV data
        'sentiment': 48,   # News sentiment scores
        'macro': 64        # Macroeconomic indicators
    },
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'magn_shared_mlp': False,
    'd_model': 64,
    'd_ff': 256,
    'dropout': 0.15,
    'activation': 'gelu'
}

# Example 3: Price + Volume + Volatility (market microstructure)
config_market_microstructure = {
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64,       # Price series
        'volume': 48,      # Trading volume
        'volatility': 48   # Realized volatility
    },
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'magn_shared_mlp': True,  # Shared MLP for efficiency
    'd_model': 64,
    'd_ff': 256,
    'dropout': 0.1,
    'activation': 'relu'
}

# Example 4: Lightweight MAGN (single modality fallback)
config_lightweight = {
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64
    },
    'magn_use_feature_gating': False,
    'magn_use_target_conditioning': False,
    'd_model': 64,
    'd_ff': 128,
    'dropout': 0.05,
    'activation': 'gelu'
}

# Example 5: Complex multimodal with many factors
config_complex = {
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64,           # Price movements
        'volume': 48,          # Trading volume
        'volatility': 48,      # Market volatility
        'macro': 64,           # Macroeconomic data
        'sentiment': 48,       # News/social sentiment
        'correlation': 32      # Cross-asset correlations
    },
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'magn_shared_mlp': False,
    'd_model': 64,
    'd_ff': 512,
    'dropout': 0.2,
    'activation': 'gelu'
}

# Configuration notes:
# - magn_modalities: dict mapping modality name to embedding dimension
#   dimensions should match your embedding output sizes
#
# - magn_use_feature_gating: applies sigmoid gates to learn importance of features per modality
#   set to False for simpler model, True for more expressive gating
#
# - magn_use_target_conditioning: conditions fusion weights on target series
#   set to True to learn target-aware fusion, False for simple averaging
#
# - magn_shared_mlp: shares MLP weights across modalities
#   set to True for parameter efficiency, False for modality-specific processing
#
# Typical usage in Model initialization:
# model = Model(
#     task_name='long_term_forecast',
#     data_path='data/ETTm1.csv',
#     target='OT',
#     train_epochs=100,
#     batch_size=32,
#     ffn_type='magn',
#     magn_modalities={'price': 64, 'macro': 64},
#     magn_use_feature_gating=True,
#     magn_use_target_conditioning=True,
#     magn_shared_mlp=False,
#     d_model=64,
#     d_ff=256,
#     dropout=0.1,
#     activation='gelu'
# )
