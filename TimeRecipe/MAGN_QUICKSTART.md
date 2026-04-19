## MAGN Quick Start Guide

Modality-Aware Gated Network (MAGN) enables processing of multiple financial data modalities with learned importance weighting.

### Basic Usage

Import and configure:
```python
from exp.exp_long_term_forecasting import Exp

# Define MAGN configuration
args = {
    'task_name': 'long_term_forecast',
    'data_path': 'data/ETTm1.csv',
    'target': 'OT',
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64,
        'macro': 64
    },
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'd_model': 64,
    'd_ff': 256,
    'dropout': 0.1,
    'batch_size': 32,
    'train_epochs': 100
}

exp = Exp(args)
exp.train()
results = exp.test()
```

### Configuration Options

**magn_modalities** - dict mapping modality names to embedding dimensions
- Example: `{'price': 64, 'macro': 64, 'news': 48}`
- Dimensions should match your data's embedding sizes

**magn_use_feature_gating** - boolean (default: True)
- Applies sigmoid gates per modality to learn feature importance
- Helps filter noise in individual modalities

**magn_use_target_conditioning** - boolean (default: True)
- Conditions fusion weights on target forecast series
- Learns adaptive weighting based on prediction target

**magn_shared_mlp** - boolean (default: False)
- True: share MLP weights across modalities (parameter efficient)
- False: separate MLPs per modality (more expressive)

### Architecture Overview

MAGN processes each data modality through:
1. **Feature Gating**: Per-modality sigmoid gates learn which features matter
2. **Modality Processing**: Independent (or shared) MLPs transform gated embeddings
3. **Target-Conditioned Fusion**: Softmax weights fuse modalities based on target
4. **Output Projection**: Maps fused representation to forecast space

### Example: Multi-Modality Financial Forecasting

Setup with price, volatility, and sentiment:
```python
config = {
    'ffn_type': 'magn',
    'magn_modalities': {
        'price': 64,        # OHLCV features
        'volatility': 48,   # Realized/implied volatility
        'sentiment': 48     # News sentiment scores
    },
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'magn_shared_mlp': False,  # Modality-specific MLPs
    'd_model': 64,
    'd_ff': 256,
    'dropout': 0.15
}
```

### Model Behavior

- **Single Modality**: MAGN automatically handles single modality input, disables fusion
- **Many Modalities**: Scales to any number of modalities, learns optimal weights
- **Missing Modalities**: Can be handled by padding/zero-filling or implementing skip logic
- **Gradient Flow**: All components differentiable, works with standard SGD/Adam

### Performance Tips

1. Match embedding dimensions to modality complexity
   - Price (simple): d=48
   - Macro (moderate): d=64
   - Sentiment (complex): d=48

2. Use feature gating for noisy modalities
   - Helps model learn to ignore spurious signals

3. Enable target conditioning for dynamic fusion
   - Weights adapt based on prediction task

4. Use shared_mlp=True for memory efficiency
   - Reduces parameters while maintaining fusion flexibility

### Testing

All MAGN functionality is tested in `tests/test_magn.py`:
```bash
python -m unittest tests.test_magn -v
```

Tests cover:
- Single and multi-modality inputs
- Feature gating mechanisms
- Fusion weight constraints
- Gradient flow
- Deterministic behavior
- Batch independence
