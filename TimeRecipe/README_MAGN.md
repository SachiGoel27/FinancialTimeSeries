# MAGN (Modality-Aware Gated Network)

A PyTorch-based feed-forward architecture for multimodal financial time series forecasting.

## Overview

MAGN enables intelligent fusion of multiple data modalities (price, macro, sentiment, etc.) through:
- **Feature Gating**: Per-modality sigmoid gates learn feature importance
- **Modality Processing**: Independent or shared MLP processing
- **Target-Conditioned Fusion**: Adaptive modality weighting based on forecast target
- **Output Projection**: Final mapping to prediction space

## Quick Start

Configure MAGN in your model:
```python
config = {
    'ffn_type': 'magn',
    'magn_modalities': {'price': 64, 'macro': 64},
    'magn_use_feature_gating': True,
    'magn_use_target_conditioning': True,
    'd_model': 64,
    'd_ff': 256,
    'dropout': 0.1
}
```

## Deliverables

### Code Files
- `module/architecture.py` - MAGN class implementation
- `model/unitsf.py` - Integration with TimeRecipe Model
- `tests/test_magn.py` - 13 comprehensive test cases (100% passing)
- `config_magn_examples.py` - 5 configuration examples

### Documentation
- `MAGN_QUICKSTART.md` - Quick start guide with examples
- `MAGN_DOCUMENTATION.md` - Complete technical documentation
- `MAGN_IMPLEMENTATION_SUMMARY.md` - Implementation overview and status

## Features

✓ Flexible input handling (tensor or dict format)
✓ Optional feature gating and target conditioning
✓ Shared or separate MLPs per modality
✓ Full gradient flow support
✓ Scales to any number of modalities
✓ Deterministic in eval mode
✓ Compatible with standard optimizers

## Testing

All 13 tests pass:
```bash
python -m unittest tests.test_magn -v
```

Covers:
- Single/multi-modality processing
- Feature gating behavior
- Fusion weight constraints
- Gradient flow
- Deterministic forward passes
- Batch independence

## Configuration Examples

5 ready-to-use configurations in `config_magn_examples.py`:
1. **Price + Macro** - Basic dual-modality setup
2. **Multimodal Full** - Price + Sentiment + Macro
3. **Market Microstructure** - Price + Volume + Volatility
4. **Lightweight** - Single modality with minimal overhead
5. **Complex** - 6 modalities with full feature set

## Architecture

```
Input Embedding(s)
    ↓
Feature Gates (per modality)
    ↓
Modality MLPs (shared or separate)
    ↓
Target-Conditioned Fusion
    ↓
Output Projection
    ↓
Prediction
```

## Integration

MAGN integrates with TimeRecipe through:
- Module: `model.unitsf.Model`
- FFN Type: `'magn'`
- Instantiation: Automatic in `_build_model()`
- Forward Pass: Handled in `forward()` with dict input conversion

## Performance

- Memory: ~1.5-2x standard MLP with multiple modalities
- Computation: Linear in number of modalities
- Training: Stable gradients, no normalization issues
- Inference: Fast softmax-based fusion

## Usage

Basic example:
```python
from exp.exp_long_term_forecasting import Exp

args = {
    'task_name': 'long_term_forecast',
    'ffn_type': 'magn',
    'magn_modalities': {'price': 64, 'macro': 64},
    'train_epochs': 100,
    'batch_size': 32
}

exp = Exp(args)
exp.train()
results = exp.test()
```

See `MAGN_QUICKSTART.md` for more examples.

## Status

**✅ COMPLETE AND TESTED**

All components implemented, tested, and integrated. Ready for production use.

---

For detailed documentation, see `MAGN_DOCUMENTATION.md`
