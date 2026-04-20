# Residual Embedding Quick Start Guide

## 🎯 5-Minute Setup

### Step 1: Enable Residual Embedding in Config
```python
# In your config file or args
configs.use_residual_embedding = True
configs.residual_window = 60  # 60 trading days
```

### Step 2: Prepare Your Data
```python
from utils.residual_utils import compute_log_returns
import pandas as pd

# Load price data
asset_prices = pd.read_csv('your_asset.csv')['price'].values
market_prices = pd.read_csv('market_index.csv')['price'].values

# Compute log returns
asset_returns = compute_log_returns(asset_prices)
market_returns = compute_log_returns(market_prices)

print(f"Asset returns shape: {asset_returns.shape}")
print(f"Market returns shape: {market_returns.shape}")
```

### Step 3: Train Your Model
```python
from model.unitsf import Model
import torch

# Convert to tensors
asset_tensor = torch.from_numpy(asset_returns).float().unsqueeze(0)
market_tensor = torch.from_numpy(market_returns).float().unsqueeze(0).unsqueeze(-1)

# Create model with residual embedding enabled
model = Model(configs)

# Forward pass - residuals computed automatically
predictions = model(asset_tensor)
print(f"Predictions shape: {predictions.shape}")
```

---

## 📊 What Residual Embedding Does

**Before**: Model learns to predict market-wide movements
```
Returns = (0.02 from market) + (-0.001 stock-specific)
Model: "Predict the 0.02 market move"  ← Not interesting
```

**After**: Model learns stock-specific signals
```
Returns = (0.02 from market) + (-0.001 stock-specific)
Residuals = -0.001  ← Only the interesting part
Model: "Predict the -0.001 mean reversion"  ← Much better!
```

---

## 🔧 Configuration Options

### Minimal (Just Market Beta)
```python
configs.use_residual_embedding = True
configs.residual_window = 60
```

### Standard (Market Beta + Beta Inclusion)
```python
configs.use_residual_embedding = True
configs.residual_window = 60
configs.include_beta = True
configs.include_market = False
```

### Full (All Components)
```python
configs.use_residual_embedding = True
configs.residual_window = 60
configs.include_beta = True
configs.include_market = True
configs.residual_factors = ['market']
```

### Advanced (Multi-Factor)
```python
configs.use_residual_embedding = True
configs.residual_window = 60
configs.residual_factors = ['market', 'sector', 'quality']
configs.include_beta = True
configs.include_market = True
```

---

## 🧪 Verify It Works

### 1. Check Shapes
```python
from module.norm import ResidualPreprocessor
import torch

# Create test data
asset_ret = torch.randn(2, 100, 3)  # [Batch, Time, Assets]
market_ret = torch.randn(2, 100, 1) # [Batch, Time, Factors]

# Compute residuals
preprocessor = ResidualPreprocessor(window_size=20)
residuals, info = preprocessor(asset_ret, [market_ret])

# Check output
print(f"Input shape: {asset_ret.shape}")
print(f"Residual shape: {residuals.shape}")  # Should match
print(f"Beta shape: {info['betas'][0].shape}")  # Should match
assert residuals.shape == asset_ret.shape, "Shape mismatch!"
print("✓ Shapes correct!")
```

### 2. Check Computation
```python
from utils.residual_utils import validate_residual_output

# Validate that residuals are correctly computed
is_valid, max_error = validate_residual_output(
    residuals=residuals.numpy(),
    raw_returns=asset_ret.numpy(),
    betas=info['betas'][0].numpy(),
    market_returns=market_ret.numpy(),
    atol=1e-5
)

print(f"Residuals valid: {is_valid}")
print(f"Max error: {max_error}")
assert is_valid, "Residual computation error!"
print("✓ Computation correct!")
```

### 3. Check Training
```python
import torch.nn as nn
from model.unitsf import Model

# Create model
model = Model(configs)

# Forward pass
x = torch.randn(2, 96, 5)  # [Batch, Seq, Features]
y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
assert y.shape == (2, 24, 5), "Output shape incorrect!"

# Check gradients
loss = y.mean()
loss.backward()
print("✓ Training works!")
```

---

## 📈 Expected Improvements

### Metrics to Track
```python
# Out-of-sample R²
r2_residual = calculate_r2(residual_predictions, targets)
r2_raw = calculate_r2(raw_predictions, targets)
print(f"Improvement: {r2_residual - r2_raw:.4f}")

# Sharpe Ratio (if using for portfolio)
sharpe_residual = calculate_sharpe(residual_returns)
sharpe_raw = calculate_sharpe(raw_returns)
print(f"Sharpe Improvement: {sharpe_residual - sharpe_raw:.4f}")

# Information Ratio (relative to market)
ir_residual = calculate_ir(residual_predictions)
ir_raw = calculate_ir(raw_predictions)
print(f"IR Improvement: {ir_residual - ir_raw:.4f}")
```

### Success Indicators
- ✅ Better prediction of mean reversion
- ✅ Lower correlation with market
- ✅ Smoother out-of-sample performance
- ✅ Better performance in sideways/choppy markets

---

## 🚨 Common Issues

### Issue: "No decomposition in forward"
```
Error: x_enc = self.decompsition(x_enc)  # Returns list
       x_emb = [self.emb[i](x_enc[i]) for i in range(len(x_enc))]
       # Index error!
```
**Fix**: This is expected behavior. Residuals are preprocessed before decomposition.

### Issue: NaN values appear
```
residuals contain NaN values in first few timesteps
```
**Fix**: Use smaller window size
```python
preprocessor = ResidualPreprocessor(window_size=10)  # Smaller window
```

### Issue: Model training unstable
```
Loss diverging or oscillating wildly
```
**Fix**: Check data quality and try larger window
```python
# Data validation
from utils.residual_utils import forward_fill_nans
residuals = forward_fill_nans(residuals.numpy())
```

### Issue: No improvement over baseline
```
Residual model performs worse than raw model
```
**Fix**: Try different configurations
```python
# Experiment with window sizes
for window in [20, 60, 120, 252]:
    print(f"Testing window={window}")
    # Train and evaluate
    
# Add market returns to embedding
configs.include_market = True

# Try different factors
configs.residual_factors = ['market', 'sector']
```

---

## 💬 Debugging Tips

### Print Beta Values
```python
residuals, info = preprocessor(asset_ret, [market_ret])
betas = info['betas'][0]  # [B, T, D]

print(f"Average beta: {betas.mean().item():.4f}")
print(f"Std beta: {betas.std().item():.4f}")
print(f"Min beta: {betas.min().item():.4f}")
print(f"Max beta: {betas.max().item():.4f}")

# Beta should be between 0.5 and 2.0 for stocks
assert 0.1 < betas.mean() < 3.0, "Beta seems unreasonable"
```

### Check Correlation
```python
import numpy as np

residuals_np = residuals.numpy()
raw_np = asset_ret.numpy()
market_np = market_ret.numpy()

# Residuals should have lower correlation with market
corr_raw = np.corrcoef(raw_np.flatten(), market_np.flatten())[0, 1]
corr_residual = np.corrcoef(residuals_np.flatten(), market_np.flatten())[0, 1]

print(f"Correlation (raw vs market): {corr_raw:.4f}")
print(f"Correlation (residual vs market): {corr_residual:.4f}")
print(f"Reduction: {corr_raw - corr_residual:.4f}")

# Residuals should have much lower correlation
assert abs(corr_residual) < abs(corr_raw), "Residuals not properly decorrelated!"
```

### Visualize Beta Over Time
```python
import matplotlib.pyplot as plt

betas = info['betas'][0][0, :, 0].detach().numpy()  # [T] for first asset

plt.figure(figsize=(12, 5))
plt.plot(betas, label='Rolling Beta')
plt.axhline(y=betas.mean(), color='r', linestyle='--', label='Mean')
plt.xlabel('Time')
plt.ylabel('Beta')
plt.legend()
plt.title('Market Beta Over Time')
plt.show()

# Beta should be relatively smooth and stable
print(f"Beta volatility: {betas.std():.4f}")
```

---

## 📚 Next Steps

1. **Experiment with window sizes**: Try 20, 60, 120, 252 days
2. **Add more factors**: Sector ETFs, macro indices
3. **Analyze results**: Check correlation with market, Sharpe ratio
4. **Compare with baseline**: Quantify the improvement
5. **Optimize for your data**: Adjust configurations based on results

---

## 🎓 Learning Resources

### Files to Read
1. `RESIDUAL_EMBEDDING_README.md` - Full documentation
2. `IMPLEMENTATION_SUMMARY.md` - Technical details
3. `tests/test_residual_embedding.py` - Usage examples
4. `utils/residual_utils.py` - Utility functions

### Key Concepts
- **Log Returns**: `log(P_t / P_{t-1})`
- **Rolling Regression**: `beta = cov(X, Y) / var(Y)`
- **Causal Window**: Only uses past data (no lookahead)
- **Idiosyncratic Risk**: Risk specific to an asset (not market-wide)

---

## ✨ Tips & Tricks

### For Best Results
1. **Use 60 trading days** (3 months) as default window
2. **Fill NaN early values** with forward-fill
3. **Include beta** in embeddings for interpretability
4. **Start with single factor**, add others gradually
5. **Monitor correlation** with market

### Performance Tips
1. **Batch processing**: Process multiple assets together
2. **GPU acceleration**: ResidualPreprocessor runs on GPU
3. **Caching**: Pre-compute residuals if not changing
4. **Profiling**: Measure preprocessing overhead

### Experimentation
```python
# Grid search over configurations
configs_to_try = [
    {'window': 20, 'include_beta': True, 'factors': ['market']},
    {'window': 60, 'include_beta': True, 'factors': ['market']},
    {'window': 120, 'include_beta': True, 'factors': ['market']},
    {'window': 60, 'include_beta': False, 'factors': ['market']},
    {'window': 60, 'include_beta': True, 'factors': ['market', 'sector']},
]

best_config = None
best_score = -float('inf')

for config in configs_to_try:
    model = train_with_config(config)
    score = evaluate(model)
    print(f"Config {config}: Score {score:.4f}")
    if score > best_score:
        best_score = score
        best_config = config

print(f"\nBest config: {best_config}")
```

---

## 🎉 You're Ready!

Now you have everything you need to:
1. ✅ Understand residual embeddings
2. ✅ Implement them in TimeRecipe
3. ✅ Validate the implementation
4. ✅ Train models with residuals
5. ✅ Analyze and improve results

**Happy forecasting! 🚀**
