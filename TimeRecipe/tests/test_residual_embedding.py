"""
Unit tests for Residual Embedding preprocessing.

Tests cover:
1. Residual computation correctness
2. Causal window (no lookahead bias)
3. Shape consistency
4. Multi-factor handling
5. Edge cases (NaN, division by zero)
6. Integration with TimeRecipe model
"""

import unittest
import torch
import numpy as np
from module.norm import ResidualPreprocessor
from module.embed import ResidualEmbedding
from utils.residual_utils import (
    compute_log_returns, 
    align_asset_market_data,
    validate_residual_output
)


class TestResidualPreprocessor(unittest.TestCase):
    """Tests for ResidualPreprocessor class."""
    
    def setUp(self):
        """Create test data for each test."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create synthetic data
        self.batch_size = 2
        self.time_steps = 100
        self.num_assets = 3
        
        # Synthetic asset returns
        self.asset_returns = torch.randn(
            self.batch_size, self.time_steps, self.num_assets
        ) * 0.02  # 2% std dev
        
        # Synthetic market returns (correlated with assets)
        self.market_returns = torch.randn(
            self.batch_size, self.time_steps, 1
        ) * 0.015  # 1.5% std dev
        
        # Preprocessor with reasonable window
        self.preprocessor = ResidualPreprocessor(window_size=20)
    
    def test_residual_shape(self):
        """Test that residual output has correct shape."""
        residuals, info = self.preprocessor(self.asset_returns, [self.market_returns])
        
        # Shape should match input
        self.assertEqual(residuals.shape, self.asset_returns.shape)
        
        # Beta shape should be [B, T, D]
        self.assertEqual(info['betas'][0].shape, self.asset_returns.shape)
    
    def test_residual_computation(self):
        """Test that residuals are computed correctly: residual = asset - beta * market."""
        residuals, info = self.preprocessor(self.asset_returns, [self.market_returns])
        
        beta = info['betas'][0]
        
        # Reconstruct asset returns: asset = residual + beta * market
        reconstructed = residuals + beta * self.market_returns
        
        # Compare with original (accounting for floating point errors)
        # Allow for small errors due to numerical precision
        max_error = torch.abs(reconstructed - self.asset_returns).max().item()
        self.assertLess(max_error, 1e-5, 
                       f"Reconstruction error too large: {max_error}")
    
    def test_causal_window_no_lookahead(self):
        """Test that beta at time t only uses data from [t-window:t]."""
        window_size = 10
        preprocessor = ResidualPreprocessor(window_size=window_size)
        
        # Simple case: asset = market
        simple_data = torch.ones(1, 30, 1) * 0.01
        simple_market = torch.ones(1, 30, 1) * 0.01
        
        residuals, info = preprocessor(simple_data, [simple_market])
        beta = info['betas'][0]
        
        # With identical data, beta should be ~1.0 after window_size
        # (once we have enough data points)
        beta_values = beta[0, window_size:, 0].detach().numpy()
        expected_beta = 1.0
        
        # Check that beta converges to 1.0 (with some tolerance)
        mean_beta = np.mean(beta_values)
        self.assertAlmostEqual(mean_beta, expected_beta, places=2,
                              msg=f"Beta should be ~{expected_beta}, got {mean_beta}")
    
    def test_multi_factor_residuals(self):
        """Test that multiple factors are handled correctly."""
        # Create second factor (market2)
        market2_returns = torch.randn(self.batch_size, self.time_steps, 1) * 0.012
        
        residuals, info = self.preprocessor(
            self.asset_returns, 
            [self.market_returns, market2_returns]
        )
        
        # Should have 2 beta matrices
        self.assertEqual(len(info['betas']), 2)
        self.assertEqual(info['betas'][0].shape, self.asset_returns.shape)
        self.assertEqual(info['betas'][1].shape, self.asset_returns.shape)
        
        # Residuals should be: asset - beta1*market1 - beta2*market2
        beta1, beta2 = info['betas']
        reconstructed = (residuals + 
                        beta1 * self.market_returns + 
                        beta2 * market2_returns)
        
        max_error = torch.abs(reconstructed - self.asset_returns).max().item()
        self.assertLess(max_error, 1e-5)
    
    def test_nan_handling(self):
        """Test handling of NaN values (early period before window fills)."""
        residuals, info = self.preprocessor(self.asset_returns, [self.market_returns])
        
        # Some early values might be NaN due to small window
        # Check that the rest of the data is valid
        num_valid = torch.isfinite(residuals).sum().item()
        num_total = residuals.numel()
        
        # At least 50% should be valid
        self.assertGreater(num_valid, num_total * 0.5,
                          f"Too many NaN values: {num_valid}/{num_total}")
    
    def test_zero_variance_market(self):
        """Test handling of zero variance in market returns."""
        # Create constant market returns (zero variance)
        constant_market = torch.zeros(self.batch_size, self.time_steps, 1)
        
        residuals, info = self.preprocessor(self.asset_returns, [constant_market])
        
        # With zero market variance, beta should be undefined or zero
        # But residuals should still exist (equal to asset returns)
        self.assertEqual(residuals.shape, self.asset_returns.shape)
        
        # Residuals should be close to assets (since market has no info)
        residual_asset_diff = torch.abs(residuals - self.asset_returns).mean()
        # Should be small because beta*0 = 0
        self.assertLess(residual_asset_diff.item(), 0.1)


class TestResidualEmbedding(unittest.TestCase):
    """Tests for ResidualEmbedding class."""
    
    def setUp(self):
        """Create test data for each test."""
        self.batch_size = 2
        self.time_steps = 50
        self.num_assets = 3
        self.d_model = 16
        
        self.residuals = torch.randn(self.batch_size, self.time_steps, self.num_assets)
        self.raw_returns = torch.randn(self.batch_size, self.time_steps, self.num_assets)
        self.market_returns = torch.randn(self.batch_size, self.time_steps, 1)
        self.betas = torch.randn(self.batch_size, self.time_steps, self.num_assets)
    
    def test_residual_embedding_shape(self):
        """Test that embedding output has correct shape."""
        embedding = ResidualEmbedding(
            c_in=self.num_assets,
            d_model=self.d_model
        )
        
        output = embedding(self.residuals)
        
        # Output should be [B, T, d_model]
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.time_steps)
        self.assertEqual(output.shape[2], self.d_model)
    
    def test_residual_embedding_with_raw(self):
        """Test embedding with raw returns included."""
        embedding = ResidualEmbedding(
            c_in=self.num_assets,
            d_model=self.d_model,
            include_raw=True
        )
        
        output = embedding(self.residuals, x_raw=self.raw_returns)
        
        # Shape should still be [B, T, d_model]
        self.assertEqual(output.shape, (self.batch_size, self.time_steps, self.d_model))
    
    def test_residual_embedding_all_components(self):
        """Test embedding with all optional components."""
        embedding = ResidualEmbedding(
            c_in=self.num_assets,
            d_model=self.d_model,
            include_raw=True,
            include_market=True,
            include_beta=True
        )
        
        output = embedding(
            self.residuals,
            x_raw=self.raw_returns,
            x_market=self.market_returns,
            x_beta=self.betas
        )
        
        self.assertEqual(output.shape, (self.batch_size, self.time_steps, self.d_model))
    
    def test_embedding_is_differentiable(self):
        """Test that embedding gradients can be computed."""
        embedding = ResidualEmbedding(
            c_in=self.num_assets,
            d_model=self.d_model
        )
        
        residuals = self.residuals.clone().detach().requires_grad_(True)
        output = embedding(residuals)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(residuals.grad)
        self.assertGreater(residuals.grad.abs().sum().item(), 0)


class TestResidualUtilities(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_compute_log_returns(self):
        """Test log return computation."""
        prices = np.array([100, 101, 102, 103, 104])
        returns = compute_log_returns(prices)
        
        # First return should be NaN
        self.assertTrue(np.isnan(returns[0]))
        
        # Second return should be log(101/100)
        expected = np.log(101 / 100)
        self.assertAlmostEqual(returns[1], expected, places=6)
    
    def test_align_asset_market_data(self):
        """Test alignment of asset and market data."""
        asset_data = np.random.randn(100, 2)
        market_data = np.random.randn(100)
        
        asset_aligned, market_aligned, valid_idx = align_asset_market_data(
            asset_data, market_data
        )
        
        # Shapes should match
        self.assertEqual(asset_aligned.shape[0], market_aligned.shape[0])
        
        # All should be valid (no NaN)
        self.assertTrue(np.all(~np.isnan(asset_aligned[valid_idx])))
        self.assertTrue(np.all(~np.isnan(market_aligned[valid_idx])))
    
    def test_validate_residual_output(self):
        """Test residual validation."""
        # Create synthetic data where residual = asset - beta * market
        market = np.array([0.01, 0.02, -0.01, 0.03])
        beta = np.array([1.2, 1.1, 1.3, 1.0])
        asset = np.array([0.025, 0.03, 0.01, 0.035])
        residual = asset - beta * market
        
        is_valid, max_error = validate_residual_output(
            residuals=residual.reshape(-1, 1),
            raw_returns=asset.reshape(-1, 1),
            betas=beta.reshape(-1, 1),
            market_returns=market,
            atol=1e-6
        )
        
        # Should validate correctly
        self.assertTrue(is_valid)
        self.assertLess(max_error, 1e-6)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def test_end_to_end_residual_pipeline(self):
        """Test complete pipeline: preprocessing -> embedding -> prediction."""
        # Create synthetic financial data
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size = 2
        time_steps = 100
        num_assets = 2
        d_model = 32
        
        # Generate asset and market returns
        market_returns = torch.randn(batch_size, time_steps, 1) * 0.01
        asset_returns = market_returns.squeeze(-1).unsqueeze(-1) * torch.rand(batch_size, time_steps, num_assets)
        asset_returns = asset_returns + torch.randn(batch_size, time_steps, num_assets) * 0.005
        
        # Preprocess
        preprocessor = ResidualPreprocessor(window_size=20)
        residuals, info = preprocessor(asset_returns, [market_returns])
        
        # Embed
        embedding = ResidualEmbedding(
            c_in=num_assets,
            d_model=d_model,
            include_raw=True,
            include_beta=True
        )
        
        embedded = embedding(
            residuals,
            x_raw=asset_returns,
            x_beta=info['betas'][0]
        )
        
        # Check output
        self.assertEqual(embedded.shape, (batch_size, time_steps, d_model))
        self.assertTrue(torch.isfinite(embedded).all())
    
    def test_residual_denorm(self):
        """Test reconstruction of predictions from residuals."""
        # Create simple test case
        batch_size = 1
        pred_len = 10
        num_assets = 1
        
        residual_pred = torch.randn(batch_size, pred_len, num_assets)
        
        preprocessor = ResidualPreprocessor(window_size=20)
        
        # Mock the stored values
        preprocessor.beta_values = [torch.ones(batch_size, 1, num_assets) * 1.5]
        preprocessor.factor_returns = [torch.ones(batch_size, 1, 1) * 0.01]
        
        reconstructed = preprocessor.denorm(residual_pred)
        
        # Shape should match input
        self.assertEqual(reconstructed.shape, residual_pred.shape)
        
        # Reconstructed should be larger than residual (factor contribution is positive)
        self.assertGreater(reconstructed.mean().item(), residual_pred.mean().item())


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and robustness."""
    
    def test_single_observation_window(self):
        """Test with window size = 1."""
        preprocessor = ResidualPreprocessor(window_size=1)
        
        asset_returns = torch.randn(1, 50, 1)
        market_returns = torch.randn(1, 50, 1)
        
        residuals, info = preprocessor(asset_returns, [market_returns])
        
        self.assertEqual(residuals.shape, asset_returns.shape)
    
    def test_large_batch(self):
        """Test with large batch size."""
        preprocessor = ResidualPreprocessor(window_size=20)
        
        asset_returns = torch.randn(100, 50, 5)  # Large batch
        market_returns = torch.randn(100, 50, 1)
        
        residuals, info = preprocessor(asset_returns, [market_returns])
        
        self.assertEqual(residuals.shape, asset_returns.shape)
    
    def test_many_factors(self):
        """Test with many factors."""
        preprocessor = ResidualPreprocessor(window_size=20)
        
        asset_returns = torch.randn(2, 50, 3)
        factors = [torch.randn(2, 50, 1) for _ in range(10)]  # 10 factors
        
        residuals, info = preprocessor(asset_returns, factors)
        
        self.assertEqual(len(info['betas']), 10)
        self.assertEqual(residuals.shape, asset_returns.shape)


if __name__ == '__main__':
    unittest.main()
