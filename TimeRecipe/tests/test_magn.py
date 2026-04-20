import unittest
import torch
import torch.nn as nn
from module.architecture import MAGN


class TestMAGN(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 96
        self.d_model = 64
        self.d_ff = 256
        self.model_in_size = 96
        
    def test_single_modality(self):
        modalities = {'price': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        
        price_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn({'price': price_emb}, target_emb)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
    
    def test_multi_modality(self):
        modalities = {'price': self.d_model, 'macro': self.d_model, 'news': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        
        embeddings = {
            'price': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'macro': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'news': torch.randn(self.batch_size, self.seq_len, self.d_model)
        }
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn(embeddings, target_emb)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
    
    def test_fusion_weights_sum_to_one(self):
        modalities = {'price': self.d_model, 'macro': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        
        embeddings = {
            'price': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'macro': torch.randn(self.batch_size, self.seq_len, self.d_model)
        }
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn(embeddings, target_emb)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
    
    def test_without_target_conditioning(self):
        modalities = {'price': self.d_model, 'macro': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size, 
                   use_target_conditioning=False)
        
        embeddings = {
            'price': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'macro': torch.randn(self.batch_size, self.seq_len, self.d_model)
        }
        
        output = magn(embeddings, None)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
    
    def test_without_feature_gating(self):
        modalities = {'price': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size,
                   use_feature_gating=False)
        
        price_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn({'price': price_emb}, target_emb)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
        self.assertFalse(hasattr(magn, 'feature_gates') or magn.use_feature_gating)
    
    def test_shared_mlp(self):
        modalities = {'price': self.d_model, 'macro': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size,
                   shared_mlp=True)
        
        embeddings = {
            'price': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'macro': torch.randn(self.batch_size, self.seq_len, self.d_model)
        }
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn(embeddings, target_emb)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
        self.assertTrue(hasattr(magn, 'mlp'))
    
    def test_gradient_flow(self):
        modalities = {'price': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        
        price_emb = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn({'price': price_emb}, target_emb)
        loss = output.mean()
        loss.backward()
        
        self.assertIsNotNone(price_emb.grad)
        self.assertGreater(price_emb.grad.abs().sum().item(), 0)
    
    def test_different_modality_inputs(self):
        modalities = {'price': self.d_model, 'vol': self.d_model, 'sentiment': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        
        embeddings = {
            'price': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'vol': torch.randn(self.batch_size, self.seq_len, self.d_model),
            'sentiment': torch.randn(self.batch_size, self.seq_len, self.d_model)
        }
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = magn(embeddings, target_emb)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.model_in_size))
    
    def test_batch_independence(self):
        modalities = {'price': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        
        torch.manual_seed(123)
        price_emb_1 = torch.randn(2, self.seq_len, self.d_model)
        target_emb_1 = torch.randn(2, self.seq_len, self.d_model)
        
        torch.manual_seed(456)
        price_emb_2 = torch.randn(2, self.seq_len, self.d_model)
        target_emb_2 = torch.randn(2, self.seq_len, self.d_model)
        
        output_1 = magn({'price': price_emb_1}, target_emb_1)
        output_2 = magn({'price': price_emb_2}, target_emb_2)
        
        self.assertFalse(torch.allclose(output_1, output_2))
    
    def test_deterministic_forward(self):
        modalities = {'price': self.d_model}
        magn = MAGN(modalities, self.d_model, self.d_ff, self.model_in_size)
        magn.eval()
        
        price_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        target_emb = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        with torch.no_grad():
            output_1 = magn({'price': price_emb}, target_emb)
            output_2 = magn({'price': price_emb}, target_emb)
        
        self.assertTrue(torch.allclose(output_1, output_2))


class TestMAGNIntegration(unittest.TestCase):
    
    def test_magn_with_multiple_timesteps(self):
        modalities = {'price': 64, 'macro': 64}
        magn = MAGN(modalities, 64, 256, 96)
        
        embeddings = {
            'price': torch.randn(8, 96, 64),
            'macro': torch.randn(8, 96, 64)
        }
        target = torch.randn(8, 96, 64)
        
        output = magn(embeddings, target)
        
        self.assertEqual(output.shape, (8, 96, 96))
    
    def test_magn_list_modalities(self):
        modalities = [64, 64, 64]
        magn = MAGN(modalities, 64, 256, 96)
        
        embeddings = {
            'modality_0': torch.randn(4, 96, 64),
            'modality_1': torch.randn(4, 96, 64),
            'modality_2': torch.randn(4, 96, 64)
        }
        target = torch.randn(4, 96, 64)
        
        output = magn(embeddings, target)
        
        self.assertEqual(output.shape, (4, 96, 96))
    
    def test_magn_sequential_pass(self):
        modalities = {'price': 64, 'macro': 64}
        magn = MAGN(modalities, 64, 256, 96)
        
        embeddings = {
            'price': torch.randn(4, 96, 64),
            'macro': torch.randn(4, 96, 64)
        }
        target = torch.randn(4, 96, 64)
        
        output_1 = magn(embeddings, target)
        output_2 = magn(embeddings, target)
        
        self.assertEqual(output_1.shape, output_2.shape)


if __name__ == '__main__':
    unittest.main()
