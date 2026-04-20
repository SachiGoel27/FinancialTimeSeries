# Test 1: Global Attention (Optimized for sudden volatility shocks/spikes)
python run.py \
  --is_training 1 \
  --data custom \
  --data_path vix_futures.csv \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --ff_type trans \
  --target VIX \
  --des 'Exp_Transformer_VIX'

# Baseline: Sequential Decay (Optimized for slow mean reversion)
python run.py \
  --is_training 1 \
  --data custom \
  --data_path vix_futures.csv \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --ff_type rnn \
  --target VIX \
  --des 'Exp_RNN_VIX'