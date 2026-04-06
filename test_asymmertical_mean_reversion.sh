# Test 1: Global Attention (Optimized for sudden volatility shocks/spikes)
python main.py \
  --is_training 1 \
  --data vix_futures \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --ff_type transformer \
  --des 'Exp_Transformer_VIX'

# Baseline: Sequential Decay (Optimized for slow mean reversion)
python main.py \
  --is_training 1 \
  --data vix_futures \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --ff_type rnn \
  --des 'Exp_RNN_VIX'