# Test 1: With Instance Normalization (Handling regime shifts)
python run.py \
  --is_training 1 \
  --data custom \
  --data_path credit_spreads.csv \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --use_norm True \
  --target "OAS (bps)" \
  --des 'Exp_Norm_Credit'

# Baseline: Without Normalization (Susceptible to volatility clustering)
python run.py \
  --is_training 1 \
  --data custom \
  --data_path credit_spreads.csv \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --use_norm False \
  --target "OAS (bps)" \
  --des 'Exp_NoNorm_Credit'