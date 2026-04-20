# Test 1: Long Sequence (Capturing slow build-up and long memory)
python run.py \
  --is_training 1 \
  --data custom \
  --data_path credit_spreads.csv \
  --features M \
  --seq_len 336 \
  --pred_len 24 \
  --target "OAS (bps)" \
  --des 'Exp_LongSeq_Credit'

# Baseline: Standard/Short Sequence (Typical equity look-back)
python run.py \
  --is_training 1 \
  --data custom \
  --data_path credit_spreads.csv \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --target "OAS (bps)" \
  --des 'Exp_ShortSeq_Credit'