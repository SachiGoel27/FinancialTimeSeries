# Test 1: Long Sequence (Capturing slow build-up and long memory)
python main.py \
  --is_training 1 \
  --data credit_spreads \
  --features M \
  --seq_len 336 \
  --pred_len 24 \
  --des 'Exp_LongSeq_Credit'

# Baseline: Standard/Short Sequence (Typical equity look-back)
python main.py \
  --is_training 1 \
  --data credit_spreads \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --des 'Exp_ShortSeq_Credit'