#!/bin/bash
# Test on BTC: Compare performance with and without Normalization
# High Shifting Data -> Recipe says use_norm=True
for norm in True False; do
  python -u run.py --model_id BTC_Norm_Test_$norm --model unitsf --data custom --data_path bitcoin.csv \
    --features S --seq_len 60 --pred_len 15 --use_norm $norm --emb_type none --ff_type rnn --target Open
done