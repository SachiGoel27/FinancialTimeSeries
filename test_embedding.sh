#!/bin/bash
for emb in token none; do
  python -u run.py --model_id BTC_Emb_Test_$emb --model unitsf --data custom --data_path bitcoin.csv \
    --features S --seq_len 60 --pred_len 15 --use_norm True --emb_type $emb --ff_type rnn --target Open
done