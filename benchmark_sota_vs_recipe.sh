#!/bin/bash

# --- BITCOIN (Short-Term Task) ---
# SOTA Baseline: Standard Transformer with Token Embedding
python -u run.py --model_id BTC_SOTA_Baseline --model unitsf --data custom --data_path bitcoin.csv \
  --features S --seq_len 60 --pred_len 15 --use_norm False --use_decomp False --emb_type token --ff_type trans --target Open

# TIMERECIPE: RNN + Instance Norm + No Embedding (Optimized for Low HL-Ratio & High Trend)
python -u run.py --model_id BTC_Optimized_Recipe --model unitsf --data custom --data_path bitcoin.csv \
  --features S --seq_len 60 --pred_len 15 --use_norm True --use_decomp False --emb_type none --ff_type rnn --target Open

# --- SPY (Long-Term Task) ---
# SOTA Baseline: Standard Transformer
python -u run.py --model_id SPY_SOTA_Baseline --model unitsf --data custom --data_path spy.csv \
  --features S --seq_len 96 --pred_len 96 --use_norm False --use_decomp False --emb_type token --ff_type trans --target SPY

# TIMERECIPE: MLP + Series Decomp + No Embedding (Optimized for High Trend & Low Seasonality)
python -u run.py --model_id SPY_Optimized_Recipe --model unitsf --data custom --data_path spy.csv \
  --features S --seq_len 96 --pred_len 96 --use_norm True --use_decomp True --emb_type none --ff_type mlp --target SPY