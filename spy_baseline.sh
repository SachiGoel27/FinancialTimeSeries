#!/bin/bash

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path spy.csv \
  --model_id SPY_Baseline \
  --model unitsf \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --use_norm False \
  --use_decomp False \
  --emb_type token \
  --ff_type trans \
  --train_epochs 5 \
  --target SPY \
  --freq d