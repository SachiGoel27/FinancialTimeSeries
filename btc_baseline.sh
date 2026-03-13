#!/bin/bash

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path bitcoin.csv \
  --model_id BTC_Baseline \
  --model unitsf \
  --data custom \
  --features S \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 15 \
  --use_norm False \
  --use_decomp False \
  --emb_type token \
  --ff_type trans \
  --train_epochs 5 \
  --target Open \
  --freq t