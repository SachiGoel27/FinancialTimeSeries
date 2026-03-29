#!/bin/bash
# Test on SPY: Is Transformer really best for low-seasonality? (Spoiler: Paper says No)
for arch in trans mlp; do
  python -u run.py --model_id SPY_Arch_Test_$arch --model unitsf --data custom --data_path spy.csv \
    --features S --seq_len 96 --pred_len 96 --use_norm True --use_decomp True --emb_type none --ff_type $arch --target SPY
done