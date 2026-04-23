#!/bin/bash
# ============================================================
# Financial Time Series Experiments - PACE SLURM Runner
# ============================================================
# Usage: bash run_financial_experiments.sh
# This script submits all experiments as individual SLURM jobs.
# Results are saved to ./results/ and logs to ./logs/
# ============================================================

mkdir -p logs results

# ============================================================
# CONFIGURATION — edit these paths for your PACE environment
# ============================================================
REPO_DIR="$HOME/FinancialTimeSeries/TimeRecipe"
DATA_DIR="$REPO_DIR/../data"
CONDA_ENV="timeseries"   # change to your conda env name

# Fixed hyperparameters
SEQ_LEN=252              # 1 year lookback (business days)
LABEL_LEN=126
PRED_LEN=5               # 1 week ahead forecast
BATCH_SIZE=32
EPOCHS=20
LR=1e-4
D_MODEL=256
D_FF=512
E_LAYERS=2
PATIENCE=5

# ============================================================
# DATASETS
# ============================================================
declare -A DATASETS
DATASETS["vix"]="vix_futures_clean.csv:VIX"
DATASETS["credit"]="credit_spreads_clean.csv:OAS_bps"
DATASETS["euro_hy"]="euro_hy_oas_clean.csv:euro_hy_oas"
DATASETS["mortgage"]="mortgage30_clean.csv:mortgage_rate_30y"

# ============================================================
# SUBMIT FUNCTION
# ============================================================
submit_job() {
    local job_name=$1
    local data_key=$2
    local ff_type=$3
    local emb_type=$4
    local use_frac_diff=$5
    local use_vol_norm=$6
    local use_fourier=$7
    local use_residual=$8

    # Parse dataset info
    local data_info=${DATASETS[$data_key]}
    local data_path="${data_info%%:*}"
    local target="${data_info##*:}"

    local full_job_name="${job_name}_${data_key}_${ff_type}_${emb_type}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${full_job_name}
#SBATCH --output=logs/${full_job_name}_%j.out
#SBATCH --error=logs/${full_job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load environment
module load anaconda3
conda activate ${CONDA_ENV}

cd ${REPO_DIR}

python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ${full_job_name} \
    --model unitsf \
    --data custom \
    --root_path ${DATA_DIR} \
    --data_path ${data_path} \
    --target ${target} \
    --features S \
    --freq b \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --e_layers ${E_LAYERS} \
    --n_heads 4 \
    --factor 1 \
    --moving_avg 25 \
    --patch_len 16 \
    --dropout 0.1 \
    --embed timeF \
    --activation gelu \
    --fusion temporal \
    --emb_type ${emb_type} \
    --ff_type ${ff_type} \
    --use_norm True \
    --use_decomp False \
    --use_frac_diff ${use_frac_diff} \
    --use_vol_norm ${use_vol_norm} \
    --use_fourier ${use_fourier} \
    --use_residual_embedding ${use_residual} \
    --nbeats_blocks 3 \
    --num_regimes 4 \
    --train_epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --patience ${PATIENCE} \
    --loss MSE \
    --lradj type3 \
    --num_workers 4 \
    --des ${full_job_name}
EOF

    echo "Submitted: ${full_job_name}"
}

# ============================================================
# PHASE 1 — Discovery
# All 4 datasets x 5 ff_types x 2 embeddings = 40 jobs
# Fixed: use_frac_diff=True, use_vol_norm=True, use_fourier=False
# ============================================================
echo "=== Submitting Phase 1: Discovery ==="

FF_TYPES=("mlp" "grn" "nbeats" "regime" "magn")
EMB_TYPES=("token" "residual")

for dataset in "${!DATASETS[@]}"; do
    for ff in "${FF_TYPES[@]}"; do
        for emb in "${EMB_TYPES[@]}"; do
            submit_job "p1" "$dataset" "$ff" "$emb" "True" "True" "False" "False"
        done
    done
done

# ============================================================
# PHASE 2 — Ablation on best modules
# Fix ff_type=grn, emb_type=residual, sweep preprocessing flags
# 4 datasets x 8 preprocessing combos = 32 jobs
# ============================================================
echo "=== Submitting Phase 2: Ablation ==="

for dataset in "${!DATASETS[@]}"; do
    # Baseline - no financial preprocessing
    submit_job "p2_base"     "$dataset" "grn" "residual" "False" "False" "False" "False"
    # Single flags
    submit_job "p2_fd"       "$dataset" "grn" "residual" "True"  "False" "False" "False"
    submit_job "p2_vn"       "$dataset" "grn" "residual" "False" "True"  "False" "False"
    submit_job "p2_fou"      "$dataset" "grn" "residual" "False" "False" "True"  "False"
    submit_job "p2_res"      "$dataset" "grn" "residual" "False" "False" "False" "True"
    # Combinations
    submit_job "p2_fd_vn"    "$dataset" "grn" "residual" "True"  "True"  "False" "False"
    submit_job "p2_fd_fou"   "$dataset" "grn" "residual" "True"  "False" "True"  "False"
    submit_job "p2_all"      "$dataset" "grn" "residual" "True"  "True"  "True"  "True"
done

echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: ${REPO_DIR}/results/"
