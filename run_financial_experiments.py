#!/usr/bin/env python3
"""
Financial Time Series Experiments - PACE SLURM Runner
======================================================
Usage: python run_financial_experiments.py
       python run_financial_experiments.py --phase 1
       python run_financial_experiments.py --phase 2
       python run_financial_experiments.py --dry_run

Submits experiments as individual SLURM jobs.
Results saved to ./results/, logs to ./logs/
"""

import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION — edit these for your PACE environment
# ============================================================
HOME = Path.home()
REPO_DIR = HOME / "FinancialTimeSeries" / "TimeRecipe"
DATA_DIR = REPO_DIR / ".."          # data files are in repo root
CONDA_ENV = "timeseries"            # your conda env name
PARTITION = "gpu"                   # check with: sinfo
TIME_LIMIT = "02:00:00"
MEM = "16G"
CPUS = 4

# Fixed hyperparameters
HPARAMS = {
    "seq_len": 252,       # 1 year lookback
    "label_len": 126,
    "pred_len": 5,        # 1 week ahead
    "batch_size": 32,
    "train_epochs": 20,
    "learning_rate": 1e-4,
    "d_model": 256,
    "d_ff": 512,
    "e_layers": 2,
    "n_heads": 4,
    "patience": 5,
    "dropout": 0.1,
    "moving_avg": 25,
    "patch_len": 16,
    "factor": 1,
    "nbeats_blocks": 3,
    "num_regimes": 4,
}

# ============================================================
# DATASETS — {key: (filename, target_column)}
# ============================================================
DATASETS = {
    "vix":      ("vix_futures_clean.csv",    "VIX"),
    "credit":   ("credit_spreads_clean.csv", "OAS_bps"),
    "euro_hy":  ("euro_hy_oas_clean.csv",    "euro_hy_oas"),
    "mortgage": ("mortgage30_clean.csv",     "mortgage_rate_30y"),
}

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
def get_phase1_experiments():
    """Phase 1: Discovery — all ff_types x embeddings x datasets"""
    experiments = []
    ff_types = ["mlp", "grn", "nbeats", "regime", "magn"]
    emb_types = ["token", "residual"]

    for dataset in DATASETS:
        for ff in ff_types:
            for emb in emb_types:
                experiments.append({
                    "phase": "p1",
                    "dataset": dataset,
                    "ff_type": ff,
                    "emb_type": emb,
                    "use_frac_diff": True,
                    "use_vol_norm": True,
                    "use_fourier": False,
                    "use_residual_embedding": False,
                })
    return experiments


def get_phase2_experiments():
    """Phase 2: Ablation — sweep preprocessing flags with best modules"""
    experiments = []
    preprocessing_combos = [
        # name suffix,       frac_diff, vol_norm, fourier, residual
        ("base",             False,     False,    False,   False),
        ("fd",               True,      False,    False,   False),
        ("vn",               False,     True,     False,   False),
        ("fou",              False,     False,    True,    False),
        ("res",              False,     False,    False,   True),
        ("fd_vn",            True,      True,     False,   False),
        ("fd_fou",           True,      False,    True,    False),
        ("all",              True,      True,     True,    True),
    ]

    for dataset in DATASETS:
        for suffix, fd, vn, fou, res in preprocessing_combos:
            experiments.append({
                "phase": f"p2_{suffix}",
                "dataset": dataset,
                "ff_type": "grn",
                "emb_type": "residual",
                "use_frac_diff": fd,
                "use_vol_norm": vn,
                "use_fourier": fou,
                "use_residual_embedding": res,
            })
    return experiments


# ============================================================
# JOB SUBMISSION
# ============================================================
def build_job_name(exp):
    return f"{exp['phase']}_{exp['dataset']}_{exp['ff_type']}_{exp['emb_type']}"


def build_slurm_script(exp):
    job_name = build_job_name(exp)
    data_file, target = DATASETS[exp["dataset"]]

    def b(val):
        return "True" if val else "False"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={REPO_DIR}/../logs/{job_name}_%j.out
#SBATCH --error={REPO_DIR}/../logs/{job_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}
#SBATCH --time={TIME_LIMIT}
#SBATCH --gres=gpu:1
#SBATCH --partition={PARTITION}

# Load environment
module load anaconda3
conda activate {CONDA_ENV}

cd {REPO_DIR}

python run.py \\
    --task_name long_term_forecast \\
    --is_training 1 \\
    --model_id {job_name} \\
    --model unitsf \\
    --data custom \\
    --root_path {DATA_DIR} \\
    --data_path {data_file} \\
    --target {target} \\
    --features S \\
    --freq b \\
    --seq_len {HPARAMS['seq_len']} \\
    --label_len {HPARAMS['label_len']} \\
    --pred_len {HPARAMS['pred_len']} \\
    --enc_in 1 \\
    --dec_in 1 \\
    --c_out 1 \\
    --d_model {HPARAMS['d_model']} \\
    --d_ff {HPARAMS['d_ff']} \\
    --e_layers {HPARAMS['e_layers']} \\
    --n_heads {HPARAMS['n_heads']} \\
    --factor {HPARAMS['factor']} \\
    --moving_avg {HPARAMS['moving_avg']} \\
    --patch_len {HPARAMS['patch_len']} \\
    --dropout {HPARAMS['dropout']} \\
    --embed timeF \\
    --activation gelu \\
    --fusion temporal \\
    --emb_type {exp['emb_type']} \\
    --ff_type {exp['ff_type']} \\
    --use_norm True \\
    --use_decomp False \\
    --use_frac_diff {b(exp['use_frac_diff'])} \\
    --use_vol_norm {b(exp['use_vol_norm'])} \\
    --use_fourier {b(exp['use_fourier'])} \\
    --use_residual_embedding {b(exp['use_residual_embedding'])} \\
    --nbeats_blocks {HPARAMS['nbeats_blocks']} \\
    --num_regimes {HPARAMS['num_regimes']} \\
    --train_epochs {HPARAMS['train_epochs']} \\
    --batch_size {HPARAMS['batch_size']} \\
    --learning_rate {HPARAMS['learning_rate']} \\
    --patience {HPARAMS['patience']} \\
    --loss MSE \\
    --lradj type3 \\
    --num_workers {CPUS} \\
    --des {job_name}
"""
    return script


def submit_job(exp, dry_run=False):
    job_name = build_job_name(exp)
    script = build_slurm_script(exp)

    if dry_run:
        print(f"[DRY RUN] Would submit: {job_name}")
        return None

    result = subprocess.run(
        ["sbatch"],
        input=script,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted {job_name} -> job {job_id}")
        return job_id
    else:
        print(f"FAILED {job_name}: {result.stderr.strip()}")
        return None


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PACE experiment runner")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 or phase 2 (default: both)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print jobs without submitting")
    args = parser.parse_args()

    # Create directories
    log_dir = REPO_DIR / ".." / "logs"
    results_dir = REPO_DIR / ".." / "results"
    log_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Build experiment list
    experiments = []
    if args.phase is None or args.phase == 1:
        p1 = get_phase1_experiments()
        experiments.extend(p1)
        print(f"Phase 1: {len(p1)} experiments")

    if args.phase is None or args.phase == 2:
        p2 = get_phase2_experiments()
        experiments.extend(p2)
        print(f"Phase 2: {len(p2)} experiments")

    print(f"Total: {len(experiments)} experiments")
    print()

    # Submit
    submitted = []
    failed = []

    for exp in experiments:
        job_id = submit_job(exp, dry_run=args.dry_run)
        if job_id:
            submitted.append({"job_id": job_id, "exp": exp})
        elif not args.dry_run:
            failed.append(exp)

    # Save submission log
    if not args.dry_run:
        log_path = results_dir / f"submission_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump({"submitted": submitted, "failed": failed}, f, indent=2, default=str)
        print(f"\nSubmission log saved to {log_path}")
        print(f"Submitted: {len(submitted)} | Failed: {len(failed)}")
        print(f"\nMonitor with: squeue -u $USER")
        print(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
