#!/bin/bash
#SBATCH --job-name=vla-adapter-end-removal-latency
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=rc_general
#SBATCH --output=/work/%u/vla-adapter/CSCE585-Project/latency_bench/end_removal_%j.out
#SBATCH --error=/work/%u/vla-adapter/CSCE585-Project/latency_bench/end_removal_%j.err

# Latency benchmark for END removal (drop 0 / 1 / 3 final LLM layers).
# All three configs run back-to-back in ONE job so they share the same GPU/node.
# Read-only w.r.t. the experiment repos and checkpoints (see bench_end_removal.py docstring).

set -euo pipefail

source /home/apps/anaconda/3.12/etc/profile.d/conda.sh
conda activate vla-adapter

ROOT=/work/codyam/vla-adapter/CSCE585-Project
BENCH=$ROOT/latency_bench
OUT=$BENCH/results/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

hostname
nvidia-smi

export PYTHONDONTWRITEBYTECODE=1   # don't drop __pycache__ into repos that running jobs are using
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# --- Checkpoints: VERIFY these before submitting. Each must have been trained with the matching repo
# --- (action-head block count = 24 - k is checked at runtime). Weights don't affect latency, but the
# --- action head shape and hidden-state slicing must match.
CKPT_K0=$ROOT/EfficientVLA/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops_A100----40000_chkpt
CKPT_K1=$ROOT/EfficientVLALayerDrop/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--spatial_layerdrop_half_dataset--40000_chkpt
CKPT_K3=$ROOT/EfficientVLAManyLayerDrop/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--spatial_many_layerdrop_half_dataset--40000_chkpt

COMMON="--task_suite_name libero_spatial --num_warmup 20 --num_iters 100 --repeats 3"

run() {  # $1=k  $2=repo  $3=checkpoint
  cd "$BENCH"   # never cd into the experiment repos
  python bench_end_removal.py \
    --num_end_layers_drop "$1" \
    --repo_dir "$ROOT/$2" \
    --pretrained_checkpoint "$3" \
    --output_json "$OUT/end_removal_k$1.json" \
    $COMMON
}

run 0 EfficientVLA              "$CKPT_K0"
run 1 EfficientVLALayerDrop     "$CKPT_K1"
run 3 EfficientVLAManyLayerDrop "$CKPT_K3"

python "$BENCH/summarize_end_removal.py" --results_dir "$OUT"
