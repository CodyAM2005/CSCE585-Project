#!/bin/bash
#SBATCH --job-name=vla-adapter-twelve-layer-drop-finetune
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --account=rc_general
#SBATCH --output=/work/%u/vla-adapter/CSCE585-Project/EfficientVLATwelveLayerDrop/hpc_logs/qwt_%j/accuracy.out
#SBATCH --error=/work/%u/vla-adapter/CSCE585-Project/EfficientVLATwelveLayerDrop/hpc_logs/qwt_%j/log.err

set -euo pipefail

source /home/apps/anaconda/3.12/etc/profile.d/conda.sh
conda activate vla-adapter

cd /work/codyam/vla-adapter/CSCE585-Project/EfficientVLATwelveLayerDrop
mkdir -p logs

hostname
nvidia-smi

data_name=libero_spatial_no_noops
current_time=$(date +%Y%m%d_%H%M%S)

export PYTHONPATH=${PYTHONPATH:-}:/work/codyam/vla-adapter/CSCE585-Project/EfficientVLA/LIBERO
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

export KEEP_TASK_IDS="0,1,2,3,4,5,6,7,8,9"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir libero_rlds \
  --dataset_name $data_name \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 200000 \
  --max_steps 200005 \
  --save_freq 2500 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "vla-adapter" \
  --wandb_project "$data_name" \
  --run_id_note spatial_twelve_last_layerdrop_full_dataset \
  > logs/VLA-Adapter--libero_spatial_no_noops_A100_twelve_last_layer_drop_full_dataset--$current_time.log