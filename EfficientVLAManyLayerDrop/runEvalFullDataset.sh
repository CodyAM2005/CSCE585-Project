#!/bin/bash
#SBATCH --job-name=vla-adapter
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=rc_general
#SBATCH --output=/work/%u/vla-adapter/CSCE585-Project/EfficientVLAManyLayerDrop/hpc_logs/qwt_%j/accuracy.out
#SBATCH --error=/work/%u/vla-adapter/CSCE585-Project/EfficientVLAManyLayerDrop/hpc_logs/qwt_%j/log.err

set -euo pipefail

source /home/apps/anaconda/3.12/etc/profile.d/conda.sh
conda activate vla-adapter

cd /work/codyam/vla-adapter/CSCE585-Project/EfficientVLAManyLayerDrop
mkdir -p eval_logs

hostname
nvidia-smi

export PYTHONPATH=${PYTHONPATH:-}:/work/codyam/vla-adapter/CSCE585-Project/EfficientVLA/LIBERO
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

export KEEP_TASK_IDS="0,1,2,3,4,5,6,7,8,9"

python experiments/robot/libero/run_libero_eval.py \
   --use_proprio True \
   --num_images_in_input 2 \
   --use_film False \
   --pretrained_checkpoint /work/codyam/vla-adapter/CSCE585-Project/EfficientVLA/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops_A100----40000_chkpt \
   --task_suite_name libero_spatial \
   --task_ids "$KEEP_TASK_IDS" \
   --use_pro_version True \
   > eval_logs/Spatial_A100_many_layer_drop_full_dataset_TEST_RUN--chkpt.log 