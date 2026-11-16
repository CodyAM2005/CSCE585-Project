#!/bin/bash
#SBATCH --job-name=vla-adapter
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=rc_general
#SBATCH --output=/work/%u/vla-adapter/EfficientVLA/hpc_logs/qwt_%j/accuracy.out
#SBATCH --error=/work/%u/vla-adapter/EfficientVLA/hpc_logs/qwt_%j/log.err

set -euo pipefail

source /home/apps/anaconda/3.12/etc/profile.d/conda.sh
conda activate vla-adapter

cd /work/codyam/vla-adapter/EfficientVLA
mkdir -p eval_logs

hostname
nvidia-smi

export PYTHONPATH=${PYTHONPATH:-}:/work/codyam/vla-adapter/EfficientVLA/LIBERO
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

python experiments/robot/libero/run_libero_eval.py \
   --use_proprio True \
   --num_images_in_input 2 \
   --use_film False \
   --pretrained_checkpoint /work/codyam/vla-adapter/EfficientVLA/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--spatial_layerdrop_half_dataset--40000_chkpt \
   --task_suite_name libero_spatial \
   --use_pro_version True \
   > eval_logs/Spatial_A100_Layer_Drop_40k_Dataset--chkpt.log