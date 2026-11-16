#!/bin/sh
#SBATCH --job-name=vla-adapter-layer-drop-finetune
#SBATCH --output /work/%u/vla-adapter/EfficientVLA/hpc_logs/qwt_%j/accuray.out
#SBATCH --error /work/%u/vla-adapter/EfficientVLA/hpc_logs/qwt_%j/log.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -p gpu-A100
#SBATCH --time=48:00:00                   # total run time limit (HH:MM:SS)
#SBATCH --account rc_general              # add your PI's account name

#gpu-H200,gpu-A100,gpu-A100-3g.20gb

source /home/apps/anaconda/3.12/etc/profile.d/conda.sh

user=$(whoami)

pwd
hostname
conda init
conda activate vla-adapter

data_name=libero_spatial_no_noops
export PYTHONPATH=$PYTHONPATH:/work/codyam/vla-adapter/EfficientVLA/LIBERO    
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
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
--run_id_note VLA-Adapter--libero_spatial_no_noops_A100_layer_drop--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops_A100_layer_drop--$current_time.log