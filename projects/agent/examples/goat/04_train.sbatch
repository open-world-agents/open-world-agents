#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=goat_%j.out
#SBATCH --error=goat_%j.err
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
# #SBATCH --nodelist=DGX-H100-2,DGX-H100-3,DGX-H100-4
#SBATCH --time=24:00:00  # Add an appropriate time limit

accelerate launch 04_train_agent.py \
  --dataset_path /mnt/raid11/datasets/owa/data/super-hexagon \
  --model_id /mnt/raid11/datasets/owa/checkpoints/SmolVLM2-256M-Video-Instruct-expanded \
  --output_dir /mnt/raid11/datasets/owa/checkpoints/SmolVLM2-256M-Video-Instruct-super_hexagon-1e4-repeat5-epoch20 \
  --repeat_n 5 \
  --num_epochs 20 \
  --learning_rate 1e-4 \
  --batch_size 8