#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=goat_%j.out
#SBATCH --error=goat_%j.err
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --nodelist=DGX-H100-12
#SBATCH --time=24:00:00  # Add an appropriate time limit

accelerate launch 04_train_agent.py \
  --dataset_path  /mnt/home/haebin/data/owa_dataset_vpt \
  --model_id /mnt/home/haebin/data/checkpoint/SmolVLM2-256M-Video-Instruct-expanded \
  --output_dir /mnt/home/haebin/data/checkpoint/vpt-mcap \
  --repeat_n 1 \
  --num_epochs 5 \
  --learning_rate 1e-4 \
  --batch_size 16