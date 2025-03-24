#!/bin/bash
#SBATCH --job-name=super_hexagon
#SBATCH --output=super_hexagon_%j.out
#SBATCH --error=super_hexagon_%j.err
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --nodelist=DGX-H100-2,DGX-H100-3,DGX-H100-4
#SBATCH --time=24:00:00  # Add an appropriate time limit

accelerate launch scripts/03_train_agent.py \
  --query_path super_hexagon.jsonl \
  --model_id /mnt/raid11/datasets/owa/SmolVLM2-500M-Video-Instruct-expanded \
  --output_dir /mnt/raid11/datasets/owa/checkpoints/super_hexagon/repeat_10-epoch_20-lr1e-4-seminit_arrow \
  --repeat_n 10 \
  --num_epochs 20 \
  --learning_rate 1e-4 \
  --batch_size 8