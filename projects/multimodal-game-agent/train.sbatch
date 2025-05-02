#!/bin/bash
#SBATCH --job-name=super_hexagon
#SBATCH --output=super_hexagon_%j.out
#SBATCH --error=super_hexagon_%j.err
#SBATCH --gpus=6
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
# #SBATCH --nodelist=DGX-H100-2,DGX-H100-3,DGX-H100-4
#SBATCH --time=24:00:00  # Add an appropriate time limit

accelerate launch scripts/03_train_agent.py \
  --query_path /mnt/cephfs/worv/datasets/owa/super-hexagon/hexagon.jsonl \
  --eval_query_path /mnt/cephfs/worv/datasets/owa/super-hexagon-30s/hexagon-val.jsonl\
  --model_id /mnt/cephfs/worv/checkpoints/owa/base-model \
  --output_dir /mnt/cephfs/worv/checkpoints/owa/super-hexagon/SmolVLM2-512M-Video-Instruct-1e4-norotaug-repeat5-epoch20 \
  --repeat_n 5 \
  --num_epochs 20 \
  --learning_rate 1e-4 \
  --batch_size 8