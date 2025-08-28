#!/bin/bash
#SBATCH --job-name=evaluate-owa
#SBATCH --gpus=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1TB
#SBATCH --time=7-00:00:00
#SBATCH --exclude=DGX-H100-1
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

# Load any necessary modules or activate environment here
# module load cuda/11.8
# source activate your_env
. ~/miniforge3/bin/activate owa

# $ ls /mnt/raid12/datasets/owa_game_dataset_filtered_448_fsl-100ms/                                                                                   
# apex-legends  counter-strike-2  euro-truck-simulator-2  grand-theft-auto-v  rainbow-six

GAME_NAME="rainbow-six"

# Run the evaluation
accelerate launch --config_file=accelerate_configs/multi_gpu.yaml evaluate.py --config evaluate_config.yaml \
    --model_name_or_path /mnt/harbor/users/jyjung/checkpoints/gidm/InternVL3-1B-hf_100ms-batch128-lr2e5-GIDM/checkpoint-8910 \
    --output-dir /mnt/harbor/projects/owa/evals/InternVL3-1B-hf_100ms-batch128-lr2e5-GIDM/checkpoint-8910/$GAME_NAME \
    --dataset-paths /mnt/raid12/datasets/owa_game_dataset_filtered_448_fsl-100ms/$GAME_NAME
