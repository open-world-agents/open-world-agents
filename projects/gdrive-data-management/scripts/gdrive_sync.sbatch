#!/bin/bash
#SBATCH --job-name=gdrive_sync
#SBATCH --partition=a100
#SBATCH --nodelist=A100-Zebra
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/drive_sync_%j.out
#SBATCH --error=logs/drive_sync_%j.err

# Load Conda (Miniforge) environment
conda init
conda activate gdrive-data-management

# Confirm environment is working
echo "Using Python from: $(which python)"
python --version

# Create logs directory if it doesn't exist
mkdir -p logs

# Run your sync script
python gdrive_sync.py

echo "Done!"
