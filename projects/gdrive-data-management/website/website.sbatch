#!/bin/bash
#SBATCH --job-name=website
#SBATCH --partition=a100
#SBATCH --nodelist=A100-Zebra
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=logs/website_%j.out
#SBATCH --error=logs/website_%j.err

# Load Conda (Miniforge) environment
conda init
conda activate gdrive-data-management

# Start the web server
python app.py

echo "Done!"
