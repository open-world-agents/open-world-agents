#!/bin/bash
#SBATCH --job-name=filter_datasets
#SBATCH --partition=a100
#SBATCH --nodelist=A100-Zebra
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=logs/filter_datasets_%j.out
#SBATCH --error=logs/filter_datasets_%j.err

# Load Conda (Miniforge) environment
conda init
conda activate gdrive-data-management

# Confirm environment is working
echo "Using Python from: $(which python)"
python --version

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
SOURCE_ROOT="/mnt/raid12/datasets/owa_game_dataset"
DEST_ROOT="/mnt/raid12/datasets/owa_game_dataset_filtered"

# TODO: Make this configurable
MIN_VERSION="0.5.5"
TARGET_VERSION="0.5.5"

# Run dataset filtering and migration script
echo "Starting dataset filtering and migration..."
echo "Source: $SOURCE_ROOT"
echo "Destination: $DEST_ROOT"
echo "Minimum version: $MIN_VERSION"
echo "Target version: $TARGET_VERSION"
echo "Start time: $(date)"

# Check if source directory exists
if [ ! -d "$SOURCE_ROOT" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_ROOT"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_ROOT"

# Check available disk space
echo ""
echo "=== DISK SPACE CHECK ==="
echo "Source directory size:"
du -sh "$SOURCE_ROOT"
echo "Available space in destination:"
df -h "$DEST_ROOT"
echo ""

python filter_datasets.py \
    --source-root "$SOURCE_ROOT" \
    --dest-root "$DEST_ROOT" \
    --min-version "$MIN_VERSION" \
    --target-version "$TARGET_VERSION"

echo "Filtering completed at: $(date)"

# Show summary of results
echo ""
echo "=== FILTERING SUMMARY ==="
if [ -d "$DEST_ROOT" ]; then
    echo "Filtered dataset location: $DEST_ROOT"
    echo "Filtered dataset size:"
    du -sh "$DEST_ROOT"
    
    echo ""
    echo "File counts by user:"
    find "$DEST_ROOT" -name "*.mcap" | cut -d'/' -f6 | sort | uniq -c | sort -nr
    
    echo ""
    echo "Total files in filtered dataset:"
    echo "MCAP files: $(find "$DEST_ROOT" -name "*.mcap" | wc -l)"
    echo "MKV files: $(find "$DEST_ROOT" -name "*.mkv" | wc -l)"
else
    echo "No filtered dataset created (no compatible files found)"
fi

echo "Done!"
