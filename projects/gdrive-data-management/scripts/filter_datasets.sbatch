#!/bin/bash
#SBATCH --job-name=filter_datasets
#SBATCH --nodelist=DGX-H100-12
#SBATCH --cpus-per-task=200
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=logs/filter_datasets_%j.out
#SBATCH --error=logs/filter_datasets_%j.err

# Load Conda (Miniforge) environment
conda init
conda activate owa

# Confirm environment is working
echo "Using Python from: $(which python)"
python --version

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
SOURCE_ROOT="/raid/datasets/owa_game_dataset"
DEST_ROOT="/raid/datasets/owa_game_dataset_filtered_448"
DB_PATH="/raid/datasets/owa_game_dataset/dataset_analysis.db"

# Filtering parameters
BANNED_USER=""
TARGET_VERSION="0.5.6"
MIN_ORIG_VERSION="0.5.3"
MIN_EFFICIENCY="70.0"
MIN_DURATION="300.0"  # 5 minutes minimum duration
BANNED_GAMES="Stardew Valley"  # Space-separated list of banned games
NUM_WORKERS="8"
# TARGET_WIDTH="854"
# TARGET_HEIGHT="480"
TARGET_WIDTH="448"
TARGET_HEIGHT="448"
RESCAN_EXISTING="false"  # Set to "true" to force rescan of existing files

# Run dataset filtering script
echo "Starting dataset filtering with video downscaling..."
echo "Source: $SOURCE_ROOT"
echo "Destination: $DEST_ROOT"
echo "Database: $DB_PATH"
echo "Banned user: $BANNED_USER"
echo "Target version (migrated): $TARGET_VERSION"
echo "Min original version: $MIN_ORIG_VERSION"
echo "Min efficiency: $MIN_EFFICIENCY%"
echo "Min duration: $MIN_DURATION seconds"
echo "Banned games: $BANNED_GAMES"
echo "Number of workers: $NUM_WORKERS"
echo "Target width: $TARGET_WIDTH"
echo "Target height: $TARGET_HEIGHT"
echo "Rescan existing: $RESCAN_EXISTING"
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

RESCAN_FLAG=""
if [ "$RESCAN_EXISTING" = "true" ]; then
    RESCAN_FLAG="--rescan-existing"
fi

python filter_datasets.py \
    --source-root "$SOURCE_ROOT" \
    --dest-root "$DEST_ROOT" \
    --db-path "$DB_PATH" \
    --banned-user "$BANNED_USER" \
    --target-version "$TARGET_VERSION" \
    --min-orig-version "$MIN_ORIG_VERSION" \
    --min-efficiency "$MIN_EFFICIENCY" \
    --min-duration "$MIN_DURATION" \
    --banned-games "$BANNED_GAMES" \
    --num-workers "$NUM_WORKERS" \
    --target-width "$TARGET_WIDTH" \
    --target-height "$TARGET_HEIGHT" \
    $RESCAN_FLAG

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
    echo "MKV files (downscaled to 448x448): $(find "$DEST_ROOT" -name "*.mkv" | wc -l)"

    echo ""
    echo "Sample of downscaled video info:"
    find "$DEST_ROOT" -name "*.mkv" | head -3 | while read mkv_file; do
        echo "File: $(basename "$mkv_file")"
        ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$mkv_file" 2>/dev/null || echo "Could not probe video"
    done
else
    echo "No filtered dataset created (no compatible files found)"
fi

echo "Done!"
