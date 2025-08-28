#!/bin/bash
#SBATCH --job-name=process_datasets
#SBATCH --partition=a100
#SBATCH --nodelist=A100-Zebra
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/process_datasets_%j.out
#SBATCH --error=logs/process_datasets_%j.err

# Load Conda (Miniforge) environment
conda init
conda activate owa

# Confirm environment is working
echo "Using Python from: $(which python)"
python --version

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
FILTERED_ROOT="/mnt/raid12/datasets/owa_game_dataset_filtered_448"
OUTPUT_ROOT="/mnt/raid12/datasets/owa_game_dataset_filtered_448_events-100ms"
DB_PATH="/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db"

# Filtering parameters (can be different from filter_datasets.py)
BANNED_USER=""
TARGET_VERSION="0.5.6"
MIN_ORIG_VERSION="0.5.3"
MIN_EFFICIENCY="70.0"
MIN_DURATION="300.0"  # 5 minutes minimum duration
BANNED_GAMES="Stardew Valley"  # Space-separated list of banned games
NUM_WORKERS="8"
TIME_SHIFT="0.1"

# Run event dataset processing script
echo "Starting event dataset processing with database filtering..."
echo "Filtered root: $FILTERED_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Database: $DB_PATH"
echo "Banned user: $BANNED_USER"
echo "Target version: $TARGET_VERSION"
echo "Min efficiency: $MIN_EFFICIENCY%"
echo "Min duration: $MIN_DURATION seconds"
echo "Banned games: $BANNED_GAMES"
echo "Number of workers: $NUM_WORKERS"
echo "Start time: $(date)"

# Check if filtered directory exists
if [ ! -d "$FILTERED_ROOT" ]; then
    echo "ERROR: Filtered directory does not exist: $FILTERED_ROOT"
    echo "Please run filter_datasets.py first to create filtered datasets."
    exit 1
fi

# Check available disk space
echo ""
echo "=== DISK SPACE CHECK ==="
echo "Filtered directory size:"
du -sh "$FILTERED_ROOT"
echo "Available space in output location:"
df -h "$OUTPUT_ROOT"
echo ""

# Count MCAP files to process
MCAP_COUNT=$(find "$FILTERED_ROOT" -name "*.mcap" | wc -l)
echo "Found $MCAP_COUNT MCAP files to process"

if [ "$MCAP_COUNT" -eq 0 ]; then
    echo "ERROR: No MCAP files found in filtered directory"
    echo "Please run filter_datasets.py first to create filtered datasets."
    exit 1
fi

echo ""
echo "=== STARTING EVENT DATASET PROCESSING ==="

python process_datasets.py \
    --filtered-root "$FILTERED_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --db-path "$DB_PATH" \
    --banned-user "$BANNED_USER" \
    --target-version "$TARGET_VERSION" \
    --min-orig-version "$MIN_ORIG_VERSION" \
    --min-efficiency "$MIN_EFFICIENCY" \
    --min-duration "$MIN_DURATION" \
    --banned-games "$BANNED_GAMES" \
    --num-workers "$NUM_WORKERS" \
    --time-shift "$TIME_SHIFT"

echo "Processing completed at: $(date)"

# Show summary of results
echo ""
echo "=== PROCESSING SUMMARY ==="

# Count event datasets created
EVENT_DATASETS=$(find "$OUTPUT_ROOT" -name "*-event" -type d | wc -l)
echo "Event datasets created: $EVENT_DATASETS"

if [ "$EVENT_DATASETS" -gt 0 ]; then
    echo ""
    echo "Event datasets by game:"
    find "$OUTPUT_ROOT" -name "*-event" -type d | while read event_dir; do
        game_name=$(basename "$event_dir" | sed 's/-event$//')
        train_files=$(find "$event_dir/train" -name "*.npz" 2>/dev/null | wc -l)
        test_files=$(find "$event_dir/test" -name "*.npz" 2>/dev/null | wc -l)
        echo "  $game_name: $train_files train files, $test_files test files"
    done
    
    echo ""
    echo "Total event dataset size:"
    find "$OUTPUT_ROOT" -name "*-event" -type d -exec du -sh {} \; | awk '{sum+=$1} END {print "Total: " sum}'
    
    echo ""
    echo "Sample dataset structure:"
    find "$OUTPUT_ROOT" -name "*-event" -type d | head -1 | while read event_dir; do
        echo "Structure of $(basename "$event_dir"):"
        ls -la "$event_dir" 2>/dev/null || echo "Could not list directory"
        if [ -d "$event_dir/train" ]; then
            echo "Train files: $(ls "$event_dir/train" | wc -l)"
        fi
        if [ -d "$event_dir/test" ]; then
            echo "Test files: $(ls "$event_dir/test" | wc -l)"
        fi
    done
else
    echo "No event datasets were created successfully."
    echo "Check the processing logs for errors."
fi

echo ""
echo "Done!"
