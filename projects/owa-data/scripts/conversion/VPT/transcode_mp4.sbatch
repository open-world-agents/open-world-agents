#!/bin/bash
#SBATCH --job-name=process_files
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-63     # Set to desired number of parallel tasks. (tasks per node)=112/7=16, (num nodes)=4
#SBATCH --cpus-per-task=7   # Request 7 CPUs per array task
#SBATCH --mem-per-cpu=4G  # Adjust as needed

# Configuration
FILELIST="filelist.txt"

# Calculate file ranges for this task
TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT}
TASK_ID=${SLURM_ARRAY_TASK_ID}

TOTAL_FILES=$(wc -l < "$FILELIST")
FILES_PER_TASK=$(( (TOTAL_FILES + TOTAL_TASKS - 1) / TOTAL_TASKS ))
START_LINE=$(( TASK_ID * FILES_PER_TASK + 1 ))
END_LINE=$(( (TASK_ID + 1) * FILES_PER_TASK ))
if [ "$END_LINE" -gt "$TOTAL_FILES" ]; then
    END_LINE=$TOTAL_FILES
fi

# Debug info
echo "Task $TASK_ID processing lines $START_LINE-$END_LINE of $TOTAL_FILES"

# Process assigned files
sed -n "${START_LINE},${END_LINE}p" "$FILELIST" | while read -r FILE; do
    # Replace .mp4 extension with .mkv
    OUTPUT_FILE="${FILE%.mp4}.mkv"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "SKIP: $(basename "$OUTPUT_FILE")"
        continue
    fi
    echo "START: $(basename "$FILE")"
    if owl --no-update-check video transcode "$FILE" "$OUTPUT_FILE"; then
        echo "SUCCESS: $(basename "$FILE")"
    else
        echo "FAILED: $(basename "$FILE")"
    fi
done