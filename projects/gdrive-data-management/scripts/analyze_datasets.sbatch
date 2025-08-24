#!/bin/bash
#SBATCH --job-name=analyze_datasets
#SBATCH --partition=a100
#SBATCH --nodelist=A100-Zebra
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/analyze_datasets_%j.out
#SBATCH --error=logs/analyze_datasets_%j.err

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
DB_PATH="/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db"
TARGET_VERSION="0.5.6"  # Target MCAP version for migration

# Run dataset analysis script
echo "Starting dataset analysis..."
echo "Dataset root: $SOURCE_ROOT"
echo "Database path: $DB_PATH"
echo "Target version: $TARGET_VERSION"
echo "Start time: $(date)"

python analyze_datasets.py \
    --dataset-root "$SOURCE_ROOT" \
    --db-path "$DB_PATH" \
    --target-version "$TARGET_VERSION"

echo "Analysis completed at: $(date)"
echo "Database saved to: $DB_PATH"

# Show summary of results
if [ -f "$DB_PATH" ]; then
    echo ""
    echo "=== ANALYSIS SUMMARY ==="
    python -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
cursor = conn.cursor()

# Get total counts
cursor.execute('SELECT COUNT(*) FROM dataset_analysis')
total_files = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM corrupted_files')
corrupted_files = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(DISTINCT user_email) FROM dataset_analysis')
total_users = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM dataset_analysis WHERE sanitized = 1')
sanitized_files = cursor.fetchone()[0]

cursor.execute('SELECT SUM(duration_seconds) FROM dataset_analysis')
total_duration = cursor.fetchone()[0] or 0

cursor.execute('SELECT SUM(accepted_duration) FROM dataset_analysis')
total_accepted = cursor.fetchone()[0] or 0

print(f'Successfully analyzed files: {total_files}')
print(f'Corrupted/unreadable files: {corrupted_files}')
print(f'Total files processed: {total_files + corrupted_files}')
print(f'Total users: {total_users}')
print(f'Files requiring sanitization: {sanitized_files}')
print(f'Total recording duration: {total_duration/3600:.2f} hours')
print(f'Total accepted duration: {total_accepted/3600:.2f} hours')
print(f'Data efficiency: {(total_accepted/total_duration*100):.1f}%' if total_duration > 0 else 'Data efficiency: N/A')

# Show top games
print('')
print('Top 5 games by file count:')
cursor.execute('SELECT game_name, COUNT(*) as count FROM dataset_analysis GROUP BY game_name ORDER BY count DESC LIMIT 5')
for game, count in cursor.fetchall():
    print(f'  {game}: {count} files')

# Show corrupted files by error type if any exist
if corrupted_files > 0:
    print('')
    print('Corrupted files by error type:')
    cursor.execute('''
        SELECT
            CASE
                WHEN error_message LIKE '%record has length%exceeds limit%' THEN 'File corruption (invalid record length)'
                WHEN error_message LIKE '%Unsupported message type%' THEN 'Unsupported message type'
                ELSE 'Other error'
            END as error_type,
            COUNT(*) as count
        FROM corrupted_files
        GROUP BY error_type
        ORDER BY count DESC
    ''')
    for error_type, count in cursor.fetchall():
        print(f'  {error_type}: {count} files')

conn.close()
"
fi

echo "Done!"
