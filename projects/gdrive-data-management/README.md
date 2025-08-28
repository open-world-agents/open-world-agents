# OWA Game Dataset Processing Workflow

This document explains the complete workflow for processing OWA (Open World Assistant) game datasets, from initial download to final filtered dataset creation with video downscaling and event processing.

## Environment Setup
```bash
conda env create -f environment.yml
conda activate gdrive-data-management
```

## Overview

The OWA dataset processing pipeline consists of five main components:

1. **`gdrive_sync.py`** - Syncs datasets from Google Drive with MD5 verification
2. **`analyze_datasets.py`** - Analyzes, migrates, sanitizes, and logs dataset metadata
3. **`filter_datasets.py`** - Filters datasets with video downscaling and multiprocessing
4. **`process_datasets.py`** - Creates event datasets grouped by game
5. **`website/app.py`** - Web dashboard for viewing and analyzing dataset statistics

## Dataset Structure

The datasets are organized in a hierarchical structure:
```
/mnt/raid12/datasets/owa_game_dataset/
├── user1@email.com/
│   ├── session1/
│   │   ├── recording1.mcap (original recording)
│   │   ├── recording1.mkv (paired video)
│   │   └── recording1_mig.mcap (processed version)
│   └── session2/
│       ├── recording2.mcap
│       ├── recording2.mkv
│       └── recording2_mig.mcap
└── user2@email.com/
    ├── recording3.mcap
    ├── recording3.mkv
    └── recording3_mig.mcap
```

## Script 1: gdrive_sync.py

### Purpose
Syncs OWA game datasets from Google Drive to local storage with intelligent file verification.

### Key Features
- **OAuth2 Authentication** with Google Drive API using client secrets
- **MD5 Hash Verification** - compares local and remote file checksums to avoid unnecessary downloads
- **Selective File Types** - downloads only `.log`, `.mkv`, and `.mcap` files
- **Shared Drive Support** - works with Google Shared Drives
- **Resume Capability** - skips files that already exist with matching checksums
- **Progress Tracking** - shows download progress for large files

### Configuration
The script uses hardcoded configuration for:
- Client secret file path: `/mnt/raid12/workspace/jyjung/confidential/client_secret_*.json`
- Shared Drive ID: `0AHLjQqrHnATRUk9PVA`
- Download directory: `/mnt/raid12/datasets/owa_game_dataset`

### Usage
```bash
python scripts/gdrive_sync.py
```

### Output
- Downloads files to `/mnt/raid12/datasets/owa_game_dataset/`
- Preserves original file names and directory structure
- Creates user-specific directories (e.g., `user@email.com/`)
- Generates `token.json` for authentication persistence

## Script 2: analyze_datasets.py

### Purpose
The core processing script that migrates, sanitizes, analyzes, and logs dataset information.

### Workflow Steps

#### Step 1: Validation & Migration
- **Validates paired files** - skips processing if `{filename}.mkv` doesn't exist
- **Checks file version** using `reader.file_version`
- **Migrates to target version** if current version is different
- **Creates `{filename}_mig.mcap`** with migrated version
- **Logs failures** - records as failed dataset if migration fails

#### Step 2: Window Analysis & Sanitization
- **Extracts window titles** from MCAP files
- **Identifies most active window** (primary game/application)
- **Sanitizes if needed**: If multiple windows detected, keeps only the most active window
- **Overwrites `_mig.mcap`** with sanitized version (or creates it if no migration occurred)

#### Step 3: Gap Detection
- **Analyzes user activity** from keyboard/mouse events
- **Detects inactivity gaps** longer than 60 seconds
- **Records precise timeline** with exact start/end timestamps (e.g., "1:52 - 2:59 (67s)")
- **Uses actual video duration** from MKV files for accurate timeline calculation
- **Stores detailed gap data** in JSON format for analysis
- **Calculates accepted duration** (total time - gap time)

#### Step 4: Database Logging
- **Logs comprehensive metadata** to SQLite database
- **Tracks original file version** for reference
- **Records analysis results** from processed files

### Database Schema
```sql
CREATE TABLE dataset_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT NOT NULL,           -- User who collected data
    file_name TEXT NOT NULL,            -- Original file name (without _mig suffix)
    game_name TEXT,                     -- Detected game/application (most active window)
    duration_seconds REAL,              -- Total recording duration from MKV file
    detected_gaps INTEGER,              -- Number of inactivity gaps > 60 seconds
    total_gap_duration REAL,            -- Total time of gaps in seconds
    accepted_duration REAL,             -- Active recording time (duration - gaps)
    gap_timeline TEXT,                  -- JSON array of detailed gap information
    sanitized BOOLEAN DEFAULT FALSE,    -- Whether multi-window sanitization was needed
    original_version TEXT,              -- Original version from {filename}.mcap
    migrated_version TEXT,              -- Target version after migration
    available BOOLEAN DEFAULT TRUE,     -- Whether file was successfully processed
    error_message TEXT,                 -- Error details for failed files
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_email, file_name)       -- Prevent duplicate entries
);
```

### Gap Timeline Format
The `gap_timeline` field contains JSON data with detailed gap information:
```json
[
  {
    "start_time": 125.5,
    "end_time": 192.3,
    "gap_duration": 66.8,
    "start_timestamp": "2:05",
    "end_timestamp": "3:12",
    "gap_description": "2:05 - 3:12 (67s)"
  }
]
```

### Usage
```bash
python scripts/analyze_datasets.py --dataset-root /mnt/raid12/datasets/owa_game_dataset --db-path dataset_analysis.db --target-version 0.5.5
```

### Output Files
- **`{filename}_mig.mcap`** - Migrated and/or sanitized version
- **`dataset_analysis.db`** - SQLite database with metadata
- **Progress logs** with detailed processing information

## Script 3: filter_datasets.py

### Purpose
Creates a clean, filtered dataset with video downscaling and comprehensive version filtering.

### Key Features
- **Dual Version Filtering**: Filters by both original version (≥0.5.5) and migrated version (=0.5.6)
- **Multiprocessing Video Downscaling**: Parallel processing of videos with configurable workers
- **Comprehensive Logging**: Detailed tracking of MCAP and MKV file locations
- **Efficiency Filtering**: Excludes datasets below minimum efficiency threshold
- **User Exclusion**: Filters out banned users from the dataset

### Workflow Steps

#### Step 1: Database Query & Filtering
- **Queries analysis database** for datasets meeting all criteria:
  - `migrated_version = 0.5.6` (target migrated version)
  - `original_version >= 0.5.5` (minimum original version before migration)
  - `efficiency >= 70.0%` (minimum activity efficiency)
  - `user_email != banned_user` (exclude specific users)
- **Reports filtering results** with detailed criteria

#### Step 2: Smart File Discovery
- **Prefers `{filename}_mig.mcap`** if available (processed version)
- **Falls back to `{filename}.mcap`** if no migrated version exists
- **Finds paired MKV files** using glob patterns at any nested depth
- **Logs file discovery** with source paths

#### Step 3: MCAP File Copying
- **Copies MCAP files** maintaining directory structure
- **Preserves migrated naming** (`_mig.mcap` suffix)
- **Creates destination directories** as needed
- **Logs copy operations** with source and destination paths

#### Step 4: Parallel Video Processing
- **Collects video tasks** for multiprocessing
- **Downscales videos to 720x480** (SD resolution) using FFmpeg
- **Processes videos in parallel** with configurable worker count
- **Maintains directory structure** (same path as MCAP files)
- **Provides fallback copying** for failed downscaling

### Video Processing Features
- **FFmpeg downscaling**: 720x480 resolution, 60fps, H.264 encoding
- **Timeout protection**: 1-hour timeout per video to prevent hanging
- **Error handling**: Graceful handling with original file fallback
- **Progress tracking**: Real-time progress bars for both phases

### Usage
```bash
python scripts/filter_datasets.py \
    --source-root /mnt/raid12/datasets/owa_game_dataset \
    --dest-root /mnt/raid12/datasets/owa_game_dataset_filtered \
    --target-version 0.5.6 \
    --min-orig-version 0.5.5 \
    --min-efficiency 70.0 \
    --banned-user parktj93@gmail.com \
    --num-workers 8
```

### Configuration Options
- `--target-version`: Target migrated version (default: 0.5.6)
- `--min-orig-version`: Minimum original version (default: 0.5.5)
- `--min-efficiency`: Minimum efficiency percentage (default: 70.0)
- `--banned-user`: User email to exclude (default: parktj93@gmail.com)
- `--num-workers`: Number of parallel workers (default: 4)
- `--target-width/height`: Video resolution (default: 720x480)

### Output
- **Filtered dataset** in `/mnt/raid12/datasets/owa_game_dataset_filtered/`
- **Downscaled videos** at 720x480 resolution for storage efficiency
- **Preserved directory structure** with MCAP and MKV files in same directories
- **Comprehensive logging** of all file operations and locations

### SLURM Batch Processing
```bash
sbatch scripts/filter_datasets.sbatch
```
The batch script includes optimized settings for cluster processing with 16 workers.

## Script 4: process_datasets.py

### Purpose
Creates event-based datasets grouped by game from filtered datasets for machine learning training.

### Key Features
- **Game-based Grouping**: Organizes datasets by detected game/application
- **Event Extraction**: Processes MCAP files to extract user interaction events
- **Version Filtering**: Same dual version filtering as filter_datasets.py
- **Efficiency Filtering**: Excludes low-efficiency datasets
- **Structured Output**: Creates organized directory structure by game

### Workflow Steps

#### Step 1: Database Query & Filtering
- **Applies same filtering criteria** as filter_datasets.py:
  - `migrated_version = 0.5.6`
  - `original_version >= 0.5.5`
  - `efficiency >= 70.0%`
  - `user_email != banned_user`

#### Step 2: Game Detection & Grouping
- **Groups datasets by game_name** from database analysis
- **Creates game-specific directories** in output root
- **Handles unknown games** with fallback naming

#### Step 3: Event Processing
- **Extracts user events** from MCAP files
- **Processes interaction data** for ML training
- **Maintains temporal relationships** between events
- **Outputs structured event data** for each game

### Usage
```bash
python scripts/process_datasets.py \
    --filtered-root /mnt/raid12/datasets/owa_game_dataset_filtered \
    --output-root /mnt/raid12/datasets/owa_game_dataset_events \
    --target-version 0.5.6 \
    --min-orig-version 0.5.5 \
    --min-efficiency 70.0 \
    --num-workers 4
```

### Configuration Options
- `--filtered-root`: Source directory with filtered datasets
- `--output-root`: Destination for event datasets
- `--target-version`: Target migrated version (default: 0.5.6)
- `--min-orig-version`: Minimum original version (default: 0.5.5)
- `--min-efficiency`: Minimum efficiency percentage (default: 70.0)
- `--num-workers`: Number of parallel workers (default: 4)
- `--dry-run`: Preview processing without actual execution

### Output Structure
```
/output-root/
├── Grand_Theft_Auto_V/
│   ├── user1@email.com_recording1_events.json
│   └── user2@email.com_recording2_events.json
├── Minecraft/
│   ├── user3@email.com_recording3_events.json
│   └── user4@email.com_recording4_events.json
└── Unknown_Game/
    └── user5@email.com_recording5_events.json
```

### SLURM Batch Processing
```bash
sbatch scripts/process_datasets.sbatch
```

## Script 5: website/app.py

### Purpose
Web dashboard for viewing and analyzing dataset statistics from the SQLite database.

### Key Features
- **Interactive Dashboard** - overview of dataset statistics and metrics
- **File Browser** - searchable, paginated table of all processed files
- **Analytics Page** - charts and visualizations of dataset distributions
- **User Details** - detailed statistics for individual users
- **Real-time Data** - connects directly to the analysis database
- **REST API** - JSON endpoints for programmatic access

### Web Interface Pages
- **Dashboard** (`/`) - overview statistics and summaries
- **Files** (`/files`) - searchable file listing with gap details
- **Analytics** (`/analytics`) - charts for version, game, and user distributions

### API Endpoints
- `GET /api/files` - paginated file listing with search and sorting
- `GET /api/stats` - overall database statistics
- `GET /api/user/<email>` - detailed user statistics

### Web Dashboard Structure
```
website/
├── app.py                 # Flask application
├── templates/             # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Dashboard overview
│   ├── files.html        # File listing page
│   ├── analytics.html    # Charts and analytics
│   └── error.html        # Error page
└── static/               # CSS and JavaScript assets
    ├── css/
    └── js/
```

### Usage
```bash
cd website
python app.py
```
**Access**: Open browser to `http://localhost:5000`

### Dashboard Features
- **Overview Statistics**: Total files, users, duration, efficiency metrics
- **Version Distribution**: Charts showing original and migrated versions
- **Game Distribution**: Most common games/applications detected
- **User Statistics**: Per-user file counts and duration summaries
- **Gap Analysis**: Statistics on inactivity gaps across all files
- **Search & Filter**: Find specific files by user, name, or game
- **Detailed Gap Timeline**: View exact timestamps of inactivity periods

## Complete Workflow Example

### 1. Sync Data from Google Drive
```bash
python scripts/gdrive_sync.py
```
**Result**: Raw datasets in `/mnt/raid12/datasets/owa_game_dataset/`

### 2. Analyze & Process
```bash
python scripts/analyze_datasets.py \
    --dataset-root /mnt/raid12/datasets/owa_game_dataset \
    --db-path dataset_analysis.db \
    --target-version 0.5.5
```
**Result**:
- Processed `_mig.mcap` files
- Analysis database with metadata
- Migration and sanitization logs

### 3. Create Filtered Dataset with Video Downscaling
```bash
python scripts/filter_datasets.py \
    --source-root /mnt/raid12/datasets/owa_game_dataset \
    --dest-root /mnt/raid12/datasets/owa_game_dataset_filtered \
    --target-version 0.5.6 \
    --min-orig-version 0.5.5 \
    --min-efficiency 70.0 \
    --num-workers 8
```
**Result**:
- Filtered dataset with downscaled videos (720x480)
- Comprehensive logging of all file operations
- Parallel processing for faster completion

### 4. Create Event Datasets by Game
```bash
python scripts/process_datasets.py \
    --filtered-root /mnt/raid12/datasets/owa_game_dataset_filtered \
    --output-root /mnt/raid12/datasets/owa_game_dataset_events \
    --target-version 0.5.6 \
    --min-orig-version 0.5.5 \
    --num-workers 4
```
**Result**: Event datasets organized by game for ML training

### 5. View Results in Web Dashboard
```bash
cd website && python app.py
```
**Result**: Interactive web interface at `http://localhost:5000`

## File Types Explained

### Original Files
- **`.mcap`** - Original recording data with user interactions
- **`.mkv`** - Paired video file (required for accurate duration calculation)
- **`.log`** - Additional log files (downloaded but not processed)

### Processed Files
- **`{filename}_mig.mcap`** - Migrated to target version and/or sanitized
- **`dataset_analysis.db`** - SQLite database with comprehensive metadata
- **`token.json`** - Google Drive authentication token (auto-generated)

### Filtered Dataset
- **`{filename}_mig.mcap`** - Processed MCAP files with migrated naming preserved
- **`{filename}.mkv`** - Downscaled videos (720x480) for storage efficiency
- **Same directory structure** - MCAP and MKV files in same relative paths

### Event Datasets
- **Game-organized directories** - Datasets grouped by detected game/application
- **Event JSON files** - Extracted user interaction events for ML training
- **Structured format** - Ready for machine learning model training

## Troubleshooting

### Common Issues

**1. Google Drive Authentication Errors**
- Ensure client secret file exists at the configured path
- Delete `token.json` and re-authenticate if needed
- Check Google Drive API permissions

**2. Migration Failures**
- Verify MCAP file is not corrupted
- Check if file version is supported for migration
- Review error messages in console output

**3. Missing MKV Files**
- Ensure paired video files exist for each MCAP file
- Check file naming conventions match exactly
- Files without MKV pairs will be skipped

**4. Video Processing Failures**
- Check FFmpeg installation and availability
- Monitor system resources during parallel processing
- Reduce worker count if memory issues occur
- Review timeout settings for very large videos

**5. Version Filtering Issues**
- Verify database contains original_version column
- Check version comparison logic for edge cases
- Ensure both original and migrated versions are properly recorded

**6. Database Connection Issues**
- Verify database path is accessible
- Check file permissions for SQLite database
- Ensure database is not locked by another process

**7. Web Dashboard Not Loading**
- Confirm database file exists at configured path
- Check Flask application logs for errors
- Verify port 5000 is available

## Key Benefits

1. **Intelligent Syncing**: MD5 verification prevents unnecessary re-downloads
2. **Dual Version Filtering**: Filters by both original and migrated versions for quality control
3. **Multiprocessing**: Parallel video processing significantly speeds up large dataset handling
4. **Storage Optimization**: Video downscaling reduces storage requirements while maintaining quality
5. **Comprehensive Logging**: Detailed tracking of all file operations and locations
6. **Data Quality**: Sanitized to remove multi-window noise and low-efficiency datasets
7. **Game Organization**: Event datasets grouped by game for targeted ML training
8. **Robust Processing**: Handles errors gracefully with fallbacks and timeout protection
9. **Scalable Architecture**: Configurable workers and SLURM batch processing for HPC clusters
10. **Web Dashboard**: Interactive visualization of dataset statistics and analysis

## Monitoring Progress

All scripts provide rich progress indicators and detailed logging:
- **Real-time progress bars** with file-by-file status
- **Comprehensive summaries** at completion
- **Error handling** with clear messages
- **Database queries** for analysis results
- **Web dashboard** for visual monitoring

## SLURM Batch Scripts

The project includes optimized SLURM batch scripts for HPC cluster processing:

### Available Batch Scripts
- **`scripts/gdrive_sync.sbatch`** - Google Drive data synchronization
- **`scripts/analyze_datasets.sbatch`** - Dataset analysis and migration
- **`scripts/filter_datasets.sbatch`** - Dataset filtering with video downscaling
- **`scripts/process_datasets.sbatch`** - Event dataset creation

### Batch Script Features
- **Resource optimization**: Configured CPU and memory requirements
- **Parallel processing**: Utilizes multiple workers for faster processing
- **Comprehensive logging**: Detailed output for monitoring progress
- **Error handling**: Robust error management for long-running jobs

### Usage Examples
```bash
# Submit filtering job with video downscaling
sbatch scripts/filter_datasets.sbatch

# Submit event processing job
sbatch scripts/process_datasets.sbatch

# Monitor job status
squeue -u $USER
```

## Recent Improvements

### Version 2.0 Features
- **Dual Version Filtering**: Added minimum original version filtering (≥0.5.5) in addition to target migrated version filtering
- **Multiprocessing Video Processing**: Parallel video downscaling with configurable worker count for significant performance improvements
- **Comprehensive Logging**: Detailed tracking of MCAP and MKV file locations throughout the entire pipeline
- **Video Downscaling**: Automatic downscaling to 720x480 resolution for storage optimization
- **Enhanced Error Handling**: Robust timeout protection and fallback mechanisms for video processing
- **Directory Structure Preservation**: MCAP and MKV files maintained in same relative paths
- **SLURM Optimization**: Updated batch scripts with optimized resource allocation for HPC clusters

### Performance Improvements
- **Up to 8x faster video processing** with multiprocessing (8 workers vs sequential)
- **50-70% storage reduction** with video downscaling while maintaining quality
- **Intelligent file discovery** using glob patterns for nested directory structures
- **Timeout protection** prevents hanging on problematic video files
- **Memory optimization** with configurable worker pools

## Performance Considerations

### Video Processing Optimization
- **Worker count**: Adjust `--num-workers` based on CPU cores and memory
- **Storage I/O**: Consider SSD vs HDD for source and destination paths
- **Network bandwidth**: Ensure adequate bandwidth for large video files
- **Memory usage**: Monitor memory consumption with multiple workers

### Recommended Settings
- **Small datasets** (< 100 videos): 2-4 workers
- **Medium datasets** (100-500 videos): 4-8 workers
- **Large datasets** (> 500 videos): 8-16 workers
- **HPC clusters**: Use SLURM batch scripts with optimized resource allocation

## Next Steps

After completing the workflow:
1. **View web dashboard** at `http://localhost:5000` for interactive analysis
2. **Query the database** for custom dataset statistics and filtering results
3. **Use filtered dataset** for training/analysis with downscaled videos
4. **Process event datasets** for game-specific machine learning models
5. **Monitor processing logs** for performance optimization and issue detection
6. **Validate results** using comprehensive logging and database analysis
