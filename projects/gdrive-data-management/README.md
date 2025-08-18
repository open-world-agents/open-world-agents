# OWA Game Dataset Processing Workflow

This document explains the complete workflow for processing OWA (Open World Assistant) game datasets, from initial download to final filtered dataset creation.

## Environment Setup
```bash
conda env create -f environment.yml
conda activate gdrive-data-management
```

## Overview

The OWA dataset processing pipeline consists of four main components:

1. **`gdrive_sync.py`** - Syncs datasets from Google Drive with MD5 verification
2. **`analyze_datasets.py`** - Analyzes, migrates, sanitizes, and logs dataset metadata
3. **`filter_datasets.py`** - Filters and copies compatible datasets to a clean directory
4. **`website/app.py`** - Web dashboard for viewing and analyzing dataset statistics

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
Creates a clean, filtered dataset containing only compatible files with consistent versions.

### Workflow Steps

#### Step 1: Version Filtering
- **Checks file versions** against minimum requirement (default: ≥0.5.5)
- **Identifies compatible files** for copying
- **Reports incompatible files** that don't meet version requirements

#### Step 2: Smart File Selection
- **Prefers `{filename}_mig.mcap`** if available (processed version)
- **Falls back to `{filename}.mcap`** if no processed version exists
- **Ensures optimal file selection** for filtered dataset

#### Step 3: Clean Copying
- **Copies MCAP files** with clean names (removes `_mig` suffix)
- **Copies paired MKV files** maintaining original names
- **Preserves directory structure** in filtered dataset
- **No migration needed** (already done in analyze step)

### Usage
```bash
python scripts/filter_datasets.py \
    --source-root /mnt/raid12/datasets/owa_game_dataset \
    --dest-root /mnt/raid12/datasets/owa_game_dataset_filtered \
    --min-version 0.5.5 \
    --target-version 0.5.5
```

### Output
- **Filtered dataset** in `/mnt/raid12/datasets/owa_game_dataset_filtered/`
- **Clean file names** (no processing suffixes)
- **Only compatible versions** (≥0.5.5)
- **Paired MCAP/MKV files** maintained

## Script 4: website/app.py

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

### 3. Create Filtered Dataset
```bash
python scripts/filter_datasets.py \
    --source-root /mnt/raid12/datasets/owa_game_dataset \
    --dest-root /mnt/raid12/datasets/owa_game_dataset_filtered \
    --min-version 0.5.5 \
    --target-version 0.5.5
```
**Result**: Clean, filtered dataset ready for use

### 4. View Results in Web Dashboard
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
- **Clean `.mcap`** - Compatible, processed files with clean names (no `_mig` suffix)
- **Paired `.mkv`** - Original video files maintained for reference

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

**4. Database Connection Issues**
- Verify database path is accessible
- Check file permissions for SQLite database
- Ensure database is not locked by another process

**5. Web Dashboard Not Loading**
- Confirm database file exists at configured path
- Check Flask application logs for errors
- Verify port 5000 is available

## Key Benefits

1. **Intelligent Syncing**: MD5 verification prevents unnecessary re-downloads
2. **Version Consistency**: All files migrated to target version for compatibility
3. **Data Quality**: Sanitized to remove multi-window noise
4. **Comprehensive Tracking**: Database logs all processing metadata
5. **Clean Output**: Filtered dataset with consistent naming
6. **Robust Processing**: Handles errors gracefully with fallbacks
7. **Efficient Workflow**: Migration done once, reused in filtering
8. **Web Dashboard**: Interactive visualization of dataset statistics

## Monitoring Progress

All scripts provide rich progress indicators and detailed logging:
- **Real-time progress bars** with file-by-file status
- **Comprehensive summaries** at completion
- **Error handling** with clear messages
- **Database queries** for analysis results
- **Web dashboard** for visual monitoring

## SLURM Batch Scripts

The project includes SLURM batch scripts for running on HPC clusters:
- `scripts/gdrive_sync.sbatch` - for syncing data
- `scripts/analyze_datasets.sbatch` - for dataset analysis
- `scripts/filter_datasets.sbatch` - for dataset filtering

## Next Steps

After completing the workflow:
1. **View web dashboard** at `http://localhost:5000` for interactive analysis
2. **Query the database** for custom dataset statistics
3. **Use filtered dataset** for training/analysis
4. **Monitor processing logs** for any issues
5. **Validate results** using the analysis database and web interface
