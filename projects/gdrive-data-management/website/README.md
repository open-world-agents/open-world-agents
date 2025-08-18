# OWA Dataset Analysis Web Viewer

A Flask-based web application for viewing and analyzing the OWA game dataset collection results stored in `dataset_analysis.db`.

## Features

### 📊 **Dashboard**
- **Overview Statistics**: Total files, users, duration, and data efficiency
- **Interactive Charts**: Game distribution, version breakdown, user contributions
- **Top Contributors**: Ranking of most active data collectors
- **Data Quality Metrics**: Sanitization rates and efficiency indicators

### 📋 **Files Browser**
- **Searchable Table**: Filter by user, file name, or game
- **Pagination**: Handle large datasets efficiently
- **Sortable Columns**: Sort by any metric
- **Export Functionality**: Download data as CSV
- **Detailed View**: Click on files for more information

### 📈 **Analytics**
- **Comprehensive Charts**: Multiple visualization types
- **Detailed Statistics**: In-depth analysis of collection patterns
- **Export Reports**: Generate CSV and text summaries
- **User Analysis**: Individual contributor breakdowns

## Installation

### Prerequisites
- Python 3.7+
- Conda environment `gdrive-data-management` (see main README.md)
- Flask and dependencies (included in conda environment)
- A `dataset_analysis.db` file (created by `analyze_datasets.py`)

### Setup Steps

1. **Activate Environment**
   ```bash
   conda activate gdrive-data-management
   cd website
   ```

2. **Ensure Database Exists**
   ```bash
   # Make sure dataset_analysis.db exists at the configured location
   ls -la /mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db

   # If not, create it by running the analysis script
   python ../scripts/analyze_datasets.py \
       --dataset-root /mnt/raid12/datasets/owa_game_dataset \
       --db-path /mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db
   ```

3. **Start the Web Server**
   ```bash
   python app.py
   ```

4. **Access the Application**
   - Open your browser to `http://localhost:5000`
   - The application will automatically detect and load the database

## Usage

### Navigation
- **Dashboard**: Overview and key metrics
- **Files**: Browse and search all dataset files
- **Analytics**: Detailed charts and statistics

### Features

#### Search and Filter
- Use the search box to filter by user email, file name, or game name
- Results update in real-time
- Export filtered results to CSV

#### Interactive Charts
- Hover over chart elements for detailed information
- Charts automatically update when data changes
- Responsive design works on all screen sizes

#### Data Export
- **CSV Export**: Download complete datasets
- **Summary Reports**: Generate text-based summaries
- **Analytics Export**: Comprehensive data analysis

### Keyboard Shortcuts
- `Ctrl/Cmd + R`: Refresh current page
- `Ctrl/Cmd + /`: Focus search input
- `Escape`: Close open modals

## Database Schema

The application reads from the `dataset_analysis` table with the following structure:

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
The `gap_timeline` field contains JSON data with detailed gap information displayed in the web interface:
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

## API Endpoints

The application provides several API endpoints for programmatic access:

- `GET /api/stats` - Overall database statistics
- `GET /api/files` - Paginated file listing with search/sort
- `GET /api/user/<email>` - Detailed user information

### Example API Usage

```bash
# Get overall statistics
curl http://localhost:5000/api/stats

# Get files with search
curl "http://localhost:5000/api/files?search=game_name&page=1&per_page=50"

# Get user details
curl http://localhost:5000/api/user/user@example.com
```

## Configuration

### Database Path
The application is configured to use the database at `/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db`. This path is set in the `DB_PATH` variable in `app.py`:

```python
DB_PATH = "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db"
```

To use a different database location, modify this variable in `app.py`.

### Server Settings
To run on a different host/port, modify the last line in `app.py`:

```python
app.run(debug=False, host='0.0.0.0', port=8080)
```

### Production Deployment
For production use:

1. **Set debug to False**
2. **Use a proper WSGI server** (e.g., Gunicorn)
3. **Configure reverse proxy** (e.g., Nginx)
4. **Set up SSL/HTTPS**

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### Common Issues

1. **Database not found**
   - Ensure `dataset_analysis.db` exists at `/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db`
   - Check file permissions for the database file
   - Verify the database was created by running `../scripts/analyze_datasets.py`
   - Confirm the `DB_PATH` variable in `app.py` points to the correct location

2. **No data showing**
   - Confirm the database contains data
   - Check that the analysis script completed successfully
   - Verify database schema matches expectations

3. **Charts not loading**
   - Ensure internet connection (CDN dependencies)
   - Check browser console for JavaScript errors
   - Verify Chart.js is loading properly

4. **Performance issues**
   - Large datasets may load slowly
   - Consider adding database indexes
   - Implement caching for frequently accessed data

### Debug Mode
Enable debug mode for development:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

This provides detailed error messages and auto-reloading.

## File Structure

```
website/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Dashboard page
│   ├── files.html        # Files browser
│   ├── analytics.html    # Analytics page
│   └── error.html        # Error page
└── static/               # Static assets
    ├── css/
    │   └── style.css     # Custom styles
    └── js/
        └── main.js       # Custom JavaScript
```

## Contributing

To extend the application:

1. **Add new routes** in `app.py`
2. **Create templates** in `templates/`
3. **Add styles** to `static/css/style.css`
4. **Add JavaScript** to `static/js/main.js`

The application uses Bootstrap 5 for styling and Chart.js for visualizations.

## License

This web application is part of the OWA dataset processing pipeline.
