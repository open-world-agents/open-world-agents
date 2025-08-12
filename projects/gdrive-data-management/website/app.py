#!/usr/bin/env python3
"""
OWA Dataset Analysis Web Viewer
A Flask web application to view and analyze the dataset_analysis.db database.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__)

# Add custom Jinja2 filter for JSON serialization
@app.template_filter('tojson')
def to_json_filter(obj):
    return json.dumps(obj)

# Configuration
DB_PATH = "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db"  # Relative to website folder
app.config['SECRET_KEY'] = 'owa-dataset-viewer-2024'

def get_db_connection():
    """Get database connection with error handling."""
    try:
        if not os.path.exists(DB_PATH):
            return None
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_database_stats():
    """Get overall database statistics."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute('SELECT COUNT(*) as total_files FROM dataset_analysis')
        total_files = cursor.fetchone()['total_files']
        
        cursor.execute('SELECT COUNT(DISTINCT user_email) as total_users FROM dataset_analysis')
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute('SELECT COUNT(*) as sanitized_files FROM dataset_analysis WHERE sanitized = 1')
        sanitized_files = cursor.fetchone()['sanitized_files']

        cursor.execute('SELECT COUNT(*) as failed_files FROM dataset_analysis WHERE available = 0')
        failed_files = cursor.fetchone()['failed_files']
        
        # Duration statistics
        cursor.execute('SELECT SUM(duration_seconds) as total_duration, SUM(accepted_duration) as total_accepted FROM dataset_analysis')
        duration_stats = cursor.fetchone()
        total_duration = duration_stats['total_duration'] or 0
        total_accepted = duration_stats['total_accepted'] or 0
        
        # Version distribution
        cursor.execute('SELECT original_version, COUNT(*) as count FROM dataset_analysis GROUP BY original_version ORDER BY count DESC')
        version_dist = cursor.fetchall()

        # Migration status
        cursor.execute('SELECT migrated_version, COUNT(*) as count FROM dataset_analysis WHERE migrated_version IS NOT NULL GROUP BY migrated_version ORDER BY count DESC')
        migration_dist = cursor.fetchall()
        
        # Game distribution (only available files)
        cursor.execute('SELECT game_name, COUNT(*) as count FROM dataset_analysis WHERE available = 1 GROUP BY game_name ORDER BY count DESC LIMIT 10')
        game_dist = cursor.fetchall()
        
        # User statistics
        cursor.execute('''
            SELECT user_email,
                   COUNT(*) as file_count,
                   SUM(duration_seconds) as total_duration,
                   SUM(accepted_duration) as accepted_duration,
                   AVG(detected_gaps) as avg_gaps
            FROM dataset_analysis
            WHERE available = 1
            GROUP BY user_email
            ORDER BY file_count DESC
        ''')
        user_stats = cursor.fetchall()

        # Gap statistics
        cursor.execute('''
            SELECT
                COUNT(*) as files_with_gaps,
                AVG(detected_gaps) as avg_gaps_per_file,
                SUM(detected_gaps) as total_gaps,
                AVG(total_gap_duration) as avg_gap_duration
            FROM dataset_analysis
            WHERE available = 1 AND detected_gaps > 0
        ''')
        gap_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_files': total_files,
            'total_users': total_users,
            'sanitized_files': sanitized_files,
            'failed_files': failed_files,
            'available_files': total_files - failed_files,
            'total_duration_hours': total_duration / 3600,
            'total_accepted_hours': total_accepted / 3600,
            'efficiency_percent': (total_accepted / total_duration * 100) if total_duration > 0 else 0,
            'version_distribution': [dict(row) for row in version_dist],
            'migration_distribution': [dict(row) for row in migration_dist],
            'game_distribution': [dict(row) for row in game_dist],
            'user_statistics': [dict(row) for row in user_stats],
            'gap_statistics': dict(gap_stats) if gap_stats else {}
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        conn.close()
        return None

@app.route('/')
def index():
    """Main dashboard page."""
    stats = get_database_stats()
    if not stats:
        return render_template('error.html', 
                             error="Could not connect to database. Make sure dataset_analysis.db exists.")
    
    return render_template('index.html', stats=stats)

@app.route('/api/files')
def api_files():
    """API endpoint to get all files with pagination and filtering."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        search = request.args.get('search', '')
        sort_by = request.args.get('sort_by', 'analysis_date')
        sort_order = request.args.get('sort_order', 'DESC')
        
        # Build WHERE clause for search
        where_clause = ""
        params = []
        if search:
            where_clause = """WHERE user_email LIKE ? OR file_name LIKE ? OR game_name LIKE ?"""
            search_param = f"%{search}%"
            params = [search_param, search_param, search_param]
        
        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM dataset_analysis {where_clause}"
        cursor = conn.cursor()
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['total']
        
        # Get paginated data
        offset = (page - 1) * per_page
        data_query = f"""
            SELECT * FROM dataset_analysis 
            {where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_query, params + [per_page, offset])
        files = [dict(row) for row in cursor.fetchall()]
        
        # Format data for display
        for file in files:
            file['duration_minutes'] = round(file['duration_seconds'] / 60, 1) if file['duration_seconds'] else 0
            file['accepted_minutes'] = round(file['accepted_duration'] / 60, 1) if file['accepted_duration'] else 0
            file['efficiency'] = round((file['accepted_duration'] / file['duration_seconds'] * 100), 1) if file['duration_seconds'] and file['duration_seconds'] > 0 else 0
            file['gap_duration_minutes'] = round(file['total_gap_duration'] / 60, 1) if file['total_gap_duration'] else 0

            # Parse gap timeline if available
            if file['gap_timeline']:
                try:
                    import json
                    file['gaps_detail'] = json.loads(file['gap_timeline'])
                except:
                    file['gaps_detail'] = []
            else:
                file['gaps_detail'] = []
        
        conn.close()
        
        return jsonify({
            'files': files,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page
        })
    
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint to get database statistics."""
    stats = get_database_stats()
    if not stats:
        return jsonify({'error': 'Could not get database statistics'}), 500
    return jsonify(stats)

@app.route('/files')
def files_page():
    """Files listing page with search and pagination."""
    return render_template('files.html')

@app.route('/analytics')
def analytics_page():
    """Analytics and charts page."""
    stats = get_database_stats()
    if not stats:
        return render_template('error.html', 
                             error="Could not connect to database for analytics.")
    return render_template('analytics.html', stats=stats)

@app.route('/api/user/<user_email>')
def api_user_details(user_email):
    """API endpoint to get detailed information about a specific user."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor()
        
        # Get user files
        cursor.execute('''
            SELECT * FROM dataset_analysis 
            WHERE user_email = ? 
            ORDER BY analysis_date DESC
        ''', (user_email,))
        files = [dict(row) for row in cursor.fetchall()]
        
        # Calculate user statistics
        if files:
            total_files = len(files)
            total_duration = sum(f['duration_seconds'] or 0 for f in files)
            total_accepted = sum(f['accepted_duration'] or 0 for f in files)
            total_gaps = sum(f['detected_gaps'] or 0 for f in files)
            sanitized_count = sum(1 for f in files if f['sanitized'])
            
            # Game distribution for this user
            games = {}
            for f in files:
                game = f['game_name'] or 'Unknown'
                games[game] = games.get(game, 0) + 1
            
            user_stats = {
                'user_email': user_email,
                'total_files': total_files,
                'total_duration_hours': total_duration / 3600,
                'total_accepted_hours': total_accepted / 3600,
                'efficiency_percent': (total_accepted / total_duration * 100) if total_duration > 0 else 0,
                'total_gaps': total_gaps,
                'sanitized_files': sanitized_count,
                'games': games,
                'files': files
            }
        else:
            user_stats = None
        
        conn.close()
        return jsonify(user_stats)
    
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
