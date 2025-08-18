#!/usr/bin/env python3
"""
Analyze OWA game datasets: sanitize MCAP files and log information to database.

This script:
1. Finds all MCAP files in the dataset directory
2. Extracts window titles and sanitizes files to keep only the most active window
3. Detects gaps in user activity
4. Logs all information to a SQLite database
"""

import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager
import warnings

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Import functions from reference files
import sys

sys.path.append("reference")

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.cli.mcap.migrate import MigrationOrchestrator


def get_file_version(mcap_path: Path) -> Optional[str]:
    """Get the file version from MCAP file."""
    try:
        with OWAMcapReader(mcap_path) as reader:
            return reader.file_version
    except Exception as e:
        print(f"Warning: Could not get version for {mcap_path}: {e}")
        return None


def migrate_mcap(input_file: Path, output_file: Path, target_version: str = "0.5.6") -> bool:
    """Migrate MCAP file to target version."""
    try:
        orchestrator = MigrationOrchestrator()

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if migration needed
        current_version = orchestrator.detect_version(input_file)

        if current_version == target_version:
            # Just copy if no migration needed
            shutil.copy2(input_file, output_file)
            return True

        # Migrate via temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / input_file.name
            shutil.copy2(input_file, temp_file)

            # Migrate
            results = orchestrator.migrate_file(temp_file, target_version, keep_backup=False)

            if all(r.success for r in results):
                shutil.copy2(temp_file, output_file)
                return True
            else:
                return False

    except Exception as e:
        print(f"Warning: Migration failed for {input_file}: {e}")
        return False


def extract_window_names_robust(mcap_path: Path) -> Dict[str, List[int]]:
    """Extract all unique window names and their timestamps from an MCAP file."""
    window_data = {}

    with OWAMcapReader(mcap_path, decode_args={"return_dict": True}) as reader:
        for mcap_msg in reader.iter_messages(topics=["window"]):
            try:
                decoded = mcap_msg.decoded
                window_title = None

                if isinstance(decoded, dict):
                    window_title = decoded.get("title", "")
                elif hasattr(decoded, "title"):
                    window_title = decoded.title
                else:
                    decoded_str = str(decoded)
                    if "title" in decoded_str:
                        import re

                        match = re.search(r"title['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", decoded_str)
                        if match:
                            window_title = match.group(1)

                if window_title:
                    if window_title not in window_data:
                        window_data[window_title] = []
                    window_data[window_title].append(mcap_msg.timestamp)

            except Exception:
                continue

    return window_data


def get_most_active_window(window_data: Dict[str, List[int]]) -> Optional[str]:
    """Get the window with the most activations."""
    if not window_data:
        return None

    return max(window_data.keys(), key=lambda w: len(window_data[w]))


@contextmanager
def safe_temp_file(mode="wb", suffix=".mcap"):
    """Context manager for temporary files."""
    with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        yield temp_file, temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


def sanitize_mcap_file(file_path: Path, keep_window: str, output_path: Path) -> dict:
    """Sanitize MCAP file to keep only events for the specified window."""
    total_messages = 0
    kept_messages = 0
    keep_current_events = False

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_temp_file(mode="wb", suffix=".mcap") as (temp_file, temp_path):
        with OWAMcapReader(file_path) as reader, OWAMcapWriter(temp_path) as writer:
            for mcap_msg in reader.iter_messages():
                total_messages += 1

                if mcap_msg.topic == "window":
                    if hasattr(mcap_msg.decoded, "title"):
                        window_title = mcap_msg.decoded.title
                    elif isinstance(mcap_msg.decoded, dict):
                        window_title = mcap_msg.decoded.get("title", "")
                    else:
                        window_title = ""

                    keep_current_events = keep_window.lower() in window_title.lower()

                if keep_current_events:
                    writer.write_message(mcap_msg.decoded, topic=mcap_msg.topic, timestamp=mcap_msg.timestamp)
                    kept_messages += 1

        # Copy to final location
        shutil.copy2(temp_path, output_path)

    return {
        "total_messages": total_messages,
        "kept_messages": kept_messages,
        "removed_messages": total_messages - kept_messages,
        "success": True,
    }


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def detect_activity_gaps(mcap_path: Path, gap_threshold_seconds: int = 60) -> List[dict]:
    """Detect gaps longer than threshold between keyboard/mouse events with detailed timeline."""
    activity_events = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Domain-based message .* not found in registry.*", UserWarning)

        with OWAMcapReader(mcap_path, decode_args={"return_dict_on_failure": True}) as reader:
            recording_start_time = reader.start_time if reader.start_time else 0

            for msg in reader.iter_messages(topics=["keyboard", "mouse", "mouse/raw"]):
                try:
                    # Calculate relative time from start of recording
                    relative_time_s = (msg.timestamp - recording_start_time) / 1e9
                    activity_events.append(
                        {
                            "timestamp": msg.timestamp,
                            "time_s": relative_time_s,
                        }
                    )
                except Exception:
                    continue

    if not activity_events:
        return []

    activity_events.sort(key=lambda x: x["timestamp"])

    gaps = []
    gap_threshold_ns = gap_threshold_seconds * 1e9

    for i in range(1, len(activity_events)):
        prev_event = activity_events[i - 1]
        curr_event = activity_events[i]

        gap_ns = curr_event["timestamp"] - prev_event["timestamp"]
        gap_seconds = gap_ns / 1e9

        if gap_ns > gap_threshold_ns:
            start_time_s = prev_event["time_s"]
            end_time_s = curr_event["time_s"]

            gaps.append(
                {
                    "start_time": start_time_s,
                    "end_time": end_time_s,
                    "gap_duration": gap_seconds,
                    "start_timestamp": format_timestamp(start_time_s),
                    "end_timestamp": format_timestamp(end_time_s),
                    "gap_description": f"{format_timestamp(start_time_s)} - {format_timestamp(end_time_s)} ({gap_seconds:.0f}s)",
                }
            )

    return gaps


def get_video_duration_from_mkv(mcap_path: Path) -> float:
    """Get actual video duration from paired MKV file."""
    import subprocess

    # Find the paired MKV file
    mkv_path = mcap_path.with_suffix(".mkv")
    if not mkv_path.exists():
        # Try without _mig suffix if it exists
        if mcap_path.stem.endswith("_mig"):
            base_name = mcap_path.stem[:-4]  # Remove _mig
            mkv_path = mcap_path.parent / f"{base_name}.mkv"

    if not mkv_path.exists():
        print(f"Warning: No paired MKV file found for {mcap_path.name}")
        return 0.0

    try:
        # Use ffprobe to get video duration
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(mkv_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not get video duration from {mkv_path}: {e}")

    return 0.0


def get_video_duration(mcap_path: Path) -> float:
    """Get total duration of the recording in seconds, preferring MKV duration."""
    # First try to get duration from MKV file (more accurate)
    mkv_duration = get_video_duration_from_mkv(mcap_path)
    if mkv_duration > 0:
        return mkv_duration

    # Fallback to MCAP duration
    try:
        with OWAMcapReader(mcap_path) as reader:
            if reader.start_time and reader.end_time:
                return (reader.end_time - reader.start_time) / 1e9
    except Exception:
        pass
    return 0.0


def init_database(db_path: Path):
    """Initialize the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            file_name TEXT NOT NULL,
            game_name TEXT,
            duration_seconds REAL,
            detected_gaps INTEGER,
            total_gap_duration REAL,
            accepted_duration REAL,
            gap_timeline TEXT,
            sanitized BOOLEAN DEFAULT FALSE,
            original_version TEXT,
            migrated_version TEXT,
            available BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_email, file_name)
        )
    """)

    conn.commit()
    conn.close()


def needs_processing(db_path: Path, user_email: str, file_name: str, target_version: str) -> bool:
    """Check if a file needs processing based on database records and target version."""
    if not db_path.exists():
        return True  # No database yet, need to process

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT migrated_version, available FROM dataset_analysis
            WHERE user_email = ? AND file_name = ?
        """,
            (user_email, file_name),
        )

        result = cursor.fetchone()
        if not result:
            return True  # File not in database, need to process

        migrated_version, available = result

        # Need to process if:
        # 1. File was previously failed (available = 0)
        # 2. No migrated version recorded
        # 3. Migrated version is different from target version
        if not available:
            return True  # Previously failed, retry

        if not migrated_version:
            return True  # No migration version recorded

        if migrated_version != target_version:
            return True  # Need to migrate to new version

        return False  # Already processed with correct version

    except Exception as e:
        print(f"Warning: Error checking database for {user_email}/{file_name}: {e}")
        return True  # On error, process to be safe
    finally:
        conn.close()


def log_to_database(
    db_path: Path,
    user_email: str,
    file_name: str,
    game_name: str = None,
    duration: float = 0,
    gaps: List[dict] = None,
    sanitized: bool = False,
    original_version: str = None,
    migrated_version: str = None,
    available: bool = True,
    error_message: str = None,
):
    """Log analysis results to database."""
    import json

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if gaps is None:
        gaps = []

    total_gap_duration = sum(gap["gap_duration"] for gap in gaps) if gaps else 0
    accepted_duration = duration - total_gap_duration if duration > 0 else 0

    # Convert gaps to JSON for storage
    gap_timeline = json.dumps(gaps) if gaps else None

    cursor.execute(
        """
        INSERT OR REPLACE INTO dataset_analysis
        (user_email, file_name, game_name, duration_seconds, detected_gaps,
         total_gap_duration, accepted_duration, gap_timeline, sanitized, original_version,
         migrated_version, available, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            user_email,
            file_name,
            game_name,
            duration,
            len(gaps),
            total_gap_duration,
            accepted_duration,
            gap_timeline,
            sanitized,
            original_version,
            migrated_version,
            available,
            error_message,
        ),
    )

    conn.commit()
    conn.close()


def analyze_datasets(
    dataset_root: str = "/mnt/raid12/datasets/owa_game_dataset",
    db_path: str = "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db",
    target_version: str = "0.5.6",
):
    """Main function to analyze all datasets."""
    console = Console()
    dataset_path = Path(dataset_root)
    db_path = Path(db_path)

    if not dataset_path.exists():
        console.print(f"[red]Dataset directory not found: {dataset_path}[/red]")
        return

    # Initialize database
    init_database(db_path)
    console.print(f"[green]Database initialized: {db_path}[/green]")
    console.print(f"[blue]Target migration version: {target_version}[/blue]")

    # Find all MCAP files at any depth within user directories
    all_mcap_files = list(dataset_path.glob("*/**/*.mcap"))
    console.print(f"[blue]Found {len(all_mcap_files)} total MCAP files[/blue]")

    if not all_mcap_files:
        console.print("[yellow]No MCAP files found![/yellow]")
        return

    # Filter files that need processing (avoid duplicates and _mig files)
    mcap_files = []
    skipped_count = 0
    processed_base_names = set()

    for mcap_file in all_mcap_files:
        # Extract user and file info
        parts = mcap_file.parts
        dataset_root_parts = dataset_path.parts
        relative_parts = parts[len(dataset_root_parts) :]
        user_email = relative_parts[0] if relative_parts else "unknown"

        # Get base file name (remove _mig suffix if present)
        file_stem = mcap_file.stem
        if file_stem.endswith("_mig"):
            base_name = file_stem[:-4]  # Remove _mig suffix
        else:
            base_name = file_stem

        # Create unique identifier for this base file
        file_identifier = f"{user_email}/{base_name}"

        # Skip if we've already processed this base file
        if file_identifier in processed_base_names:
            skipped_count += 1
            continue

        # Check if this base file needs processing
        if needs_processing(db_path, user_email, base_name, target_version):
            mcap_files.append(mcap_file)
            processed_base_names.add(file_identifier)
        else:
            skipped_count += 1
            processed_base_names.add(file_identifier)

    console.print(f"[green]Files to process: {len(mcap_files)}[/green]")
    console.print(f"[yellow]Files skipped (already processed): {skipped_count}[/yellow]")

    if not mcap_files:
        console.print("[blue]All files are already processed with the target version![/blue]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing datasets...", total=len(mcap_files))

        for mcap_file in mcap_files:
            # Extract user and file info first (needed for error logging)
            parts = mcap_file.parts
            dataset_root_parts = dataset_path.parts
            relative_parts = parts[len(dataset_root_parts) :]
            user_email = relative_parts[0] if relative_parts else "unknown"

            # Get base file name (remove _mig suffix if present)
            file_stem = mcap_file.stem
            if file_stem.endswith("_mig"):
                base_name = file_stem[:-4]  # Remove _mig suffix
            else:
                base_name = file_stem

            progress.update(task, description=f"Processing {user_email}/{base_name}")

            try:
                # Check if paired MKV file exists
                mkv_file = mcap_file.parent / f"{base_name}.mkv"
                if not mkv_file.exists():
                    console.print(f"[yellow]Skipping {base_name}: No paired MKV file found[/yellow]")
                    progress.advance(task)
                    continue

                # Determine which file to use for processing
                original_file = mcap_file.parent / f"{base_name}.mcap"
                mig_file = mcap_file.parent / f"{base_name}_mig.mcap"

                # Always get original version from the base .mcap file
                if original_file.exists():
                    original_version = get_file_version(original_file)
                else:
                    # If only _mig file exists, try to get version from it but note this case
                    original_version = get_file_version(mcap_file)
                    console.print(f"[yellow]Warning: Only _mig file exists for {base_name}[/yellow]")

                # Prefer _mig file if it exists, otherwise use original
                if mig_file.exists():
                    working_file = mig_file
                    console.print(f"[blue]Using existing migrated file: {mig_file.name}[/blue]")
                else:
                    working_file = original_file

                # Step 1: Check version and migrate if needed
                migrated_file = working_file
                migrated_version = original_version  # Default to original if no migration

                if original_version and original_version != target_version:
                    # Create/update migrated version (always use base_name_mig.mcap)
                    # Always migrate from the original file, not the working file
                    migrated_path = mcap_file.parent / f"{base_name}_mig.mcap"
                    source_file = original_file if original_file.exists() else working_file
                    migration_success = migrate_mcap(source_file, migrated_path, target_version)

                    if migration_success:
                        migrated_file = migrated_path
                        migrated_version = target_version
                        console.print(
                            f"[green]Migrated: {migrated_path} ({original_version} → {target_version})[/green]"
                        )
                    else:
                        error_msg = f"Migration failed from version {original_version} to {target_version}"
                        console.print(f"[red]Migration failed for {mcap_file.name}[/red]")

                        # Log as failed dataset
                        log_to_database(
                            db_path,
                            user_email,
                            base_name,
                            None,
                            0,
                            None,
                            False,
                            original_version,
                            None,
                            False,
                            error_msg,
                        )
                        progress.advance(task)
                        continue

                # Step 2: Extract window names from the working file
                window_data = extract_window_names_robust(migrated_file)
                most_active_window = get_most_active_window(window_data)

                if not most_active_window:
                    error_msg = "No window data found - unable to determine game/application"
                    console.print(f"[red]No window data found for {mcap_file}[/red]")

                    # Log as failed dataset
                    log_to_database(
                        db_path,
                        user_email,
                        base_name,
                        None,
                        0,
                        None,
                        False,
                        original_version,
                        migrated_version,
                        False,
                        error_msg,
                    )
                    progress.advance(task)
                    continue

                # Step 3: Check if we need to sanitize (more than 1 window)
                needs_sanitization = len(window_data) > 1
                final_analysis_file = migrated_file

                if needs_sanitization:
                    # Create sanitized version, always use base_name_mig.mcap
                    sanitized_path = mcap_file.parent / f"{base_name}_mig.mcap"

                    sanitize_result = sanitize_mcap_file(migrated_file, most_active_window, sanitized_path)

                    if sanitize_result["success"]:
                        final_analysis_file = sanitized_path
                        console.print(f"[green]Sanitized: {sanitized_path}[/green]")
                    else:
                        console.print(f"[yellow]Sanitization failed for {mcap_file.name}[/yellow]")

                # Step 4: Get duration and detect gaps from the final processed file
                try:
                    duration = get_video_duration(final_analysis_file)
                    gaps = detect_activity_gaps(final_analysis_file)

                    # Step 5: Log to database (use original file info but analysis from processed file)
                    log_to_database(
                        db_path,
                        user_email,
                        base_name,
                        most_active_window,
                        duration,
                        gaps,
                        needs_sanitization,
                        original_version,
                        migrated_version,
                        True,
                        None,
                    )

                except Exception as analysis_error:
                    error_msg = f"Analysis failed: {str(analysis_error)}"
                    console.print(f"[red]Analysis failed for {mcap_file.name}: {analysis_error}[/red]")

                    # Log as failed dataset
                    log_to_database(
                        db_path,
                        user_email,
                        base_name,
                        most_active_window,
                        0,
                        None,
                        needs_sanitization,
                        original_version,
                        migrated_version,
                        False,
                        error_msg,
                    )

                progress.advance(task)

            except Exception as e:
                error_msg = f"Processing failed: {str(e)}"
                console.print(f"[red]Error processing {mcap_file}: {e}[/red]")

                # Log as failed dataset
                log_to_database(
                    db_path, user_email, base_name, None, 0, None, False, original_version, None, False, error_msg
                )
                progress.advance(task)

    console.print(f"[green]Analysis complete! Results saved to {db_path}[/green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze OWA game datasets")
    parser.add_argument(
        "--dataset-root", default="/mnt/raid12/datasets/owa_game_dataset", help="Root directory of the dataset"
    )
    parser.add_argument(
        "--db-path",
        default="/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db",
        help="Path to SQLite database file",
    )
    parser.add_argument("--target-version", default="0.5.6", help="Target MCAP version for migration (default: 0.5.6)")

    args = parser.parse_args()
    analyze_datasets(args.dataset_root, args.db_path, args.target_version)
