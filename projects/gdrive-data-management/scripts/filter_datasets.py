#!/usr/bin/env python3
"""
Filter OWA game datasets based on database criteria and copy to new directory.

This script:
1. Queries the analyzed dataset database for files meeting criteria (efficiency, banned users, version)
2. Copies MCAP files that meet the criteria
3. Downscales paired MKV files to 854x480 (480p) resolution
4. Maintains original directory structure
"""

import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from multiprocessing import Pool

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.append("reference")


# Import subprocess for ffmpeg processing


@dataclass
class FilterConfig:
    """Configuration for dataset filtering."""

    banned_user_email: str = "parktj93@gmail.com"
    target_version: str = "0.5.6"
    min_orig_version: str = "0.5.5"
    min_efficiency_percent: float = 70.0
    min_duration_seconds: float = 300.0  # 5 minutes minimum duration
    banned_games: List[str] = None  # List of banned game names
    source_root: str = "/mnt/raid12/datasets/owa_game_dataset"
    dest_root: str = "/mnt/raid12/datasets/owa_game_dataset_filtered"
    db_path: str = "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db"
    filtered_db_path: str = None  # Path to filtered dataset database (auto-generated)
    num_workers: int = 4
    target_width: int = 854
    target_height: int = 480

    def __post_init__(self):
        """Initialize default banned games list and filtered DB path if not provided."""
        if self.banned_games is None:
            self.banned_games = ["Stardew Valley"]
        if self.filtered_db_path is None:
            self.filtered_db_path = str(Path(self.dest_root) / "filtered_datasets.db")


def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Get database connection with error handling."""
    try:
        if not Path(db_path).exists():
            print(f"Database not found: {db_path}")
            return None
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def init_filtered_database(db_path: str):
    """Initialize the filtered datasets database."""
    try:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filtered_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                file_name TEXT NOT NULL,
                game_name TEXT,
                duration_seconds REAL,
                total_gap_duration REAL,
                efficiency REAL,
                migrated_version TEXT,
                original_version TEXT,
                source_mcap_path TEXT NOT NULL,
                dest_mcap_path TEXT NOT NULL,
                source_mkv_path TEXT,
                dest_mkv_path TEXT,
                mcap_processed BOOLEAN DEFAULT FALSE,
                mkv_processed BOOLEAN DEFAULT FALSE,
                processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                target_width INTEGER,
                target_height INTEGER,
                UNIQUE(user_email, file_name)
            )
        """)

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error initializing filtered database: {e}")
        return False


def get_filtered_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Get filtered database connection, creating it if it doesn't exist."""
    try:
        if not Path(db_path).exists():
            if not init_filtered_database(db_path):
                return None

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        print(f"Filtered database connection error: {e}")
        return None


def is_already_processed(config: FilterConfig, user_email: str, file_name: str) -> bool:
    """Check if a file has already been processed with correct resolution."""
    conn = get_filtered_db_connection(config.filtered_db_path)
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT mcap_processed, mkv_processed, dest_mcap_path, dest_mkv_path
            FROM filtered_datasets
            WHERE user_email = ? AND file_name = ?
        """,
            (user_email, file_name),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            mcap_processed = result["mcap_processed"]
            mkv_processed = result["mkv_processed"]
            dest_mkv_path = result["dest_mkv_path"]

            # Check MCAP processing
            if not mcap_processed:
                return False

            # If no MKV file, consider processed
            if not dest_mkv_path:
                return True

            # If MKV is marked as processed, verify it has correct resolution
            if mkv_processed:
                mkv_path = Path(dest_mkv_path)
                if mkv_path.exists():
                    # Check if resolution matches target
                    if needs_reprocessing(mkv_path, config.target_width, config.target_height):
                        # Resolution doesn't match, needs reprocessing
                        return False
                    else:
                        # Resolution matches, truly processed
                        return True
                else:
                    # MKV file doesn't exist, needs processing
                    return False
            else:
                # MKV not processed
                return False

        return False
    except Exception as e:
        print(f"Error checking if file is processed: {e}")
        conn.close()
        return False


def record_processed_file(
    config: FilterConfig,
    dataset: Dict,
    mcap_file: Path,
    dest_mcap: Path,
    mkv_file: Optional[Path] = None,
    dest_mkv: Optional[Path] = None,
    mcap_success: bool = False,
    mkv_success: bool = False,
):
    """Record a processed file in the filtered database."""
    conn = get_filtered_db_connection(config.filtered_db_path)
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Insert or update the record
        cursor.execute(
            """
            INSERT OR REPLACE INTO filtered_datasets (
                user_email, file_name, game_name, duration_seconds, total_gap_duration,
                efficiency, migrated_version, original_version, source_mcap_path, dest_mcap_path,
                source_mkv_path, dest_mkv_path, mcap_processed, mkv_processed,
                target_width, target_height, processing_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                dataset["user_email"],
                dataset["file_name"],
                dataset["game_name"],
                dataset["duration_seconds"],
                dataset["total_gap_duration"],
                dataset["efficiency"],
                dataset["migrated_version"],
                dataset["original_version"],
                str(mcap_file),
                str(dest_mcap),
                str(mkv_file) if mkv_file else None,
                str(dest_mkv) if dest_mkv else None,
                mcap_success,
                mkv_success or (mkv_file is None),  # Consider MKV processed if no MKV file
                config.target_width,
                config.target_height,
            ),
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error recording processed file: {e}")
        conn.close()
        return False


def scan_and_populate_existing_files(config: FilterConfig, datasets: List[Dict], console: Console) -> int:
    """
    Scan the destination directory for already processed files and populate the database.

    This is useful when the database was deleted but files were already processed,
    to avoid reprocessing existing files.

    Returns the number of existing files found and recorded.
    """
    dest_path = Path(config.dest_root)
    if not dest_path.exists():
        return 0

    console.print("[bold]Scanning destination directory for existing processed files...[/bold]")

    existing_count = 0
    conn = get_filtered_db_connection(config.filtered_db_path)
    if not conn:
        console.print("[red]Could not connect to filtered database for scanning[/red]")
        return 0

    try:
        cursor = conn.cursor()

        # Create a lookup dictionary for datasets by user_email/file_name
        dataset_lookup = {}
        for dataset in datasets:
            # The dataset file_name from database doesn't have .mcap extension
            file_name_with_ext = dataset["file_name"]
            if not file_name_with_ext.endswith(".mcap"):
                file_name_with_ext = f"{dataset['file_name']}.mcap"
            key = (dataset["user_email"], file_name_with_ext)
            dataset_lookup[key] = dataset

        # Scan for existing MCAP files in destination
        mcap_files = list(dest_path.glob("**/*.mcap"))
        console.print(f"Found {len(mcap_files)} MCAP files in destination directory")

        for dest_mcap in mcap_files:
            try:
                # Extract user_email and file_name from path
                # Expected structure: dest_root/user_email/.../filename_mig.mcap or filename.mcap
                relative_path = dest_mcap.relative_to(dest_path)
                path_parts = relative_path.parts

                if len(path_parts) < 2:
                    continue

                user_email = path_parts[0]
                file_name = dest_mcap.stem

                # Handle _mig suffix - the database stores the original name without _mig
                original_file_name = file_name
                if file_name.endswith("_mig"):
                    original_file_name = file_name[:-4]  # Remove _mig suffix

                # Add .mcap extension for lookup
                file_name_with_ext = f"{original_file_name}.mcap"

                # Check if this file is in our datasets
                lookup_key = (user_email, file_name_with_ext)
                if lookup_key not in dataset_lookup:
                    # Also try with the full filename as stored in database
                    console.print(
                        f"[yellow]Could not find dataset entry for {user_email}/{file_name_with_ext}[/yellow]"
                    )
                    continue

                dataset = dataset_lookup[lookup_key]

                # Check if already in database
                cursor.execute(
                    """
                    SELECT id FROM filtered_datasets
                    WHERE user_email = ? AND file_name = ?
                """,
                    (user_email, file_name_with_ext),
                )

                if cursor.fetchone():
                    console.print(f"[blue]Already in database: {user_email}/{file_name_with_ext}[/blue]")
                    continue  # Already in database

                # Check for paired MKV file and if it's properly processed
                mkv_file = find_paired_mkv(dest_mcap)
                mkv_processed = False
                if mkv_file and mkv_file.exists():
                    # Check if MKV has correct resolution
                    mkv_processed = not needs_reprocessing(mkv_file, config.target_width, config.target_height)

                # Determine source paths (best guess)
                source_mcap = Path(config.source_root) / relative_path
                source_mkv = None
                if mkv_file:
                    mkv_relative = mkv_file.relative_to(dest_path)
                    source_mkv = Path(config.source_root) / mkv_relative

                # Insert record for existing file
                cursor.execute(
                    """
                    INSERT INTO filtered_datasets (
                        user_email, file_name, game_name, duration_seconds, total_gap_duration,
                        efficiency, migrated_version, original_version, source_mcap_path, dest_mcap_path,
                        source_mkv_path, dest_mkv_path, mcap_processed, mkv_processed,
                        target_width, target_height, processing_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (
                        dataset["user_email"],
                        dataset["file_name"],
                        dataset["game_name"],
                        dataset["duration_seconds"],
                        dataset["total_gap_duration"],
                        dataset["efficiency"],
                        dataset["migrated_version"],
                        dataset["original_version"],
                        str(source_mcap),
                        str(dest_mcap),
                        str(source_mkv) if source_mkv else None,
                        str(mkv_file) if mkv_file else None,
                        True,  # MCAP is processed (file exists)
                        mkv_processed,  # MKV processed if file exists
                        config.target_width,
                        config.target_height,
                    ),
                )

                existing_count += 1
                console.print(f"[blue]Recorded existing file: {user_email}/{file_name_with_ext}[/blue]")

            except Exception as e:
                console.print(f"[yellow]Warning: Could not process existing file {dest_mcap}: {e}[/yellow]")
                continue

        conn.commit()
        conn.close()

        if existing_count > 0:
            console.print(f"[green]Successfully recorded {existing_count} existing processed files[/green]")
        else:
            console.print("[blue]No existing processed files found to record[/blue]")

        return existing_count

    except Exception as e:
        console.print(f"[red]Error scanning existing files: {e}[/red]")
        conn.close()
        return 0


def calculate_efficiency(duration_seconds: float, total_gap_duration: float) -> float:
    """Calculate efficiency percentage based on duration and gaps."""
    if duration_seconds <= 0:
        return 0.0
    accepted_duration = duration_seconds - total_gap_duration
    return (accepted_duration / duration_seconds) * 100.0


def compare_version(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    if not version1 or not version2:
        return 0

    try:
        # Split versions into parts and convert to integers
        v1_parts = [int(x) for x in version1.split(".")]
        v2_parts = [int(x) for x in version2.split(".")]

        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))

        # Compare each part
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1

        return 0
    except (ValueError, AttributeError):
        # Fallback to string comparison if parsing fails
        return -1 if version1 < version2 else (1 if version1 > version2 else 0)


def query_filtered_datasets(config: FilterConfig, console: Console) -> List[Dict]:
    """
    Query the database for datasets meeting the filtering criteria.

    Returns list of dataset records with file paths and metadata.
    """
    conn = get_db_connection(config.db_path)
    if not conn:
        return []

    try:
        cursor = conn.cursor()

        # Query for datasets meeting all criteria
        query = """
        SELECT user_email, file_name, game_name, duration_seconds, total_gap_duration,
               migrated_version, original_version, available
        FROM dataset_analysis
        WHERE available = 1
          AND migrated_version = ?
          AND user_email != ?
          AND duration_seconds > 0
          AND game_name IS NOT NULL
          AND game_name != ''
          AND original_version IS NOT NULL
        """

        cursor.execute(query, (config.target_version, config.banned_user_email))
        results = cursor.fetchall()

        # Filter by efficiency, original version, duration, and banned games
        filtered_results = []
        for row in results:
            efficiency = calculate_efficiency(row["duration_seconds"], row["total_gap_duration"] or 0)
            original_version = row["original_version"]
            duration_seconds = row["duration_seconds"]
            game_name = row["game_name"]

            # Check efficiency
            if efficiency < config.min_efficiency_percent:
                continue

            # Check original version (must be >= min_orig_version)
            if compare_version(original_version, config.min_orig_version) < 0:
                continue

            # Check minimum duration
            if duration_seconds < config.min_duration_seconds:
                continue

            # Check banned games
            if game_name in config.banned_games:
                continue

            filtered_results.append(
                {
                    "user_email": row["user_email"],
                    "file_name": row["file_name"],
                    "game_name": row["game_name"],
                    "duration_seconds": row["duration_seconds"],
                    "total_gap_duration": row["total_gap_duration"] or 0,
                    "efficiency": efficiency,
                    "migrated_version": row["migrated_version"],
                    "original_version": original_version,
                }
            )

        console.print(f"[green]Found {len(filtered_results)} datasets meeting criteria[/green]")
        console.print(
            f"[blue]Criteria: efficiency >= {config.min_efficiency_percent}%, migrated version = {config.target_version}, original version >= {config.min_orig_version}, duration >= {config.min_duration_seconds}s, excluding user {config.banned_user_email}, banned games: {config.banned_games}[/blue]"
        )

        return filtered_results

    except Exception as e:
        print(f"Error querying database: {e}")
        return []
    finally:
        conn.close()


def find_paired_mkv(mcap_file: Path) -> Optional[Path]:
    """Find the paired MKV file for a given MCAP file (migrated or original) in the same directory."""
    # Extract the base name from the MCAP file
    if "_mig" in mcap_file.stem:
        # For migrated MCAP: filename_mig.mcap -> filename.mkv
        base_name = mcap_file.stem.replace("_mig", "")
    else:
        # For original MCAP: filename.mcap -> filename.mkv
        base_name = mcap_file.stem

    # The MKV file should be in the same directory as the MCAP file
    # e.g., if MCAP is at path/to/filename_mig.mcap or path/to/filename.mcap,
    #      MKV should be at path/to/filename.mkv
    mkv_file = mcap_file.parent / f"{base_name}.mkv"

    if mkv_file.exists():
        return mkv_file

    return None


def copy_with_structure(source_file: Path, source_root: Path, dest_root: Path) -> Path:
    """Copy file maintaining directory structure relative to source root.

    For MCAP files, removes _mig suffix from filename if present.
    """
    relative_path = source_file.relative_to(source_root)

    # Remove _mig suffix from MCAP files
    if source_file.suffix == ".mcap" and source_file.stem.endswith("_mig"):
        # Create new filename without _mig suffix
        new_filename = source_file.stem[:-4] + source_file.suffix  # Remove _mig, keep .mcap
        dest_file = dest_root / relative_path.parent / new_filename
    else:
        dest_file = dest_root / relative_path

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    return dest_file


def get_video_resolution(video_path: Path) -> tuple[int, int] | None:
    """Get video resolution using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            resolution = result.stdout.strip()
            if "x" in resolution:
                width, height = resolution.split("x")
                return int(width), int(height)
    except (subprocess.TimeoutExpired, ValueError, Exception) as e:
        print(f"Error getting resolution for {video_path}: {e}")

    return None


def is_video_complete(video_path: Path) -> bool:
    """Check if video file is complete and not corrupted.

    Args:
        video_path: Path to the video file

    Returns:
        True if video is complete, False if corrupted/incomplete
    """
    try:
        # Use ffprobe to check if the file can be read and has valid duration
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            duration_str = result.stdout.strip()
            if duration_str and duration_str != "N/A" and duration_str != "":
                try:
                    duration = float(duration_str)
                    # Valid duration should be positive and reasonable (> 1 second)
                    return duration > 1.0
                except ValueError:
                    return False

        # If ffprobe failed or returned invalid data, file is likely incomplete
        return False
    except (subprocess.TimeoutExpired, Exception):
        return False


def needs_reprocessing(video_path: Path, target_width: int, target_height: int) -> bool:
    """Check if video needs reprocessing based on resolution and completeness.

    Args:
        video_path: Path to the video file
        target_width: Target width
        target_height: Target height

    Returns:
        True if video needs reprocessing, False otherwise
    """
    if not video_path.exists():
        return True  # File doesn't exist, needs processing

    # Check if file is complete and not corrupted
    if not is_video_complete(video_path):
        return True  # File is incomplete/corrupted, needs reprocessing

    current_resolution = get_video_resolution(video_path)
    if current_resolution is None:
        return True  # Can't determine resolution, assume needs processing

    current_width, current_height = current_resolution
    return current_width != target_width or current_height != target_height


def process_video_worker(video_task: Tuple[Path, Path, str, FilterConfig, Dict]) -> Tuple[bool, str, str, str, str]:
    """
    Worker function for multiprocessing video downscaling.

    Args:
        video_task: Tuple of (input_path, output_path, description, config, dataset)

    Returns:
        Tuple of (success, input_name, error_message, user_email, file_name)
    """
    input_path, output_path, _, config, dataset = video_task
    user_email = dataset["user_email"]
    file_name = dataset["file_name"]

    # Log start of processing (this will be captured by the main process)
    start_time = time.time()

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # FFmpeg command for downscaling to target resolution
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vsync",
            "1",
            "-filter:v",
            f"fps=60,scale={config.target_width}:{config.target_height}",
            "-c:v",
            "libx264",
            "-x264-params",
            "keyint=30:no-scenecut=1:bframes=0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1000",
            "-c:s",
            "copy",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=sys.stdout,  # Inherit stdout
            stderr=sys.stderr,  # Inherit stderr
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        processing_time = time.time() - start_time
        if result.returncode == 0:
            return True, input_path.name, f"Completed in {processing_time:.1f}s", user_email, file_name
        else:
            # Extract meaningful error from stderr
            stderr_text = result.stderr.strip()

            # Look for common FFmpeg error patterns
            error_lines = stderr_text.split("\n")
            meaningful_errors = []

            for line in error_lines:
                line = line.strip()
                # Skip version/build info and look for actual errors
                if any(
                    keyword in line.lower()
                    for keyword in ["error", "invalid", "failed", "cannot", "unable", "not found"]
                ):
                    if not any(skip in line.lower() for skip in ["ffmpeg version", "built with", "configuration"]):
                        meaningful_errors.append(line)

            if meaningful_errors:
                error_msg = "; ".join(meaningful_errors[:3])  # Take first 3 meaningful errors
            else:
                # Fallback to last few lines if no specific errors found
                error_msg = "; ".join(error_lines[-3:]) if error_lines else "Unknown FFmpeg error"

            # Limit total length but allow more than 200 chars
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."

            return False, input_path.name, f"FFmpeg error: {error_msg}", user_email, file_name

    except subprocess.TimeoutExpired:
        processing_time = time.time() - start_time
        return False, input_path.name, f"Timeout after {processing_time:.1f}s (>1 hour)", user_email, file_name
    except Exception as e:
        processing_time = time.time() - start_time
        return False, input_path.name, f"Error after {processing_time:.1f}s: {str(e)}", user_email, file_name


def downscale_video(input_mkv: Path, output_mkv: Path, target_width: int = 854, target_height: int = 480) -> bool:
    """
    Downscale video to target resolution using ffmpeg.

    Uses the command: ffmpeg -i input.mkv -vsync 1 -filter:v "fps=60,scale=854:480"
    -c:v libx264 -x264-params keyint=30:no-scenecut=1:bframes=0 -c:a aac -b:a 192k
    -af aresample=async=1000 -c:s copy output.mkv

    Args:
        input_mkv: Path to input MKV file
        output_mkv: Path to output MKV file
        target_width: Target width (default: 854)
        target_height: Target height (default: 480)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory
        output_mkv.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command based on the reference
        cmd = [
            "ffmpeg",
            "-i",
            str(input_mkv),
            "-vsync",
            "1",
            "-filter:v",
            f"fps=60,scale={target_width}:{target_height}",
            "-c:v",
            "libx264",
            "-x264-params",
            "keyint=30:no-scenecut=1:bframes=0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1000",
            "-c:s",
            "copy",
            "-y",  # Overwrite output file if it exists
            str(output_mkv),
        ]

        # Run ffmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for video processing
        )

        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error processing {input_mkv.name}:")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"FFmpeg timeout processing {input_mkv.name} (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"Error downscaling video {input_mkv}: {e}")
        return False


def filter_datasets(
    banned_user: str = "parktj93@gmail.com",
    target_version: str = "0.5.6",
    min_orig_version: str = "0.5.5",
    min_efficiency: float = 70.0,
    min_duration: float = 300.0,
    banned_games: List[str] = None,
    source_root: str = "/mnt/raid12/datasets/owa_game_dataset",
    dest_root: str = "/mnt/raid12/datasets/owa_game_dataset_filtered",
    db_path: str = "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db",
    num_workers: int = 4,
    target_width: int = 854,
    target_height: int = 480,
    rescan_existing: bool = False,
):
    """Main function to filter datasets based on database criteria."""
    console = Console()

    # Create configuration
    config = FilterConfig(
        banned_user_email=banned_user,
        target_version=target_version,
        min_orig_version=min_orig_version,
        min_efficiency_percent=min_efficiency,
        min_duration_seconds=min_duration,
        banned_games=banned_games if banned_games is not None else ["Stardew Valley"],
        source_root=source_root,
        dest_root=dest_root,
        db_path=db_path,
        num_workers=num_workers,
        target_width=target_width,
        target_height=target_height,
    )

    console.print("[bold blue]OWA Dataset Filtering[/bold blue]")
    console.print("Configuration:")
    console.print(f"  Banned user: {config.banned_user_email}")
    console.print(f"  Target version (migrated): {config.target_version}")
    console.print(f"  Min original version: {config.min_orig_version}")
    console.print(f"  Min efficiency: {config.min_efficiency_percent}%")
    console.print(f"  Min duration: {config.min_duration_seconds}s ({config.min_duration_seconds / 60:.1f} minutes)")
    console.print(f"  Banned games: {config.banned_games}")
    console.print(f"  Source root: {config.source_root}")
    console.print(f"  Destination root: {config.dest_root}")
    console.print(f"  Database: {config.db_path}")
    console.print(f"  Workers: {config.num_workers}")
    console.print()

    source_path = Path(source_root)
    dest_path = Path(dest_root)

    if not source_path.exists():
        console.print(f"[red]Source directory not found: {source_path}[/red]")
        return

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Destination directory: {dest_path}[/green]")

    # Step 1: Query filtered datasets from database
    console.print("[bold]Step 1: Querying filtered datasets from database...[/bold]")
    datasets = query_filtered_datasets(config, console)

    if not datasets:
        console.print("[red]No datasets found meeting criteria. Exiting.[/red]")
        return

    console.print(f"[green]Found {len(datasets)} datasets meeting criteria[/green]")
    console.print(
        f"[blue]Criteria: efficiency >= {config.min_efficiency_percent}%, version = {config.target_version}, duration >= {config.min_duration_seconds}s, excluding user {config.banned_user_email}, banned games: {config.banned_games}[/blue]"
    )

    # Step 2: Initialize filtered database
    console.print("[bold]Step 2: Initializing filtered database...[/bold]")
    db_existed = Path(config.filtered_db_path).exists()
    if not init_filtered_database(config.filtered_db_path):
        console.print(f"[red]Failed to initialize filtered database: {config.filtered_db_path}[/red]")
        return
    console.print(f"[green]Filtered database ready: {config.filtered_db_path}[/green]")

    # Step 2.5: If database was newly created or rescan requested, scan for existing processed files
    if not db_existed or rescan_existing:
        if rescan_existing:
            console.print("[yellow]Forced rescan requested - scanning for existing processed files...[/yellow]")
        existing_count = scan_and_populate_existing_files(config, datasets, console)
        if existing_count > 0:
            console.print(f"[green]Populated database with {existing_count} existing processed files[/green]")
    else:
        console.print("[blue]Using existing filtered database (skipping scan)[/blue]")

    # Step 3: Check which files actually exist using glob patterns and filter already processed
    console.print("[bold]Step 3: Checking file availability and processing status...[/bold]")

    available_files = []
    missing_files = []
    already_processed = []

    for dataset in datasets:
        user_email = dataset["user_email"]
        file_name = dataset["file_name"]

        # Check if already processed (including resolution verification)
        if is_already_processed(config, user_email, file_name):
            already_processed.append(f"{user_email}/{file_name}")
            continue

        # First try to find {file_name}_mig.mcap at any depth under user_email directory
        # Pattern: user_email/**/{file_name}_mig.mcap
        mig_glob_pattern = f"{user_email}/**/{file_name}_mig.mcap"
        matching_mig_files = list(source_path.glob(mig_glob_pattern))

        if matching_mig_files:
            # Use the first migrated match if multiple files found
            mcap_path = matching_mig_files[0]
            available_files.append((mcap_path, dataset))
            if len(matching_mig_files) > 1:
                console.print(
                    f"[yellow]Multiple migrated matches for {file_name}, like as {matching_mig_files}, using: {mcap_path}[/yellow]"
                )
        else:
            # Fallback to original {file_name}.mcap if migrated version doesn't exist
            orig_glob_pattern = f"{user_email}/**/{file_name}.mcap"
            matching_orig_files = list(source_path.glob(orig_glob_pattern))

            if matching_orig_files:
                # Use the first original match if multiple files found
                mcap_path = matching_orig_files[0]
                available_files.append((mcap_path, dataset))
                console.print(f"[cyan]Using original MCAP for {file_name}: {mcap_path}[/cyan]")
                if len(matching_orig_files) > 1:
                    console.print(f"[yellow]Multiple original matches for {file_name}, using: {mcap_path}[/yellow]")
            else:
                missing_files.append(f"{user_email}/**/{file_name}_mig.mcap or {file_name}.mcap")

    console.print(f"[green]Available files to process: {len(available_files)}[/green]")
    if already_processed:
        console.print(f"[blue]Already processed files (skipped): {len(already_processed)}[/blue]")
        for processed in already_processed[:5]:  # Show first 5 processed files
            console.print(f"  [blue]Skipped: {processed}[/blue]")
        if len(already_processed) > 5:
            console.print(f"  [blue]... and {len(already_processed) - 5} more[/blue]")

    if missing_files:
        console.print(f"[yellow]Missing files: {len(missing_files)}[/yellow]")
        for missing in missing_files[:5]:  # Show first 5 missing files
            console.print(f"  [yellow]Missing: {missing}[/yellow]")
        if len(missing_files) > 5:
            console.print(f"  [yellow]... and {len(missing_files) - 5} more[/yellow]")

    if not available_files:
        if already_processed:
            console.print("[green]All eligible files have already been processed![/green]")
        else:
            console.print("[yellow]No available files found to process![/yellow]")
        return

    # Step 4: Copy MCAP files and collect video tasks
    console.print("[bold]Step 4: Copying MCAP files and preparing video tasks...[/bold]")

    successful_copies = 0
    failed_copies = 0
    missing_mkvs = 0
    video_tasks = []  # List of (input_mkv, output_mkv, description) tuples

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        copy_task = progress.add_task("Copying MCAP files...", total=len(available_files))

        for mcap_file, dataset in available_files:
            try:
                user_email = dataset["user_email"]
                file_name = dataset["file_name"]

                progress.update(copy_task, description=f"Processing {user_email}/{file_name}")

                # Create destination path maintaining directory structure
                # Copy as _mig.mcap to maintain the migrated file naming
                dest_mcap = copy_with_structure(mcap_file, source_path, dest_path)

                # Copy MCAP file
                mcap_success = False
                mkv_file = None
                dest_mkv_path = None

                try:
                    shutil.copy2(mcap_file, dest_mcap)
                    mcap_success = True
                    successful_copies += 1
                    console.print(f"[green]✓ Copied MCAP: {mcap_file.name}[/green]")
                    console.print(f"[dim]  From: {mcap_file}[/dim]")
                    console.print(f"[dim]  To: {dest_mcap}[/dim]")

                    # Find paired MKV file and check if it needs processing
                    mkv_file = find_paired_mkv(mcap_file)
                    if mkv_file:
                        # Create destination MKV path maintaining same directory structure as MCAP
                        # Get the relative path of the MKV file from source root
                        mkv_relative_path = mkv_file.relative_to(source_path)
                        dest_mkv_path = dest_path / mkv_relative_path

                        # Ensure destination directory exists
                        dest_mkv_path.parent.mkdir(parents=True, exist_ok=True)

                        # Check if video needs processing (doesn't exist, wrong resolution, or incomplete)
                        needs_processing = needs_reprocessing(dest_mkv_path, config.target_width, config.target_height)

                        if needs_processing:
                            # Add to video processing tasks
                            video_tasks.append((mkv_file, dest_mkv_path, f"{user_email}/{file_name}", config, dataset))
                            if dest_mkv_path.exists():
                                # Determine why it needs reprocessing
                                is_complete = is_video_complete(dest_mkv_path)
                                current_res = get_video_resolution(dest_mkv_path)

                                if not is_complete:
                                    console.print(
                                        f"[cyan]→ Found paired MKV: {mkv_file.name} (needs reprocessing: incomplete/corrupted file)[/cyan]"
                                    )
                                elif current_res:
                                    console.print(
                                        f"[cyan]→ Found paired MKV: {mkv_file.name} (needs reprocessing: {current_res[0]}x{current_res[1]} → {config.target_width}x{config.target_height})[/cyan]"
                                    )
                                else:
                                    console.print(
                                        f"[cyan]→ Found paired MKV: {mkv_file.name} (needs reprocessing: invalid resolution)[/cyan]"
                                    )
                            else:
                                console.print(f"[cyan]→ Found paired MKV: {mkv_file.name} (new file)[/cyan]")
                        else:
                            console.print(
                                f"[green]→ Found paired MKV: {mkv_file.name} (already processed at {config.target_width}x{config.target_height})[/green]"
                            )
                    else:
                        missing_mkvs += 1
                        console.print(f"[yellow]No MKV found for {mcap_file.name}[/yellow]")

                except Exception as e:
                    failed_copies += 1
                    console.print(f"[red]Copy failed for {mcap_file.name}: {e}[/red]")

                # Determine MKV processing status
                mkv_already_processed = False
                if mkv_file and dest_mkv_path:
                    # Check if MKV is already processed with correct resolution
                    mkv_already_processed = not needs_reprocessing(
                        dest_mkv_path, config.target_width, config.target_height
                    )

                # Record the processing status in filtered database
                record_processed_file(
                    config, dataset, mcap_file, dest_mcap, mkv_file, dest_mkv_path, mcap_success, mkv_already_processed
                )

                progress.advance(copy_task)

            except Exception as e:
                console.print(f"[red]Error processing {mcap_file}: {e}[/red]")
                failed_copies += 1
                progress.advance(copy_task)

    # Step 5: Process videos in parallel
    processed_mkvs = 0
    failed_mkvs = 0

    if video_tasks:
        # Sort video tasks by original file size for better batch processing
        console.print(
            f"\n[bold]Step 5: Processing {len(video_tasks)} videos with {config.num_workers} workers...[/bold]"
        )
        console.print("[dim]Sorting videos by file size for optimal batch processing...[/dim]")

        def get_video_size(task):
            """Get video file size for sorting."""
            input_mkv, _, _, _, _ = task
            try:
                return input_mkv.stat().st_size if input_mkv.exists() else 0
            except Exception:
                return 0

        # Sort by file size (smallest to largest)
        video_tasks.sort(key=get_video_size)

        # Show size distribution
        if video_tasks:
            sizes = [get_video_size(task) / (1024 * 1024) for task in video_tasks]  # Convert to MB
            min_size, max_size = min(sizes), max(sizes)
            avg_size = sum(sizes) / len(sizes)
            console.print(f"[dim]Video sizes: {min_size:.1f}MB - {max_size:.1f}MB (avg: {avg_size:.1f}MB)[/dim]")

        console.print(f"[dim]Target resolution: {config.target_width}x{config.target_height}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            video_task = progress.add_task("Processing videos...", total=len(video_tasks))

            # Process videos in batches for better progress visibility
            batch_size = config.num_workers
            all_results = []

            for i in range(0, len(video_tasks), batch_size):
                batch = video_tasks[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(video_tasks) + batch_size - 1) // batch_size

                # Show which videos are starting in this batch
                console.print(
                    f"\n[bold blue]Batch {batch_num}/{total_batches} - Starting {len(batch)} videos:[/bold blue]"
                )
                for j, (input_mkv, output_mkv, _, _, task_dataset) in enumerate(batch):
                    video_size = input_mkv.stat().st_size / (1024 * 1024) if input_mkv.exists() else 0  # MB
                    console.print(
                        f"  {i + j + 1:2d}. {task_dataset['user_email']}/{input_mkv.name} ({video_size:.1f}MB)"
                    )

                # Process this batch
                with Pool(processes=config.num_workers) as pool:
                    batch_results = pool.map(process_video_worker, batch)

                # Process batch results immediately for real-time feedback
                for success, video_name, error_msg, user_email, file_name in batch_results:
                    # Find the corresponding task to get full paths and dataset
                    input_path = None
                    output_path = None
                    dataset = None

                    for input_mkv, output_mkv, _, _, task_dataset in batch:
                        if input_mkv.name == video_name:
                            input_path = input_mkv
                            output_path = output_mkv
                            dataset = task_dataset
                            break

                    if success:
                        processed_mkvs += 1
                        console.print(
                            f"[green]✓ [{processed_mkvs}/{len(video_tasks)}] Successfully processed: {user_email}/{video_name}[/green]"
                        )
                        if input_path and output_path:
                            # Get file sizes for comparison
                            original_size = input_path.stat().st_size / (1024 * 1024)  # MB
                            processed_size = (
                                output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
                            )  # MB
                            compression_ratio = (1 - processed_size / original_size) * 100 if original_size > 0 else 0
                            console.print(
                                f"[dim]  Original: {original_size:.1f}MB → Processed: {processed_size:.1f}MB ({compression_ratio:.1f}% reduction)[/dim]"
                            )
                            console.print(
                                f"[dim]  Resolution: {config.target_width}x{config.target_height} | {error_msg}[/dim]"
                            )

                        # Update database to mark MKV as processed
                        if dataset:
                            conn = get_filtered_db_connection(config.filtered_db_path)
                            if conn:
                                try:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        """
                                        UPDATE filtered_datasets
                                        SET mkv_processed = TRUE
                                        WHERE user_email = ? AND file_name = ?
                                    """,
                                        (user_email, file_name),
                                    )
                                    conn.commit()
                                    conn.close()
                                except Exception as e:
                                    console.print(
                                        f"[yellow]Warning: Could not update MKV status in database: {e}[/yellow]"
                                    )
                                    conn.close()
                    else:
                        failed_mkvs += 1
                        console.print(
                            f"[red]✗ [{processed_mkvs + failed_mkvs}/{len(video_tasks)}] Failed to process: {user_email}/{video_name}[/red]"
                        )
                        console.print(f"[red]  Error: {error_msg}[/red]")
                        console.print(
                            "[yellow]  → No MKV will be created for this file (can be retried later)[/yellow]"
                        )

                        # Do NOT mark MKV as processed in database - leave it for retry
                        # The file will remain available for reprocessing in future runs

                    # Update progress bar
                    progress.advance(video_task)

                all_results.extend(batch_results)
    else:
        console.print("[yellow]No videos to process[/yellow]")

    # Final summary
    console.print("\n[bold]Processing Summary[/bold]")
    console.print(f"[green]MCAP files copied successfully: {successful_copies}[/green]")
    console.print(f"[red]MCAP files failed: {failed_copies}[/red]")
    console.print(f"[green]MKV files processed (downscaled): {processed_mkvs}[/green]")
    console.print(f"[red]MKV files failed to downscale: {failed_mkvs}[/red]")
    console.print(f"[yellow]MKV files missing: {missing_mkvs}[/yellow]")

    if failed_mkvs > 0:
        console.print(
            f"[yellow]Note: {failed_mkvs} failed MKV files were left unprocessed and can be retried in future runs[/yellow]"
        )

    if successful_copies > 0:
        console.print(f"\n[green]✓ Filtering complete! {successful_copies} datasets copied to {dest_path}[/green]")
        console.print(
            f"[blue]Videos processed with {config.num_workers} workers, downscaled to {config.target_width}x{config.target_height} resolution[/blue]"
        )
        console.print(
            "[dim]Note: Videos are automatically reprocessed if they don't match the target resolution[/dim]"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter OWA game datasets based on database criteria")
    parser.add_argument("--banned-user", default="parktj93@gmail.com", help="User email to exclude from datasets")
    parser.add_argument("--target-version", default="0.5.6", help="Target MCAP migrated version to filter for")
    parser.add_argument("--min-orig-version", default="0.5.5", help="Minimum original version to filter for")
    parser.add_argument("--min-efficiency", type=float, default=70.0, help="Minimum efficiency percentage (0-100)")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=300.0,
        help="Minimum video duration in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--banned-games", nargs="*", default=["Stardew Valley"], help="List of game names to exclude from datasets"
    )
    parser.add_argument(
        "--source-root", default="/mnt/raid12/datasets/owa_game_dataset", help="Source dataset directory"
    )
    parser.add_argument(
        "--dest-root",
        default="/mnt/raid12/datasets/owa_game_dataset_filtered",
        help="Destination directory for filtered datasets",
    )
    parser.add_argument(
        "--db-path",
        default="/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db",
        help="Path to analysis database",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for video processing")
    parser.add_argument(
        "--target-width", type=int, default=854, help="Target video width for downscaling (default: 854)"
    )
    parser.add_argument(
        "--target-height", type=int, default=480, help="Target video height for downscaling (default: 480)"
    )
    parser.add_argument(
        "--rescan-existing",
        action="store_true",
        help="Force rescan of destination directory for existing processed files",
    )

    args = parser.parse_args()
    filter_datasets(
        args.banned_user,
        args.target_version,
        args.min_orig_version,
        args.min_efficiency,
        args.min_duration,
        args.banned_games,
        args.source_root,
        args.dest_root,
        args.db_path,
        args.num_workers,
        args.target_width,
        args.target_height,
        args.rescan_existing,
    )
