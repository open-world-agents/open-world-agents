#!/usr/bin/env python3
"""
Process filtered datasets to create event datasets.

This script:
1. Globs all MCAP files from the filtered dataset directory (or original dataset directory)
2. Applies additional filtering based on database criteria (efficiency, banned users, version)
3. Groups them by game name (extracted from MCAP file analysis)
4. Processes each game's MCAP files into event datasets using the existing event processing script
5. Saves event datasets to /mnt/raid12/datasets/owa_game_dataset_filtered/{game_name}/{game_name}-event

Note: This script can apply different filtering strategies than filter_datasets.py, allowing for
flexible dataset processing with varying criteria.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add the open-world-agents project to the path to import the event processing script
sys.path.append("/mnt/home/jyjung/closed-world-agents/open-world-agents/projects/owa-data/scripts")

from mcap_owa.highlevel import OWAMcapReader

# Import the event processing functionality
import subprocess
import tempfile

console = Console()
app = typer.Typer(add_completion=False)


@dataclass
class ProcessConfig:
    """Configuration for dataset processing."""

    banned_user_email: str = "parktj93@gmail.com"
    target_version: str = "0.5.6"
    min_orig_version: str = "0.5.5"
    min_efficiency_percent: float = 70.0
    min_duration_seconds: float = 300.0  # 5 minutes minimum duration
    banned_games: List[str] = None  # List of banned game names
    filtered_root: str = "/mnt/raid12/datasets/owa_game_dataset_filtered"
    db_path: str = "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db"
    output_root: str = "/mnt/raid12/datasets/owa_game_dataset_filtered"
    time_shift_seconds: Optional[float] = None  # Time shift in seconds to add to action topics

    def __post_init__(self):
        """Initialize default banned games list if not provided."""
        if self.banned_games is None:
            self.banned_games = ["Stardew Valley"]


def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Get database connection with error handling."""
    try:
        if not Path(db_path).exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            return None
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        console.print(f"[red]Database connection error: {e}[/red]")
        return None


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


def query_filtered_datasets(config: ProcessConfig) -> List[Dict]:
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
        console.print(f"[red]Error querying database: {e}[/red]")
        return []
    finally:
        conn.close()


def extract_game_name_from_mcap(mcap_path: Path) -> Optional[str]:
    """
    Extract game name from MCAP file by reading window data.

    Returns the most common window title as game name.
    """
    try:
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
                            window_data[window_title] = 0
                        window_data[window_title] += 1

                except Exception:
                    continue

        # Return the most common window title
        if window_data:
            return max(window_data.keys(), key=lambda w: window_data[w])
        return None

    except Exception as e:
        console.print(f"[yellow]Could not extract game name from {mcap_path.name}: {e}[/yellow]")
        return None


def collect_mcap_paths_from_filtered(config: ProcessConfig) -> Dict[str, List[str]]:
    """
    Collect MCAP files from filtered directory, apply database filtering, and group by game name.

    This allows for additional filtering even when working with pre-filtered data,
    enabling different filtering strategies between filter_datasets.py and process_datasets.py.

    Returns dictionary mapping game names to lists of MCAP file paths.
    """
    filtered_root = Path(config.filtered_root)

    if not filtered_root.exists():
        console.print(f"[red]Filtered directory not found: {filtered_root}[/red]")
        console.print("[yellow]Trying to work with original dataset directory instead...[/yellow]")
        return {}

    # Step 1: Query database for datasets meeting criteria
    console.print("[bold]Querying database for datasets meeting criteria...[/bold]")
    datasets = query_filtered_datasets(config)

    if not datasets:
        console.print("[red]No datasets found meeting criteria in database.[/red]")
        return {}

    # Step 2: Check which files actually exist using glob patterns (migrated MCAP only)
    console.print("[bold]Checking file availability using glob patterns...[/bold]")

    game_mcap_paths = {}
    available_files = []
    missing_files = []

    for dataset in datasets:
        user_email = dataset["user_email"]
        file_name = dataset["file_name"]
        game_name = dataset["game_name"]

        # First try to find {file_name}_mig.mcap at any depth under user_email directory
        # Pattern: user_email/**/{file_name}_mig.mcap
        mig_glob_pattern = f"{user_email}/**/{file_name}_mig.mcap"
        matching_mig_files = list(filtered_root.glob(mig_glob_pattern))

        if matching_mig_files:
            # Use the first migrated match if multiple files found
            mcap_path = matching_mig_files[0]
            available_files.append((mcap_path, dataset))
            if game_name not in game_mcap_paths:
                game_mcap_paths[game_name] = []
            game_mcap_paths[game_name].append(str(mcap_path))
            if len(matching_mig_files) > 1:
                console.print(f"[yellow]Multiple migrated matches for {file_name}, using: {mcap_path}[/yellow]")
        else:
            # Fallback to original {file_name}.mcap if migrated version doesn't exist
            orig_glob_pattern = f"{user_email}/**/{file_name}.mcap"
            matching_orig_files = list(filtered_root.glob(orig_glob_pattern))

            if matching_orig_files:
                # Use the first original match if multiple files found
                mcap_path = matching_orig_files[0]
                available_files.append((mcap_path, dataset))
                if game_name not in game_mcap_paths:
                    game_mcap_paths[game_name] = []
                game_mcap_paths[game_name].append(str(mcap_path))
                console.print(f"[cyan]Using original MCAP for {file_name}: {mcap_path}[/cyan]")
                if len(matching_orig_files) > 1:
                    console.print(f"[yellow]Multiple original matches for {file_name}, using: {mcap_path}[/yellow]")
            else:
                missing_files.append(f"{user_email}/**/{file_name}_mig.mcap or {file_name}.mcap")

    console.print(f"[green]Available files: {len(available_files)}[/green]")
    if missing_files:
        console.print(f"[yellow]Missing files: {len(missing_files)}[/yellow]")
        for missing in missing_files[:5]:  # Show first 5 missing files
            console.print(f"  [yellow]Missing: {missing}[/yellow]")
        if len(missing_files) > 5:
            console.print(f"  [yellow]... and {len(missing_files) - 5} more[/yellow]")

    return game_mcap_paths


def process_game_event_dataset(
    game_name: str, mcap_files: List[str], config: ProcessConfig, num_workers: int = 4
) -> bool:
    """
    Process a single game's MCAP files into an event dataset.

    Files are split into train/test sets:
    - If only 1 file: skip the entire group
    - If 2+ files: smallest file goes to --test-files, others go to --train-files

    Args:
        game_name: Name of the game
        mcap_files: List of MCAP file paths for this game
        config: Processing configuration
        num_workers: Number of worker processes

    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip groups with only 1 file
        if len(mcap_files) < 2:
            console.print(
                f"[yellow]Skipping {game_name}: Only {len(mcap_files)} file(s), need at least 2 for train/test split[/yellow]"
            )
            return True  # Return True since this is expected behavior, not an error

        # Split files into train and test sets based on file size
        file_sizes = []
        for mcap_file in mcap_files:
            try:
                path_obj = Path(mcap_file)
                if path_obj.exists():
                    size_bytes = path_obj.stat().st_size
                    file_sizes.append((mcap_file, size_bytes))
                else:
                    console.print(f"[yellow]Warning: File not found: {mcap_file}[/yellow]")
            except Exception as e:
                console.print(f"[red]Error getting size for {mcap_file}: {e}[/red]")

        if len(file_sizes) < 2:
            console.print(f"[yellow]Skipping {game_name}: Only {len(file_sizes)} valid file(s) found[/yellow]")
            return True

        # Sort by file size (smallest first)
        file_sizes.sort(key=lambda x: x[1])

        # Smallest file goes to test, others go to train
        test_file = file_sizes[0][0]  # Smallest file
        train_files = [f[0] for f in file_sizes[1:]]  # All other files

        test_size_mb = file_sizes[0][1] / (1024 * 1024)
        train_sizes_mb = [f[1] / (1024 * 1024) for f in file_sizes[1:]]

        console.print(
            f"[blue]Processing {game_name}: {len(train_files)} train + 1 test file -> {config.output_root}/{game_name}[/blue]"
        )
        console.print(f"[dim]  Test file: {Path(test_file).name} ({test_size_mb:.1f}MB)[/dim]")
        console.print(
            f"[dim]  Train files: {len(train_files)} files ({min(train_sizes_mb):.1f}-{max(train_sizes_mb):.1f}MB)[/dim]"
        )

        # Create output directory
        output_dir = Path(config.output_root) / game_name / f"{game_name}-event"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Path to the modified event processing script
        script_path = "/mnt/home/jyjung/closed-world-agents/open-world-agents/projects/owa-data/scripts/01_raw_events_to_event_dataset.py"

        # Create temporary file with the list of MCAP files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            for mcap_file in mcap_files:
                temp_file.write(f"{mcap_file}\n")
            temp_file_path = temp_file.name

        try:
            # Build command to run the event processing script
            cmd = [
                "python",
                script_path,
                "--output-dir",
                str(output_dir),
                "--num-workers",
                str(num_workers),
                "--keep-topic",
                "screen",
                "--keep-topic",
                "keyboard",
                "--keep-topic",
                "mouse/raw",
                "--rate",
                "mouse/raw=20.0",
                "--rate",
                "screen=20.0",
                "--action-topic",
                "keyboard",
                "--action-topic",
                "mouse/raw",
            ]

            # Add time-shift parameter if specified
            if config.time_shift_seconds is not None:
                cmd.extend(["--time-shift", str(config.time_shift_seconds)])

            # Add train files
            for mcap_file in train_files:
                cmd.extend(["--train-files", mcap_file])

            # Add test file
            cmd.extend(["--test-files", test_file])

            console.print(f"[dim]Running: {' '.join(cmd[:10])}...[/dim]")

            # Run the event processing script
            result = subprocess.run(
                cmd,
                stdout=sys.stdout,  # Inherit stdout
                stderr=sys.stderr,  # Inherit stderr
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                console.print(f"[green]✓ Successfully processed {game_name}[/green]")
                return True
            else:
                console.print(f"[red]✗ Failed to process {game_name}[/red]")
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False

        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)

    except Exception as e:
        console.print(f"[red]✗ Exception processing {game_name}: {e}[/red]")
        return False


def process_all_game_datasets(game_mcap_paths: Dict[str, List[str]], config: ProcessConfig, num_workers: int = 4):
    """
    Process all games' MCAP files into event datasets.

    Args:
        game_mcap_paths: Dictionary mapping game names to lists of MCAP file paths
        config: Processing configuration
        num_workers: Number of worker processes
    """
    total_games = len(game_mcap_paths)
    successful_games = 0
    failed_games = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing games...", total=total_games)

        for game_name, mcap_files in game_mcap_paths.items():
            progress.update(task, description=f"Processing {game_name}")

            success = process_game_event_dataset(game_name, mcap_files, config, num_workers)

            if success:
                successful_games += 1
            else:
                failed_games.append(game_name)

            progress.advance(task)

    console.print("\n[bold]Processing Summary:[/bold]")
    console.print(f"[green]✓ Successful: {successful_games}/{total_games} games[/green]")

    if failed_games:
        console.print(f"[red]✗ Failed: {len(failed_games)} games[/red]")
        for game in failed_games:
            console.print(f"  [red]- {game}[/red]")


@app.command()
def main(
    banned_user: str = typer.Option("parktj93@gmail.com", "--banned-user", help="User email to exclude from datasets"),
    target_version: str = typer.Option("0.5.6", "--target-version", help="Target MCAP migrated version to filter for"),
    min_orig_version: str = typer.Option("0.5.5", "--min-orig-version", help="Minimum original version to filter for"),
    min_efficiency: float = typer.Option(70.0, "--min-efficiency", help="Minimum efficiency percentage (0-100)"),
    min_duration: float = typer.Option(
        300.0, "--min-duration", help="Minimum video duration in seconds (default: 300 = 5 minutes)"
    ),
    banned_games: List[str] = typer.Option(
        ["Stardew Valley"], "--banned-games", help="List of game names to exclude from datasets"
    ),
    filtered_root: str = typer.Option(
        "/mnt/raid12/datasets/owa_game_dataset_filtered", "--filtered-root", help="Root directory of filtered datasets"
    ),
    db_path: str = typer.Option(
        "/mnt/raid12/datasets/owa_game_dataset/dataset_analysis.db", "--db-path", help="Path to analysis database"
    ),
    output_root: str = typer.Option(
        "/mnt/raid12/datasets/owa_game_dataset_filtered",
        "--output-root",
        help="Root directory for output event datasets",
    ),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel worker processes"),
    time_shift: Optional[float] = typer.Option(
        None, "--time-shift", help="Time shift in seconds to add to action topics (keyboard, mouse/raw)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed without actually processing"),
):
    """Process datasets to create event datasets grouped by game.

    This script applies database filtering to select datasets meeting criteria (efficiency, version, banned users)
    and can work with either filtered or original dataset directories, allowing for flexible filtering strategies.
    """

    # Create configuration
    config = ProcessConfig(
        banned_user_email=banned_user,
        target_version=target_version,
        min_orig_version=min_orig_version,
        min_efficiency_percent=min_efficiency,
        min_duration_seconds=min_duration,
        banned_games=banned_games,
        filtered_root=filtered_root,
        db_path=db_path,
        output_root=output_root,
        time_shift_seconds=time_shift,
    )

    console.print("[bold blue]OWA Event Dataset Processing[/bold blue]")
    console.print("Configuration:")
    console.print(f"  Banned user: {config.banned_user_email}")
    console.print(f"  Target version (migrated): {config.target_version}")
    console.print(f"  Min original version: {config.min_orig_version}")
    console.print(f"  Min efficiency: {config.min_efficiency_percent}%")
    console.print(f"  Min duration: {config.min_duration_seconds}s ({config.min_duration_seconds / 60:.1f} minutes)")
    console.print(f"  Banned games: {config.banned_games}")
    console.print(f"  Filtered root: {config.filtered_root}")
    console.print(f"  Database: {config.db_path}")
    console.print(f"  Output root: {config.output_root}")
    console.print(
        f"  Time shift: {config.time_shift_seconds}s"
        if config.time_shift_seconds is not None
        else "  Time shift: None"
    )
    console.print()

    # Step 1: Apply database filtering and collect MCAP files
    console.print("[bold]Step 1: Applying database filtering and collecting MCAP files...[/bold]")
    game_mcap_paths = collect_mcap_paths_from_filtered(config)

    if not game_mcap_paths:
        console.print("[red]No MCAP files found meeting criteria.[/red]")
        raise typer.Exit(1)

    # Step 2: Display the collected paths for debug
    console.print("[bold]Step 2: MCAP files grouped by game:[/bold]")
    for game_name, mcap_paths in game_mcap_paths.items():
        console.print(f"[cyan]{game_name}[/cyan]: {len(mcap_paths)} files")
        for path in mcap_paths[:3]:  # Show first 3 paths
            console.print(f"  {path}")
        if len(mcap_paths) > 3:
            console.print(f"  ... and {len(mcap_paths) - 3} more")
        console.print()

    if dry_run:
        console.print("[yellow]Dry run mode - stopping here[/yellow]")
        return

    # Step 3: Process event datasets
    console.print("[bold]Step 3: Processing event datasets...[/bold]")
    process_all_game_datasets(game_mcap_paths, config, num_workers)

    console.print("[bold green]✓ All processing complete![/bold green]")


if __name__ == "__main__":
    app()
