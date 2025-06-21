"""
MCAP file sanitization command for filtering events based on window activation.

This module provides functionality to sanitize MCAP files by keeping only events
that occurred when a specific window was active, with automatic backup and rollback
capabilities for data safety.

TODO: implement configurable feature which blocks sanitize if message to be removed is more than specific ratio, say 10%.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.cli.mcap.backup_utils import BackupContext


def window_matches_target(window_title: str, target_window: str, exact_match: bool) -> bool:
    """
    Check if a window title matches the target window criteria.

    Args:
        window_title: The window title to check
        target_window: The target window name to match against
        exact_match: Whether to use exact matching or substring matching

    Returns:
        True if the window matches, False otherwise
    """
    if exact_match:
        return window_title == target_window
    else:
        return target_window.lower() in window_title.lower()


def sanitize_mcap_file(
    file_path: Path,
    keep_window: str,
    exact_match: bool,
    console: Console,
    dry_run: bool = False,
    verbose: bool = False,
    keep_backup: bool = True,
) -> dict:
    """
    Sanitize a single MCAP file by filtering events based on window activation.

    Args:
        file_path: Path to the MCAP file to sanitize
        keep_window: Window name to keep events for
        exact_match: Whether to use exact window name matching
        console: Rich console for output
        dry_run: If True, only analyze without making changes
        verbose: If True, show detailed information

    Returns:
        Dictionary with sanitization results
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix != ".mcap":
        raise ValueError(f"File must be an MCAP file: {file_path}")

    total_messages = 0
    kept_messages = 0
    window_messages = 0
    matching_windows = set()
    keep_current_events = False

    # First pass: analyze the file
    with OWAMcapReader(file_path) as reader:
        for mcap_msg in reader.iter_messages():
            total_messages += 1

            if mcap_msg.topic == "window":
                window_messages += 1
                # Handle both dict and object formats
                if hasattr(mcap_msg.decoded, "title"):
                    window_title = mcap_msg.decoded.title
                elif isinstance(mcap_msg.decoded, dict):
                    window_title = mcap_msg.decoded.get("title", "")
                else:
                    window_title = ""

                # Update current window state
                keep_current_events = window_matches_target(window_title, keep_window, exact_match)

                if keep_current_events:
                    matching_windows.add(window_title)

            if keep_current_events:
                kept_messages += 1

    if verbose:
        removed_messages = total_messages - kept_messages
        removal_percentage = (removed_messages / total_messages * 100) if total_messages > 0 else 0

        console.print(f"[blue]Analysis for {file_path}:[/blue]")
        console.print(f"  Total messages: {total_messages}")
        console.print(f"  Window messages: {window_messages}")
        console.print(f"  Messages to keep: {kept_messages}")
        console.print(f"  Messages to remove: {removed_messages} ({removal_percentage:.1f}%)")
        if matching_windows:
            console.print(f"  Matching windows: {', '.join(sorted(matching_windows))}")

    if dry_run:
        return {
            "file_path": file_path,
            "total_messages": total_messages,
            "kept_messages": kept_messages,
            "removed_messages": total_messages - kept_messages,
            "matching_windows": list(matching_windows),
            "success": True,
        }

    # Use combined context managers for safe file operations
    with (
        BackupContext(file_path, console, keep_backup=keep_backup) as backup_ctx,
        tempfile.NamedTemporaryFile(mode="wb", suffix=".mcap", delete=True) as temp_file,
    ):
        temp_path = Path(temp_file.name)

        # Second pass: write sanitized file
        keep_current_events = False

        with OWAMcapReader(file_path) as reader, OWAMcapWriter(temp_path) as writer:
            for mcap_msg in reader.iter_messages():
                if mcap_msg.topic == "window":
                    # Handle both dict and object formats
                    if hasattr(mcap_msg.decoded, "title"):
                        window_title = mcap_msg.decoded.title
                    elif isinstance(mcap_msg.decoded, dict):
                        window_title = mcap_msg.decoded.get("title", "")
                    else:
                        window_title = ""
                    keep_current_events = window_matches_target(window_title, keep_window, exact_match)

                # Write message if it should be kept
                if keep_current_events:
                    writer.write_message(
                        mcap_msg.topic,
                        mcap_msg.decoded,
                        log_time=mcap_msg.timestamp,
                        publish_time=mcap_msg.timestamp,
                    )

        # Replace original file with sanitized version
        shutil.move(temp_path, file_path)

        return {
            "file_path": file_path,
            "total_messages": total_messages,
            "kept_messages": kept_messages,
            "removed_messages": total_messages - kept_messages,
            "matching_windows": list(matching_windows),
            "backup_path": backup_ctx.backup_path,
            "success": True,
        }


def sanitize(
    files: Annotated[List[Path], typer.Argument(help="MCAP files to sanitize (supports glob patterns)")],
    keep_window: Annotated[str, typer.Option("--keep-window", help="Window name to keep events for")],
    exact: Annotated[
        bool, typer.Option("--exact/--substring", help="Use exact window name matching (default: substring)")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be changed without making modifications")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed sanitization information")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
    keep_backups: Annotated[
        bool, typer.Option("--keep-backups/--no-backups", help="Keep backup files after sanitization")
    ] = True,
) -> None:
    """
    Sanitize MCAP files by keeping only events when a specific window is active.

    This command filters MCAP files to retain only the events that occurred when
    the specified window was active, effectively removing data from other applications
    for privacy or focus purposes.

    Examples:
        owl mcap sanitize recording.mcap --keep-window "Notepad"
        owl mcap sanitize *.mcap --keep-window "Work App" --exact
        owl mcap sanitize data.mcap --keep-window "Browser" --dry-run
    """
    console = Console()

    # Validate inputs
    if not files:
        console.print("[red]No files specified[/red]")
        raise typer.Exit(1)

    if not keep_window.strip():
        console.print("[red]Window name cannot be empty[/red]")
        raise typer.Exit(1)

    # Filter for valid MCAP files
    valid_files = []
    for file_path in files:
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
        elif file_path.suffix != ".mcap":
            console.print(f"[yellow]Skipping non-MCAP file: {file_path}[/yellow]")
        else:
            valid_files.append(file_path)

    if not valid_files:
        console.print("[yellow]No valid MCAP files found[/yellow]")
        return

    # Display operation summary
    console.print("[bold blue]MCAP Sanitization Tool[/bold blue]")
    console.print(f"Window filter: '{keep_window}' ({'exact' if exact else 'substring'} match)")
    console.print(f"Files to process: {len(valid_files)}")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be modified[/yellow]")

    # Show confirmation prompt unless --yes is used
    if not dry_run and not yes:
        console.print("\n[yellow]This operation will modify the specified files.[/yellow]")
        console.print("[yellow]Backups will be created automatically.[/yellow]")

        confirm = typer.confirm("Do you want to continue?")
        if not confirm:
            console.print("[yellow]Operation cancelled[/yellow]")
            return

    # Process files
    successful_sanitizations = 0
    failed_sanitizations = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, file_path in enumerate(valid_files, 1):
            task = progress.add_task(f"Processing {file_path.name} ({i}/{len(valid_files)})", total=None)

            try:
                result = sanitize_mcap_file(
                    file_path=file_path,
                    keep_window=keep_window,
                    exact_match=exact,
                    console=console,
                    dry_run=dry_run,
                    verbose=verbose,
                    keep_backup=keep_backups,
                )

                if result["success"]:
                    successful_sanitizations += 1

                    if not verbose:
                        # Calculate percentage of messages removed
                        total = result["total_messages"]
                        removed = result["removed_messages"]
                        percentage = (removed / total * 100) if total > 0 else 0

                        console.print(
                            f"[green]✓ {file_path.name}: {removed} messages removed ({percentage:.1f}%)[/green]"
                        )
                else:
                    failed_sanitizations += 1

            except Exception as e:
                console.print(f"[red]✗ {file_path.name}: {e}[/red]")
                failed_sanitizations += 1

            progress.remove_task(task)

    # Final summary
    console.print(f"\n[bold]Sanitization {'Analysis' if dry_run else 'Complete'}[/bold]")
    console.print(f"[green]Successful: {successful_sanitizations}[/green]")
    console.print(f"[red]Failed: {failed_sanitizations}[/red]")

    if failed_sanitizations > 0:
        raise typer.Exit(1)
