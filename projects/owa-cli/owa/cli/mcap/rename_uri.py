"""
MCAP file URI renaming command for screen topic events.

This module provides functionality to rename URIs in mediaref fields of screen topic events
in MCAP files, with automatic backup and rollback capabilities for data safety.
"""

import shutil
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core.utils.backup import BackupContext
from owa.core.utils.tempfile import NamedTemporaryFile
from owa.msgs.desktop.screen import MediaRef

from ..console import console


def _is_modifiable_screen_message(mcap_msg) -> bool:
    """Check if an MCAP message is a screen message with a modifiable URI."""
    message = mcap_msg.decoded
    return (
        mcap_msg.topic == "screen"
        and hasattr(message, "media_ref")
        and message.media_ref
        and hasattr(message.media_ref, "uri")
    )


def rename_uri_in_mcap_file(
    file_path: Path,
    new_uri: str,
    console: Console,
    dry_run: bool = False,
    verbose: bool = False,
    keep_backup: bool = True,
) -> dict:
    """
    Rename URIs in mediaref fields of screen topic events in a single MCAP file.

    Args:
        file_path: Path to the MCAP file to process
        new_uri: The new URI to use for all screen topic events
        console: Rich console for output
        dry_run: If True, only analyze without making changes
        verbose: If True, show detailed information
        keep_backup: Whether to keep backup files after processing

    Returns:
        Dictionary with processing results

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not an MCAP file
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix != ".mcap":
        raise ValueError(f"File must be an MCAP file: {file_path}")

    total_messages = 0
    screen_messages = 0
    modified_messages = 0
    original_uris = set()

    # First pass: analyze the file
    with OWAMcapReader(file_path) as reader:
        for mcap_msg in reader.iter_messages():
            total_messages += 1

            if mcap_msg.topic == "screen":
                screen_messages += 1

                # Check if message has media_ref with URI
                if _is_modifiable_screen_message(mcap_msg):
                    current_uri = mcap_msg.decoded.media_ref.uri
                    original_uris.add(current_uri)

                    # All screen messages with media_ref will be modified
                    modified_messages += 1

    if verbose:
        console.print(f"[blue]Analysis for {file_path}:[/blue]")
        console.print(f"  Total messages: {total_messages}")
        console.print(f"  Screen messages: {screen_messages}")
        console.print(f"  Messages to modify: {modified_messages}")
        if original_uris:
            console.print(f"  Original URIs: {', '.join(sorted(original_uris))}")

    if dry_run:
        return {
            "file_path": file_path,
            "total_messages": total_messages,
            "screen_messages": screen_messages,
            "modified_messages": modified_messages,
            "original_uris": list(original_uris),
            "success": True,
        }

    # If no modifications needed, skip processing
    if modified_messages == 0:
        return {
            "file_path": file_path,
            "total_messages": total_messages,
            "screen_messages": screen_messages,
            "modified_messages": 0,
            "original_uris": [],
            "success": True,
        }

    # Use combined context managers for safe file operations
    with (
        BackupContext(file_path, console=console, keep_backup=keep_backup) as backup_ctx,
        NamedTemporaryFile(mode="wb", suffix=".mcap") as temp_file,
    ):
        temp_path = Path(temp_file.name)
        # Second pass: write modified file
        with OWAMcapReader(file_path) as reader, OWAMcapWriter(temp_path) as writer:
            for mcap_msg in reader.iter_messages():
                message = mcap_msg.decoded

                # Modify all screen messages with media_ref
                if _is_modifiable_screen_message(mcap_msg):
                    # Create a new MediaRef with the updated URI
                    new_media_ref = MediaRef(uri=new_uri, pts_ns=message.media_ref.pts_ns)

                    # Update the message with the new media_ref
                    message.media_ref = new_media_ref

                # Write the message (modified or original)
                writer.write_message(message, topic=mcap_msg.topic, timestamp=mcap_msg.timestamp)

        # Replace original file with modified version (after writer is closed)
        shutil.move(str(temp_path), file_path)

        return {
            "file_path": file_path,
            "total_messages": total_messages,
            "screen_messages": screen_messages,
            "modified_messages": modified_messages,
            "original_uris": list(original_uris),
            "backup_path": backup_ctx.backup_path,
            "success": True,
        }


def rename_uri(
    files: Annotated[List[Path], typer.Argument(help="MCAP files to process (supports glob patterns)")],
    uri: Annotated[str, typer.Option("--uri", help="URI to use for all screen topic events")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be changed without making modifications")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed processing information")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
    keep_backups: Annotated[
        bool, typer.Option("--keep-backups/--no-backups", help="Keep backup files after processing")
    ] = True,
) -> None:
    """
    Rename URIs in mediaref fields of screen topic events in MCAP files.

    This command processes MCAP files to set all URIs in the mediaref fields
    of screen topic events to the specified new URI. This is useful for updating
    file paths or URLs when media files have been moved or renamed.

    Examples:
        owl mcap rename-uri recording.mcap --uri "new_video.mkv"
        owl mcap rename-uri *.mcap --uri "/new/path/video.mp4"
        owl mcap rename-uri data.mcap --uri "http://new.com/video" --dry-run
    """

    # Validate inputs
    if not files:
        console.print("[red]No files specified[/red]")
        raise typer.Exit(1)

    if not uri.strip():
        console.print("[red]URI cannot be empty[/red]")
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
    console.print("[bold blue]MCAP URI Renaming Tool[/bold blue]")
    console.print(f"URI: '{uri}'")
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
    successful_operations = 0
    failed_operations = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, file_path in enumerate(valid_files, 1):
            task = progress.add_task(f"Processing {file_path.name} ({i}/{len(valid_files)})", total=None)

            try:
                result = rename_uri_in_mcap_file(
                    file_path=file_path,
                    new_uri=uri,
                    console=console,
                    dry_run=dry_run,
                    verbose=verbose,
                    keep_backup=keep_backups,
                )

                if result["success"]:
                    successful_operations += 1

                    if not verbose:
                        modified = result["modified_messages"]
                        screen_total = result["screen_messages"]

                        if modified > 0:
                            console.print(
                                f"[green]✓ {file_path.name}: {modified} URIs updated out of {screen_total} screen messages[/green]"
                            )
                        else:
                            console.print(
                                f"[yellow]○ {file_path.name}: No matching URIs found in {screen_total} screen messages[/yellow]"
                            )
                else:
                    failed_operations += 1

            except Exception as e:
                console.print(f"[red]✗ {file_path.name}: {e}[/red]")
                failed_operations += 1

            progress.remove_task(task)

    # Final summary
    console.print(f"\n[bold]URI Renaming {'Analysis' if dry_run else 'Complete'}[/bold]")
    console.print(f"[green]Successful: {successful_operations}[/green]")
    console.print(f"[red]Failed: {failed_operations}[/red]")

    if failed_operations > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(rename_uri)
