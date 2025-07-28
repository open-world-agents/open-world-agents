#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.0.0",
#   "mcap>=1.0.0",
#   "easydict>=1.10",
#   "orjson>=3.8.0",
#   "typer>=0.12.0",
#   "numpy>=2.2.0",
#   "mcap-owa-support==0.5.5",
#   "owa-core==0.5.5",
#   "owa-msgs==0.5.5",
# ]
# [tool.uv]
# exclude-newer = "2025-07-28T12:00:00Z"
# ///
"""
MCAP Migrator: v0.5.1 → v0.5.5

Migrates RawMouseEvent messages from legacy dx/dy fields to new last_x/last_y/us_flags structure and fixes button_data offset error in previous versions.
Key changes:
- dx → last_x (direct mapping for relative movement)
- dy → last_y (direct mapping for relative movement)
- Add us_flags field with default MOUSE_MOVE_RELATIVE (0x0000)
- Move button_data value to button_flags and set button_data to 0 (due to offset error in previous versions)
- Preserve all other fields (device_handle, timestamp)

NOTE: These migrators are locked, separate script with separate dependency sets. DO NOT change the contents unless you know what you are doing.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.decoder import dict_decoder
from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core import MESSAGES

app = typer.Typer(help="MCAP Migration: v0.5.1 → v0.5.5")

# Version constants
FROM_VERSION = "0.5.1"
TO_VERSION = "0.5.5"


def migrate_raw_mouse_event_data(data: dict) -> dict:
    """
    Migrate RawMouseEvent message data from v0.5.1 to v0.5.5 format.

    Key transformations:
    - dx → last_x (direct mapping for relative movement)
    - dy → last_y (direct mapping for relative movement)
    - Add us_flags field with default MOUSE_MOVE_RELATIVE (0x0000)
    - Move button_data value to button_flags and set button_data to 0 (due to offset error in previous versions)
    """
    migrated_data = data.copy()

    # Check if this is legacy format with dx/dy fields
    if "dx" in migrated_data and "dy" in migrated_data:
        # Map dx/dy to last_x/last_y
        migrated_data["last_x"] = migrated_data.pop("dx")
        migrated_data["last_y"] = migrated_data.pop("dy")

        # Add us_flags with default relative movement (0x0000)
        migrated_data["us_flags"] = 0x0000  # MOUSE_MOVE_RELATIVE

        # Move button_data value to button_flags and set button_data to 0 (due to offset error in previous versions)
        if "button_data" in migrated_data:
            button_data_value = migrated_data["button_data"]

            migrated_data["button_flags"] = button_data_value
            migrated_data["button_data"] = 0

    return migrated_data


def has_legacy_raw_mouse_event(data: dict) -> bool:
    """Check if RawMouseEvent data contains legacy dx/dy fields."""
    # Legacy format has 'dx' and 'dy' fields, new format has 'last_x' and 'last_y'
    return "dx" in data and "dy" in data and "last_x" not in data and "last_y" not in data


@app.command()
def migrate(
    input_file: Path = typer.Argument(..., help="Input MCAP file path"),
    output_file: Optional[Path] = typer.Argument(
        None, help="Output MCAP file path (defaults to in-place modification)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging output"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: 'text' or 'json'"),
) -> None:
    """
    Migrate MCAP file from source version to target version.

    Transforms the input MCAP file according to the version-specific
    migration rules. If output_file is not specified, performs in-place
    modification of the input file.
    """
    console = Console()

    if not input_file.exists():
        if output_format == "json":
            result = {
                "success": False,
                "changes_made": 0,
                "error": f"Input file not found: {input_file}",
                "from_version": FROM_VERSION,
                "to_version": TO_VERSION,
            }
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Determine final output location
    final_output_file = output_file if output_file is not None else input_file

    if verbose:
        console.print(f"[blue]Migrating: {input_file} → {final_output_file}[/blue]")

    try:
        changes_made = 0

        # Collect all messages first to avoid reader/writer conflicts
        messages = []
        with OWAMcapReader(input_file) as reader:
            for message in reader.iter_messages():
                if message.message_type == "desktop/RawMouseEvent":
                    decoded_data = dict_decoder(message.message)

                    # Check if migration is needed
                    if has_legacy_raw_mouse_event(decoded_data):
                        migrated_data = migrate_raw_mouse_event_data(decoded_data)
                        new_msg = MESSAGES["desktop/RawMouseEvent"](**migrated_data)
                        messages.append((message.timestamp, message.topic, new_msg))
                        changes_made += 1

                        if output_format != "json" and verbose and changes_made % 100 == 0:
                            console.print(f"[green]Migrated {changes_made} RawMouseEvent messages...[/green]")
                    else:
                        # Already in new format
                        messages.append((message.timestamp, message.topic, message.decoded))
                else:
                    # Copy non-RawMouseEvent messages as-is
                    messages.append((message.timestamp, message.topic, message.decoded))

        # Always write to temporary file first, then move to final location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "temp.mcap"

            # Write all messages to temporary file
            with OWAMcapWriter(temp_file) as writer:
                for log_time, topic, msg in messages:
                    writer.write_message(message=msg, topic=topic, timestamp=log_time)

            # Atomically move temporary file to final location
            shutil.copy2(str(temp_file), str(final_output_file))

        # Output results according to schema
        if output_format == "json":
            result = {
                "success": True,
                "changes_made": changes_made,
                "from_version": FROM_VERSION,
                "to_version": TO_VERSION,
                "message": "Migration completed successfully",
            }
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[green]✓ Migration completed: {changes_made} changes made[/green]")

    except Exception as e:
        # Reraise typer.Exit exceptions to prevent printing duplicate error messages
        if isinstance(e, typer.Exit):
            raise e

        if output_format == "json":
            result = {
                "success": False,
                "changes_made": 0,
                "error": str(e),
                "from_version": FROM_VERSION,
                "to_version": TO_VERSION,
            }
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Migration failed: {e}[/red]")

        raise typer.Exit(1)


@app.command()
def verify(
    file_path: Path = typer.Argument(..., help="MCAP file path to verify"),
    backup_path: Optional[Path] = typer.Option(None, help="Reference backup file path (optional)"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: 'text' or 'json'"),
) -> None:
    """
    Verify migration completeness and data integrity.

    Validates that all legacy structures have been properly migrated
    and no data corruption has occurred during the transformation process.
    """
    console = Console()

    if not file_path.exists():
        if output_format == "json":
            result = {"success": False, "error": f"File not found: {file_path}"}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        # Check for legacy RawMouseEvent structures
        legacy_found = False

        with OWAMcapReader(file_path) as reader:
            for message in reader.iter_messages():
                if message.message_type == "desktop/RawMouseEvent":
                    decoded_data = dict_decoder(message.message)

                    if has_legacy_raw_mouse_event(decoded_data):
                        legacy_found = True
                        break

        # Perform integrity verification if backup is provided
        integrity_verified = True
        if backup_path is not None:
            if not backup_path.exists():
                if output_format == "json":
                    result = {"success": False, "error": f"Backup file not found: {backup_path}"}
                    print(orjson.dumps(result).decode())
                else:
                    console.print(f"[red]Backup file not found: {backup_path}[/red]")
                raise typer.Exit(1)

            verification_result = verify_migration_integrity(
                migrated_file=file_path,
                backup_file=backup_path,
                check_message_count=True,
                check_file_size=True,
                check_topics=True,
                size_tolerance_percent=10.0,
            )
            integrity_verified = verification_result.success

        # Report results according to schema
        if legacy_found:
            if output_format == "json":
                result = {"success": False, "error": "Legacy RawMouseEvent structures detected"}
                print(orjson.dumps(result).decode())
            else:
                console.print("[red]Legacy RawMouseEvent structures detected[/red]")
            raise typer.Exit(1)

        # Check if verification failed
        if backup_path is not None and not integrity_verified:
            if output_format == "json":
                result = {
                    "success": False,
                    "error": verification_result.error or "Migration integrity verification failed",
                }
                print(orjson.dumps(result).decode())
            else:
                console.print(f"[red]Migration integrity verification failed: {verification_result.error}[/red]")
            raise typer.Exit(1)

        # Success case
        success_message = "No legacy RawMouseEvent structures found"
        if backup_path is not None and integrity_verified:
            success_message += ", integrity verification passed"

        if output_format == "json":
            result = {"success": True, "message": success_message}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[green]✓ {success_message}[/green]")

    except Exception as e:
        # Reraise typer.Exit exceptions to prevent printing duplicate error messages
        if isinstance(e, typer.Exit):
            raise e

        if output_format == "json":
            result = {"success": False, "error": str(e)}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Verification failed: {e}[/red]")
        raise typer.Exit(1)


"""
Utilities for migration verification and integrity checks.

This module provides functionality to verify the integrity of migrated MCAP files
by comparing them with their backup counterparts.
"""


@dataclass
class FileStats:
    """Statistics about an MCAP file."""

    message_count: int
    file_size: int
    topics: set[str]
    schemas: set[str]


@dataclass
class VerificationResult:
    """Result of migration verification."""

    success: bool
    error: Optional[str] = None
    message_count_match: Optional[bool] = None
    file_size_diff_percent: Optional[float] = None
    topics_match: Optional[bool] = None


def get_file_stats(file_path: Path) -> FileStats:
    """Get comprehensive statistics about an MCAP file."""
    with OWAMcapReader(file_path) as reader:
        schemas = set(reader.schemas)
        topics = set(reader.topics)
        message_count = reader.message_count

    file_size = file_path.stat().st_size
    return FileStats(message_count=message_count, file_size=file_size, topics=topics, schemas=schemas)


def verify_migration_integrity(
    migrated_file: Path,
    backup_file: Path,
    check_message_count: bool = True,
    check_file_size: bool = True,
    check_topics: bool = True,
    size_tolerance_percent: float = 10.0,
) -> VerificationResult:
    """
    Verify migration integrity by comparing migrated file with backup.

    Args:
        migrated_file: Path to the migrated MCAP file
        backup_file: Path to the backup (original) MCAP file
        check_message_count: Whether to verify message count preservation
        check_file_size: Whether to verify file size is within tolerance
        check_topics: Whether to verify topic preservation
        size_tolerance_percent: Allowed file size difference percentage

    Returns:
        VerificationResult with success status and detailed information
    """
    try:
        if not migrated_file.exists():
            return VerificationResult(success=False, error=f"Migrated file not found: {migrated_file}")

        if not backup_file.exists():
            return VerificationResult(success=False, error=f"Backup file not found: {backup_file}")

        migrated_stats = get_file_stats(migrated_file)
        backup_stats = get_file_stats(backup_file)

        result = VerificationResult(success=True)

        # Check message count
        if check_message_count:
            result.message_count_match = migrated_stats.message_count == backup_stats.message_count
            if not result.message_count_match:
                result.success = False
                result.error = (
                    f"Message count mismatch: {migrated_stats.message_count} vs {backup_stats.message_count}"
                )

        # Check file size
        if check_file_size and result.success:
            result.file_size_diff_percent = (
                abs(migrated_stats.file_size - backup_stats.file_size) / backup_stats.file_size * 100
            )
            if result.file_size_diff_percent > size_tolerance_percent:
                result.success = False
                result.error = f"File size difference too large: {result.file_size_diff_percent:.1f}% (limit: {size_tolerance_percent}%)"

        # Check topics
        if check_topics and result.success:
            result.topics_match = migrated_stats.topics == backup_stats.topics
            if not result.topics_match:
                result.success = False
                result.error = f"Topic mismatch: {migrated_stats.topics} vs {backup_stats.topics}"

        return result

    except Exception as e:
        return VerificationResult(success=False, error=f"Error during integrity verification: {e}")


if __name__ == "__main__":
    app()
