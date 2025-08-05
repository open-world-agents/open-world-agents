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
#   "mcap-owa-support==0.5.6",
#   "owa-core==0.5.6",
#   "owa-msgs==0.5.6",
# ]
# [tool.uv]
# exclude-newer = "2025-08-06T12:00:00Z"
# ///
"""
MCAP Migrator: v0.5.5 → v0.5.6

Migrates PointerBallisticsConfig metadata from legacy format to new format.
Key changes:
- SmoothMouseXCurve: base64-encoded → hex-encoded binary data (or default hex if None)
- SmoothMouseYCurve: base64-encoded → hex-encoded binary data (or default hex if None)
- Fields changed from str|None to required str with default hex values
- Data encoding changed from base64 to hex format

NOTE: These migrators are locked, separate script with separate dependency sets. DO NOT change the contents unless you know what you are doing.
"""

import base64
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter

app = typer.Typer(help="MCAP Migration: v0.5.5 → v0.5.6")

# Version constants
FROM_VERSION = "0.5.5"
TO_VERSION = "0.5.6"

# Default values for SmoothMouse curves
DEFAULT_SMOOTH_MOUSE_X_CURVE = "0000000000000000156e000000000000004001000000000029dc0300000000000000280000000000"
DEFAULT_SMOOTH_MOUSE_Y_CURVE = "0000000000000000fd11010000000000002404000000000000fc12000000000000c0bb0100000000"


def migrate_pointer_ballistics_metadata(metadata: dict) -> dict:
    """
    Migrate PointerBallisticsConfig metadata from v0.5.5 to v0.5.6 format.

    Key transformations:
    - Convert SmoothMouseXCurve from base64 to hex encoding (or use default if None)
    - Convert SmoothMouseYCurve from base64 to hex encoding (or use default if None)
    """
    migrated_metadata = metadata.copy()

    # Handle SmoothMouseXCurve
    smooth_x = migrated_metadata.get("SmoothMouseXCurve")
    if smooth_x is None or smooth_x == "None" or smooth_x == "":
        migrated_metadata["SmoothMouseXCurve"] = DEFAULT_SMOOTH_MOUSE_X_CURVE
    else:
        # Convert from base64 to hex
        try:
            binary_data = base64.b64decode(smooth_x)
            migrated_metadata["SmoothMouseXCurve"] = binary_data.hex()
        except Exception:
            # If conversion fails, use default
            migrated_metadata["SmoothMouseXCurve"] = DEFAULT_SMOOTH_MOUSE_X_CURVE

    # Handle SmoothMouseYCurve
    smooth_y = migrated_metadata.get("SmoothMouseYCurve")
    if smooth_y is None or smooth_y == "None" or smooth_y == "":
        migrated_metadata["SmoothMouseYCurve"] = DEFAULT_SMOOTH_MOUSE_Y_CURVE
    else:
        # Convert from base64 to hex
        try:
            binary_data = base64.b64decode(smooth_y)
            migrated_metadata["SmoothMouseYCurve"] = binary_data.hex()
        except Exception:
            # If conversion fails, use default
            migrated_metadata["SmoothMouseYCurve"] = DEFAULT_SMOOTH_MOUSE_Y_CURVE

    return migrated_metadata


def has_legacy_pointer_ballistics_metadata(metadata: dict) -> bool:
    """Check if PointerBallisticsConfig metadata contains legacy base64 format or None values."""
    smooth_x = metadata.get("SmoothMouseXCurve")
    smooth_y = metadata.get("SmoothMouseYCurve")

    # Legacy format has None, "None", missing values, or base64-encoded data
    def is_legacy_value(value):
        if value is None or value == "None" or value == "":
            return True
        # Check if it's base64 (not hex) - base64 can contain +, /, = characters
        # Hex only contains 0-9, a-f, A-F
        if isinstance(value, str) and len(value) > 0:
            # If it contains base64-specific characters, it's legacy
            if any(c in value for c in "+/="):
                return True
            # If it's not a valid hex string, it might be base64
            try:
                int(value, 16)  # Try to parse as hex
                return False  # Valid hex, not legacy
            except ValueError:
                return True  # Not valid hex, likely base64 or other format
        return False

    return is_legacy_value(smooth_x) or is_legacy_value(smooth_y)


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

        # Collect all messages and metadata first to avoid reader/writer conflicts
        messages = []
        metadata_dict = {}

        with OWAMcapReader(input_file) as reader:
            # Copy metadata and check for pointer_ballistics_config
            for metadata_record in reader.iter_metadata():
                name = metadata_record.name
                data = metadata_record.metadata

                if name == "pointer_ballistics_config":
                    if has_legacy_pointer_ballistics_metadata(data):
                        migrated_data = migrate_pointer_ballistics_metadata(data)
                        metadata_dict[name] = migrated_data
                        changes_made += 1

                        if verbose:
                            console.print("[green]Migrated pointer_ballistics_config metadata[/green]")
                    else:
                        metadata_dict[name] = data
                else:
                    metadata_dict[name] = data

            # Copy all messages as-is
            for message in reader.iter_messages():
                messages.append((message.timestamp, message.topic, message.decoded))

        # Always write to temporary file first, then move to final location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "temp.mcap"

            # Write all messages and metadata to temporary file
            with OWAMcapWriter(temp_file) as writer:
                # Write metadata first
                for name, data in metadata_dict.items():
                    writer.write_metadata(name, data)

                # Write all messages
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
        # Check for legacy PointerBallisticsConfig metadata
        legacy_found = False

        with OWAMcapReader(file_path) as reader:
            for metadata_record in reader.iter_metadata():
                if metadata_record.name == "pointer_ballistics_config":
                    if has_legacy_pointer_ballistics_metadata(metadata_record.metadata):
                        legacy_found = True
                        break

        # Perform integrity verification if backup is provided
        integrity_verified = True
        verification_result = None
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
                result = {"success": False, "error": "Legacy PointerBallisticsConfig metadata detected"}
                print(orjson.dumps(result).decode())
            else:
                console.print("[red]Legacy PointerBallisticsConfig metadata detected[/red]")
            raise typer.Exit(1)

        # Check if verification failed
        if backup_path is not None and not integrity_verified:
            error_msg = "Migration integrity verification failed"
            if verification_result and verification_result.error:
                error_msg = verification_result.error

            if output_format == "json":
                result = {
                    "success": False,
                    "error": error_msg,
                }
                print(orjson.dumps(result).decode())
            else:
                console.print(f"[red]Migration integrity verification failed: {error_msg}[/red]")
            raise typer.Exit(1)

        # Success case
        success_message = "No legacy PointerBallisticsConfig metadata found"
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
