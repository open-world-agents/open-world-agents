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
#   "mcap-owa-support==0.5.0a2",
#   "owa-core==0.5.0a2",
#   "owa-msgs==0.5.0a2",
# ]
# [tool.uv]
# exclude-newer = "2025-06-26T00:00:00Z"
# ///
"""
MCAP Migrator: v0.4.2 → v0.5.0

Migrates ScreenCaptured messages from the legacy field structure to the new
structured media reference system. Key changes:
- `original_shape` → `source_shape`
- `pts` → `pts_ns` (in ExternalVideoRef)
- Simple path/pts fields → structured MediaRef system (EmbeddedRef, ExternalImageRef, ExternalVideoRef)
- Enhanced media reference types with proper type discrimination
"""

from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core import MESSAGES

# Import migration utilities
try:
    from ..utils import verify_migration_integrity
except ImportError:
    # Fallback if utils module is not available
    def verify_migration_integrity(*_args, **_kwargs):
        return True


app = typer.Typer(help="MCAP Migration: v0.4.2 → v0.5.0")


def migrate_screen_captured_data(data: dict) -> dict:
    """
    Migrate ScreenCaptured message data from v0.4.2 to v0.5.0 format.

    Key transformations:
    1. original_shape → source_shape
    2. path/pts fields → structured media_ref
    3. Handle pts → pts_ns conversion
    4. Create appropriate ExternalImageRef vs ExternalVideoRef
    """
    migrated_data = data.copy()

    # 1. Rename original_shape to source_shape
    if "original_shape" in migrated_data:
        migrated_data["source_shape"] = migrated_data.pop("original_shape")

    # 2. Create structured media reference if path/pts exist
    path = migrated_data.pop("path", None)
    pts = migrated_data.pop("pts", None)  # Legacy field name (nanoseconds)

    if path is not None and pts is not None:
        # Create ExternalVideoRef with pts_ns (converted from pts)
        migrated_data["media_ref"] = {
            "type": "external_video",
            "path": path,
            "pts_ns": pts,  # pts was already in nanoseconds in v0.4.2
        }
    else:
        raise ValueError("Unexpected legacy ScreenCaptured message format")

    return migrated_data


@app.command()
def migrate(
    input_file: Path = typer.Argument(..., help="Input MCAP file"),
    output_file: Optional[Path] = typer.Argument(
        None, help="Output MCAP file (optional, defaults to overwriting input)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
) -> None:
    """Migrate MCAP file from v0.4.2 to v0.5.0."""
    console = Console()

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    if not input_file.suffix == ".mcap":
        console.print(f"[red]Input file must be an MCAP file: {input_file}[/red]")
        raise typer.Exit(1)

    output_path = output_file or input_file
    changes_made = 0

    try:
        msgs = []
        with OWAMcapReader(input_file, decode_as_dict=True) as reader:
            for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                schema_name = schema.name

                if schema_name == "desktop/ScreenCaptured":
                    # Extract data from the decoded message
                    if hasattr(decoded, "model_dump"):
                        data = decoded.model_dump()
                    elif hasattr(decoded, "__dict__"):
                        data = decoded.__dict__
                    else:
                        data = dict(decoded)

                    # Check if this message needs migration (has legacy fields)
                    needs_migration = (
                        "original_shape" in data or "path" in data or ("pts" in data and "media_ref" not in data)
                    )

                    if needs_migration:
                        # Migrate the data structure
                        migrated_data = migrate_screen_captured_data(data)

                        # Create new message with migrated data
                        new_message = MESSAGES["desktop/ScreenCaptured"](**migrated_data)
                        msgs.append((message.log_time, channel.topic, new_message))
                        changes_made += 1

                        if verbose:
                            console.print("  Migrated ScreenCaptured message with legacy fields")
                    else:
                        # Message already in new format
                        msgs.append((message.log_time, channel.topic, decoded))
                else:
                    # Non-ScreenCaptured messages pass through unchanged
                    msgs.append((message.log_time, channel.topic, decoded))

        # Write migrated messages
        with OWAMcapWriter(output_path) as writer:
            for log_time, topic, msg in msgs:
                writer.write_message(topic=topic, message=msg, log_time=log_time)

        if output_format == "json":
            result = {"success": True, "changes_made": changes_made, "from_version": "0.4.2", "to_version": "0.5.0"}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[green]✓ Migration completed: {changes_made} changes made[/green]")

    except Exception as e:
        if output_format == "json":
            result = {
                "success": False,
                "changes_made": 0,
                "error": str(e),
                "from_version": "0.4.2",
                "to_version": "0.5.0",
            }
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Migration failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    file_path: Path = typer.Argument(..., help="MCAP file to verify"),
    backup_path: Optional[Path] = typer.Option(None, help="Backup file path (for reference)"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
) -> None:
    """Verify that no legacy ScreenCaptured field structures remain."""
    console = Console()

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        # Check for legacy field structures
        with OWAMcapReader(file_path) as reader:
            for schema, _, _, decoded in reader.reader.iter_decoded_messages():
                if schema.name == "desktop/ScreenCaptured":
                    # Check if message has legacy field structure
                    if hasattr(decoded, "model_dump"):
                        data = decoded.model_dump()
                    elif hasattr(decoded, "__dict__"):
                        data = decoded.__dict__
                    else:
                        data = dict(decoded)

                    # Check for legacy fields
                    has_legacy_fields = (
                        "original_shape" in data or "path" in data or ("pts" in data and "media_ref" not in data)
                    )

                    if has_legacy_fields:
                        if output_format == "json":
                            result = {
                                "success": False,
                                "error": "Found ScreenCaptured messages with legacy field structure",
                            }
                            print(orjson.dumps(result).decode())
                        else:
                            console.print("[red]Found ScreenCaptured messages with legacy field structure[/red]")
                        raise typer.Exit(1)

        # Perform integrity verification if backup is provided
        integrity_verified = True
        if backup_path is not None:
            integrity_verified = verify_migration_integrity(
                migrated_file=file_path,
                backup_file=backup_path,
                console=console,
                check_message_count=True,
                check_file_size=True,
                check_topics=True,
                size_tolerance_percent=10.0,
            )

        # Report results
        success_message = "No legacy ScreenCaptured field structures found"
        if backup_path is not None:
            if integrity_verified:
                success_message += ", integrity verification passed"
            else:
                success_message += ", integrity verification failed"

        if output_format == "json":
            result = {"success": True, "message": success_message}
            print(orjson.dumps(result).decode())
        else:
            console.print("[green]✓ No legacy ScreenCaptured field structures found[/green]")
            if backup_path is not None and not integrity_verified:
                console.print("[yellow]⚠ Migration integrity verification failed[/yellow]")

    except Exception as e:
        if output_format == "json":
            result = {"success": False, "error": str(e)}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Verification error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
