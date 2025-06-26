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
#   "mcap-owa-support==0.5.1",
#   "owa-cli==0.5.1",
#   "owa-core==0.5.1",
#   "owa-msgs==0.5.1",
# ]
# [tool.uv]
# exclude-newer = "2025-06-27T00:00:00Z"
# [tool.uv.sources]
# mcap-owa-support = { path = "../../../../../../mcap-owa-support", editable = true }
# owa-cli = { path = "../../../../../../owa-cli", editable = true }
# owa-core = { path = "../../../../../../owa-core", editable = true }
# owa-msgs = { path = "../../../../../../owa-msgs", editable = true }
# ///
"""
MCAP Migrator: v0.5.0 → v0.5.1

Migrates ScreenCaptured messages from discriminated union MediaRef to unified URI-based MediaRef.
Key changes:
- EmbeddedRef → MediaRef with data URI
- ExternalImageRef → MediaRef with path URI
- ExternalVideoRef → MediaRef with path URI + pts_ns
- Unified MediaRef class with computed properties
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.decoder import dict_decoder
from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.cli.mcap.migrate.utils import verify_migration_integrity
from owa.core import MESSAGES

app = typer.Typer(help="MCAP Migration: v0.5.0 → v0.5.1")

# Version constants
FROM_VERSION = "0.5.0"
TO_VERSION = "0.5.1"


def migrate_media_ref(media_ref_data: dict) -> dict:
    """
    Migrate MediaRef from discriminated union to unified URI format.

    Transformations:
    - EmbeddedRef → MediaRef with data URI
    - ExternalImageRef → MediaRef with path URI
    - ExternalVideoRef → MediaRef with path URI + pts_ns
    """
    if not media_ref_data:
        return media_ref_data

    ref_type = media_ref_data.get("type")

    if ref_type == "embedded":
        # Convert EmbeddedRef to data URI
        format_type = media_ref_data.get("format", "png")
        data = media_ref_data.get("data", "")
        uri = f"data:image/{format_type};base64,{data}"
        return {"uri": uri, "pts_ns": None}

    elif ref_type == "external_image":
        # Convert ExternalImageRef to path URI
        path = media_ref_data.get("path", "")
        return {"uri": path, "pts_ns": None}

    elif ref_type == "external_video":
        # Convert ExternalVideoRef to path URI with pts_ns
        path = media_ref_data.get("path", "")
        pts_ns = media_ref_data.get("pts_ns")
        return {"uri": path, "pts_ns": pts_ns}

    else:
        # Unknown type, return as-is
        return media_ref_data


def migrate_screen_captured_data(data: dict) -> dict:
    """
    Migrate ScreenCaptured message data from v0.5.0 to v0.5.1 format.

    Key transformation:
    - media_ref: discriminated union → unified MediaRef
    """
    migrated_data = data.copy()

    # Migrate media_ref if present
    if "media_ref" in migrated_data and migrated_data["media_ref"]:
        migrated_data["media_ref"] = migrate_media_ref(migrated_data["media_ref"])

    return migrated_data


def has_legacy_media_ref(data: dict) -> bool:
    """Check if ScreenCaptured data contains legacy MediaRef format."""
    media_ref = data.get("media_ref")
    if not media_ref or not isinstance(media_ref, dict):
        return False

    # Legacy format has 'type' field, new format has 'uri' field
    return "type" in media_ref and "uri" not in media_ref


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
                if message.message_type == "desktop/ScreenCaptured":
                    decoded_data = dict_decoder(message.message)

                    # Check if migration is needed
                    if has_legacy_media_ref(decoded_data):
                        migrated_data = migrate_screen_captured_data(decoded_data)
                        new_msg = MESSAGES["desktop/ScreenCaptured"](**migrated_data)
                        messages.append((message.timestamp, message.topic, new_msg))
                        changes_made += 1

                        if output_format != "json" and verbose and changes_made % 100 == 0:
                            console.print(f"[green]Migrated {changes_made} ScreenCaptured messages...[/green]")
                    else:
                        # Already in new format
                        messages.append((message.timestamp, message.topic, message.decoded))
                else:
                    # Copy non-ScreenCaptured messages as-is
                    messages.append((message.timestamp, message.topic, message.decoded))

        # Always write to temporary file first, then move to final location
        with tempfile.NamedTemporaryFile(suffix=".mcap", dir=final_output_file.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            # Write all messages to temporary file
            with OWAMcapWriter(temp_path) as writer:
                for log_time, topic, msg in messages:
                    writer.write_message(topic=topic, message=msg, log_time=log_time)

            # Atomically move temporary file to final location
            shutil.move(str(temp_path), str(final_output_file))

        # Output results according to schema
        if output_format == "json":
            result = {
                "success": True,
                "changes_made": changes_made,
                "from_version": FROM_VERSION,
                "to_version": TO_VERSION,
            }
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[green]✓ Migration completed: {changes_made} changes made[/green]")

    except Exception as e:
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
        # Check for legacy MediaRef structures
        legacy_found = False

        with OWAMcapReader(file_path) as reader:
            for message in reader.iter_messages(topics="screen"):
                decoded_data = dict_decoder(message.message)

                if has_legacy_media_ref(decoded_data):
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
                result = {"success": False, "error": "Legacy MediaRef structures detected"}
                print(orjson.dumps(result).decode())
            else:
                console.print("[red]Legacy MediaRef structures detected[/red]")
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
        success_message = "No legacy MediaRef structures found"
        if backup_path is not None and integrity_verified:
            success_message += ", integrity verification passed"

        if output_format == "json":
            result = {"success": True, "message": success_message}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[green]✓ {success_message}[/green]")

    except Exception as e:
        if output_format == "json":
            result = {"success": False, "error": str(e)}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Verification failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
