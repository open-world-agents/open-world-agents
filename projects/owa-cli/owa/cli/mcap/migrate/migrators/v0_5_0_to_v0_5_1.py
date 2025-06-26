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
#   "owa-core==0.5.1",
#   "owa-msgs==0.5.1",
# ]
# [tool.uv]
# exclude-newer = "2025-06-26T00:00:00Z"
# [tool.uv.sources]
# mcap-owa-support = { path = "../../../../../../mcap-owa-support", editable = true }
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

from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.decoder import dict_decoder
from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core import MESSAGES

# Import migration utilities
try:
    from ..utils import verify_migration_integrity
except ImportError:
    # Fallback if utils module is not available
    def verify_migration_integrity(*_args, **_kwargs):
        return True


app = typer.Typer(help="MCAP Migration: v0.5.0 → v0.5.1")


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


@app.command()
def migrate(
    input_file: Path = typer.Argument(..., help="Input MCAP file"),
    output_file: Optional[Path] = typer.Argument(
        None, help="Output MCAP file (optional, defaults to overwriting input)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
) -> None:
    """Migrate MCAP file from v0.5.0 to v0.5.1."""
    console = Console()

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Default output to input file (overwrite)
    if output_file is None:
        output_file = input_file

    if verbose:
        console.print(f"[blue]Migrating: {input_file} → {output_file}[/blue]")

    try:
        # Statistics
        total_messages = 0
        migrated_messages = 0

        with OWAMcapReader(input_file) as reader:
            with OWAMcapWriter(output_file) as writer:
                for message in reader.iter_messages():
                    total_messages += 1

                    # Check if this is a ScreenCaptured message
                    if message.schema_name == "desktop/ScreenCaptured":
                        # Decode message data
                        decoded_data = dict_decoder(message.data, message.schema)

                        # Migrate the data
                        migrated_data = migrate_screen_captured_data(decoded_data)
                        migrated_messages += 1

                        # Re-encode and write
                        writer.write_message(
                            topic=message.topic,
                            schema_name=message.schema_name,
                            data=migrated_data,
                            log_time_ns=message.log_time_ns,
                            publish_time_ns=message.publish_time_ns,
                        )

                        if verbose and migrated_messages % 100 == 0:
                            console.print(f"[green]Migrated {migrated_messages} ScreenCaptured messages...[/green]")
                    else:
                        # Copy non-ScreenCaptured messages as-is
                        writer.write_raw_message(message)

        # Output results
        result = {
            "status": "success",
            "input_file": str(input_file),
            "output_file": str(output_file),
            "total_messages": total_messages,
            "migrated_messages": migrated_messages,
        }

        if output_format == "json":
            console.print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
        else:
            console.print(f"[green]✓ Migration completed successfully[/green]")
            console.print(f"  Total messages: {total_messages}")
            console.print(f"  Migrated ScreenCaptured messages: {migrated_messages}")
            console.print(f"  Output file: {output_file}")

    except Exception as e:
        error_result = {
            "status": "error",
            "input_file": str(input_file),
            "error": str(e),
        }

        if output_format == "json":
            console.print(orjson.dumps(error_result, option=orjson.OPT_INDENT_2).decode())
        else:
            console.print(f"[red]✗ Migration failed: {e}[/red]")

        raise typer.Exit(1)


@app.command()
def verify(
    input_file: Path = typer.Argument(..., help="MCAP file to verify"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
) -> None:
    """Verify that an MCAP file uses v0.5.1 MediaRef format."""
    console = Console()

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    try:
        screen_captured_count = 0
        v0_5_1_format_count = 0
        legacy_format_count = 0

        with OWAMcapReader(input_file) as reader:
            for message in reader.iter_messages():
                if message.schema_name == "desktop/ScreenCaptured":
                    screen_captured_count += 1

                    # Decode and check format
                    decoded_data = dict_decoder(message.data, message.schema)
                    media_ref = decoded_data.get("media_ref")

                    if media_ref and isinstance(media_ref, dict):
                        if "uri" in media_ref:
                            v0_5_1_format_count += 1
                        elif "type" in media_ref:
                            legacy_format_count += 1

        # Determine if migration is needed
        needs_migration = legacy_format_count > 0

        result = {
            "status": "success",
            "file": str(input_file),
            "screen_captured_messages": screen_captured_count,
            "v0_5_1_format": v0_5_1_format_count,
            "legacy_format": legacy_format_count,
            "needs_migration": needs_migration,
        }

        if output_format == "json":
            console.print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
        else:
            console.print(f"[blue]Verification Results for {input_file}[/blue]")
            console.print(f"  ScreenCaptured messages: {screen_captured_count}")
            console.print(f"  v0.5.1 format (URI-based): {v0_5_1_format_count}")
            console.print(f"  Legacy format (type-based): {legacy_format_count}")

            if needs_migration:
                console.print(f"[yellow]⚠ Migration needed: {legacy_format_count} messages use legacy format[/yellow]")
            else:
                console.print(f"[green]✓ All messages use v0.5.1 format[/green]")

    except Exception as e:
        error_result = {
            "status": "error",
            "file": str(input_file),
            "error": str(e),
        }

        if output_format == "json":
            console.print(orjson.dumps(error_result, option=orjson.OPT_INDENT_2).decode())
        else:
            console.print(f"[red]✗ Verification failed: {e}[/red]")

        raise typer.Exit(1)


if __name__ == "__main__":
    app()
