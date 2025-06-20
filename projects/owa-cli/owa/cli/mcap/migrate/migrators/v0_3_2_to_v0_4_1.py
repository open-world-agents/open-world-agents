#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.0.0",
#   "mcap>=1.0.0",
#   "easydict>=1.10",
#   "orjson>=3.8.0",
#   "typer>=0.12.0",
#   "mcap-owa-support==0.4.1",
# ]
# [tool.uv]
# exclude-newer = "2025-06-15T16:02:47Z"
# ///
"""
MCAP Migrator: v0.3.2 → v0.4.1

Migrates schema format from module-based to domain-based. See OEP-0006.
"""

__all__ = []  # This is a uv script, not a module

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

try:
    from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
except ImportError as e:
    print(f"Error: Required packages not available: {e}")
    sys.exit(1)

app = typer.Typer(help="MCAP Migration: v0.3.2 → v0.4.1")

# Legacy to new message type mapping
LEGACY_MESSAGE_MAPPING = {
    "owa.env.desktop.msg.KeyboardEvent": "desktop/KeyboardEvent",
    "owa.env.desktop.msg.KeyboardState": "desktop/KeyboardState",
    "owa.env.desktop.msg.MouseEvent": "desktop/MouseEvent",
    "owa.env.desktop.msg.MouseState": "desktop/MouseState",
    "owa.env.desktop.msg.WindowInfo": "desktop/WindowInfo",
    "owa.env.gst.msg.ScreenEmitted": "desktop/ScreenCaptured",
}


class SimpleMessageClass:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@app.command()
def migrate(
    input_file: Path = typer.Argument(..., help="Input MCAP file"),
    output_file: Optional[Path] = typer.Argument(
        None, help="Output MCAP file (optional, defaults to overwriting input)"
    ),
    backup: Optional[Path] = typer.Option(None, "--backup", help="Backup file path (for reference)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """Migrate MCAP file from v0.3.2 to v0.4.1."""
    console = Console()

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    if not input_file.suffix == ".mcap":
        console.print(f"[red]Input file must be an MCAP file: {input_file}[/red]")
        raise typer.Exit(1)

    output_path = output_file or input_file
    changes_made = 0

    # Check if migration is needed
    with OWAMcapReader(input_file) as reader:
        has_legacy_schemas = any(schema.name in LEGACY_MESSAGE_MAPPING for schema in reader.schemas.values())

    if not has_legacy_schemas:
        console.print("[green]✓ No legacy schemas found, no migration needed[/green]")
        return

    try:
        msgs = []
        with OWAMcapReader(input_file) as reader:
            for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                schema_name = schema.name

                if schema_name in LEGACY_MESSAGE_MAPPING:
                    new_schema_name = LEGACY_MESSAGE_MAPPING[schema_name]

                    if hasattr(decoded, "model_dump"):
                        data = decoded.model_dump()
                    elif hasattr(decoded, "__dict__"):
                        data = decoded.__dict__
                    else:
                        data = dict(decoded)

                    new_message = SimpleMessageClass(**data)
                    new_message._type = new_schema_name
                    msgs.append((message.log_time, channel.topic, new_message))
                    changes_made += 1

                    if verbose:
                        console.print(f"  Migrated: {schema_name} → {new_schema_name}")
                else:
                    msgs.append((message.log_time, channel.topic, decoded))

        with OWAMcapWriter(output_path) as writer:
            for log_time, topic, msg in msgs:
                writer.write_message(topic=topic, message=msg, log_time=log_time)

        console.print(f"[green]✓ Migration completed: {changes_made} changes made[/green]")

    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    file_path: Path = typer.Argument(..., help="MCAP file to verify"),
    backup: Optional[Path] = typer.Option(None, "--backup", help="Backup file path (for reference)"),
) -> None:
    """Verify that no legacy schemas remain."""
    console = Console()

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        with OWAMcapReader(file_path) as reader:
            for schema in reader.schemas.values():
                if schema.name in LEGACY_MESSAGE_MAPPING:
                    console.print(f"[red]Legacy schema still present: {schema.name}[/red]")
                    raise typer.Exit(1)

        console.print("[green]✓ No legacy schemas found[/green]")

    except Exception as e:
        console.print(f"[red]Verification error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
