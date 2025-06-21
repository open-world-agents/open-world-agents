#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.0.0",
#   "mcap>=1.0.0",
#   "easydict>=1.10",
#   "orjson>=3.8.0",
#   "typer>=0.12.0",
#   "mcap-owa-support==0.3.2",
#   "owa-env-desktop==0.3.2",
# ]
# [tool.uv]
# exclude-newer = "2025-03-22T11:14:45Z"
# ///
"""
MCAP Migrator: v0.3.0 → v0.3.2

Migrates keyboard state field from `pressed_vk_list` to `buttons`.
Note: Before v0.3.2, mcap-owa-support version was 0.1.0, so this script must be run with manual user's decision.
"""

import importlib
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter

app = typer.Typer(help="MCAP Migration: v0.3.0 → v0.3.2")


@app.command()
def migrate(
    input_file: Path = typer.Argument(..., help="Input MCAP file"),
    output_file: Optional[Path] = typer.Argument(
        None, help="Output MCAP file (optional, defaults to overwriting input)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
) -> None:
    """Migrate MCAP file from v0.3.0 to v0.3.2."""
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
        with OWAMcapReader(input_file) as reader:
            for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                if channel.topic == "keyboard/state":
                    buttons = decoded.pop("pressed_vk_list")
                    decoded["buttons"] = buttons
                    changes_made += 1
                try:
                    module, class_name = schema.name.rsplit(".", 1)
                    module = importlib.import_module(module)
                    cls = getattr(module, class_name)

                    decoded = cls(**decoded)
                except ValueError as e:
                    # Skip schemas that don't need conversion
                    if "not found in convert_dict" not in str(e):
                        raise

                msgs.append((message.log_time, channel.topic, decoded))

        with OWAMcapWriter(output_path) as writer:
            for log_time, topic, msg in msgs:
                writer.write_message(topic=topic, message=msg, log_time=log_time)

        if output_format == "json":
            result = {"success": True, "changes_made": changes_made, "from_version": "0.3.0", "to_version": "0.3.2"}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[green]✓ Migration completed: {changes_made} changes made[/green]")

    except Exception as e:
        if output_format == "json":
            result = {
                "success": False,
                "changes_made": 0,
                "error": str(e),
                "from_version": "0.3.0",
                "to_version": "0.3.2",
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
    """Verify that pressed_vk_list fields are gone."""
    console = Console()

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        with OWAMcapReader(file_path) as reader:
            for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                if channel.topic == "keyboard/state" and hasattr(decoded, "pressed_vk_list"):
                    if output_format == "json":
                        result = {
                            "success": False,
                            "error": "pressed_vk_list field still present in keyboard/state message",
                        }
                        print(orjson.dumps(result).decode())
                    else:
                        console.print("[red]pressed_vk_list field still present in keyboard/state message[/red]")
                    raise typer.Exit(1)

        if output_format == "json":
            result = {"success": True, "message": "pressed_vk_list fields successfully migrated"}
            print(orjson.dumps(result).decode())
        else:
            console.print("[green]✓ pressed_vk_list fields successfully migrated[/green]")

    except Exception as e:
        if output_format == "json":
            result = {"success": False, "error": str(e)}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Verification error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
