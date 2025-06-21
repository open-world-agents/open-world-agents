#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.0.0",
#   "mcap>=1.0.0",
#   "easydict>=1.10",
#   "orjson>=3.8.0",
#   "typer>=0.12.0",
#   "mcap-owa-support==0.3.0",
#   "owa-env-desktop==0.3.0",
# ]
# [tool.uv]
# exclude-newer = "2025-03-14T00:00:00Z"
# ///
"""
MCAP Migrator: v0.2.0 → v0.3.0

Migrates schema names from old format (owa_env_desktop) to new format (owa.env.desktop: implicit namespace packaging, PEP 420).
Note: Before v0.3.2, mcap-owa-support version was 0.1.0, so this script must be run with manual user's decision.
"""

import importlib
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter

app = typer.Typer(help="MCAP Migration: v0.2.0 → v0.3.0")


def convert_name(name: str) -> str:
    """Convert old schema names to new format."""
    convert_dict = {
        "owa_env_desktop": "owa.env.desktop",
        "owa_env_gst": "owa.env.gst",
    }
    names = name.split(".")
    if names[0] in convert_dict:
        return convert_dict[names[0]] + "." + ".".join(names[1:])
    raise ValueError(f"Name {name} not found in convert_dict")


@app.command()
def migrate(
    input_file: Path = typer.Argument(..., help="Input MCAP file"),
    output_file: Optional[Path] = typer.Argument(
        None, help="Output MCAP file (optional, defaults to overwriting input)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
) -> None:
    """Migrate MCAP file from v0.2.0 to v0.3.0."""
    console = Console()

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    if not input_file.suffix == ".mcap":
        console.print(f"[red]Input file must be an MCAP file: {input_file}[/red]")
        raise typer.Exit(1)

    output_path = output_file or input_file
    schema_conversions = {}
    msgs = []

    try:
        with OWAMcapReader(input_file) as reader:
            for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                old_name = schema.name
                try:
                    new_name = convert_name(old_name)

                    # Track schema conversions for reporting
                    if old_name not in schema_conversions:
                        schema_conversions[old_name] = new_name

                    module, class_name = new_name.rsplit(".", 1)
                    module = importlib.import_module(module)
                    cls = getattr(module, class_name)

                    decoded = cls(**decoded)
                except ValueError as e:
                    # Skip schemas that don't need conversion
                    if "not found in convert_dict" not in str(e):
                        raise

                msgs.append((message.log_time, channel.topic, decoded))

        # Write the converted file
        with OWAMcapWriter(output_path) as writer:
            for log_time, topic, msg in msgs:
                writer.write_message(topic=topic, message=msg, log_time=log_time)

        changes_made = len(schema_conversions)

        if output_format == "json":
            result = {
                "success": True,
                "changes_made": changes_made,
                "schema_conversions": schema_conversions,
                "from_version": "0.2.0",
                "to_version": "0.3.0",
            }
            print(orjson.dumps(result).decode())
        else:
            if verbose and schema_conversions:
                console.print("[green]Schema conversions:[/green]")
                for old, new in schema_conversions.items():
                    console.print(f"  {old} → {new}")

            console.print(f"[green]✓ Migration completed: {changes_made} changes made[/green]")

    except Exception as e:
        if output_format == "json":
            result = {
                "success": False,
                "changes_made": 0,
                "error": str(e),
                "from_version": "0.2.0",
                "to_version": "0.3.0",
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
    """Verify that old schema names are gone."""
    console = Console()

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        old_schema_names = {"owa_env_desktop", "owa_env_gst"}
        found_old_schemas = set()

        with OWAMcapReader(file_path) as reader:
            for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                schema_prefix = schema.name.split(".")[0]
                if schema_prefix in old_schema_names:
                    found_old_schemas.add(schema.name)

        if found_old_schemas:
            if output_format == "json":
                result = {
                    "success": False,
                    "error": f"Old schema names still present: {', '.join(found_old_schemas)}",
                    "found_old_schemas": list(found_old_schemas),
                }
                print(orjson.dumps(result).decode())
            else:
                console.print(f"[red]Old schema names still present: {', '.join(found_old_schemas)}[/red]")
            raise typer.Exit(1)

        if output_format == "json":
            result = {"success": True, "message": "Old schema names successfully migrated"}
            print(orjson.dumps(result).decode())
        else:
            console.print("[green]✓ Old schema names successfully migrated[/green]")

    except Exception as e:
        if output_format == "json":
            result = {"success": False, "error": str(e)}
            print(orjson.dumps(result).decode())
        else:
            console.print(f"[red]Verification error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
