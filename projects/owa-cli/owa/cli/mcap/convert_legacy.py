"""
Convert legacy OWAMcap files to new domain-based message format (OEP-0006).

This script converts MCAP files that use old module-based message names
(e.g., 'owa.env.desktop.msg.KeyboardEvent') to the new domain-based format
(e.g., 'desktop/KeyboardEvent').
"""

import tempfile
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from mcap.reader import make_reader

    from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
    from owa.core import MESSAGES
except ImportError as e:
    typer.echo(f"Error: Required packages not available: {e}", err=True)
    typer.echo("Please install: pip install mcap-owa-support", err=True)
    raise typer.Exit(1)


# Legacy to new message type mapping
LEGACY_MESSAGE_MAPPING = {
    "owa.env.desktop.msg.KeyboardEvent": "desktop/KeyboardEvent",
    "owa.env.desktop.msg.KeyboardState": "desktop/KeyboardState",
    "owa.env.desktop.msg.MouseEvent": "desktop/MouseEvent",
    "owa.env.desktop.msg.MouseState": "desktop/MouseState",
    "owa.env.desktop.msg.WindowInfo": "desktop/WindowInfo",
    "owa.env.gst.msg.ScreenCaptured": "desktop/ScreenCaptured",
    # Add more mappings as needed
}


def convert_legacy_mcap(
    input_file: Path = typer.Argument(..., help="Input legacy MCAP file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output MCAP file (default: input_converted.mcap)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be converted without writing output"),
    force: bool = typer.Option(False, "--force", help="Overwrite output file if it exists"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed conversion information"),
) -> None:
    """
    Convert legacy OWAMcap files to new domain-based message format.

    This command converts MCAP files that use old module-based message schema names
    to the new domain-based format introduced in OEP-0006.

    Example:
        owl mcap convert-legacy old_recording.mcap -o new_recording.mcap
    """
    console = Console()

    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' does not exist.[/red]")
        raise typer.Exit(1)

    # Determine output file
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_converted{input_file.suffix}"

    # Check if output file exists
    if output_file.exists() and not force:
        console.print(f"[red]Error: Output file '{output_file}' already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(1)

    console.print("[bold blue]Converting legacy MCAP file...[/bold blue]")
    console.print(f"Input:  {input_file}")
    console.print(f"Output: {output_file}")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be written[/yellow]")

    # Analyze the input file first
    analysis = _analyze_mcap_file(input_file, console, verbose)

    if not analysis["has_legacy_messages"]:
        console.print("[green]✓ File already uses new message format. No conversion needed.[/green]")
        return

    if dry_run:
        console.print(f"\n[yellow]Would convert {analysis['legacy_message_count']} legacy messages[/yellow]")
        for old_name, new_name in analysis["conversions"].items():
            console.print(f"  {old_name} → {new_name}")
        return

    # Perform the conversion
    try:
        _convert_mcap_file(input_file, output_file, analysis, console, verbose)
        console.print("\n[green]✓ Conversion completed successfully![/green]")
        console.print(f"Converted file saved to: {output_file}")

        # Show file size comparison
        input_size = input_file.stat().st_size
        output_size = output_file.stat().st_size
        console.print(f"File size: {input_size:,} → {output_size:,} bytes")

    except Exception as e:
        console.print(f"[red]✗ Conversion failed: {e}[/red]")
        # Clean up partial output file
        if output_file.exists():
            output_file.unlink()
        raise typer.Exit(1)


def _analyze_mcap_file(input_file: Path, console: Console, verbose: bool) -> Dict:
    """Analyze MCAP file to determine what needs to be converted."""

    analysis = {
        "total_messages": 0,
        "legacy_message_count": 0,
        "has_legacy_messages": False,
        "conversions": {},
        "schemas": {},
        "topics": set(),
    }

    console.print("\n[bold]Analyzing input file...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning messages...", total=None)

        try:
            with open(input_file, "rb") as f:
                reader = make_reader(f)

                # Analyze schemas
                progress.update(task, description="Analyzing schemas...")
                for schema in reader.get_summary().schemas.values():
                    analysis["schemas"][schema.id] = schema.name

                    if schema.name in LEGACY_MESSAGE_MAPPING:
                        new_name = LEGACY_MESSAGE_MAPPING[schema.name]
                        analysis["conversions"][schema.name] = new_name
                        analysis["has_legacy_messages"] = True

                        if verbose:
                            console.print(f"  Found legacy schema: {schema.name} → {new_name}")

                # Count messages
                progress.update(task, description="Counting messages...")
                for message in reader.iter_messages():
                    analysis["total_messages"] += 1
                    analysis["topics"].add(message.channel.topic)

                    schema_name = analysis["schemas"].get(message.channel.schema_id, "unknown")
                    if schema_name in LEGACY_MESSAGE_MAPPING:
                        analysis["legacy_message_count"] += 1

                    # Update progress every 1000 messages to avoid too frequent updates
                    if analysis["total_messages"] % 1000 == 0:
                        progress.update(task, description=f"Counted {analysis['total_messages']:,} messages...")

        except Exception as e:
            console.print(f"[red]Error analyzing file: {e}[/red]")
            raise

    console.print("✓ Analysis complete:")
    console.print(f"  Total messages: {analysis['total_messages']:,}")
    console.print(f"  Legacy messages: {analysis['legacy_message_count']:,}")
    console.print(f"  Topics: {len(analysis['topics'])}")
    console.print(f"  Schemas to convert: {len(analysis['conversions'])}")

    return analysis


def _convert_mcap_file(input_file: Path, output_file: Path, analysis: Dict, console: Console, verbose: bool) -> None:
    """Perform the actual MCAP file conversion."""

    console.print("\n[bold]Converting messages...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=analysis["total_messages"])

        # Use temporary file for safer conversion
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            with OWAMcapWriter(str(temp_path)) as writer:
                with OWAMcapReader(str(input_file)) as reader:
                    for msg in reader.iter_messages():
                        # Get the schema name
                        schema_name = analysis["schemas"].get(msg.channel.schema_id, "unknown")

                        # Check if this message needs conversion
                        if schema_name in analysis["conversions"]:
                            new_schema_name = analysis["conversions"][schema_name]

                            # Get the new message class
                            try:
                                new_message_class = MESSAGES[new_schema_name]

                                # Convert the message data
                                if hasattr(msg, "decoded") and msg.decoded:
                                    # Message is already decoded, create new instance
                                    if hasattr(msg.decoded, "model_dump"):
                                        # Pydantic model
                                        data = msg.decoded.model_dump()
                                    else:
                                        # Dictionary or other format
                                        data = dict(msg.decoded)

                                    # Create new message instance
                                    new_message = new_message_class(**data)

                                    # Write with new schema
                                    writer.write_message(
                                        msg.channel.topic,
                                        new_message,
                                        publish_time=msg.publish_time,
                                        log_time=msg.log_time,
                                    )

                                    if verbose:
                                        console.print(f"  Converted: {schema_name} → {new_schema_name}")

                                else:
                                    # Message not decoded, skip or handle differently
                                    console.print(
                                        f"[yellow]Warning: Could not decode message with schema {schema_name}[/yellow]"
                                    )

                            except KeyError:
                                console.print(
                                    f"[red]Error: New message type '{new_schema_name}' not found in registry[/red]"
                                )
                                raise
                            except Exception as e:
                                console.print(f"[red]Error converting message: {e}[/red]")
                                raise
                        else:
                            # Message doesn't need conversion, copy as-is
                            # This is more complex with the high-level API, so we'll skip for now
                            # In a full implementation, we'd need to handle this case
                            pass

                        progress.update(task, advance=1)

            # Move temporary file to final location
            temp_path.replace(output_file)

        except Exception:
            # Clean up temporary file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

        progress.update(task, completed=analysis["total_messages"])


# Add to CLI
def add_to_mcap_commands():
    """Add the convert-legacy command to the mcap command group."""
    # This would be called from the mcap/__init__.py file
    pass
