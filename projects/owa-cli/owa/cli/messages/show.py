"""
Show command for message details.
"""

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

try:
    from owa.core import MESSAGES
except ImportError:
    MESSAGES = None


def show_message(
    message_type: str = typer.Argument(..., help="Message type to show (e.g., 'desktop/KeyboardEvent')"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json, schema"),
    example: bool = typer.Option(False, "--example", "-e", help="Show usage example"),
) -> None:
    """
    Show detailed information about a specific message type.

    This command displays comprehensive information about a message type,
    including its schema, properties, and usage examples.
    """
    if MESSAGES is None:
        typer.echo("Error: owa.core not available. Please install owa-core package.", err=True)
        raise typer.Exit(1)

    console = Console()

    # Get the message class
    try:
        message_class = MESSAGES[message_type]
    except KeyError:
        console.print(f"[red]Error: Message type '{message_type}' not found.[/red]")
        console.print("Use 'owl messages list' to see available message types.")
        raise typer.Exit(1)

    if format == "rich":
        _show_rich_format(console, message_type, message_class, example)
    elif format == "json":
        _show_json_format(message_type, message_class)
    elif format == "schema":
        _show_schema_format(message_type, message_class)
    else:
        typer.echo(f"Error: Unknown format '{format}'. Use: rich, json, schema", err=True)
        raise typer.Exit(1)


def _show_rich_format(console: Console, message_type: str, message_class, show_example: bool) -> None:
    """Show message information in rich format."""

    # Header
    console.print(f"\n[bold cyan]Message Type: {message_type}[/bold cyan]")

    # Basic information
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value")

    info_table.add_row("Class Name", message_class.__name__)
    info_table.add_row("Module", message_class.__module__)
    info_table.add_row("Domain", message_type.split("/")[0] if "/" in message_type else "unknown")

    # Get _type value
    type_attr = message_class._type
    if hasattr(type_attr, "default"):
        type_value = type_attr.default
    else:
        type_value = type_attr
    info_table.add_row("Type ID", type_value)

    console.print(Panel(info_table, title="Basic Information", border_style="blue"))

    # Schema information
    try:
        schema = message_class.get_schema()

        # Properties table
        if "properties" in schema:
            props_table = Table(title="Schema Properties")
            props_table.add_column("Property", style="cyan")
            props_table.add_column("Type", style="green")
            props_table.add_column("Required", style="yellow")
            props_table.add_column("Description", style="dim")

            required_props = set(schema.get("required", []))

            for prop_name, prop_info in schema["properties"].items():
                prop_type = prop_info.get("type", "unknown")
                if "anyOf" in prop_info:
                    # Handle union types
                    types = [t.get("type", str(t)) for t in prop_info["anyOf"]]
                    prop_type = " | ".join(types)

                is_required = "✓" if prop_name in required_props else ""
                description = prop_info.get("description", "")

                props_table.add_row(prop_name, prop_type, is_required, description)

            console.print(props_table)

        # Full schema (collapsed)
        schema_json = json.dumps(schema, indent=2)
        schema_syntax = Syntax(schema_json, "json", theme="monokai", line_numbers=True)
        console.print(Panel(schema_syntax, title="JSON Schema", border_style="green", expand=False))

    except Exception as e:
        console.print(f"[red]Error getting schema: {e}[/red]")

    # Usage example
    if show_example:
        _show_usage_example(console, message_type, message_class)


def _show_usage_example(console: Console, message_type: str, message_class) -> None:
    """Show usage example for the message."""

    # Generate example based on message type
    if "KeyboardEvent" in message_class.__name__:
        example_code = f"""from owa.core import MESSAGES

# Access via registry
{message_class.__name__} = MESSAGES['{message_type}']

# Create instance
event = {message_class.__name__}(
    event_type="press",
    vk=65,  # 'A' key
    timestamp=1234567890
)

# Use in OWAMcap
from mcap_owa.highlevel import OWAMcapWriter
with OWAMcapWriter("events.mcap") as writer:
    writer.write_message("keyboard", event)"""

    elif "MouseEvent" in message_class.__name__:
        example_code = f"""from owa.core import MESSAGES

# Access via registry
{message_class.__name__} = MESSAGES['{message_type}']

# Create instance
event = {message_class.__name__}(
    event_type="click",
    x=100,
    y=200,
    button="left",
    pressed=True,
    timestamp=1234567890
)

# Use in OWAMcap
from mcap_owa.highlevel import OWAMcapWriter
with OWAMcapWriter("events.mcap") as writer:
    writer.write_message("mouse", event)"""

    else:
        example_code = f"""from owa.core import MESSAGES

# Access via registry
{message_class.__name__} = MESSAGES['{message_type}']

# Create instance (adjust parameters as needed)
message = {message_class.__name__}(
    # Add required parameters here
)

# Use in OWAMcap
from mcap_owa.highlevel import OWAMcapWriter
with OWAMcapWriter("data.mcap") as writer:
    writer.write_message("topic", message)"""

    syntax = Syntax(example_code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Usage Example", border_style="yellow"))


def _show_json_format(message_type: str, message_class) -> None:
    """Show message information in JSON format."""

    # Get _type value
    type_attr = message_class._type
    if hasattr(type_attr, "default"):
        type_value = type_attr.default
    else:
        type_value = type_attr

    info = {
        "message_type": message_type,
        "class_name": message_class.__name__,
        "module": message_class.__module__,
        "type_id": type_value,
        "domain": message_type.split("/")[0] if "/" in message_type else "unknown",
    }

    try:
        info["schema"] = message_class.get_schema()
    except Exception as e:
        info["schema_error"] = str(e)

    print(json.dumps(info, indent=2))


def _show_schema_format(message_type: str, message_class) -> None:
    """Show only the JSON schema."""
    try:
        schema = message_class.get_schema()
        print(json.dumps(schema, indent=2))
    except Exception as e:
        typer.echo(f"Error getting schema: {e}", err=True)
        raise typer.Exit(1)
