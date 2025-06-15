"""
List command for message registry.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

try:
    from owa.core import MESSAGES
except ImportError:
    MESSAGES = None


def list_messages(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Filter by domain (e.g., 'desktop')"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """
    List all available message types in the registry.
    
    This command displays all message types discovered through entry points,
    with optional filtering by domain and different output formats.
    """
    if MESSAGES is None:
        typer.echo("Error: owa.core not available. Please install owa-core package.", err=True)
        raise typer.Exit(1)
    
    console = Console()
    
    # Get all message types
    try:
        all_messages = dict(MESSAGES.items())
    except Exception as e:
        typer.echo(f"Error loading messages: {e}", err=True)
        raise typer.Exit(1)
    
    if not all_messages:
        console.print("[yellow]No message types found in registry.[/yellow]")
        return
    
    # Filter by domain if specified
    if domain:
        filtered_messages = {
            name: cls for name, cls in all_messages.items() 
            if name.startswith(f"{domain}/")
        }
        if not filtered_messages:
            console.print(f"[yellow]No message types found for domain '{domain}'.[/yellow]")
            return
        all_messages = filtered_messages
    
    # Output in different formats
    if format == "table":
        _output_table(console, all_messages, verbose)
    elif format == "json":
        _output_json(all_messages, verbose)
    elif format == "yaml":
        _output_yaml(all_messages, verbose)
    else:
        typer.echo(f"Error: Unknown format '{format}'. Use: table, json, yaml", err=True)
        raise typer.Exit(1)


def _output_table(console: Console, messages: dict, verbose: bool) -> None:
    """Output messages in table format."""
    table = Table(title="Available Message Types")
    
    table.add_column("Message Type", style="cyan", no_wrap=True)
    table.add_column("Domain", style="green")
    table.add_column("Class Name", style="blue")
    
    if verbose:
        table.add_column("Module", style="dim")
        table.add_column("Schema Properties", style="yellow")
    
    for message_type, message_class in sorted(messages.items()):
        domain = message_type.split("/")[0] if "/" in message_type else "unknown"
        class_name = message_class.__name__
        
        row = [message_type, domain, class_name]
        
        if verbose:
            module = message_class.__module__
            
            # Get schema properties
            try:
                schema = message_class.get_schema()
                properties = list(schema.get("properties", {}).keys())
                properties_str = ", ".join(properties[:5])  # Limit to first 5
                if len(properties) > 5:
                    properties_str += f" (+{len(properties) - 5} more)"
            except Exception:
                properties_str = "N/A"
            
            row.extend([module, properties_str])
        
        table.add_row(*row)
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(messages)} message types[/dim]")


def _output_json(messages: dict, verbose: bool) -> None:
    """Output messages in JSON format."""
    import json
    
    output = {}
    for message_type, message_class in messages.items():
        info = {
            "class_name": message_class.__name__,
            "module": message_class.__module__,
            "domain": message_type.split("/")[0] if "/" in message_type else "unknown"
        }
        
        if verbose:
            try:
                schema = message_class.get_schema()
                info["schema"] = schema
            except Exception as e:
                info["schema_error"] = str(e)
        
        output[message_type] = info
    
    print(json.dumps(output, indent=2))


def _output_yaml(messages: dict, verbose: bool) -> None:
    """Output messages in YAML format."""
    try:
        import yaml
    except ImportError:
        typer.echo("Error: PyYAML not installed. Use 'pip install pyyaml'", err=True)
        raise typer.Exit(1)
    
    output = {}
    for message_type, message_class in messages.items():
        info = {
            "class_name": message_class.__name__,
            "module": message_class.__module__,
            "domain": message_type.split("/")[0] if "/" in message_type else "unknown"
        }
        
        if verbose:
            try:
                schema = message_class.get_schema()
                info["schema"] = schema
            except Exception as e:
                info["schema_error"] = str(e)
        
        output[message_type] = info
    
    print(yaml.dump(output, default_flow_style=False))
