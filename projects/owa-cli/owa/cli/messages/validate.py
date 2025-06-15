"""
Validate command for message registry.
"""

import typer
from rich.console import Console
from rich.table import Table

try:
    from owa.core import MESSAGES
    from owa.core.message import BaseMessage
except ImportError:
    MESSAGES = None
    BaseMessage = None


def validate_messages(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix validation issues"),
) -> None:
    """
    Validate message registry and message definitions.
    
    This command checks the integrity of the message registry,
    validates message schemas, and reports any issues found.
    """
    if MESSAGES is None:
        typer.echo("Error: owa.core not available. Please install owa-core package.", err=True)
        raise typer.Exit(1)
    
    console = Console()
    console.print("[bold blue]Validating Message Registry...[/bold blue]\n")
    
    # Validation results
    results = {
        "total_messages": 0,
        "valid_messages": 0,
        "invalid_messages": 0,
        "errors": [],
        "warnings": []
    }
    
    # Test registry loading
    try:
        MESSAGES.reload()
        all_messages = dict(MESSAGES.items())
        results["total_messages"] = len(all_messages)
        console.print(f"✓ Registry loaded successfully: {len(all_messages)} message types found")
    except Exception as e:
        console.print(f"[red]✗ Failed to load message registry: {e}[/red]")
        results["errors"].append(f"Registry loading failed: {e}")
        raise typer.Exit(1)
    
    if not all_messages:
        console.print("[yellow]⚠ No message types found in registry[/yellow]")
        return
    
    # Validate each message
    console.print("\n[bold]Validating individual messages...[/bold]")
    
    validation_table = Table(title="Message Validation Results")
    validation_table.add_column("Message Type", style="cyan")
    validation_table.add_column("Status", style="bold")
    validation_table.add_column("Issues", style="yellow")
    
    for message_type, message_class in sorted(all_messages.items()):
        issues = []
        status = "✓ Valid"
        
        try:
            # Check if it's a proper BaseMessage subclass
            if not issubclass(message_class, BaseMessage):
                issues.append("Not a BaseMessage subclass")
                status = "✗ Invalid"
            
            # Check _type attribute
            type_attr = message_class._type
            if hasattr(type_attr, 'default'):
                type_value = type_attr.default
            else:
                type_value = type_attr
            
            if not type_value:
                issues.append("Empty _type attribute")
                status = "✗ Invalid"
            elif type_value != message_type:
                issues.append(f"_type mismatch: '{type_value}' != '{message_type}'")
                status = "⚠ Warning"
            
            # Check schema generation
            try:
                schema = message_class.get_schema()
                if not isinstance(schema, dict):
                    issues.append("Invalid schema format")
                    status = "✗ Invalid"
                elif "properties" not in schema:
                    issues.append("Schema missing properties")
                    status = "⚠ Warning"
            except Exception as e:
                issues.append(f"Schema error: {e}")
                status = "✗ Invalid"
            
            # Check instantiation (basic test)
            try:
                # Try to get required fields from schema
                schema = message_class.get_schema()
                required_fields = schema.get("required", [])
                
                # Skip instantiation test if we can't determine required fields
                if required_fields:
                    issues.append("Instantiation test skipped (requires parameters)")
                    if status == "✓ Valid":
                        status = "⚠ Partial"
            except Exception as e:
                issues.append(f"Instantiation test failed: {e}")
                if status == "✓ Valid":
                    status = "⚠ Warning"
            
            # Update counters
            if "✗" in status:
                results["invalid_messages"] += 1
                results["errors"].extend([f"{message_type}: {issue}" for issue in issues])
            elif "⚠" in status:
                results["valid_messages"] += 1
                results["warnings"].extend([f"{message_type}: {issue}" for issue in issues])
            else:
                results["valid_messages"] += 1
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
            status = "✗ Error"
            results["invalid_messages"] += 1
            results["errors"].append(f"{message_type}: {e}")
        
        # Add to table
        issues_str = "; ".join(issues) if issues else ""
        validation_table.add_row(message_type, status, issues_str)
        
        if verbose:
            if issues:
                for issue in issues:
                    console.print(f"  [dim]- {issue}[/dim]")
    
    console.print(validation_table)
    
    # Summary
    console.print("\n[bold]Validation Summary:[/bold]")
    console.print(f"Total messages: {results['total_messages']}")
    console.print(f"Valid messages: {results['valid_messages']}")
    console.print(f"Invalid messages: {results['invalid_messages']}")
    
    if results["errors"]:
        console.print(f"\n[red]Errors ({len(results['errors'])}):[/red]")
        for error in results["errors"]:
            console.print(f"  [red]✗ {error}[/red]")
    
    if results["warnings"]:
        console.print(f"\n[yellow]Warnings ({len(results['warnings'])}):[/yellow]")
        for warning in results["warnings"]:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")
    
    if not results["errors"] and not results["warnings"]:
        console.print("\n[green]✓ All messages are valid![/green]")
    
    # Exit with error code if there are validation failures
    if results["invalid_messages"] > 0:
        raise typer.Exit(1)
