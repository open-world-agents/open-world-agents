"""Message discovery and management commands for OEP-0005."""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from owa.core import get_component_info, list_components, get_plugin_discovery, MESSAGES

console = Console()


def list_messages(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search messages by name pattern"),
    details: bool = typer.Option(False, "--details", "-d", help="Show detailed message information"),
    table_format: bool = typer.Option(False, "--table", help="Display results in table format"),
):
    """List all available message types from installed plugins."""
    
    # Get all message components
    message_components = list_components("messages", namespace)
    
    if not message_components.get("messages"):
        search_msg = f" matching '{search}'" if search else ""
        namespace_msg = f" in namespace '{namespace}'" if namespace else ""
        console.print(f"[yellow]No message types found{namespace_msg}{search_msg}[/yellow]")
        return

    messages = message_components["messages"]
    
    # Apply search filter
    if search:
        messages = [msg for msg in messages if search.lower() in msg.lower()]
        
    if not messages:
        console.print(f"[yellow]No message types found matching '{search}'[/yellow]")
        return

    if table_format:
        _display_messages_table(messages, details)
    else:
        _display_messages_tree(messages, details)


def show_message(
    message_type: str = typer.Argument(..., help="Message type to show (namespace/MessageType)"),
    schema: bool = typer.Option(False, "--schema", "-s", help="Show message schema"),
):
    """Show detailed information about a specific message type."""
    
    try:
        # Try to load the message class to get detailed information
        message_class = MESSAGES[message_type]
        
        # Display message information
        tree = Tree(f"📨 Message Type: {message_type}")
        
        # Basic information
        tree.add(f"├── Type: {getattr(message_class, '_type', 'Unknown')}")
        tree.add(f"├── Class: {message_class.__name__}")
        tree.add(f"└── Module: {message_class.__module__}")
        
        # Show schema if requested
        if schema:
            try:
                schema_data = message_class.get_schema()
                schema_tree = tree.add("📋 Schema")
                
                # Show properties
                if "properties" in schema_data:
                    props_tree = schema_tree.add("Properties")
                    for prop_name, prop_info in schema_data["properties"].items():
                        prop_type = prop_info.get("type", "unknown")
                        required = prop_name in schema_data.get("required", [])
                        req_marker = " (required)" if required else " (optional)"
                        props_tree.add(f"{prop_name}: {prop_type}{req_marker}")
                
                # Show required fields
                if "required" in schema_data:
                    req_tree = schema_tree.add("Required Fields")
                    for field in schema_data["required"]:
                        req_tree.add(field)
                        
            except Exception as e:
                tree.add(f"[red]Error getting schema: {e}[/red]")
        
        console.print(tree)
        
    except KeyError:
        console.print(f"[red]Error: Message type '{message_type}' not found[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading message type '{message_type}': {e}[/red]")
        sys.exit(1)


def validate_messages(
    namespace: Optional[str] = typer.Argument(None, help="Namespace to validate (optional, validates all if not specified)"),
):
    """Validate message definitions in plugins."""
    
    plugin_discovery = get_plugin_discovery()
    
    if namespace:
        # Validate specific namespace
        plugins_to_check = {name: spec for name, spec in plugin_discovery.discovered_plugins.items() 
                          if spec.namespace == namespace}
        if not plugins_to_check:
            console.print(f"[red]Error: No plugin found with namespace '{namespace}'[/red]")
            sys.exit(1)
    else:
        # Validate all plugins
        plugins_to_check = plugin_discovery.discovered_plugins
    
    total_messages = 0
    valid_messages = 0
    errors = []
    
    for plugin_name, plugin_spec in plugins_to_check.items():
        if not plugin_spec.messages:
            continue
            
        for msg_name, import_path in plugin_spec.messages.items():
            total_messages += 1
            full_name = f"{plugin_spec.namespace}/{msg_name}"
            
            try:
                # Try to load the message class
                message_class = MESSAGES[full_name]
                
                # Validate _type attribute
                expected_type = full_name
                actual_type = getattr(message_class, '_type', None)
                
                if actual_type != expected_type:
                    errors.append(f"Message {full_name}: _type mismatch. Expected '{expected_type}', got '{actual_type}'")
                    continue
                
                # Validate schema generation
                try:
                    schema = message_class.get_schema()
                    if not isinstance(schema, dict):
                        errors.append(f"Message {full_name}: get_schema() must return a dictionary")
                        continue
                except Exception as e:
                    errors.append(f"Message {full_name}: Schema generation failed: {e}")
                    continue
                
                valid_messages += 1
                
            except Exception as e:
                errors.append(f"Message {full_name}: Failed to load: {e}")
    
    # Display results
    if total_messages == 0:
        console.print("[yellow]No message types found to validate[/yellow]")
        return
    
    tree = Tree("📨 Message Validation Results")
    tree.add(f"├── Total Messages: {total_messages}")
    tree.add(f"├── Valid: {valid_messages}")
    tree.add(f"└── Errors: {len(errors)}")
    
    if errors:
        error_tree = tree.add("[red]Validation Errors[/red]")
        for error in errors:
            error_tree.add(f"[red]• {error}[/red]")
    
    console.print(tree)
    
    # Exit with error code if there were validation failures
    if errors:
        sys.exit(1)


def _display_messages_tree(messages: list[str], show_details: bool = False):
    """Display messages in tree format."""
    
    # Group by namespace
    namespaces = {}
    for msg in messages:
        if "/" in msg:
            namespace, name = msg.split("/", 1)
            if namespace not in namespaces:
                namespaces[namespace] = []
            namespaces[namespace].append(name)
    
    tree = Tree(f"📨 Message Types ({len(messages)})")
    
    for namespace in sorted(namespaces.keys()):
        namespace_tree = tree.add(f"📦 {namespace} ({len(namespaces[namespace])} messages)")
        
        for msg_name in sorted(namespaces[namespace]):
            full_name = f"{namespace}/{msg_name}"
            if show_details:
                # Get detailed information
                msg_info = get_component_info("messages").get(full_name, {})
                status = "✅ loaded" if msg_info.get("loaded", False) else "⏳ lazy"
                import_path = msg_info.get("import_path", "unknown")
                namespace_tree.add(f"{msg_name} [{status}] ({import_path})")
            else:
                namespace_tree.add(msg_name)
    
    console.print(tree)


def _display_messages_table(messages: list[str], show_details: bool = False):
    """Display messages in table format."""
    
    if show_details:
        table = Table("Message Type", "Namespace", "Name", "Status", "Import Path")
        msg_info = get_component_info("messages")
    else:
        table = Table("Message Type", "Namespace", "Name")
    
    for msg in sorted(messages):
        if "/" in msg:
            namespace, name = msg.split("/", 1)
        else:
            namespace, name = "unknown", msg
        
        row_data = [msg, namespace, name]
        
        if show_details:
            info = msg_info.get(msg, {})
            status = "Loaded" if info.get("loaded", False) else "Lazy"
            import_path = info.get("import_path", "unknown")
            row_data.extend([status, import_path])
        
        table.add_row(*row_data)
    
    console.print(table)
