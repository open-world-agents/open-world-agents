import re
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from owa.core import get_component, get_component_info, list_components

console = Console()


def show_plugin(
    namespace: str = typer.Argument(..., help="Namespace of the plugin to show"),
    components: bool = typer.Option(False, "--components", "-c", help="Show individual components"),
    component_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by component type"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search components by name pattern"),
    details: bool = typer.Option(False, "--details", "-d", help="Show import paths and load status"),
    table_format: bool = typer.Option(False, "--table", help="Display components in table format"),
    inspect: Optional[str] = typer.Option(None, "--inspect", "-i", help="Inspect specific component (show docstring/signature)"),
):
    """Show detailed information about a plugin with enhanced filtering and inspection options."""

    # Validate component type
    if component_type and component_type not in ["callables", "listeners", "runnables"]:
        console.print(
            f"[red]Error: Invalid component type '{component_type}'. Must be one of: callables, listeners, runnables[/red]"
        )
        sys.exit(1)

    # Get all components for the namespace
    plugin_components = {}
    comp_types_to_check = [component_type] if component_type else ["callables", "listeners", "runnables"]

    for comp_type in comp_types_to_check:
        comps = list_components(comp_type, namespace=namespace)
        if comps and comps.get(comp_type):
            # Apply search filter
            comp_list = comps[comp_type]
            if search:
                pattern = re.compile(search, re.IGNORECASE)
                comp_list = [name for name in comp_list if pattern.search(name)]

            if comp_list:
                plugin_components[comp_type] = comp_list

    if not plugin_components:
        search_msg = f" matching '{search}'" if search else ""
        type_msg = f" of type '{component_type}'" if component_type else ""
        console.print(f"[red]Error: No plugin found with namespace '{namespace}'{type_msg}{search_msg}[/red]")
        sys.exit(1)

    # Handle component inspection
    if inspect:
        _inspect_component(namespace, inspect)
        return

    # Always show plugin overview
    _display_plugin_overview(namespace, plugin_components)

    # Show individual components if requested OR if details flag is used
    # (details flag implies user wants to see component details)
    if components or details:
        if table_format:
            _display_components_table_detailed(plugin_components, details)
        else:
            _display_components_tree_detailed(plugin_components, details)


def _display_plugin_overview(namespace: str, components: dict):
    """Display plugin overview with component counts."""
    total_count = sum(len(comps) for comps in components.values())
    tree = Tree(f"📦 Plugin: {namespace} ({total_count} components)")

    # Add component counts with icons
    icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}
    for comp_type, comps in components.items():
        icon = icons.get(comp_type, "🔧")
        tree.add(f"{icon} {comp_type.title()}: {len(comps)}")

    console.print(tree)


def _display_components_tree_detailed(components: dict, details: bool):
    """Display components in detailed tree format."""
    icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}

    for comp_type, comps in components.items():
        icon = icons.get(comp_type, "🔧")
        comp_tree = Tree(f"{icon} {comp_type.title()} ({len(comps)})")

        if details:
            comp_info = get_component_info(comp_type)
            for comp_name in sorted(comps):
                info = comp_info.get(comp_name, {})
                status = "✅ loaded" if info.get("loaded", False) else "⏳ lazy"
                import_path = info.get("import_path", "unknown")
                comp_tree.add(f"{comp_name} [{status}] ({import_path})")
        else:
            for comp_name in sorted(comps):
                comp_tree.add(comp_name)

        console.print(comp_tree)


def _display_components_table_detailed(components: dict, details: bool):
    """Display components in detailed table format."""
    for comp_type, comps in components.items():
        table = Table(title=f"{comp_type.title()} Components")
        table.add_column("Component", style="cyan")
        table.add_column("Name", style="yellow")

        if details:
            table.add_column("Status", style="blue")
            table.add_column("Import Path", style="magenta")
            comp_info = get_component_info(comp_type)

        for comp_name in sorted(comps):
            # Extract just the name part (after namespace/)
            name_part = comp_name.split("/", 1)[1] if "/" in comp_name else comp_name

            if details:
                info = comp_info.get(comp_name, {})
                status = "Loaded" if info.get("loaded", False) else "Lazy"
                import_path = info.get("import_path", "unknown")
                table.add_row(comp_name, name_part, status, import_path)
            else:
                table.add_row(comp_name, name_part)

        console.print(table)


def _inspect_component(namespace: str, component_name: str):
    """Inspect a specific component to show its docstring and signature."""
    # Try to find the component in any type
    full_name = f"{namespace}/{component_name}"
    component = None
    comp_type_found = None

    for comp_type in ["callables", "listeners", "runnables"]:
        try:
            component = get_component(comp_type, namespace=namespace, name=component_name)
            comp_type_found = comp_type
            break
        except (KeyError, ValueError):
            continue

    if component is None:
        console.print(f"[red]Error: Component '{full_name}' not found[/red]")
        sys.exit(1)

    # Display component information
    tree = Tree(f"🔍 Component: {full_name}")
    tree.add(f"Type: {comp_type_found}")
    tree.add(f"Class: {component.__class__.__name__}")

    # Show docstring if available
    if hasattr(component, '__doc__') and component.__doc__:
        doc_lines = component.__doc__.strip().split('\n')
        doc_tree = tree.add("📝 Documentation")
        for line in doc_lines[:10]:  # Limit to first 10 lines
            doc_tree.add(line.strip())
        if len(doc_lines) > 10:
            doc_tree.add("... (truncated)")
    else:
        tree.add("📝 Documentation: None")

    # Show signature for callables
    if comp_type_found == "callables" and hasattr(component, '__call__'):
        try:
            import inspect
            sig = inspect.signature(component)
            tree.add(f"🔧 Signature: {component.__name__}{sig}")
        except (ValueError, TypeError):
            tree.add("🔧 Signature: Unable to inspect")

    console.print(tree)
