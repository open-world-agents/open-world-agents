import re
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from owa.core import get_component_info, list_components

console = Console()


def list_plugins(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter plugins by namespace"),
    component_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Type of components to list (callables/listeners/runnables)"
    ),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search components by name pattern"),
    components: bool = typer.Option(False, "--components", "-c", help="Show individual components"),
    details: bool = typer.Option(False, "--details", "-d", help="Show detailed component information"),
    table_format: bool = typer.Option(False, "--table", help="Display results in table format"),
    sort_by: str = typer.Option("namespace", "--sort", help="Sort by: namespace, name, type"),
):
    """List discovered plugins and their components with enhanced filtering and display options."""

    # Validate component type
    if component_type and component_type not in ["callables", "listeners", "runnables"]:
        console.print(
            f"[red]Error: Invalid component type '{component_type}'. Must be one of: callables, listeners, runnables[/red]"
        )
        sys.exit(1)

    # Validate sort option
    if sort_by not in ["namespace", "name", "type"]:
        console.print(
            f"[red]Error: Invalid sort option '{sort_by}'. Must be one of: namespace, name, type[/red]"
        )
        sys.exit(1)

    if component_type:
        # List components of specific type
        _list_components_by_type(component_type, namespace, search, details, table_format, sort_by)
    else:
        # List all plugins
        _list_all_plugins(namespace, search, components, details, table_format, sort_by)


def _list_components_by_type(component_type: str, namespace: Optional[str], search: Optional[str],
                           details: bool, table_format: bool, sort_by: str):
    """List components of a specific type."""
    components = list_components(component_type, namespace=namespace)
    if not components or not components.get(component_type):
        console.print("[yellow]No components found[/yellow]")
        return

    # Filter by search pattern
    comp_list = components[component_type]
    if search:
        pattern = re.compile(search, re.IGNORECASE)
        comp_list = [name for name in comp_list if pattern.search(name)]

    if not comp_list:
        console.print(f"[yellow]No components found matching '{search}'[/yellow]")
        return

    # Sort components
    comp_list = _sort_components(comp_list, sort_by)

    if table_format:
        _display_components_table(component_type, comp_list, details)
    else:
        _display_components_tree(component_type, comp_list, details)


def _list_all_plugins(namespace: Optional[str], search: Optional[str], components: bool,
                     details: bool, table_format: bool, sort_by: str):
    """List all plugins with their components."""
    # Collect all plugins
    plugins = {}
    for comp_type in ["callables", "listeners", "runnables"]:
        plugin_components = list_components(comp_type, namespace=namespace)
        if plugin_components:
            for comp_name in plugin_components[comp_type]:
                # Apply search filter
                if search:
                    pattern = re.compile(search, re.IGNORECASE)
                    if not pattern.search(comp_name):
                        continue

                ns = comp_name.split("/")[0]
                if ns not in plugins:
                    plugins[ns] = {"callables": [], "listeners": [], "runnables": []}
                plugins[ns][comp_type].append(comp_name)

    if not plugins:
        search_msg = f" matching '{search}'" if search else ""
        console.print(f"[yellow]No plugins found{search_msg}[/yellow]")
        return

    if table_format and (components or details):
        # Show individual components in table format
        _display_all_components_table(plugins, details)
    elif table_format:
        # Show plugin overview in table format
        _display_plugins_table(plugins, components, details)
    else:
        # Show in tree format
        _display_plugins_tree(plugins, components, details)


def _sort_components(components: list, sort_by: str) -> list:
    """Sort components based on the specified criteria."""
    if sort_by == "namespace":
        return sorted(components, key=lambda x: (x.split("/")[0], x.split("/", 1)[1]))
    elif sort_by == "name":
        return sorted(components, key=lambda x: x.split("/", 1)[1] if "/" in x else x)
    elif sort_by == "type":
        # This doesn't apply to single-type lists, so fall back to namespace
        return sorted(components, key=lambda x: (x.split("/")[0], x.split("/", 1)[1]))
    return components


def _display_components_tree(component_type: str, components: list, details: bool):
    """Display components in tree format."""
    icon = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}
    tree = Tree(f"{icon.get(component_type, '�')} {component_type.title()} ({len(components)})")

    if details:
        comp_info = get_component_info(component_type)
        for comp_name in components:
            info = comp_info.get(comp_name, {})
            status = "✅ loaded" if info.get("loaded", False) else "⏳ lazy"
            import_path = info.get("import_path", "unknown")
            tree.add(f"{comp_name} [{status}] ({import_path})")
    else:
        for comp_name in components:
            tree.add(comp_name)

    console.print(tree)


def _display_components_table(component_type: str, components: list, details: bool):
    """Display components in table format."""
    table = Table(title=f"{component_type.title()} Components")
    table.add_column("Component", style="cyan")
    table.add_column("Namespace", style="green")
    table.add_column("Name", style="yellow")

    if details:
        table.add_column("Status", style="blue")
        table.add_column("Import Path", style="magenta")
        comp_info = get_component_info(component_type)

    for comp_name in components:
        parts = comp_name.split("/", 1)
        namespace = parts[0]
        name = parts[1] if len(parts) > 1 else ""

        if details:
            info = comp_info.get(comp_name, {})
            status = "Loaded" if info.get("loaded", False) else "Lazy"
            import_path = info.get("import_path", "unknown")
            table.add_row(comp_name, namespace, name, status, import_path)
        else:
            table.add_row(comp_name, namespace, name)

    console.print(table)


def _display_plugins_tree(plugins: dict, show_components: bool, show_details: bool):
    """Display plugins in tree format."""
    tree = Tree(f"📦 Discovered Plugins ({len(plugins)})")

    for ns in sorted(plugins.keys()):
        components = plugins[ns]
        total_count = sum(len(comps) for comps in components.values())
        plugin_branch = tree.add(f"{ns} ({total_count} components)")

        # Add component counts with proper tree formatting
        comp_types = []
        if components["callables"]:
            comp_types.append(f"📞 Callables: {len(components['callables'])}")
        if components["listeners"]:
            comp_types.append(f"👂 Listeners: {len(components['listeners'])}")
        if components["runnables"]:
            comp_types.append(f"🏃 Runnables: {len(components['runnables'])}")

        for i, comp_type in enumerate(comp_types):
            plugin_branch.add(comp_type)

        # Show individual components if requested OR if details requested
        if show_components or show_details:
            for comp_type in ["callables", "listeners", "runnables"]:
                if components[comp_type]:
                    type_branch = plugin_branch.add(f"🔧 {comp_type.title()} Details")

                    if show_details:
                        # Show with detailed information
                        comp_info = get_component_info(comp_type)
                        for comp_name in sorted(components[comp_type]):
                            info = comp_info.get(comp_name, {})
                            status = "✅ loaded" if info.get("loaded", False) else "⏳ lazy"
                            import_path = info.get("import_path", "unknown")
                            type_branch.add(f"{comp_name} [{status}] ({import_path})")
                    else:
                        # Show just component names
                        for comp_name in sorted(components[comp_type]):
                            type_branch.add(comp_name)

    console.print(tree)


def _display_all_components_table(plugins: dict, show_details: bool):
    """Display all components in table format."""
    table = Table(title="All Components")
    table.add_column("Component", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Namespace", style="yellow")
    table.add_column("Name", style="blue")

    if show_details:
        table.add_column("Status", style="magenta")
        table.add_column("Import Path", style="dim")

    # Collect all components
    all_components = []
    for ns in sorted(plugins.keys()):
        components = plugins[ns]
        for comp_type in ["callables", "listeners", "runnables"]:
            if components[comp_type]:
                comp_info = get_component_info(comp_type) if show_details else {}
                for comp_name in sorted(components[comp_type]):
                    parts = comp_name.split("/", 1)
                    namespace = parts[0]
                    name = parts[1] if len(parts) > 1 else ""

                    row_data = [comp_name, comp_type, namespace, name]

                    if show_details:
                        info = comp_info.get(comp_name, {})
                        status = "Loaded" if info.get("loaded", False) else "Lazy"
                        import_path = info.get("import_path", "unknown")
                        row_data.extend([status, import_path])

                    all_components.append(row_data)

    # Add rows to table
    for row_data in all_components:
        table.add_row(*row_data)

    console.print(table)


def _display_plugins_table(plugins: dict, show_components: bool, show_details: bool):
    """Display plugins in table format."""
    table = Table(title="Plugin Overview")
    table.add_column("Namespace", style="cyan")
    table.add_column("Callables", justify="right", style="green")
    table.add_column("Listeners", justify="right", style="yellow")
    table.add_column("Runnables", justify="right", style="blue")
    table.add_column("Total", justify="right", style="bold magenta")

    if show_details:
        table.add_column("Loaded", justify="right", style="blue")
        table.add_column("Load %", justify="right", style="magenta")

    for ns in sorted(plugins.keys()):
        components = plugins[ns]
        callables_count = len(components["callables"])
        listeners_count = len(components["listeners"])
        runnables_count = len(components["runnables"])
        total_count = callables_count + listeners_count + runnables_count

        row_data = [
            ns,
            str(callables_count) if callables_count > 0 else "-",
            str(listeners_count) if listeners_count > 0 else "-",
            str(runnables_count) if runnables_count > 0 else "-",
            str(total_count)
        ]

        if show_details:
            # Calculate loaded components
            loaded_count = 0
            for comp_type in ["callables", "listeners", "runnables"]:
                if components[comp_type]:
                    comp_info = get_component_info(comp_type)
                    loaded_count += sum(1 for comp_name in components[comp_type]
                                      if comp_info.get(comp_name, {}).get("loaded", False))

            load_percentage = (loaded_count / total_count) * 100 if total_count > 0 else 0
            row_data.extend([str(loaded_count), f"{load_percentage:.1f}%"])

        table.add_row(*row_data)

    console.print(table)

    # If showing components but not in table format, show individual components
    if show_components and not show_details:
        console.print("\n[dim]💡 Use --details (-d) to see import paths and load status[/dim]")
