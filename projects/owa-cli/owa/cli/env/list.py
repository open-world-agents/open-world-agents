import re
import sys
from typing import List, Optional

import typer
from rich.table import Table
from rich.tree import Tree

from owa.core import get_component, get_component_info, list_components

from ..console import console


def list_env(
    namespaces: Optional[List[str]] = typer.Argument(None, help="Plugin namespace(s) to show"),
    components: bool = typer.Option(False, "--components", "-c", help="Show individual components"),
    details: bool = typer.Option(False, "--details", "-d", help="Show import paths and load status"),
    table: bool = typer.Option(False, "--table", help="Display in table format"),
    type_filter: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by component type (callables/listeners/runnables)"
    ),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search components by name pattern"),
    inspect: Optional[str] = typer.Option(
        None, "--inspect", help="Inspect specific component (show docstring/signature)"
    ),
):
    """List environment plugins and components."""

    # Validate component type
    if type_filter and type_filter not in ["callables", "listeners", "runnables"]:
        console.print(
            f"[red]Error: Invalid component type '{type_filter}'. Must be one of: callables, listeners, runnables[/red]"
        )
        sys.exit(1)

    # Smart defaults
    if namespaces:
        # When specific namespaces are provided, auto-show components
        components = True

    if inspect:
        # When inspecting, auto-enable details and focus on single namespace
        details = True
        if not namespaces:
            console.print("[red]Error: --inspect requires a namespace to be specified[/red]")
            sys.exit(1)
        if len(namespaces) > 1:
            console.print("[red]Error: --inspect can only be used with a single namespace[/red]")
            sys.exit(1)

    # Handle component inspection
    if inspect:
        _inspect_component(namespaces[0], inspect)
        return

    # Main display logic
    _display_plugins(namespaces, components, details, table, type_filter, search)


def _display_plugins(
    namespaces: Optional[List[str]],
    show_components: bool,
    show_details: bool,
    table_format: bool,
    type_filter: Optional[str],
    search: Optional[str],
):
    """Main display logic for plugins."""
    # Collect plugin data
    plugins_data = _collect_plugin_data(namespaces, type_filter, search)

    if not plugins_data:
        search_msg = f" matching '{search}'" if search else ""
        ns_msg = f" in namespace(s) {', '.join(namespaces)}" if namespaces else ""
        type_msg = f" of type '{type_filter}'" if type_filter else ""
        console.print(f"[yellow]No plugins found{ns_msg}{type_msg}{search_msg}[/yellow]")
        return

    if namespaces:
        # Specific namespace(s) view
        _display_specific_plugins(plugins_data, show_components, show_details, table_format)
    else:
        # All plugins overview
        _display_all_plugins_overview(plugins_data, show_components, show_details, table_format)


def _collect_plugin_data(namespaces: Optional[List[str]], type_filter: Optional[str], search: Optional[str]) -> dict:
    """Collect and filter plugin data efficiently."""
    plugins = {}
    comp_types = [type_filter] if type_filter else ["callables", "listeners", "runnables"]

    for comp_type in comp_types:
        namespace_filter = namespaces[0] if namespaces and len(namespaces) == 1 else None
        comp_data = list_components(comp_type, namespace=namespace_filter)

        if comp_data and comp_data.get(comp_type):
            for comp_name in comp_data[comp_type]:
                # Apply namespace filter for multiple namespaces
                if namespaces and len(namespaces) > 1:
                    comp_namespace = comp_name.split("/")[0]
                    if comp_namespace not in namespaces:
                        continue

                # Apply search filter
                if search:
                    pattern = re.compile(search, re.IGNORECASE)
                    if not pattern.search(comp_name):
                        continue

                ns = comp_name.split("/")[0]
                if ns not in plugins:
                    plugins[ns] = {"callables": [], "listeners": [], "runnables": []}
                plugins[ns][comp_type].append(comp_name)

    return plugins


def _display_specific_plugins(plugins_data: dict, show_components: bool, show_details: bool, table_format: bool):
    """Display specific plugin(s) with detailed view."""
    for namespace in sorted(plugins_data.keys()):
        components = plugins_data[namespace]
        total_count = sum(len(comps) for comps in components.values())

        # Plugin overview
        tree = Tree(f"📦 Plugin: {namespace} ({total_count} components)")
        icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}
        for comp_type, comps in components.items():
            if comps:
                icon = icons.get(comp_type, "🔧")
                tree.add(f"{icon} {comp_type.title()}: {len(comps)}")
        console.print(tree)

        # Show components if requested
        if show_components:
            if table_format:
                _display_components_table_detailed(components, show_details)
            else:
                _display_components_tree_detailed(components, show_details)


def _display_all_plugins_overview(plugins_data: dict, show_components: bool, show_details: bool, table_format: bool):
    """Display overview of all plugins."""
    if table_format and (show_components or show_details):
        _display_all_components_table(plugins_data, show_details)
    elif table_format:
        _display_plugins_table(plugins_data, show_components, show_details)
    else:
        _display_plugins_tree(plugins_data, show_components, show_details)


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


def _display_components_tree_detailed(components: dict, show_details: bool):
    """Display components in detailed tree format."""
    icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}

    for comp_type, comps in components.items():
        if not comps:
            continue

        icon = icons.get(comp_type, "🔧")
        comp_tree = Tree(f"{icon} {comp_type.title()} ({len(comps)})")

        if show_details:
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


def _display_components_table_detailed(components: dict, show_details: bool):
    """Display components in detailed table format."""
    if show_details:
        table = Table(title="Component Details")
        table.add_column("Component", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Import Path", style="blue")
    else:
        table = Table(title="Components")
        table.add_column("Component", style="cyan")
        table.add_column("Type", style="green")

    for comp_type, comps in components.items():
        if not comps:
            continue

        if show_details:
            comp_info = get_component_info(comp_type)

        for comp_name in sorted(comps):
            if show_details:
                info = comp_info.get(comp_name, {})
                status = "Loaded" if info.get("loaded", False) else "Lazy"
                import_path = info.get("import_path", "unknown")
                table.add_row(comp_name, comp_type, status, import_path)
            else:
                table.add_row(comp_name, comp_type)

    console.print(table)


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
            str(total_count),
        ]

        if show_details:
            # Calculate loaded components
            loaded_count = 0
            for comp_type in ["callables", "listeners", "runnables"]:
                if components[comp_type]:
                    comp_info = get_component_info(comp_type)
                    loaded_count += sum(
                        1 for comp_name in components[comp_type] if comp_info.get(comp_name, {}).get("loaded", False)
                    )

            load_percentage = (loaded_count / total_count) * 100 if total_count > 0 else 0
            row_data.extend([str(loaded_count), f"{load_percentage:.1f}%"])

        table.add_row(*row_data)

    console.print(table)

    # If showing components but not in table format, show individual components
    if show_components and not show_details:
        console.print("\n[dim]💡 Use --details (-d) to see import paths and load status[/dim]")


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
    if hasattr(component, "__doc__") and component.__doc__:
        doc_lines = component.__doc__.strip().split("\n")
        doc_tree = tree.add("📝 Documentation")
        for line in doc_lines[:10]:  # Limit to first 10 lines
            doc_tree.add(line.strip())
        if len(doc_lines) > 10:
            doc_tree.add("... (truncated)")
    else:
        tree.add("📝 Documentation: None")

    # Show signature for callables
    if comp_type_found == "callables" and hasattr(component, "__call__"):
        try:
            import inspect

            sig = inspect.signature(component)
            tree.add(f"🔧 Signature: {component.__name__}{sig}")
        except (ValueError, TypeError):
            tree.add("🔧 Signature: Unable to inspect")

    console.print(tree)
