import sys
from typing import Optional

import typer
from rich.console import Console
from rich.tree import Tree

from owa.core import CALLABLES, LISTENERS, RUNNABLES, get_component, list_components

console = Console()


def list_plugins(
    namespace: Optional[str] = typer.Option(None, help="Filter plugins by namespace"),
    component_type: Optional[str] = typer.Option(
        None, help="Type of components to list (callables/listeners/runnables)"
    ),
):
    """List discovered plugins and their components."""
    if component_type:
        # List components of specific type
        if component_type not in ["callables", "listeners", "runnables"]:
            console.print(
                f"[red]Error: Invalid component type '{component_type}'. Must be one of: callables, listeners, runnables[/red]"
            )
            sys.exit(1)

        components = list_components(component_type, namespace=namespace)
        if not components:
            console.print("[yellow]No components found[/yellow]")
            return

        # Create tree for components
        tree = Tree(f"📞 {component_type.title()}")
        for comp_name in components[component_type]:
            tree.add(f"├── {comp_name}")

        console.print(tree)
        return

    # List all plugins
    plugins = {}
    for comp_type in ["callables", "listeners", "runnables"]:
        components = list_components(comp_type, namespace=namespace)
        if components:
            for comp_name in components[comp_type]:
                ns = comp_name.split("/")[0]
                if ns not in plugins:
                    plugins[ns] = {"callables": [], "listeners": [], "runnables": []}
                plugins[ns][comp_type].append(comp_name)

    if not plugins:
        console.print("[yellow]No plugins found[/yellow]")
        return

    # Create tree for plugins
    tree = Tree("📦 Discovered Plugins")
    for ns, components in sorted(plugins.items()):
        plugin_branch = tree.add(f"├── {ns}")

        # Add component counts
        if components["callables"]:
            plugin_branch.add(f"├── Callables: {len(components['callables'])}")
        if components["listeners"]:
            plugin_branch.add(f"├── Listeners: {len(components['listeners'])}")
        if components["runnables"]:
            plugin_branch.add(f"└── Runnables: {len(components['runnables'])}")

    console.print(tree)
