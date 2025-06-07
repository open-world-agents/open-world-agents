import sys

import typer
from rich.console import Console
from rich.tree import Tree

from owa.core import list_components

console = Console()


def show_plugin(
    namespace: str = typer.Argument(..., help="Namespace of the plugin to show"),
    show_components: bool = typer.Option(False, "--components", help="Show detailed component information"),
):
    """Show detailed information about a plugin."""
    # Get all components for the namespace
    components = {}
    for comp_type in ["callables", "listeners", "runnables"]:
        comps = list_components(comp_type, namespace=namespace)
        if comps:
            components[comp_type] = comps[comp_type]

    if not components:
        console.print(f"[red]Error: No plugin found with namespace '{namespace}'[/red]")
        sys.exit(1)

    # Create tree for plugin info
    tree = Tree(f"📦 Plugin: {namespace}")

    # Add component counts
    if "callables" in components:
        tree.add(f"├── Callables: {len(components['callables'])}")
    if "listeners" in components:
        tree.add(f"├── Listeners: {len(components['listeners'])}")
    if "runnables" in components:
        tree.add(f"└── Runnables: {len(components['runnables'])}")

    console.print(tree)

    if show_components:
        # Show detailed component information
        for comp_type, comps in components.items():
            comp_tree = Tree(f"🔧 {comp_type.title()}")
            for comp_name in comps:
                comp_tree.add(f"├── {comp_name}")
            console.print(comp_tree)
