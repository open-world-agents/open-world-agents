import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.tree import Tree

from owa.core.plugin_spec import PluginSpec

console = Console()


def validate_plugin(
    spec_path: str = typer.Argument(..., help="Path to plugin specification file (plugin.yaml)"),
):
    """Validate a plugin specification file."""
    spec_file = Path(spec_path)
    if not spec_file.exists():
        console.print(f"[red]Error: Specification file not found: {spec_path}[/red]")
        sys.exit(1)

    try:
        # Load and validate the specification
        spec = PluginSpec.from_yaml(spec_file)

        # Create validation tree
        tree = Tree(f"✅ Plugin Specification Valid")
        tree.add(f"├── Namespace: {spec.namespace}")
        tree.add(f"├── Version: {spec.version}")
        tree.add(f"├── Author: {spec.author}")
        tree.add(f"└── Description: {spec.description}")

        # Add component counts
        comp_tree = tree.add("🔧 Components")
        if spec.components.get("callables"):
            comp_tree.add(f"├── Callables: {len(spec.components['callables'])}")
        if spec.components.get("listeners"):
            comp_tree.add(f"├── Listeners: {len(spec.components['listeners'])}")
        if spec.components.get("runnables"):
            comp_tree.add(f"└── Runnables: {len(spec.components['runnables'])}")

        console.print(tree)

    except Exception as e:
        console.print(f"[red]Error: Invalid plugin specification: {str(e)}[/red]")
        sys.exit(1)
