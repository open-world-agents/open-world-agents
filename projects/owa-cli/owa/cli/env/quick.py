"""Quick access commands for common operations."""

import typer
from rich.console import Console

from . import list as list_module
from . import search as search_module
from . import show as show_module

console = Console()


def ls(
    namespace: str = typer.Argument(None, help="Namespace to list (optional)"),
):
    """Quick list - show plugins or components in a namespace (alias for 'list')."""
    if namespace:
        # Show specific namespace with components
        show_module.show_plugin(
            namespace=namespace,
            show_components=True,
            component_type=None,
            search=None,
            details=False,
            table_format=False,
            inspect=None
        )
    else:
        # Show all plugins
        list_module.list_plugins(
            namespace=None,
            component_type=None,
            search=None,
            details=False,
            table_format=False,
            sort_by="namespace"
        )


def find(
    pattern: str = typer.Argument(..., help="Pattern to search for"),
):
    """Quick search - find components by pattern (alias for 'search')."""
    search_module.search_components(
        pattern=pattern,
        component_type=None,
        namespace=None,
        case_sensitive=False,
        details=False,
        table_format=True,
        limit=50
    )





def namespaces():
    """List all available namespaces."""
    from owa.core import list_components
    from collections import Counter
    
    # Collect all namespaces
    all_namespaces = set()
    namespace_counts = Counter()
    
    for comp_type in ["callables", "listeners", "runnables"]:
        components = list_components(comp_type)
        if components and components.get(comp_type):
            for comp_name in components[comp_type]:
                namespace = comp_name.split("/")[0]
                all_namespaces.add(namespace)
                namespace_counts[namespace] += 1
    
    if not all_namespaces:
        console.print("[yellow]No namespaces found[/yellow]")
        return
    
    from rich.table import Table
    
    table = Table(title="Available Namespaces")
    table.add_column("Namespace", style="cyan")
    table.add_column("Components", justify="right", style="green")
    table.add_column("Quick Access", style="dim")
    
    for namespace in sorted(all_namespaces):
        count = namespace_counts[namespace]
        table.add_row(
            namespace,
            str(count),
            f"owl env ls {namespace}"
        )
    
    console.print(table)
    console.print(f"\n[dim]💡 Found {len(all_namespaces)} namespaces with {sum(namespace_counts.values())} total components[/dim]")



