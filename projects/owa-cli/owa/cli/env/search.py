import re
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from owa.core import get_component_info, list_components

console = Console()


def search_components(
    pattern: str = typer.Argument(..., help="Search pattern (regex supported)"),
    component_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by component type"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", "-c", help="Case sensitive search"),
    details: bool = typer.Option(False, "--details", "-d", help="Show detailed component information"),
    table_format: bool = typer.Option(False, "--table", help="Display results in table format"),
    limit: int = typer.Option(50, "--limit", "-l", help="Limit number of results"),
):
    """Search for components across all plugins using pattern matching."""
    
    # Validate component type
    if component_type and component_type not in ["callables", "listeners", "runnables"]:
        console.print(
            f"[red]Error: Invalid component type '{component_type}'. Must be one of: callables, listeners, runnables[/red]"
        )
        sys.exit(1)

    # Compile search pattern
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
    except re.error as e:
        console.print(f"[red]Error: Invalid regex pattern '{pattern}': {e}[/red]")
        sys.exit(1)

    # Search across component types
    results = {}
    comp_types_to_search = [component_type] if component_type else ["callables", "listeners", "runnables"]
    
    for comp_type in comp_types_to_search:
        components = list_components(comp_type, namespace=namespace)
        if components and components.get(comp_type):
            matches = []
            for comp_name in components[comp_type]:
                if regex.search(comp_name):
                    matches.append(comp_name)
            
            if matches:
                # Sort by relevance (exact matches first, then by length)
                matches.sort(key=lambda x: (pattern.lower() not in x.lower(), len(x), x))
                results[comp_type] = matches[:limit]

    if not results:
        console.print(f"[yellow]No components found matching pattern '{pattern}'[/yellow]")
        return

    # Display results
    total_matches = sum(len(matches) for matches in results.values())
    
    if table_format:
        _display_search_results_table(results, details, pattern, total_matches)
    else:
        _display_search_results_tree(results, details, pattern, total_matches)


def _display_search_results_tree(results: dict, details: bool, pattern: str, total_matches: int):
    """Display search results in tree format."""
    tree = Tree(f"🔍 Search Results for '{pattern}' ({total_matches} matches)")
    
    icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}
    
    for comp_type, matches in results.items():
        icon = icons.get(comp_type, "🔧")
        type_tree = tree.add(f"{icon} {comp_type.title()} ({len(matches)})")
        
        if details:
            comp_info = get_component_info(comp_type)
            for comp_name in matches:
                info = comp_info.get(comp_name, {})
                status = "✅ loaded" if info.get("loaded", False) else "⏳ lazy"
                import_path = info.get("import_path", "unknown")
                type_tree.add(f"{comp_name} [{status}] ({import_path})")
        else:
            for comp_name in matches:
                type_tree.add(comp_name)
    
    console.print(tree)


def _display_search_results_table(results: dict, details: bool, pattern: str, total_matches: int):
    """Display search results in table format."""
    table = Table(title=f"Search Results for '{pattern}' ({total_matches} matches)")
    table.add_column("Component", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Namespace", style="yellow")
    table.add_column("Name", style="blue")
    
    if details:
        table.add_column("Status", style="magenta")
        table.add_column("Import Path", style="dim")

    # Collect all results with type information
    all_results = []
    for comp_type, matches in results.items():
        comp_info = get_component_info(comp_type) if details else {}
        for comp_name in matches:
            parts = comp_name.split("/", 1)
            namespace = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            
            result_data = {
                "component": comp_name,
                "type": comp_type,
                "namespace": namespace,
                "name": name,
            }
            
            if details:
                info = comp_info.get(comp_name, {})
                result_data["status"] = "Loaded" if info.get("loaded", False) else "Lazy"
                result_data["import_path"] = info.get("import_path", "unknown")
            
            all_results.append(result_data)

    # Sort results by relevance
    all_results.sort(key=lambda x: (pattern.lower() not in x["component"].lower(), len(x["component"]), x["component"]))

    # Add rows to table
    for result in all_results:
        if details:
            table.add_row(
                result["component"],
                result["type"],
                result["namespace"],
                result["name"],
                result["status"],
                result["import_path"]
            )
        else:
            table.add_row(
                result["component"],
                result["type"],
                result["namespace"],
                result["name"]
            )

    console.print(table)



