from collections import Counter
from typing import Dict, List

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from owa.core import get_component_info, list_components

console = Console()


def show_stats(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
    by_namespace: bool = typer.Option(False, "--by-namespace", "-n", help="Group statistics by namespace"),
    by_type: bool = typer.Option(False, "--by-type", "-t", help="Group statistics by component type"),
    loaded_only: bool = typer.Option(False, "--loaded-only", "-l", help="Show statistics for loaded components only"),
):
    """Show comprehensive statistics about the plugin ecosystem."""
    
    # Collect all component data
    all_data = _collect_component_data()
    
    if not all_data:
        console.print("[yellow]No components found[/yellow]")
        return

    # Filter for loaded components if requested
    if loaded_only:
        all_data = [item for item in all_data if item["loaded"]]
        if not all_data:
            console.print("[yellow]No loaded components found[/yellow]")
            return

    # Display different views based on options
    if by_namespace:
        _show_namespace_stats(all_data, detailed)
    elif by_type:
        _show_type_stats(all_data, detailed)
    else:
        _show_overview_stats(all_data, detailed)


def _collect_component_data() -> List[Dict]:
    """Collect comprehensive data about all components."""
    all_data = []
    
    for comp_type in ["callables", "listeners", "runnables"]:
        components = list_components(comp_type)
        if not components or not components.get(comp_type):
            continue
            
        comp_info = get_component_info(comp_type)
        
        for comp_name in components[comp_type]:
            namespace = comp_name.split("/")[0]
            name = comp_name.split("/", 1)[1] if "/" in comp_name else comp_name
            info = comp_info.get(comp_name, {})
            
            all_data.append({
                "full_name": comp_name,
                "namespace": namespace,
                "name": name,
                "type": comp_type,
                "loaded": info.get("loaded", False),
                "import_path": info.get("import_path", "unknown"),
            })
    
    return all_data


def _show_overview_stats(all_data: List[Dict], detailed: bool):
    """Show overall statistics overview."""
    total_components = len(all_data)
    loaded_components = sum(1 for item in all_data if item["loaded"])
    lazy_components = total_components - loaded_components
    
    # Count by type
    type_counts = Counter(item["type"] for item in all_data)
    
    # Count by namespace
    namespace_counts = Counter(item["namespace"] for item in all_data)
    
    # Create overview tree
    tree = Tree("📊 Plugin Ecosystem Statistics")
    
    # Overall counts
    overview = tree.add("📈 Overview")
    overview.add(f"Total Components: {total_components}")
    overview.add(f"✅ Loaded: {loaded_components}")
    overview.add(f"⏳ Lazy: {lazy_components}")
    overview.add(f"📦 Namespaces: {len(namespace_counts)}")
    
    # Type breakdown
    types = tree.add("🔧 By Component Type")
    for comp_type, count in type_counts.most_common():
        icon = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}.get(comp_type, "🔧")
        percentage = (count / total_components) * 100
        types.add(f"{icon} {comp_type.title()}: {count} ({percentage:.1f}%)")
    
    # Top namespaces
    namespaces = tree.add("📦 Top Namespaces")
    for namespace, count in namespace_counts.most_common(5):
        percentage = (count / total_components) * 100
        namespaces.add(f"{namespace}: {count} ({percentage:.1f}%)")
    
    console.print(tree)
    
    if detailed:
        _show_detailed_tables(all_data)


def _show_namespace_stats(all_data: List[Dict], detailed: bool):
    """Show statistics grouped by namespace."""
    namespace_data = {}
    
    for item in all_data:
        ns = item["namespace"]
        if ns not in namespace_data:
            namespace_data[ns] = {"callables": 0, "listeners": 0, "runnables": 0, "loaded": 0, "total": 0}
        
        namespace_data[ns][item["type"]] += 1
        namespace_data[ns]["total"] += 1
        if item["loaded"]:
            namespace_data[ns]["loaded"] += 1

    # Create table
    table = Table(title="Statistics by Namespace")
    table.add_column("Namespace", style="cyan")
    table.add_column("📞 Callables", justify="right", style="green")
    table.add_column("👂 Listeners", justify="right", style="yellow")
    table.add_column("🏃 Runnables", justify="right", style="blue")
    table.add_column("Total", justify="right", style="bold white")
    table.add_column("✅ Loaded", justify="right", style="green")
    table.add_column("Load %", justify="right", style="magenta")

    # Sort by total components (descending)
    sorted_namespaces = sorted(namespace_data.items(), key=lambda x: x[1]["total"], reverse=True)
    
    for namespace, data in sorted_namespaces:
        load_percentage = (data["loaded"] / data["total"]) * 100 if data["total"] > 0 else 0
        
        table.add_row(
            namespace,
            str(data["callables"]) if data["callables"] > 0 else "-",
            str(data["listeners"]) if data["listeners"] > 0 else "-",
            str(data["runnables"]) if data["runnables"] > 0 else "-",
            str(data["total"]),
            str(data["loaded"]),
            f"{load_percentage:.1f}%"
        )

    console.print(table)


def _show_type_stats(all_data: List[Dict], detailed: bool):
    """Show statistics grouped by component type."""
    type_data = {}
    
    for item in all_data:
        comp_type = item["type"]
        if comp_type not in type_data:
            type_data[comp_type] = {"namespaces": set(), "loaded": 0, "total": 0, "components": []}
        
        type_data[comp_type]["namespaces"].add(item["namespace"])
        type_data[comp_type]["total"] += 1
        type_data[comp_type]["components"].append(item)
        if item["loaded"]:
            type_data[comp_type]["loaded"] += 1

    # Create table
    table = Table(title="Statistics by Component Type")
    table.add_column("Type", style="cyan")
    table.add_column("Icon", justify="center")
    table.add_column("Total", justify="right", style="bold white")
    table.add_column("✅ Loaded", justify="right", style="green")
    table.add_column("Load %", justify="right", style="magenta")
    table.add_column("📦 Namespaces", justify="right", style="yellow")
    table.add_column("Avg per NS", justify="right", style="blue")

    icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}
    
    for comp_type in ["callables", "listeners", "runnables"]:
        if comp_type not in type_data:
            continue
            
        data = type_data[comp_type]
        load_percentage = (data["loaded"] / data["total"]) * 100 if data["total"] > 0 else 0
        avg_per_ns = data["total"] / len(data["namespaces"]) if data["namespaces"] else 0
        
        table.add_row(
            comp_type.title(),
            icons.get(comp_type, "🔧"),
            str(data["total"]),
            str(data["loaded"]),
            f"{load_percentage:.1f}%",
            str(len(data["namespaces"])),
            f"{avg_per_ns:.1f}"
        )

    console.print(table)

    if detailed:
        # Show top components by type
        for comp_type, data in type_data.items():
            if not data["components"]:
                continue
                
            console.print(f"\n[bold]{icons.get(comp_type, '🔧')} Top {comp_type.title()}:[/bold]")
            
            # Group by namespace and show counts
            ns_counts = Counter(item["namespace"] for item in data["components"])
            for namespace, count in ns_counts.most_common(5):
                console.print(f"  {namespace}: {count} components")


def _show_detailed_tables(all_data: List[Dict]):
    """Show additional detailed tables."""
    console.print("\n")
    
    # Import path analysis
    import_paths = [item["import_path"] for item in all_data if item["import_path"] != "unknown"]
    if import_paths:
        # Extract package names from import paths
        packages = []
        for path in import_paths:
            if "." in path:
                # Take the first two parts of the module path
                parts = path.split(".")
                if len(parts) >= 2:
                    packages.append(f"{parts[0]}.{parts[1]}")
                else:
                    packages.append(parts[0])
        
        package_counts = Counter(packages)
        
        table = Table(title="Top Source Packages")
        table.add_column("Package", style="cyan")
        table.add_column("Components", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        
        total_with_paths = len(import_paths)
        for package, count in package_counts.most_common(10):
            percentage = (count / total_with_paths) * 100
            table.add_row(package, str(count), f"{percentage:.1f}%")
        
        console.print(table)


def health_check():
    """Perform a health check on the plugin ecosystem."""
    console.print("🏥 [bold]Plugin Ecosystem Health Check[/bold]\n")
    
    all_data = _collect_component_data()
    issues = []
    warnings = []
    
    if not all_data:
        issues.append("No components found - plugin discovery may not be working")
        console.print("[red]❌ Critical: No components discovered[/red]")
        return
    
    # Check for basic health indicators
    total_components = len(all_data)
    loaded_components = sum(1 for item in all_data if item["loaded"])
    unknown_imports = sum(1 for item in all_data if item["import_path"] == "unknown")
    
    # Health checks
    if loaded_components == 0:
        warnings.append("No components are currently loaded")
    
    if unknown_imports > total_components * 0.1:  # More than 10% unknown
        warnings.append(f"{unknown_imports} components have unknown import paths")
    
    namespace_counts = Counter(item["namespace"] for item in all_data)
    if len(namespace_counts) == 1:
        warnings.append("Only one namespace found - consider organizing into multiple namespaces")
    
    # Display results
    if not issues and not warnings:
        console.print("✅ [green]All health checks passed![/green]")
    else:
        if issues:
            console.print("[red]❌ Issues found:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
        
        if warnings:
            console.print("[yellow]⚠️  Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")
    
    # Summary
    console.print("\n📊 [bold]Summary:[/bold]")
    console.print(f"  • Total components: {total_components}")
    console.print(f"  • Loaded components: {loaded_components}")
    console.print(f"  • Namespaces: {len(namespace_counts)}")
    console.print(f"  • Load ratio: {(loaded_components/total_components)*100:.1f}%")
