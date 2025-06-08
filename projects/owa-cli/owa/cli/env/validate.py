import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from owa.core.documentation import PluginDocumentationGenerator
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
        tree = Tree("✅ Plugin Specification Valid")
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


def validate_docs_command(
    namespace: Optional[str] = typer.Argument(None, help="Plugin namespace to validate (or --all for all plugins)"),
    all_plugins: bool = typer.Option(False, "--all", help="Validate documentation for all plugins"),
    show_recommendations: bool = typer.Option(
        True, "--recommendations/--no-recommendations", help="Show improvement recommendations"
    ),
):
    """Validate plugin documentation quality and completeness."""

    if not namespace and not all_plugins:
        console.print("[red]Error: Must specify either a plugin namespace or --all[/red]")
        sys.exit(1)

    generator = PluginDocumentationGenerator()

    if all_plugins:
        # Validate all plugins
        ecosystem_docs = generator.generate_ecosystem_documentation()
        if not ecosystem_docs:
            console.print("[yellow]No plugins found to validate[/yellow]")
            return

        console.print(f"[blue]Validating documentation for {len(ecosystem_docs)} plugins...[/blue]\n")

        overall_results = []
        for plugin_namespace in ecosystem_docs.keys():
            result = generator.validate_plugin_documentation(plugin_namespace)
            overall_results.append((plugin_namespace, result))
            _display_validation_result(plugin_namespace, result, show_recommendations)
            console.print()  # Add spacing between plugins

        # Show summary
        _display_validation_summary(overall_results)
    else:
        # Validate specific plugin
        result = generator.validate_plugin_documentation(namespace)
        if not result:
            console.print(f"[red]Error: Plugin '{namespace}' not found[/red]")
            sys.exit(1)

        _display_validation_result(namespace, result, show_recommendations)


def _display_validation_result(namespace: str, result: dict, show_recommendations: bool):
    """Display validation results for a single plugin."""
    status_icon = "✅" if result["valid"] else "❌"

    tree = Tree(f"{status_icon} Plugin: {namespace}")

    # Add basic stats
    stats_node = tree.add("📊 Documentation Stats")
    stats_node.add(f"Total Components: {result['total_components']}")
    stats_node.add(f"Documented Components: {result['documented_components']}")

    if result["total_components"] > 0:
        coverage = (result["documented_components"] / result["total_components"]) * 100
        stats_node.add(f"Documentation Coverage: {coverage:.1f}%")

    # Add errors
    if result["errors"]:
        error_node = tree.add(f"[red]❌ Errors ({len(result['errors'])})[/red]")
        for error in result["errors"]:
            error_node.add(f"[red]• {error}[/red]")

    # Add warnings
    if result["warnings"]:
        warning_node = tree.add(f"[yellow]⚠️ Warnings ({len(result['warnings'])})[/yellow]")
        for warning in result["warnings"]:
            warning_node.add(f"[yellow]• {warning}[/yellow]")

    # Add recommendations
    if show_recommendations and result["recommendations"]:
        rec_node = tree.add(f"[blue]💡 Recommendations ({len(result['recommendations'])})[/blue]")
        for rec in result["recommendations"]:
            rec_node.add(f"[blue]• {rec}[/blue]")

    console.print(tree)


def _display_validation_summary(results: list):
    """Display summary of validation results for multiple plugins."""
    table = Table(title="Documentation Validation Summary")
    table.add_column("Plugin", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Components", justify="right")
    table.add_column("Documented", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Issues", justify="right")

    total_plugins = len(results)
    valid_plugins = 0
    total_components = 0
    total_documented = 0
    total_issues = 0

    for namespace, result in results:
        status = "✅ Valid" if result["valid"] else "❌ Invalid"
        if result["valid"]:
            valid_plugins += 1

        components = result["total_components"]
        documented = result["documented_components"]
        coverage = f"{(documented / components) * 100:.1f}%" if components > 0 else "N/A"
        issues = len(result["errors"]) + len(result["warnings"])

        total_components += components
        total_documented += documented
        total_issues += issues

        table.add_row(namespace, status, str(components), str(documented), coverage, str(issues))

    console.print(table)

    # Overall summary
    overall_coverage = (total_documented / total_components * 100) if total_components > 0 else 0
    console.print(f"\n[bold]Overall Summary:[/bold]")
    console.print(f"• Valid Plugins: {valid_plugins}/{total_plugins} ({valid_plugins / total_plugins * 100:.1f}%)")
    console.print(f"• Total Components: {total_components}")
    console.print(f"• Documented Components: {total_documented}")
    console.print(f"• Overall Coverage: {overall_coverage:.1f}%")
    console.print(f"• Total Issues: {total_issues}")
