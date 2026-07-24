"""
Documentation validation and statistics for OWA plugins.

This module implements the `owl env docs` command that validates plugin documentation
quality and provides statistics with proper exit codes for CI/CD integration.
"""

import json
import sys
from typing import Optional

import typer
from rich.table import Table

from owa.core.documentation import DocumentationValidator
from owa.core.documentation.validator import PluginStatus

from ..console import console


def docs(
    plugin_namespace: Optional[str] = typer.Argument(None, help="Specific plugin namespace (optional)"),
    strict: bool = typer.Option(False, "--strict", help="Enable strict mode (100% coverage + 100% quality)"),
    min_coverage_pass: float = typer.Option(0.8, "--min-coverage-pass", help="Minimum coverage for PASS status"),
    min_coverage_fail: float = typer.Option(0.6, "--min-coverage-fail", help="Minimum coverage to avoid FAIL status"),
    min_quality_pass: float = typer.Option(
        0.6, "--min-quality-pass", help="Minimum good quality ratio for PASS status"
    ),
    min_quality_fail: float = typer.Option(
        0.0, "--min-quality-fail", help="Minimum good quality ratio to avoid FAIL status"
    ),
    output_format: str = typer.Option("table", "--output-format", help="Output format: table or json"),
    by_type: bool = typer.Option(False, "--by-type", help="Group statistics by component type (table format only)"),
) -> None:
    """
    Validate plugin documentation quality and show statistics.

    Always validates documentation and returns meaningful exit codes:
    - 0: All validations passed
    - 1: Documentation issues found (warnings or failures)
    - 2: Command error (invalid arguments, plugin not found, etc.)
    """
    try:
        # Validate format
        if output_format not in ("table", "json"):
            console.print(f"[red]❌ ERROR: Invalid format '{output_format}'. Must be 'table' or 'json'[/red]")
            sys.exit(2)

        validator = DocumentationValidator()

        # Validate specific plugin or all plugins
        if plugin_namespace:
            try:
                results = {plugin_namespace: validator.validate_plugin(plugin_namespace)}
            except KeyError:
                console.print(f"[red]❌ ERROR: Plugin '{plugin_namespace}' not found[/red]")
                sys.exit(2)
        else:
            results = validator.validate_all_plugins()

        if not results:
            console.print("[yellow]⚠️  No plugins found to validate[/yellow]")
            sys.exit(0)

        # Apply strict mode
        if strict:
            min_coverage_pass = min_coverage_fail = min_quality_pass = min_quality_fail = 1.0

        # Check overall status
        all_pass = True
        for result in results.values():
            plugin_status = result.get_status(min_coverage_pass, min_coverage_fail, min_quality_pass, min_quality_fail)
            if plugin_status == PluginStatus.FAIL:
                all_pass = False
                break

        # Output results
        if output_format == "json":
            _output_json(results, all_pass)
        else:
            _output_table(results, all_pass, by_type)

        # Exit with appropriate code
        sys.exit(0 if all_pass else 1)

    except Exception as e:
        console.print(f"[red]❌ ERROR: {e}[/red]")
        sys.exit(2)


def _output_table(results, all_pass, by_type):
    """Output validation results in table format with improvements."""
    if by_type:
        table = Table(title="Documentation Statistics by Type")
        table.add_column("Plugin", style="cyan")
        table.add_column("Coverage", justify="right")
        table.add_column("Documented", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Note", style="dim")

        for name, result in results.items():
            coverage = result.coverage
            status = "✅" if coverage == 1.0 else "⚠️" if coverage >= 0.75 else "❌"
            table.add_row(name, f"{coverage:.1%}", str(result.documented), str(result.total), status, "by-type view")
    else:
        table = Table(title="Documentation Statistics")
        table.add_column("Plugin", style="cyan")
        table.add_column("Coverage", justify="right")
        table.add_column("Documented", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Quality", justify="right")
        table.add_column("Status", justify="center")

        for name, result in results.items():
            status_icon = (
                "✅" if result.status == PluginStatus.PASS else "⚠️" if result.status == PluginStatus.WARNING else "❌"
            )

            table.add_row(
                name,
                f"{result.coverage:.1%}",
                str(result.documented),
                str(result.total),
                f"{result.quality_ratio:.1%}",
                status_icon,
            )

    console.print(table)

    # Overall statistics
    total_components = sum(r.total for r in results.values())
    documented_components = sum(r.documented for r in results.values())
    overall_coverage = documented_components / total_components if total_components > 0 else 0

    console.print(f"\n📊 Overall Coverage: {overall_coverage:.1%} ({documented_components}/{total_components})")

    overall_status = "PASS" if all_pass else "FAIL"
    status_color = "green" if all_pass else "red"
    console.print(f"📋 Overall Result: [{status_color}]{overall_status}[/{status_color}]")

    # Show improvements for failed plugins
    failed_plugins = [name for name, result in results.items() if result.status == PluginStatus.FAIL]
    if failed_plugins:
        console.print(f"\n[red]📝 Improvements needed for {len(failed_plugins)} plugin(s):[/red]")
        for plugin_name in failed_plugins:
            result = results[plugin_name]
            console.print(f"\n❌ **{plugin_name}**:")
            for component in result.components:
                if component.quality_grade in ("poor", "acceptable") and component.improvements:
                    console.print(f"   • {component.component}: {', '.join(component.improvements)}")


def _output_json(results, all_pass):
    """Output results in JSON format for tooling integration."""
    output = {
        "result": "PASS" if all_pass else "FAIL",
        "plugins": {},
    }

    for name, result in results.items():
        output["plugins"][name] = {
            "status": result.status.value,
            "coverage": f"{result.coverage:.0%}",
            "quality": f"{result.quality_ratio:.0%}",
            "improvements": result.all_improvements,
        }

    print(json.dumps(output, indent=2))
