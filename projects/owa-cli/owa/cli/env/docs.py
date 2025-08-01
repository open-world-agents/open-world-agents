"""
Documentation management commands for OWA plugins.

This module implements the `owl env docs` command with validation and statistics,
providing comprehensive documentation quality checks with CI/CD integration.
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
    validate: bool = typer.Option(False, "--validate", help="Validate plugin documentation"),
    strict: bool = typer.Option(False, "--strict", help="Enable strict mode (100% coverage + 100% quality)"),
    min_coverage_pass: float = typer.Option(0.8, "--min-coverage-pass", help="Minimum coverage for PASS status"),
    min_coverage_fail: float = typer.Option(0.6, "--min-coverage-fail", help="Minimum coverage to avoid FAIL status"),
    min_quality_pass: float = typer.Option(
        0.6, "--min-quality-pass", help="Minimum good quality ratio for PASS status"
    ),
    min_quality_fail: float = typer.Option(
        0.0, "--min-quality-fail", help="Minimum good quality ratio to avoid FAIL status"
    ),
    format: str = typer.Option("text", "--output-format", help="Output format: text or json"),
    by_type: bool = typer.Option(False, "--by-type", help="Group statistics by component type"),
) -> None:
    """
    Manage plugin documentation - validate quality and show statistics.

    This command provides comprehensive documentation management for plugins,
    including validation for CI/CD integration and statistics for development.
    """

    if validate:
        _validate_docs(
            plugin_namespace, strict, min_coverage_pass, min_coverage_fail, min_quality_pass, min_quality_fail, format
        )
    else:
        # Default behavior: show statistics
        _docs_stats(plugin_namespace, by_type)


def _validate_docs(
    plugin_namespace: Optional[str],
    strict: bool,
    min_coverage_pass: float,
    min_coverage_fail: float,
    min_quality_pass: float,
    min_quality_fail: float,
    format: str,
) -> None:
    """
    Validate plugin documentation with proper exit codes for CI/CD integration.

    This command serves as a test utility that can be integrated into CI/CD pipelines,
    ensuring consistent documentation quality across all plugins.

    Features:
    - Per-component quality grading (GOOD/ACCEPTABLE/POOR)
    - Per-plugin quality thresholds (PASS/WARN/FAIL)
    - Skip quality check support (@skip-quality-check in docstrings)
    - Flexible threshold configuration

    Exit codes:
    - 0: All validations passed
    - 1: Documentation issues found (warnings or failures)
    - 2: Command error (invalid arguments, plugin not found, etc.)
    """
    try:
        validator = DocumentationValidator()

        # Validate specific plugin or all plugins
        if plugin_namespace:
            try:
                results = {plugin_namespace: validator.validate_plugin(plugin_namespace)}
            except KeyError:
                console.print(f"[red]❌ ERROR: Plugin '{plugin_namespace}' not found[/red]")
                sys.exit(2)
        else:
            # Default behavior: validate all plugins
            results = validator.validate_all_plugins()

        if not results:
            console.print("[yellow]⚠️  No plugins found to validate[/yellow]")
            sys.exit(0)

        # Apply strict mode adjustments
        if strict:
            # In strict mode, require high standards
            min_coverage_pass = min_coverage_fail = min_quality_pass = min_quality_fail = 1.0

        # Check plugin status using configurable thresholds
        all_pass = True
        for result in results.values():
            plugin_status = result.get_status(min_coverage_pass, min_coverage_fail, min_quality_pass, min_quality_fail)
            if plugin_status == PluginStatus.FAIL:
                all_pass = False
                break

        # Output results based on format
        if format == "json":
            _output_json(results, all_pass)
        else:
            _output_text(results, all_pass)

        # Determine exit code
        if all_pass:
            sys.exit(0)  # All validations passed
        else:
            sys.exit(1)  # Documentation issues found

    except Exception as e:
        console.print(f"[red]❌ ERROR: {e}[/red]")
        sys.exit(2)  # Command error


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


def _output_text(results, all_pass):
    """Output results in human-readable text format (same data as JSON)."""
    # Overall result (same as JSON)
    console.print(f"Result: {'PASS' if all_pass else 'FAIL'}\n", style="bold")

    # Display per-plugin results (same data as JSON)
    for name, result in results.items():
        # Status with icon
        status_icons = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
        status_colors = {"pass": "green", "warning": "yellow", "fail": "red"}

        status_value = result.status.value
        icon = status_icons.get(status_value, "❌")
        color = status_colors.get(status_value, "red")

        console.print(
            f"{icon} {name}:",
            style=f"bold {color}",
        )
        console.print(f"  Status: {status_value}")
        console.print(f"  Coverage: {result.coverage:.0%}")
        console.print(f"  Quality: {result.quality_ratio:.0%}")

        # Show improvement issues
        console.print("  Improvements:", style="yellow")
        for improvement in result.all_improvements:
            console.print(f"    - {improvement}", style="yellow")

        console.print()  # Empty line between plugins


def _docs_stats(plugin_namespace: Optional[str], by_type: bool) -> None:
    """
    Show documentation statistics for plugins.

    This is a helper command for development and analysis.
    """
    try:
        validator = DocumentationValidator()

        if plugin_namespace:
            try:
                results = {plugin_namespace: validator.validate_plugin(plugin_namespace)}
            except KeyError:
                console.print(f"[red]❌ ERROR: Plugin '{plugin_namespace}' not found[/red]")
                sys.exit(1)
        else:
            results = validator.validate_all_plugins()

        if not results:
            console.print("[yellow]No plugins found[/yellow]")
            return

        # Create statistics table
        if by_type:
            # Group by component type - simplified implementation
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
                table.add_row(
                    name, f"{coverage:.1%}", str(result.documented), str(result.total), status, "by-type view"
                )
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
                    "✅"
                    if result.status == PluginStatus.PASS
                    else "⚠️"
                    if result.status == PluginStatus.WARNING
                    else "❌"
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

    except Exception as e:
        console.print(f"[red]❌ ERROR: {e}[/red]")
        sys.exit(1)
