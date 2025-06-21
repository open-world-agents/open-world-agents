"""
Documentation validation commands for OWA plugins.

This module implements the `owl env validate-docs` command specified in OEP-0004,
providing comprehensive documentation quality checks with CI/CD integration.
"""

import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from owa.core.documentation import DocumentationValidator

console = Console()


def validate_docs(
    plugin_namespace: Optional[str] = typer.Argument(None, help="Specific plugin namespace to validate (optional)"),
    strict: bool = typer.Option(False, "--strict", help="Enable strict mode (100% coverage + 100% quality)"),
    min_coverage_pass: float = typer.Option(0.8, "--min-coverage-pass", help="Minimum coverage for PASS status"),
    min_coverage_fail: float = typer.Option(0.6, "--min-coverage-fail", help="Minimum coverage to avoid FAIL status"),
    min_quality_pass: float = typer.Option(
        0.6, "--min-quality-pass", help="Minimum good quality ratio for PASS status"
    ),
    min_quality_fail: float = typer.Option(
        0.0, "--min-quality-fail", help="Minimum good quality ratio to avoid FAIL status"
    ),
    format: str = typer.Option("text", "--format", help="Output format: text or json"),
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

        # Calculate overall statistics as minimum per-plugin results
        if results:
            min_coverage = min(r.coverage for r in results.values())
            min_quality = min(r.quality_ratio for r in results.values())
            min_coverage_plugin = min(results.items(), key=lambda x: x[1].coverage)[0]
            min_quality_plugin = min(results.items(), key=lambda x: x[1].quality_ratio)[0]
        else:
            min_coverage = min_quality = 0
            min_coverage_plugin = min_quality_plugin = "none"

        # Apply strict mode adjustments
        if strict:
            # In strict mode, require high standards
            min_coverage_pass = min_coverage_fail = min_quality_pass = min_quality_fail = 1.0

        # Check thresholds using minimum per-plugin results
        coverage_pass = min_coverage >= min_coverage_pass
        quality_pass = min_quality >= min_quality_pass

        # Check plugin status using configurable thresholds
        plugin_pass = True
        for result in results.values():
            plugin_status = result.get_status(min_coverage_pass, min_coverage_fail, min_quality_pass, min_quality_fail)
            if plugin_status == "fail":
                plugin_pass = False
                break

        # Output results based on format
        all_pass = coverage_pass and quality_pass and plugin_pass
        if format == "json":
            _output_json(
                results,
                min_coverage,
                min_quality,
                all_pass,
                min_coverage_pass,
                min_quality_pass,
                min_coverage_plugin,
                min_quality_plugin,
            )
        else:
            _output_text(
                results,
                min_coverage,
                min_quality,
                coverage_pass,
                quality_pass,
                plugin_pass,
                min_coverage_pass,
                min_coverage_fail,
                min_quality_pass,
                min_quality_fail,
                min_coverage_plugin,
                min_quality_plugin,
            )

        # Determine exit code
        if all_pass:
            sys.exit(0)  # All validations passed
        else:
            sys.exit(1)  # Documentation issues found

    except Exception as e:
        console.print(f"[red]❌ ERROR: {e}[/red]")
        sys.exit(2)  # Command error


def _output_json(
    results,
    min_coverage,
    min_quality,
    all_pass,
    min_coverage_pass,
    min_quality_pass,
    min_coverage_plugin,
    min_quality_plugin,
):
    """Output results in JSON format for tooling integration."""
    output = {
        "min_plugin_coverage": min_coverage,
        "min_plugin_quality": min_quality,
        "worst_coverage_plugin": min_coverage_plugin,
        "worst_quality_plugin": min_quality_plugin,
        "thresholds": {
            "min_coverage_pass": min_coverage_pass,
            "min_quality_pass": min_quality_pass,
        },
        "plugins": {},
        "exit_code": 0 if all_pass else 1,
    }

    for name, result in results.items():
        output["plugins"][name] = {
            "documented": result.documented,
            "total": result.total,
            "coverage": result.coverage,
            "good_quality": result.good_quality,
            "quality_ratio": result.quality_ratio,
            "skipped": result.skipped,
            "status": result.status,
            "issues": [],
        }

        # Add component-level issues
        for comp_result in result.components:
            if comp_result.issues:
                output["plugins"][name]["issues"].extend(
                    [f"{comp_result.component}: {issue}" for issue in comp_result.issues]
                )

    print(json.dumps(output, indent=2))


def _output_text(
    results,
    min_coverage,
    min_quality,
    coverage_pass,
    quality_pass,
    plugin_pass,
    min_coverage_pass,
    min_coverage_fail,
    min_quality_pass,
    min_quality_fail,
    min_coverage_plugin,
    min_quality_plugin,
):
    """Output results in human-readable text format."""
    # Display per-plugin results
    for name, result in results.items():
        # Determine status icon based on configurable plugin status
        plugin_status = result.get_status(min_coverage_pass, min_coverage_fail, min_quality_pass, min_quality_fail)
        if plugin_status == "pass":
            status_icon = "✅"
            status_color = "green"
        elif plugin_status == "warning":
            status_icon = "⚠️"
            status_color = "yellow"
        else:
            status_icon = "❌"
            status_color = "red"

        console.print(
            f"{status_icon} {name} plugin: {result.documented}/{result.total} documented ({result.coverage:.0%}), "
            f"{result.good_quality}/{result.total} good quality ({result.quality_ratio:.0%})",
            style=status_color,
        )

        if result.skipped > 0:
            console.print(f"  📝 {result.skipped} components skipped quality check", style="dim")

    # Display overall summary (minimum per-plugin results)
    console.print(
        f"\n📊 Overall: Minimum plugin coverage: {min_coverage:.0%} ({min_coverage_plugin}), "
        f"Minimum plugin quality: {min_quality:.0%} ({min_quality_plugin})"
    )

    # Show status messages
    if coverage_pass and quality_pass and plugin_pass:
        console.print("✅ PASS: All quality thresholds met", style="green")
    else:
        if not coverage_pass:
            console.print(
                f"❌ FAIL: Minimum plugin coverage {min_coverage:.0%} ({min_coverage_plugin}) below threshold ({min_coverage_pass:.0%})",
                style="red",
            )
        if not quality_pass:
            console.print(
                f"❌ FAIL: Minimum plugin quality {min_quality:.0%} ({min_quality_plugin}) below threshold ({min_quality_pass:.0%})",
                style="red",
            )
        if not plugin_pass:
            console.print(
                f"❌ FAIL: Some plugins below quality thresholds (PASS: {min_coverage_pass:.0%}/{min_quality_pass:.0%}, FAIL: {min_coverage_fail:.0%}/{min_quality_fail:.0%})",
                style="red",
            )


# Additional helper command for development
def docs_stats(
    plugin_namespace: Optional[str] = typer.Argument(None, help="Specific plugin namespace (optional)"),
    by_type: bool = typer.Option(False, "--by-type", help="Group statistics by component type"),
) -> None:
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
                status_icon = "✅" if result.status == "pass" else "⚠️" if result.status == "warning" else "❌"

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
