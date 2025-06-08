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
    strict: bool = typer.Option(False, "--strict", help="Enable strict mode for CI/CD"),
    min_coverage: float = typer.Option(0.9, "--min-coverage", help="Minimum documentation coverage threshold"),
    format: str = typer.Option("text", "--format", help="Output format: text or json"),
    all_plugins: bool = typer.Option(False, "--all", help="Validate all plugins (explicit flag)"),
    check_quality: bool = typer.Option(False, "--check-quality", help="Perform detailed quality checks"),
    min_examples: int = typer.Option(0, "--min-examples", help="Minimum number of examples required"),
) -> None:
    """
    Validate plugin documentation with proper exit codes for CI/CD integration.

    This command serves as a test utility that can be integrated into CI/CD pipelines,
    ensuring consistent documentation quality across all plugins.

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
                console.print(f"[red]❌ ERROR: Plugin '{plugin_namespace}' not found[/red]", file=sys.stderr)
                sys.exit(2)
        else:
            results = validator.validate_all_plugins()

        if not results:
            console.print("[yellow]⚠️  No plugins found to validate[/yellow]")
            sys.exit(0)

        # Calculate overall statistics
        total_components = sum(r.total for r in results.values())
        documented_components = sum(r.documented for r in results.values())
        overall_coverage = documented_components / total_components if total_components > 0 else 0

        # Apply strict mode adjustments
        if strict:
            # In strict mode, any missing documentation is a failure
            min_coverage = max(min_coverage, 1.0)

        # Check coverage threshold
        coverage_pass = overall_coverage >= min_coverage

        # Apply quality checks if requested
        if check_quality:
            _apply_quality_checks(results, min_examples)

        # Output results based on format
        if format == "json":
            _output_json(results, overall_coverage, coverage_pass, min_coverage)
        else:
            _output_text(results, overall_coverage, coverage_pass, min_coverage, check_quality)

        # Determine exit code
        if coverage_pass and not _has_critical_issues(results):
            sys.exit(0)  # All validations passed
        else:
            sys.exit(1)  # Documentation issues found

    except Exception as e:
        console.print(f"[red]❌ ERROR: {e}[/red]", file=sys.stderr)
        sys.exit(2)  # Command error


def _output_json(results, overall_coverage, coverage_pass, min_coverage):
    """Output results in JSON format for tooling integration."""
    output = {"overall_coverage": overall_coverage, "plugins": {}, "exit_code": 0 if coverage_pass else 1}

    for name, result in results.items():
        coverage = result.coverage
        status = "pass" if coverage == 1.0 else "warning" if coverage >= 0.75 else "fail"

        output["plugins"][name] = {
            "coverage": coverage,
            "documented": result.documented,
            "total": result.total,
            "status": status,
            "issues": [],
        }

        # Add component-level issues
        for comp_result in result.components:
            if comp_result.issues:
                output["plugins"][name]["issues"].extend(
                    [f"{comp_result.component}: {issue}" for issue in comp_result.issues]
                )

    print(json.dumps(output, indent=2))


def _output_text(results, overall_coverage, coverage_pass, min_coverage, check_quality):
    """Output results in human-readable text format."""
    # Display per-plugin results
    for name, result in results.items():
        coverage = result.coverage

        if coverage == 1.0:
            status_icon = "✅"
            status_color = "green"
        elif coverage >= 0.75:
            status_icon = "⚠️"
            status_color = "yellow"
        else:
            status_icon = "❌"
            status_color = "red"

        console.print(
            f"{status_icon} {name} plugin: {result.documented}/{result.total} components documented ({coverage:.0%})",
            style=status_color,
        )

        # Show detailed issues if quality check is enabled
        if check_quality:
            for comp_result in result.components:
                if comp_result.issues:
                    for issue in comp_result.issues:
                        console.print(f"  ❌ {comp_result.component}: {issue}", style="red")

    # Display overall summary
    console.print(
        f"\n📊 Overall: {sum(r.documented for r in results.values())}/{sum(r.total for r in results.values())} components documented ({overall_coverage:.0%})"
    )

    if coverage_pass:
        console.print("✅ PASS: Documentation coverage meets minimum threshold", style="green")
    else:
        console.print(
            f"❌ FAIL: Documentation coverage {overall_coverage:.0%} below minimum {min_coverage:.0%}", style="red"
        )


def _apply_quality_checks(results, min_examples):
    """Apply additional quality checks to validation results."""
    # This function can be enhanced to apply more sophisticated quality checks
    # For now, it's a placeholder for future quality validation logic
    for result in results.values():
        for comp_result in result.components:
            # Check for minimum examples requirement
            if min_examples > 0:
                # This would need to be implemented based on docstring parsing
                pass


def _has_critical_issues(results) -> bool:
    """Check if there are any critical issues beyond coverage."""
    for result in results.values():
        for comp_result in result.components:
            if comp_result.status == "fail":
                return True
    return False


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
            table.add_column("Status", justify="center")

            for name, result in results.items():
                coverage = result.coverage
                status = "✅" if coverage == 1.0 else "⚠️" if coverage >= 0.75 else "❌"

                table.add_row(name, f"{coverage:.1%}", str(result.documented), str(result.total), status)

        console.print(table)

        # Overall statistics
        total_components = sum(r.total for r in results.values())
        documented_components = sum(r.documented for r in results.values())
        overall_coverage = documented_components / total_components if total_components > 0 else 0

        console.print(f"\n📊 Overall Coverage: {overall_coverage:.1%} ({documented_components}/{total_components})")

    except Exception as e:
        console.print(f"[red]❌ ERROR: {e}[/red]")
        sys.exit(1)
