import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from owa.core.plugin_spec import PluginSpec

console = Console()


def _detect_input_type(spec_input: str) -> str:
    """
    Detect whether the input is a YAML file path or an entry point specification.

    Args:
        spec_input: Input string to analyze

    Returns:
        "yaml" if it's a file path, "entry_point" if it's an entry point spec
    """
    # Check if it looks like a file path
    if "/" in spec_input or "\\" in spec_input or spec_input.endswith((".yaml", ".yml")):
        return "yaml"

    # Check if it contains a colon (entry point format)
    if ":" in spec_input:
        return "entry_point"

    # Default to checking if it's a file that exists
    if Path(spec_input).exists():
        return "yaml"

    # If no clear indicators, assume entry point
    return "entry_point"


def _validate_component_imports(spec: PluginSpec) -> list[str]:
    """
    Validate that all component import paths can be loaded.

    Args:
        spec: PluginSpec to validate

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []

    for component_type, components in spec.components.items():
        for name, import_path in components.items():
            try:
                # Try to parse the import path
                if ":" not in import_path:
                    errors.append(f"{component_type}/{name}: Invalid import path format '{import_path}' (missing ':')")
                    continue

                module_path, _ = import_path.split(":", 1)

                # Try to import the module (but don't load the object to avoid side effects)
                import importlib.util

                spec_obj = importlib.util.find_spec(module_path)
                if spec_obj is None:
                    errors.append(f"{component_type}/{name}: Module '{module_path}' not found")

            except Exception as e:
                errors.append(f"{component_type}/{name}: Import validation failed - {str(e)}")

    return errors


def _create_validation_display(spec: PluginSpec, import_errors: list[str], source_info: str) -> None:
    """
    Create and display the validation results.

    Args:
        spec: Validated PluginSpec
        import_errors: List of import validation errors
        source_info: Information about the source (file path or entry point)
    """
    # Main validation tree
    if import_errors:
        tree = Tree("⚠️  Plugin Specification Valid (with warnings)")
        tree.add(f"[yellow]Source: {source_info}[/yellow]")
    else:
        tree = Tree("✅ Plugin Specification Valid")
        tree.add(f"[green]Source: {source_info}[/green]")

    # Plugin metadata
    metadata_tree = tree.add("📋 Plugin Metadata")
    metadata_tree.add(f"├── Namespace: [bold]{spec.namespace}[/bold]")
    metadata_tree.add(f"├── Version: {spec.version}")
    metadata_tree.add(f"├── Author: {spec.author or '[dim]Not specified[/dim]'}")
    metadata_tree.add(f"└── Description: {spec.description}")

    # Component summary
    comp_tree = tree.add("🔧 Components Summary")
    total_components = sum(len(components) for components in spec.components.values())
    comp_tree.add(f"├── Total Components: [bold]{total_components}[/bold]")

    if spec.components.get("callables"):
        comp_tree.add(f"├── 📞 Callables: {len(spec.components['callables'])}")
    if spec.components.get("listeners"):
        comp_tree.add(f"├── 👂 Listeners: {len(spec.components['listeners'])}")
    if spec.components.get("runnables"):
        comp_tree.add(f"└── 🏃 Runnables: {len(spec.components['runnables'])}")

    console.print(tree)

    # Show detailed component table if there are components
    if total_components > 0:
        console.print()
        table = Table(title="Component Details", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Import Path", style="yellow")
        table.add_column("Status", justify="center")

        for component_type, components in spec.components.items():
            for name, import_path in components.items():
                # Check if this component has import errors
                component_key = f"{component_type}/{name}"
                has_error = any(error.startswith(component_key + ":") for error in import_errors)
                status = "❌" if has_error else "✅"

                table.add_row(component_type.title(), f"{spec.namespace}/{name}", import_path, status)

        console.print(table)

    # Show import errors if any
    if import_errors:
        console.print()
        error_panel = Panel(
            "\n".join(f"• {error}" for error in import_errors),
            title="⚠️  Import Validation Warnings",
            border_style="yellow",
        )
        console.print(error_panel)


def validate_plugin(
    spec_input: str = typer.Argument(
        ...,
        help="Plugin specification to validate. Can be:\n"
        "• Path to YAML file (e.g., './plugin.yaml')\n"
        "• Entry point specification (e.g., 'owa.env.plugins.desktop:plugin_spec')",
    ),
    check_imports: bool = typer.Option(
        True, "--check-imports/--no-check-imports", help="Validate that component import paths are accessible"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation information"),
):
    """
    Validate a plugin specification from YAML file or entry point.

    This command can validate plugin specifications in two ways:

    1. From YAML files: owl env validate ./plugin.yaml
    2. From entry points: owl env validate owa.env.plugins.desktop:plugin_spec

    The command automatically detects the input type and validates accordingly.
    """
    try:
        # Detect input type
        input_type = _detect_input_type(spec_input)

        if verbose:
            console.print(f"[dim]Detected input type: {input_type}[/dim]")

        # Load the specification
        if input_type == "yaml":
            if not Path(spec_input).exists():
                console.print(f"[red]Error: YAML file not found: {spec_input}[/red]")
                sys.exit(1)
            spec = PluginSpec.from_yaml(spec_input)
            source_info = f"YAML file: {spec_input}"
        else:
            spec = PluginSpec.from_entry_point(spec_input)
            source_info = f"Entry point: {spec_input}"

        # Validate component imports if requested
        import_errors = []
        if check_imports:
            if verbose:
                console.print("[dim]Validating component imports...[/dim]")
            import_errors = _validate_component_imports(spec)

        # Display results
        _create_validation_display(spec, import_errors, source_info)

        # Exit with appropriate code
        if import_errors:
            console.print(f"\n[yellow]Validation completed with {len(import_errors)} warnings.[/yellow]")
            sys.exit(0)  # Warnings don't cause failure
        else:
            console.print("\n[green]✅ Validation successful![/green]")
            sys.exit(0)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]Error: Invalid YAML - {str(e)}[/red]")
        sys.exit(1)
    except (ImportError, AttributeError, TypeError) as e:
        console.print(f"[red]Error: Entry point validation failed - {str(e)}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: Plugin specification validation failed - {str(e)}[/red]")
        sys.exit(1)
