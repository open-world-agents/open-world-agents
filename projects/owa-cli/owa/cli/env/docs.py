"""
Documentation commands for OEP-0004.

Provides CLI commands for viewing and generating plugin documentation.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from owa.core.documentation import PluginDocumentationGenerator, TemplateEngine

console = Console()


def docs_command(
    namespace: str = typer.Argument(..., help="Plugin namespace to show documentation for"),
    component: Optional[str] = typer.Option(None, "--component", "-c", help="Specific component to document"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, markdown, json"),
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save output to file"),
    type_filter: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by component type"),
):
    """Show comprehensive plugin documentation extracted from source code."""
    
    # Validate format
    if format not in ["text", "markdown", "json"]:
        console.print(f"[red]Error: Invalid format '{format}'. Must be one of: text, markdown, json[/red]")
        sys.exit(1)
    
    # Validate component type filter
    if type_filter and type_filter not in ["callables", "listeners", "runnables"]:
        console.print(f"[red]Error: Invalid type '{type_filter}'. Must be one of: callables, listeners, runnables[/red]")
        sys.exit(1)
    
    generator = PluginDocumentationGenerator()
    template_engine = TemplateEngine()
    
    if component:
        # Show specific component documentation
        component_doc = generator.generate_component_documentation(namespace, component)
        if not component_doc:
            console.print(f"[red]Error: Component '{namespace}/{component}' not found[/red]")
            sys.exit(1)
        
        if format == "json":
            output = _component_to_json(component_doc)
        else:
            output = template_engine.render_component_detail(component_doc, format)
    else:
        # Show plugin documentation
        plugin_doc = generator.generate_plugin_documentation(namespace)
        if not plugin_doc:
            console.print(f"[red]Error: Plugin '{namespace}' not found[/red]")
            sys.exit(1)
        
        # Apply type filter if specified
        if type_filter:
            filtered_components = {type_filter: plugin_doc.components.get(type_filter, [])}
            plugin_doc.components = filtered_components
        
        if format == "json":
            output = _plugin_to_json(plugin_doc)
        else:
            output = template_engine.render_plugin_overview(plugin_doc, format)
    
    # Output or save
    if save:
        try:
            Path(save).write_text(output)
            console.print(f"[green]Documentation saved to {save}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving file: {e}[/red]")
            sys.exit(1)
    else:
        if format == "json":
            console.print_json(output)
        else:
            console.print(output)


def ecosystem_docs_command(
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, markdown, json"),
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save output to file"),
):
    """Show documentation for all discovered plugins in the ecosystem."""
    
    if format not in ["text", "markdown", "json"]:
        console.print(f"[red]Error: Invalid format '{format}'. Must be one of: text, markdown, json[/red]")
        sys.exit(1)
    
    generator = PluginDocumentationGenerator()
    template_engine = TemplateEngine()
    
    console.print("[blue]Generating ecosystem documentation...[/blue]")
    ecosystem_docs = generator.generate_ecosystem_documentation()
    
    if not ecosystem_docs:
        console.print("[yellow]No plugins found in the ecosystem[/yellow]")
        return
    
    if format == "json":
        output = _ecosystem_to_json(ecosystem_docs)
    else:
        output = template_engine.render_ecosystem_index(ecosystem_docs, format)
    
    # Output or save
    if save:
        try:
            Path(save).write_text(output)
            console.print(f"[green]Ecosystem documentation saved to {save}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving file: {e}[/red]")
            sys.exit(1)
    else:
        if format == "json":
            console.print_json(output)
        else:
            console.print(output)


def _plugin_to_json(plugin_doc) -> str:
    """Convert plugin documentation to JSON."""
    data = {
        "namespace": plugin_doc.namespace,
        "version": plugin_doc.version,
        "description": plugin_doc.description,
        "author": plugin_doc.author,
        "generated_at": plugin_doc.generated_at.isoformat(),
        "source_files": plugin_doc.source_files,
        "components": {}
    }
    
    for comp_type, components in plugin_doc.components.items():
        data["components"][comp_type] = [_component_to_dict(comp) for comp in components]
    
    return json.dumps(data, indent=2)


def _component_to_json(component_doc) -> str:
    """Convert component documentation to JSON."""
    return json.dumps(_component_to_dict(component_doc), indent=2)


def _component_to_dict(component_doc) -> dict:
    """Convert component documentation to dictionary."""
    return {
        "name": component_doc.name,
        "full_name": component_doc.full_name,
        "type": component_doc.type,
        "source_file": component_doc.source_file,
        "line_number": component_doc.line_number,
        "summary": component_doc.summary,
        "description": component_doc.description,
        "signature": component_doc.signature,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "default": p.default,
                "description": p.description,
                "is_optional": p.is_optional
            }
            for p in component_doc.parameters
        ],
        "returns": {
            "type": component_doc.returns.type,
            "description": component_doc.returns.description
        } if component_doc.returns else None,
        "raises": [
            {
                "exception_type": e.exception_type,
                "description": e.description
            }
            for e in component_doc.raises
        ],
        "examples": [
            {
                "code": e.code,
                "description": e.description,
                "expected_output": e.expected_output,
                "is_doctest": e.is_doctest
            }
            for e in component_doc.examples
        ],
        "notes": component_doc.notes,
        "type_hints": component_doc.type_hints,
        "is_async": component_doc.is_async,
        "is_method": component_doc.is_method,
        "is_classmethod": component_doc.is_classmethod,
        "is_staticmethod": component_doc.is_staticmethod,
        "methods": [_component_to_dict(m) for m in component_doc.methods],
        "attributes": [
            {
                "name": a.name,
                "type": a.type,
                "description": a.description
            }
            for a in component_doc.attributes
        ],
        "inheritance": component_doc.inheritance
    }


def _ecosystem_to_json(ecosystem_docs) -> str:
    """Convert ecosystem documentation to JSON."""
    data = {
        "total_plugins": len(ecosystem_docs),
        "plugins": {}
    }
    
    for namespace, plugin_doc in ecosystem_docs.items():
        data["plugins"][namespace] = {
            "version": plugin_doc.version,
            "description": plugin_doc.description,
            "author": plugin_doc.author,
            "total_components": sum(len(components) for components in plugin_doc.components.values()),
            "component_counts": {
                comp_type: len(components) 
                for comp_type, components in plugin_doc.components.items()
            }
        }
    
    return json.dumps(data, indent=2)
