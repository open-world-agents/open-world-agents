"""
Documentation site generation for OEP-0004.

Provides CLI commands for generating MkDocs-compatible documentation sites.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from owa.core.documentation import PluginDocumentationGenerator, TemplateEngine

console = Console()


def generate_docs_command(
    output: str = typer.Option("./docs", "--output", "-o", help="Output directory for generated documentation"),
    format: str = typer.Option("mkdocs", "--format", "-f", help="Documentation format: mkdocs, html"),
    theme: str = typer.Option("material", "--theme", help="MkDocs theme to use"),
    include_source: bool = typer.Option(True, "--include-source/--no-source", help="Include source code links"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing documentation"),
):
    """Generate comprehensive documentation site from plugin source code."""
    
    if format not in ["mkdocs", "html"]:
        console.print(f"[red]Error: Invalid format '{format}'. Must be one of: mkdocs, html[/red]")
        sys.exit(1)
    
    output_path = Path(output)
    
    # Check if output directory exists
    if output_path.exists() and not overwrite:
        if any(output_path.iterdir()):
            console.print(f"[red]Error: Output directory '{output}' is not empty. Use --overwrite to overwrite.[/red]")
            sys.exit(1)
    
    generator = PluginDocumentationGenerator()
    template_engine = TemplateEngine()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Discover plugins
        task = progress.add_task("Discovering plugins via entry points...", total=None)
        ecosystem_docs = generator.generate_ecosystem_documentation()
        
        if not ecosystem_docs:
            console.print("[yellow]No plugins found to document[/yellow]")
            return
        
        progress.update(task, description=f"Found {len(ecosystem_docs)} plugins")
        
        # Create output directory structure
        progress.update(task, description="Creating directory structure...")
        _create_directory_structure(output_path, format)
        
        # Generate documentation files
        progress.update(task, description="Analyzing source code and generating documentation...")
        
        if format == "mkdocs":
            _generate_mkdocs_site(output_path, ecosystem_docs, template_engine, theme, include_source, progress, task)
        else:
            _generate_html_site(output_path, ecosystem_docs, template_engine, include_source, progress, task)
        
        progress.update(task, description="✅ Documentation site generated successfully")
    
    console.print(f"\n[green]📚 Documentation site generated at {output_path}[/green]")
    
    if format == "mkdocs":
        console.print(f"\n[blue]To serve the documentation:[/blue]")
        console.print(f"  cd {output} && mkdocs serve")
        console.print(f"\n[blue]To build static site:[/blue]")
        console.print(f"  cd {output} && mkdocs build")


def _create_directory_structure(output_path: Path, format: str):
    """Create the basic directory structure."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "mkdocs":
        docs_dir = output_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / "plugins").mkdir(exist_ok=True)
        (docs_dir / "components").mkdir(exist_ok=True)
        (docs_dir / "api").mkdir(exist_ok=True)


def _generate_mkdocs_site(output_path: Path, ecosystem_docs: dict, template_engine: TemplateEngine, 
                         theme: str, include_source: bool, progress, task):
    """Generate MkDocs-compatible documentation site."""
    docs_dir = output_path / "docs"
    
    # Generate mkdocs.yml
    progress.update(task, description="Generating MkDocs configuration...")
    _generate_mkdocs_config(output_path, ecosystem_docs, theme)
    
    # Generate index page
    progress.update(task, description="Generating ecosystem overview...")
    index_content = template_engine.render_ecosystem_index(ecosystem_docs, "markdown")
    (docs_dir / "index.md").write_text(index_content)
    
    # Generate plugin pages
    plugins_dir = docs_dir / "plugins"
    api_dir = docs_dir / "api"
    
    for namespace, plugin_doc in ecosystem_docs.items():
        progress.update(task, description=f"Generating documentation for {namespace} plugin...")
        
        # Plugin overview page
        plugin_content = template_engine.render_plugin_overview(plugin_doc, "markdown")
        (plugins_dir / f"{namespace}.md").write_text(plugin_content)
        
        # Create API directory for this plugin
        plugin_api_dir = api_dir / namespace
        plugin_api_dir.mkdir(exist_ok=True)
        
        # Generate component detail pages
        for comp_type, components in plugin_doc.components.items():
            for component in components:
                component_content = template_engine.render_component_detail(component, "markdown")
                
                # Create safe filename
                safe_name = component.name.replace("/", "_").replace(".", "_")
                filename = f"{safe_name}.md"
                
                (plugin_api_dir / filename).write_text(component_content)
    
    # Generate component type overview pages
    progress.update(task, description="Generating component type overviews...")
    components_dir = docs_dir / "components"
    
    for comp_type in ["callables", "listeners", "runnables"]:
        _generate_component_type_overview(components_dir, comp_type, ecosystem_docs, template_engine)


def _generate_mkdocs_config(output_path: Path, ecosystem_docs: dict, theme: str):
    """Generate mkdocs.yml configuration file."""
    config_lines = [
        "site_name: OWA Plugin Documentation",
        "site_description: Auto-generated documentation for OWA plugins",
        "",
        "theme:",
        f"  name: {theme}",
        "  features:",
        "    - navigation.tabs",
        "    - navigation.sections",
        "    - navigation.expand",
        "    - search.highlight",
        "",
        "plugins:",
        "  - search",
        "",
        "nav:",
        "  - Home: index.md",
        "  - Plugins:",
    ]
    
    # Add plugin pages to navigation
    for namespace in sorted(ecosystem_docs.keys()):
        config_lines.append(f"    - {namespace}: plugins/{namespace}.md")
    
    config_lines.extend([
        "  - Components:",
        "    - Callables: components/callables.md",
        "    - Listeners: components/listeners.md", 
        "    - Runnables: components/runnables.md",
        "  - API Reference:",
    ])
    
    # Add API reference pages
    for namespace in sorted(ecosystem_docs.keys()):
        config_lines.append(f"    - {namespace}: api/{namespace}/")
    
    (output_path / "mkdocs.yml").write_text('\n'.join(config_lines))


def _generate_component_type_overview(components_dir: Path, comp_type: str, ecosystem_docs: dict, template_engine: TemplateEngine):
    """Generate overview page for a component type."""
    lines = [
        f"# {comp_type.title()} Components",
        "",
        f"Overview of all {comp_type} components across the OWA ecosystem.",
        "",
    ]
    
    for namespace, plugin_doc in ecosystem_docs.items():
        components = plugin_doc.components.get(comp_type, [])
        if not components:
            continue
            
        lines.append(f"## {namespace}")
        lines.append("")
        
        for component in components:
            lines.append(f"### {component.full_name}")
            lines.append("")
            lines.append(component.summary)
            lines.append("")
            
            if component.signature:
                lines.append("```python")
                lines.append(component.signature)
                lines.append("```")
                lines.append("")
    
    (components_dir / f"{comp_type}.md").write_text('\n'.join(lines))


def _generate_html_site(output_path: Path, ecosystem_docs: dict, template_engine: TemplateEngine, 
                       include_source: bool, progress, task):
    """Generate standalone HTML documentation site."""
    # Basic HTML generation - can be enhanced
    progress.update(task, description="Generating HTML pages...")
    
    # Generate index.html
    index_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OWA Plugin Documentation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .plugin {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>OWA Plugin Documentation</h1>
        <p>Auto-generated documentation for {len(ecosystem_docs)} plugins.</p>
        
        <h2>Plugins</h2>
        {"".join(f'<div class="plugin"><h3>{ns}</h3><p>{doc.description}</p></div>' 
                 for ns, doc in ecosystem_docs.items())}
    </body>
    </html>
    """
    
    (output_path / "index.html").write_text(index_content)
