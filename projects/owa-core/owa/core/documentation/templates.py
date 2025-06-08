"""
Template engine for OEP-0004 documentation generation.

Provides customizable templates for generating documentation in various formats.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


class TemplateEngine:
    """
    Template engine for generating documentation from extracted data.
    
    Supports both built-in templates and custom Jinja2 templates.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or self._get_default_template_dir()
        self.env = None
        
        if JINJA2_AVAILABLE:
            self.env = Environment(
                loader=FileSystemLoader(self.template_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
    
    def render_plugin_overview(self, plugin_doc: Any, format: str = "text") -> str:
        """
        Render plugin overview documentation.
        
        Args:
            plugin_doc: PluginDocumentation object
            format: Output format ("text", "markdown", "html")
            
        Returns:
            Rendered documentation
        """
        if format == "markdown":
            return self._render_plugin_overview_markdown(plugin_doc)
        elif format == "html":
            return self._render_plugin_overview_html(plugin_doc)
        else:
            return self._render_plugin_overview_text(plugin_doc)
    
    def render_component_detail(self, component_doc: Any, format: str = "text") -> str:
        """
        Render detailed component documentation.
        
        Args:
            component_doc: ComponentDocumentation object
            format: Output format ("text", "markdown", "html")
            
        Returns:
            Rendered documentation
        """
        if format == "markdown":
            return self._render_component_detail_markdown(component_doc)
        elif format == "html":
            return self._render_component_detail_html(component_doc)
        else:
            return self._render_component_detail_text(component_doc)
    
    def render_ecosystem_index(self, ecosystem_docs: Dict[str, Any], format: str = "text") -> str:
        """
        Render ecosystem overview documentation.
        
        Args:
            ecosystem_docs: Dictionary of plugin documentation
            format: Output format ("text", "markdown", "html")
            
        Returns:
            Rendered documentation
        """
        if format == "markdown":
            return self._render_ecosystem_index_markdown(ecosystem_docs)
        else:
            return self._render_ecosystem_index_text(ecosystem_docs)
    
    def _render_plugin_overview_text(self, plugin_doc: Any) -> str:
        """Render plugin overview in text format."""
        lines = []
        
        # Header
        lines.append(f"📚 Plugin Documentation: {plugin_doc.namespace} v{plugin_doc.version}")
        lines.append(f"├── 👤 Author: {plugin_doc.author}")
        lines.append(f"├── 📝 Description: {plugin_doc.description}")
        lines.append(f"├── 📁 Source Files: {len(plugin_doc.source_files)} files analyzed")
        
        for source_file in plugin_doc.source_files:
            lines.append(f"│   ├── {source_file}")
        
        # Component sections
        icons = {"callables": "📞", "listeners": "👂", "runnables": "🏃"}
        
        for comp_type, components in plugin_doc.components.items():
            if not components:
                continue
                
            icon = icons.get(comp_type, "🔧")
            lines.append(f"├── {icon} {comp_type.title()} ({len(components)})")
            
            for i, component in enumerate(components):
                is_last = i == len(components) - 1
                prefix = "└──" if is_last else "├──"
                
                lines.append(f"│   {prefix} {component.full_name}")
                lines.append(f"│   │   ├── 📝 {component.summary}")
                lines.append(f"│   │   ├── 📍 Source: {os.path.basename(component.source_file)} (line {component.line_number})")
                
                if component.signature:
                    lines.append(f"│   │   ├── 🔧 Signature: {component.signature}")
                
                if component.parameters:
                    lines.append(f"│   │   ├── 📋 Parameters:")
                    for param in component.parameters:
                        param_type = f" ({param.type})" if param.type else ""
                        default = f", default={param.default}" if param.default else ""
                        optional = ", optional" if param.is_optional else ""
                        lines.append(f"│   │   │   ├── {param.name}{param_type}{optional}{default}: {param.description}")
                
                if component.returns:
                    return_type = f" ({component.returns.type})" if component.returns.type else ""
                    lines.append(f"│   │   ├── 🔄 Returns{return_type}: {component.returns.description}")
                
                if component.raises:
                    lines.append(f"│   │   ├── ⚠️ Raises:")
                    for exc in component.raises:
                        lines.append(f"│   │   │   ├── {exc.exception_type}: {exc.description}")
                
                if component.examples:
                    lines.append(f"│   │   ├── 💡 Examples ({len(component.examples)}):")
                    for j, example in enumerate(component.examples[:3]):  # Limit to 3 examples
                        example_lines = example.code.split('\n')
                        lines.append(f"│   │   │   ├── Example {j+1}:")
                        for line in example_lines[:3]:  # Limit lines per example
                            lines.append(f"│   │   │   │   {line}")
                        if len(example_lines) > 3:
                            lines.append(f"│   │   │   │   ... (truncated)")
                
                if component.notes:
                    lines.append(f"│   │   └── 📝 Notes:")
                    for note in component.notes:
                        lines.append(f"│   │       ├── {note}")
        
        return '\n'.join(lines)
    
    def _render_plugin_overview_markdown(self, plugin_doc: Any) -> str:
        """Render plugin overview in markdown format."""
        lines = []
        
        # Header
        lines.append(f"# {plugin_doc.namespace} Plugin")
        lines.append("")
        lines.append(f"**Version:** {plugin_doc.version}")
        lines.append(f"**Author:** {plugin_doc.author}")
        lines.append(f"**Description:** {plugin_doc.description}")
        lines.append("")
        
        # Source files
        lines.append("## Source Files")
        lines.append("")
        for source_file in plugin_doc.source_files:
            lines.append(f"- `{source_file}`")
        lines.append("")
        
        # Components
        for comp_type, components in plugin_doc.components.items():
            if not components:
                continue
                
            lines.append(f"## {comp_type.title()}")
            lines.append("")
            
            for component in components:
                lines.append(f"### {component.full_name}")
                lines.append("")
                lines.append(component.summary)
                lines.append("")
                
                if component.description:
                    lines.append(component.description)
                    lines.append("")
                
                if component.signature:
                    lines.append("**Signature:**")
                    lines.append(f"```python")
                    lines.append(component.signature)
                    lines.append("```")
                    lines.append("")
                
                if component.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    for param in component.parameters:
                        param_type = f" ({param.type})" if param.type else ""
                        lines.append(f"- `{param.name}`{param_type}: {param.description}")
                    lines.append("")
                
                if component.examples:
                    lines.append("**Examples:**")
                    lines.append("")
                    for example in component.examples:
                        lines.append("```python")
                        lines.append(example.code)
                        lines.append("```")
                        lines.append("")
        
        return '\n'.join(lines)
    
    def _render_component_detail_text(self, component_doc: Any) -> str:
        """Render component detail in text format."""
        lines = []
        
        lines.append(f"🔍 Component Documentation: {component_doc.full_name}")
        lines.append(f"├── 📝 Description: {component_doc.summary}")
        
        if component_doc.signature:
            lines.append(f"├── 🔧 Signature: {component_doc.signature}")
        
        if component_doc.parameters:
            lines.append("├── 📋 Parameters")
            for param in component_doc.parameters:
                param_type = f" ({param.type})" if param.type else ""
                optional = ", optional" if param.is_optional else ""
                lines.append(f"│   ├── {param.name}{param_type}{optional}: {param.description}")
        
        if component_doc.returns:
            return_type = f" ({component_doc.returns.type})" if component_doc.returns.type else ""
            lines.append(f"├── 🔄 Returns{return_type}: {component_doc.returns.description}")
        
        lines.append(f"├── 📍 Source: {component_doc.source_file}:{component_doc.line_number}")
        
        if component_doc.examples:
            lines.append("├── 💡 Examples")
            for i, example in enumerate(component_doc.examples):
                lines.append(f"│   ├── Example {i+1}:")
                for line in example.code.split('\n'):
                    lines.append(f"│   │   {line}")
        
        if component_doc.notes:
            lines.append("└── 📝 Notes")
            for note in component_doc.notes:
                lines.append(f"    {note}")
        
        return '\n'.join(lines)
    
    def _render_component_detail_markdown(self, component_doc: Any) -> str:
        """Render component detail in markdown format."""
        lines = []
        
        lines.append(f"# {component_doc.full_name}")
        lines.append("")
        lines.append(component_doc.summary)
        lines.append("")
        
        if component_doc.description:
            lines.append(component_doc.description)
            lines.append("")
        
        if component_doc.signature:
            lines.append("## Signature")
            lines.append("")
            lines.append("```python")
            lines.append(component_doc.signature)
            lines.append("```")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _render_plugin_overview_html(self, plugin_doc: Any) -> str:
        """Render plugin overview in HTML format."""
        # Basic HTML rendering - can be enhanced with proper templates
        return f"<h1>{plugin_doc.namespace} Plugin</h1><p>{plugin_doc.description}</p>"
    
    def _render_component_detail_html(self, component_doc: Any) -> str:
        """Render component detail in HTML format."""
        return f"<h1>{component_doc.full_name}</h1><p>{component_doc.summary}</p>"
    
    def _render_ecosystem_index_text(self, ecosystem_docs: Dict[str, Any]) -> str:
        """Render ecosystem index in text format."""
        lines = []
        
        lines.append("🌍 OWA Plugin Ecosystem")
        lines.append(f"├── Total Plugins: {len(ecosystem_docs)}")
        
        for namespace, plugin_doc in ecosystem_docs.items():
            total_components = sum(len(components) for components in plugin_doc.components.values())
            lines.append(f"├── {namespace} v{plugin_doc.version} ({total_components} components)")
            lines.append(f"│   └── {plugin_doc.description}")
        
        return '\n'.join(lines)
    
    def _render_ecosystem_index_markdown(self, ecosystem_docs: Dict[str, Any]) -> str:
        """Render ecosystem index in markdown format."""
        lines = []
        
        lines.append("# OWA Plugin Ecosystem")
        lines.append("")
        
        for namespace, plugin_doc in ecosystem_docs.items():
            total_components = sum(len(components) for components in plugin_doc.components.values())
            lines.append(f"## {namespace}")
            lines.append("")
            lines.append(f"**Version:** {plugin_doc.version}")
            lines.append(f"**Components:** {total_components}")
            lines.append(f"**Description:** {plugin_doc.description}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _get_default_template_dir(self) -> str:
        """Get the default template directory."""
        return str(Path(__file__).parent / "templates")
