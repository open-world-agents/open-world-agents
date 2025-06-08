"""
Main documentation generator for OEP-0004.

Provides the primary interface for generating plugin documentation from source code.
"""

from datetime import datetime
from typing import Dict, List, Optional

from ..plugin_discovery import get_plugin_discovery
from .inspector import SignatureInspector
from .models import PluginDocumentation


class PluginDocumentationGenerator:
    """
    Main documentation generation engine for OEP-0004.
    
    This class provides mkdocstrings-like functionality for OWA plugins,
    automatically extracting and organizing documentation from source code.
    """
    
    def __init__(self):
        self.inspector = SignatureInspector()
        self.plugin_discovery = get_plugin_discovery()
    
    def generate_plugin_documentation(self, namespace: str) -> Optional[PluginDocumentation]:
        """
        Generate complete documentation for a specific plugin.
        
        Args:
            namespace: Plugin namespace to generate documentation for
            
        Returns:
            Complete plugin documentation or None if plugin not found
        """
        # Get plugin spec
        plugin_spec = self.plugin_discovery.discovered_plugins.get(namespace)
        if not plugin_spec:
            return None
        
        # Extract component documentation
        components = self.inspector.get_all_plugin_components(namespace)
        
        # Get source files
        source_files = self._get_plugin_source_files(namespace, components)
        
        return PluginDocumentation(
            namespace=namespace,
            version=plugin_spec.version,
            description=plugin_spec.description,
            author=plugin_spec.author or "Unknown",
            components=components,
            generated_at=datetime.now(),
            source_files=source_files
        )
    
    def generate_component_documentation(self, namespace: str, component_name: str):
        """
        Generate documentation for a specific component.
        
        Args:
            namespace: Plugin namespace
            component_name: Component name (e.g., "mouse.click")
            
        Returns:
            Component documentation or None if not found
        """
        full_name = f"{namespace}/{component_name}"
        
        # Try each component type
        for component_type in ['callables', 'listeners', 'runnables']:
            doc = self.inspector.inspect_component(component_type, full_name)
            if doc:
                return doc
        
        return None
    
    def generate_ecosystem_documentation(self) -> Dict[str, PluginDocumentation]:
        """
        Generate documentation for all discovered plugins.
        
        Returns:
            Dictionary mapping plugin namespaces to their documentation
        """
        ecosystem_docs = {}
        
        for namespace in self.plugin_discovery.discovered_plugins.keys():
            plugin_doc = self.generate_plugin_documentation(namespace)
            if plugin_doc:
                ecosystem_docs[namespace] = plugin_doc
        
        return ecosystem_docs
    
    def get_plugin_overview(self, namespace: str) -> Optional[Dict[str, any]]:
        """
        Get a quick overview of a plugin without full documentation generation.
        
        Args:
            namespace: Plugin namespace
            
        Returns:
            Plugin overview information
        """
        plugin_spec = self.plugin_discovery.discovered_plugins.get(namespace)
        if not plugin_spec:
            return None
        
        # Count components by type
        component_counts = {}
        total_components = 0
        
        for comp_type, components in plugin_spec.components.items():
            count = len(components)
            component_counts[comp_type] = count
            total_components += count
        
        return {
            'namespace': namespace,
            'version': plugin_spec.version,
            'description': plugin_spec.description,
            'author': plugin_spec.author,
            'component_counts': component_counts,
            'total_components': total_components
        }
    
    def _get_plugin_source_files(self, namespace: str, components: Dict[str, List]) -> List[str]:
        """Get list of source files for a plugin."""
        source_files = set()
        
        for component_list in components.values():
            for component in component_list:
                if hasattr(component, 'source_file') and component.source_file != "unknown":
                    source_files.add(component.source_file)
        
        return sorted(list(source_files))
    
    def validate_plugin_documentation(self, namespace: str) -> Dict[str, any]:
        """
        Validate the documentation quality for a plugin.
        
        Args:
            namespace: Plugin namespace to validate
            
        Returns:
            Validation results with issues and recommendations
        """
        plugin_doc = self.generate_plugin_documentation(namespace)
        if not plugin_doc:
            return {
                'valid': False,
                'errors': [f"Plugin '{namespace}' not found"],
                'warnings': [],
                'recommendations': []
            }
        
        errors = []
        warnings = []
        recommendations = []
        
        # Check plugin-level documentation
        if not plugin_doc.description:
            warnings.append("Plugin description is missing")
        
        if not plugin_doc.author:
            recommendations.append("Consider adding author information")
        
        # Check component documentation
        for comp_type, components in plugin_doc.components.items():
            for component in components:
                comp_name = component.full_name
                
                if not component.summary:
                    warnings.append(f"Component '{comp_name}' has no summary")
                
                if not component.description:
                    recommendations.append(f"Component '{comp_name}' could use a detailed description")
                
                if comp_type == 'callables' and not component.parameters:
                    recommendations.append(f"Callable '{comp_name}' should document its parameters")
                
                if not component.examples:
                    recommendations.append(f"Component '{comp_name}' could benefit from usage examples")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations,
            'total_components': sum(len(components) for components in plugin_doc.components.values()),
            'documented_components': sum(
                len([c for c in components if c.summary]) 
                for components in plugin_doc.components.values()
            )
        }
