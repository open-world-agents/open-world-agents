"""
Plugin specification for Open World Agents environment plugins.

This module defines the PluginSpec class used for declaring plugin metadata
and component mappings in the entry points-based plugin discovery system.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PluginSpec:
    """
    Specification for an OWA environment plugin.
    
    This class defines the metadata and component mappings for a plugin,
    used by the entry points-based discovery system to automatically
    register plugin components.
    
    Attributes:
        namespace: Unique namespace for the plugin (e.g., "example", "std")
        version: Plugin version string
        description: Human-readable description of the plugin
        author: Plugin author/maintainer
        components: Dictionary mapping component types to their implementations
    """
    namespace: str
    version: str
    description: str
    author: str
    components: Dict[str, Dict[str, str]]
    
    def __post_init__(self):
        """Validate plugin specification after initialization."""
        if not self.namespace:
            raise ValueError("Plugin namespace cannot be empty")
        
        if not self.version:
            raise ValueError("Plugin version cannot be empty")
            
        # Validate component types
        valid_component_types = {"callables", "listeners", "runnables"}
        for component_type in self.components.keys():
            if component_type not in valid_component_types:
                raise ValueError(f"Invalid component type: {component_type}. "
                               f"Must be one of: {valid_component_types}")
        
        # Validate component module paths
        for component_type, components in self.components.items():
            for name, module_path in components.items():
                if ":" not in module_path:
                    raise ValueError(f"Invalid module path '{module_path}' for {component_type}.{name}. "
                                   "Expected format: 'module_name:component_name'")
    
    def get_component_names(self, component_type: Optional[str] = None) -> Dict[str, list]:
        """
        Get component names, optionally filtered by type.
        
        Args:
            component_type: Optional component type to filter by
            
        Returns:
            Dictionary mapping component types to lists of component names
        """
        if component_type:
            if component_type not in self.components:
                return {component_type: []}
            return {component_type: list(self.components[component_type].keys())}
        
        return {
            comp_type: list(components.keys())
            for comp_type, components in self.components.items()
        }
    
    def get_full_component_names(self, component_type: Optional[str] = None) -> Dict[str, list]:
        """
        Get full component names (namespace/name), optionally filtered by type.
        
        Args:
            component_type: Optional component type to filter by
            
        Returns:
            Dictionary mapping component types to lists of full component names
        """
        result = {}
        component_names = self.get_component_names(component_type)
        
        for comp_type, names in component_names.items():
            result[comp_type] = [f"{self.namespace}/{name}" for name in names]
            
        return result
