"""
OWA mkdocstrings handler for EnvPlugin components.

This handler enables automatic documentation generation for OWA plugins
using the familiar mkdocstrings syntax.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mkdocstrings.handlers.base import BaseHandler
    MKDOCSTRINGS_AVAILABLE = True
except ImportError:
    # Graceful degradation when mkdocstrings is not available
    BaseHandler = object
    MKDOCSTRINGS_AVAILABLE = False

try:
    from owa.core.plugin_discovery import get_plugin_discovery
    OWA_AVAILABLE = True
except ImportError:
    OWA_AVAILABLE = False


class OWAHandler(BaseHandler):
    """Minimal mkdocstrings handler for OWA EnvPlugin components."""

    name = "owa"
    domain = "py"
    fallback_theme = "material"

    def __init__(self, **kwargs):
        """Initialize the OWA handler."""
        if not MKDOCSTRINGS_AVAILABLE:
            raise ImportError("mkdocstrings is required to use the OWA handler")
        if not OWA_AVAILABLE:
            raise ImportError("OWA core is required to use the OWA handler")
        
        super().__init__(**kwargs)

    def collect(self, identifier: str, options: dict) -> dict:
        """
        Collect documentation data for an OWA EnvPlugin component.
        
        Args:
            identifier: Plugin namespace or namespace/component
            options: Collection options
            
        Returns:
            Dictionary containing collected documentation data
        """
        try:
            plugin_discovery = get_plugin_discovery()
            
            if "/" in identifier:
                namespace, component_name = identifier.split("/", 1)
                return self._collect_component(namespace, component_name, plugin_discovery, options)
            else:
                return self._collect_plugin(identifier, plugin_discovery, options)
                
        except Exception as e:
            return {"error": f"Failed to collect documentation for '{identifier}': {e}"}

    def _collect_plugin(self, namespace: str, plugin_discovery, options: dict) -> dict:
        """Collect plugin-level documentation."""
        plugin_spec = plugin_discovery.discovered_plugins.get(namespace)
        if not plugin_spec:
            return {"error": f"Plugin '{namespace}' not found"}

        # Extract components information
        components = {}
        for component_type, component_dict in plugin_spec.components.items():
            components[component_type] = list(component_dict.keys())

        return {
            "type": "plugin",
            "namespace": namespace,
            "version": getattr(plugin_spec, 'version', 'unknown'),
            "description": getattr(plugin_spec, 'description', ''),
            "author": getattr(plugin_spec, 'author', ''),
            "components": components,
            "total_components": sum(len(comps) for comps in components.values())
        }

    def _collect_component(self, namespace: str, component_name: str, plugin_discovery, options: dict) -> dict:
        """Collect component-level documentation."""
        plugin_spec = plugin_discovery.discovered_plugins.get(namespace)
        if not plugin_spec:
            return {"error": f"Plugin '{namespace}' not found"}

        # Find the component in the plugin spec
        component_obj = None
        component_type = None
        
        for comp_type, components in plugin_spec.components.items():
            if component_name in components:
                component_type = comp_type
                try:
                    # Load the component
                    from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES
                    full_name = f"{namespace}/{component_name}"
                    
                    if comp_type == "callables":
                        component_obj = CALLABLES.get(full_name)
                    elif comp_type == "listeners":
                        component_obj = LISTENERS.get(full_name)
                    elif comp_type == "runnables":
                        component_obj = RUNNABLES.get(full_name)
                        
                except Exception as e:
                    return {"error": f"Failed to load component '{full_name}': {e}"}
                break

        if not component_obj:
            return {"error": f"Component '{component_name}' not found in plugin '{namespace}'"}

        # Extract documentation information
        docstring = inspect.getdoc(component_obj) or ""
        signature = None
        
        try:
            if inspect.isfunction(component_obj) or inspect.ismethod(component_obj):
                signature = str(inspect.signature(component_obj))
        except (ValueError, TypeError):
            pass

        return {
            "type": "component",
            "namespace": namespace,
            "name": component_name,
            "component_type": component_type,
            "full_name": f"{namespace}/{component_name}",
            "docstring": docstring,
            "signature": signature,
            "is_function": inspect.isfunction(component_obj),
            "is_class": inspect.isclass(component_obj),
            "is_method": inspect.ismethod(component_obj)
        }

    def render(self, data: dict, options: dict) -> str:
        """
        Render collected data using templates.
        
        Args:
            data: Collected documentation data
            options: Rendering options
            
        Returns:
            Rendered HTML string
        """
        if data.get("error"):
            return f'<div class="error">Error: {data["error"]}</div>'

        template_name = "plugin.html" if data.get("type") == "plugin" else "component.html"
        
        try:
            template = self.env.get_template(template_name)
            return template.render(data=data, options=options)
        except Exception as e:
            return f'<div class="error">Template error: {e}</div>'

    def get_templates_dir(self, handler: str = None) -> Path:
        """Get the templates directory for this handler."""
        return Path(__file__).parent / "templates"


def get_handler(**kwargs) -> OWAHandler:
    """
    Entry point function for mkdocstrings.
    
    Returns:
        Configured OWA handler instance
    """
    return OWAHandler(**kwargs)
