# ================ Plugin Specification System ================================
# Defines the PluginSpec class for entry points-based plugin discovery

from typing import Dict, Optional

from pydantic import BaseModel


class PluginSpec(BaseModel):
    """
    Plugin specification for entry points-based plugin discovery.

    This class defines the structure that plugins must follow when declaring
    their components via entry points.
    """

    namespace: str
    version: str
    description: str
    author: Optional[str] = None
    components: Dict[str, Dict[str, str]]

    model_config = {
        "extra": "forbid",  # Don't allow extra fields
        "str_strip_whitespace": True,  # Strip whitespace from strings
    }

    def validate_components(self) -> None:
        """
        Validate that component types are supported.

        Raises:
            ValueError: If unsupported component types are found
        """
        supported_types = {"callables", "listeners", "runnables"}
        for component_type in self.components.keys():
            if component_type not in supported_types:
                raise ValueError(f"Unsupported component type '{component_type}'. Supported types: {supported_types}")

    def get_component_names(self, component_type: str) -> list[str]:
        """
        Get all component names for a given type.

        Args:
            component_type: Type of components to list

        Returns:
            List of component names with namespace prefix
        """
        if component_type not in self.components:
            return []

        return [f"{self.namespace}/{name}" for name in self.components[component_type].keys()]

    def get_import_path(self, component_type: str, name: str) -> Optional[str]:
        """
        Get the import path for a specific component.

        Args:
            component_type: Type of component
            name: Name of component (without namespace)

        Returns:
            Import path or None if not found
        """
        if component_type not in self.components:
            return None

        return self.components[component_type].get(name)
