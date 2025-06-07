# ================ Plugin Specification System ================================
# Defines the PluginSpec class for entry points-based plugin discovery

import re
from typing import Dict, Optional

from pydantic import BaseModel, field_validator


class PluginSpec(BaseModel):
    """
    Plugin specification for entry points-based plugin discovery.

    This class defines the structure that plugins must follow when declaring
    their components via entry points.

    Validation Rules (OEP-0003):
    - namespace MUST consist of only letters, numbers, underscores, and hyphens
    - component names SHOULD consist of only letters, numbers, underscores, and dots
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

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """
        Validate namespace according to OEP-0003 rules.

        namespace MUST consist of only letters, numbers, underscores, and hyphens.
        """
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                f"Namespace '{v}' is invalid. "
                "Namespace MUST consist of only letters, numbers, underscores, and hyphens."
            )
        return v

    @field_validator("components")
    @classmethod
    def validate_component_names(cls, v: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Validate component names according to OEP-0003 rules.

        OEP-0003 Naming Convention:
        - Components consist of letters, numbers, underscores, and dots
        - Examples: "screen_capture", "mouse.click", "omnimodal.recorder"
        """
        for component_type, components in v.items():
            for name in components.keys():
                # Check allowed characters: alphanumeric, underscores, dots
                if not re.match(r"^[a-zA-Z0-9_.]+$", name):
                    raise ValueError(
                        f"Component name '{name}' in '{component_type}' is invalid. "
                        "Component names SHOULD consist of only letters, numbers, underscores, and dots."
                    )
        return v

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
