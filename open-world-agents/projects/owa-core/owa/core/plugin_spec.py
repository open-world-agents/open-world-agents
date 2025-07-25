# ================ Plugin Specification System ================================
# Defines the PluginSpec class for entry points-based plugin discovery

import re
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
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

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PluginSpec":
        """
        Load a PluginSpec from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            PluginSpec instance

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValueError: If the plugin specification is invalid
        """
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {yaml_path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"YAML file must contain a dictionary, got {type(data)}")

        return cls(**data)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save the PluginSpec to a YAML file.

        Args:
            yaml_path: Path where to save the YAML file
        """
        yaml_file = Path(yaml_path)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_entry_point(cls, entry_point_spec: str) -> "PluginSpec":
        """
        Load a PluginSpec from an entry point specification.

        Args:
            entry_point_spec: Entry point specification in format "module.path:object_name"

        Returns:
            PluginSpec instance

        Raises:
            ValueError: If the entry point specification is invalid
            ImportError: If the module cannot be imported
            AttributeError: If the object doesn't exist in the module
            TypeError: If the object is not a PluginSpec instance
        """
        import importlib

        if ":" not in entry_point_spec:
            raise ValueError(
                f"Invalid entry point format: '{entry_point_spec}'. Must be in format 'module.path:object_name'"
            )

        module_path, object_name = entry_point_spec.split(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module_path}': {e}")

        try:
            plugin_spec_obj = getattr(module, object_name)
        except AttributeError:
            raise AttributeError(f"Object '{object_name}' not found in module '{module_path}'")

        if not isinstance(plugin_spec_obj, cls):
            raise TypeError(f"Object '{entry_point_spec}' must be a PluginSpec instance, got {type(plugin_spec_obj)}")

        return plugin_spec_obj
