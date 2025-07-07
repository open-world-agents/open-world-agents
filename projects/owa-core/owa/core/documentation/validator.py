"""
Documentation validation system for OWA plugins.

This module implements the core validation logic for OEP-0004,
providing comprehensive documentation quality checks for plugin components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, cast

import griffe

from ..plugin_discovery import get_plugin_discovery

# Type alias for griffe objects that we work with
GriffeObject = Union[griffe.Object, griffe.Function, griffe.Class, griffe.Module, griffe.Attribute, griffe.Alias]


class PluginStatus(Enum):
    """Plugin documentation validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ComponentValidationResult:
    """Result of documentation validation for a component."""

    component: str
    quality_grade: str  # "good", "acceptable", "poor", "skipped"
    improvements: List[str]  # Issues that need improvement


@dataclass
class PluginValidationResult:
    """Aggregated validation result for a plugin."""

    plugin_name: str
    documented: int  # good + acceptable
    total: int  # total components (excluding skipped)
    good_quality: int  # only good quality components
    skipped: int  # components with @skip-quality-check
    components: List[ComponentValidationResult]

    @property
    def coverage(self) -> float:
        """Calculate documentation coverage percentage (documented/total)."""
        return self.documented / self.total if self.total > 0 else 0.0

    @property
    def quality_ratio(self) -> float:
        """Calculate good quality ratio (good/total)."""
        return self.good_quality / self.total if self.total > 0 else 0.0

    def get_status(
        self,
        min_coverage_pass: float = 0.8,
        min_coverage_fail: float = 0.6,
        min_quality_pass: float = 0.6,
        min_quality_fail: float = 0.0,
    ) -> PluginStatus:
        """Determine overall plugin status based on configurable quality thresholds."""
        # PASS: ≥ coverage_pass AND ≥ quality_pass
        if self.coverage >= min_coverage_pass and self.quality_ratio >= min_quality_pass:
            return PluginStatus.PASS
        # FAIL: < coverage_fail OR < quality_fail
        elif self.coverage < min_coverage_fail or self.quality_ratio < min_quality_fail:
            return PluginStatus.FAIL
        # WARN: between thresholds
        else:
            return PluginStatus.WARNING

    @property
    def status(self) -> PluginStatus:
        """Determine overall plugin status based on default quality thresholds."""
        return self.get_status()

    @property
    def all_improvements(self) -> List[str]:
        """Get all improvement issues across all components."""
        issues = []
        for comp in self.components:
            for issue in comp.improvements:
                issues.append(f"{comp.component}: {issue}")
        return issues


class DocumentationValidator:
    """
    Documentation validator for OWA plugin components.

    This class implements the validation logic specified in OEP-0004,
    checking for docstring presence, quality, type hints, and examples.
    """

    def __init__(self):
        self.plugin_discovery = get_plugin_discovery()

    def validate_all_plugins(self) -> Dict[str, PluginValidationResult]:
        """
        Validate documentation for all discovered plugins.

        Returns:
            Dictionary mapping plugin names to their validation results
        """
        results = {}

        for plugin_name in self.plugin_discovery.discovered_plugins.keys():
            results[plugin_name] = self.validate_plugin(plugin_name)

        return results

    def validate_plugin(self, plugin_name: str) -> PluginValidationResult:
        """
        Validate documentation for a specific plugin.

        Args:
            plugin_name: Name of the plugin to validate

        Returns:
            Validation result for the plugin

        Raises:
            KeyError: If plugin is not found
        """
        if plugin_name not in self.plugin_discovery.discovered_plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found")

        plugin_spec = self.plugin_discovery.discovered_plugins[plugin_name]
        component_results = []
        documented_count = 0  # good + acceptable
        good_quality_count = 0  # only good
        total_count = 0  # excluding skipped
        skipped_count = 0

        # Validate each component type
        for component_type, components in plugin_spec.components.items():
            for component_name, import_path in components.items():
                full_name = f"{plugin_spec.namespace}/{component_name}"

                try:
                    # Load the component to inspect it
                    component = self._load_component(import_path)
                    result = self.validate_component(component, full_name, component_type)

                    if result.quality_grade == "skipped":
                        skipped_count += 1
                    else:
                        total_count += 1
                        if result.quality_grade in ("good", "acceptable"):
                            documented_count += 1
                        if result.quality_grade == "good":
                            good_quality_count += 1

                    component_results.append(result)

                except Exception as e:
                    # Component failed to load
                    result = ComponentValidationResult(
                        component=full_name,
                        quality_grade="poor",
                        improvements=[f"Failed to load component: {e}"],
                    )
                    component_results.append(result)
                    total_count += 1

        return PluginValidationResult(
            plugin_name=plugin_name,
            documented=documented_count,
            total=total_count,
            good_quality=good_quality_count,
            skipped=skipped_count,
            components=component_results,
        )

    def validate_component(
        self, component: GriffeObject, full_name: str, component_type: str
    ) -> ComponentValidationResult:
        """Validate documentation for a single component."""
        docstring = self._get_docstring(component)

        # No docstring = poor
        if not docstring:
            return ComponentValidationResult(
                component=full_name, quality_grade="poor", improvements=["Missing docstring"]
            )

        # Skip if requested
        if "@skip-quality-check" in docstring:
            return ComponentValidationResult(component=full_name, quality_grade="skipped", improvements=[])

        # Check requirements for "good" status
        improvements = []

        # Issues that need improvement: Examples
        if "Example" not in docstring and "Examples" not in docstring:
            improvements.append("Missing usage examples")

        # Issues that need improvement: Type hints
        if component_type == "callables":
            # For callables, check the function itself
            component = cast(griffe.Function, component)
            if not self._has_complete_type_hints(component):
                improvements.append("Missing type hints")
        elif component_type in ("listeners", "runnables"):
            # For listeners/runnables, check on_configure method
            if hasattr(component, "members") and "on_configure" in component.members:
                on_configure = cast(griffe.Function, component.members["on_configure"])
                if not self._has_complete_type_hints(on_configure, validate_return=False):
                    improvements.append("Missing type hints in on_configure method")

        # Issues that need improvement: Comprehensive description
        if len(docstring.strip()) <= 50:
            improvements.append("Missing comprehensive description")

        # Determine quality grade
        quality_grade = "good" if not improvements else "acceptable"

        return ComponentValidationResult(component=full_name, quality_grade=quality_grade, improvements=improvements)

    def _load_component(self, import_path: str) -> GriffeObject:
        """Load a component using griffe for static analysis."""
        if ":" not in import_path:
            raise ValueError(f"Invalid import path format: {import_path}")

        module_path, object_name = import_path.split(":", 1)
        module = griffe.load(module_path, allow_inspection=False)

        if "." in object_name:
            # Handle nested objects like "ClassName.method_name"
            parts = object_name.split(".")
            obj = module
            for part in parts:
                obj = obj[part]
            return obj
        else:
            return module[object_name]

    def _get_docstring(self, component: GriffeObject) -> str:
        """Get docstring from griffe component object."""
        if hasattr(component, "docstring") and component.docstring:
            return component.docstring.value if hasattr(component.docstring, "value") else str(component.docstring)
        return ""

    def _has_complete_type_hints(self, func: griffe.Function, validate_return: bool = True) -> bool:
        """Check if function has complete type hints."""
        if not hasattr(func, "parameters"):
            return False

        # Check parameters
        for param in func.parameters:
            if param.name in ("self", "cls"):
                continue
            if not param.annotation:
                return False

        # Check return type
        if validate_return:
            return bool(func.annotation)
        return True
