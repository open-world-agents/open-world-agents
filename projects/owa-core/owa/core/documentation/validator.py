"""
Documentation validation system for OWA plugins.

This module implements the core validation logic for OEP-0004,
providing comprehensive documentation quality checks for plugin components.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

from ..plugin_discovery import get_plugin_discovery
from ..registry import CALLABLES, LISTENERS, RUNNABLES


@dataclass
class ValidationResult:
    """Result of documentation validation for a component or plugin."""

    component: str
    status: str  # "pass", "warning", "fail"
    issues: List[str]
    documented: int = 0
    total: int = 0


@dataclass
class PluginValidationResult:
    """Aggregated validation result for a plugin."""

    plugin_name: str
    documented: int
    total: int
    components: List[ValidationResult]

    @property
    def coverage(self) -> float:
        """Calculate documentation coverage percentage."""
        return self.documented / self.total if self.total > 0 else 0.0

    @property
    def status(self) -> str:
        """Determine overall status based on coverage."""
        if self.coverage == 1.0:
            return "pass"
        elif self.coverage >= 0.75:
            return "warning"
        else:
            return "fail"


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
        documented_count = 0
        total_count = 0

        # Validate each component type
        for component_type, components in plugin_spec.components.items():
            for component_name in components.keys():
                full_name = f"{plugin_spec.namespace}/{component_name}"

                try:
                    # Load the component to inspect it
                    component = self._load_component(component_type, full_name)
                    result = self.validate_component(component, full_name)

                    if result.status == "pass":
                        documented_count += 1

                    component_results.append(result)
                    total_count += 1

                except Exception as e:
                    # Component failed to load
                    result = ValidationResult(
                        component=full_name, status="fail", issues=[f"Failed to load component: {e}"]
                    )
                    component_results.append(result)
                    total_count += 1

        return PluginValidationResult(
            plugin_name=plugin_name, documented=documented_count, total=total_count, components=component_results
        )

    def validate_component(self, component: Any, full_name: str) -> ValidationResult:
        """
        Validate documentation for a single component.

        Args:
            component: The component object to validate
            full_name: Full name of the component (namespace/name)

        Returns:
            Validation result for the component
        """
        issues = []

        # Check docstring presence
        docstring = inspect.getdoc(component)
        if not docstring:
            issues.append("Missing docstring")
            status = "fail"
        else:
            # Component has docstring, so it's documented (basic level)
            status = "pass"

            # Check docstring quality (these are warnings, not failures)
            quality_issues = self._validate_docstring_quality(docstring)
            issues.extend(quality_issues)

            # Check type hints for functions/methods (warnings)
            if inspect.isfunction(component) or inspect.ismethod(component):
                type_issues = self._validate_type_hints(component)
                issues.extend(type_issues)
            elif inspect.isclass(component):
                # For classes, check __init__ and key methods (warnings)
                class_issues = self._validate_class_documentation(component)
                issues.extend(class_issues)

        return ValidationResult(component=full_name, status=status, issues=issues)

    def _load_component(self, component_type: str, full_name: str) -> Any:
        """Load a component from the appropriate registry."""
        if component_type == "callables":
            return CALLABLES[full_name]
        elif component_type == "listeners":
            return LISTENERS[full_name]
        elif component_type == "runnables":
            return RUNNABLES[full_name]
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def _validate_docstring_quality(self, docstring: str) -> List[str]:
        """Validate the quality of a docstring."""
        issues = []

        # Check for summary (first line should be a summary)
        lines = docstring.strip().split("\n")
        if not lines or not lines[0].strip():
            issues.append("Missing summary in docstring")
        elif len(lines[0].strip()) < 10:
            issues.append("Summary too short (should be descriptive)")

        # Check for parameter documentation (if Args: section exists)
        if "Args:" in docstring or "Arguments:" in docstring:
            # Basic check - could be enhanced
            pass

        # Check for examples
        if "Example" not in docstring and "Examples" not in docstring:
            issues.append("Missing usage examples")

        # Check for return documentation
        if "Returns:" not in docstring and "Return:" not in docstring:
            # Only flag if it's likely a function that should return something
            if any(keyword in docstring.lower() for keyword in ["return", "result", "output"]):
                issues.append("Missing return value documentation")

        return issues

    def _validate_type_hints(self, func: Any) -> List[str]:
        """Validate type hints for a function."""
        issues = []

        try:
            sig = inspect.signature(func)

            # Check parameters have type hints
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                if param.annotation == inspect.Parameter.empty:
                    issues.append(f"Parameter '{param_name}' missing type hint")

            # Check return type hint
            if sig.return_annotation == inspect.Signature.empty:
                issues.append("Missing return type hint")

        except (ValueError, TypeError):
            # Can't inspect signature
            issues.append("Unable to inspect function signature")

        return issues

    def _validate_class_documentation(self, cls: type) -> List[str]:
        """Validate documentation for a class."""
        issues = []

        # Check __init__ method if it exists
        if hasattr(cls, "__init__"):
            init_method = getattr(cls, "__init__")
            if init_method != object.__init__:  # Not the default object.__init__
                init_issues = self._validate_type_hints(init_method)
                issues.extend([f"__init__ {issue}" for issue in init_issues])

        return issues
