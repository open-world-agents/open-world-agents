"""
Documentation validation system for OWA plugins.

This module implements the core validation logic for OEP-0004,
providing comprehensive documentation quality checks for plugin components.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import griffe

from ..plugin_discovery import get_plugin_discovery


@dataclass
class ValidationResult:
    """Result of documentation validation for a component."""

    component: str
    quality_grade: str  # "good", "acceptable", "poor", "skipped"
    issues: List[str]
    skip_reason: str = ""


@dataclass
class PluginValidationResult:
    """Aggregated validation result for a plugin."""

    plugin_name: str
    documented: int  # good + acceptable
    total: int  # total components (excluding skipped)
    good_quality: int  # only good quality components
    skipped: int  # components with @skip-quality-check
    components: List[ValidationResult]

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
    ) -> str:
        """Determine overall plugin status based on configurable quality thresholds."""
        # PASS: ≥ coverage_pass AND ≥ quality_pass
        if self.coverage >= min_coverage_pass and self.quality_ratio >= min_quality_pass:
            return "pass"
        # FAIL: < coverage_fail OR < quality_fail
        elif self.coverage < min_coverage_fail or self.quality_ratio < min_quality_fail:
            return "fail"
        # WARN: between thresholds
        else:
            return "warning"

    @property
    def status(self) -> str:
        """Determine overall plugin status based on default quality thresholds."""
        return self.get_status()


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
        for components in plugin_spec.components.values():
            for component_name, import_path in components.items():
                full_name = f"{plugin_spec.namespace}/{component_name}"

                try:
                    # Load the component to inspect it
                    component = self._load_component(import_path)
                    result = self.validate_component(component, full_name)

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
                    result = ValidationResult(
                        component=full_name, quality_grade="poor", issues=[f"Failed to load component: {e}"]
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

    def validate_component(self, component: Any, full_name: str) -> ValidationResult:
        """
        Validate documentation for a single component.

        Args:
            component: The component object to validate (griffe object)
            full_name: Full name of the component (namespace/name)

        Returns:
            Validation result for the component
        """
        issues = []

        # Check docstring presence
        docstring = self._get_docstring(component)
        if not docstring:
            return ValidationResult(component=full_name, quality_grade="poor", issues=["Missing docstring"])

        # Check for skip quality check directive
        skip_reason = self._check_skip_directive(docstring)
        if skip_reason:
            return ValidationResult(component=full_name, quality_grade="skipped", issues=[], skip_reason=skip_reason)

        # Component has docstring, determine quality grade
        quality_issues = self._validate_docstring_quality(docstring)
        issues.extend(quality_issues)

        # Check type hints for functions/methods
        if self._is_function_or_method(component):
            type_issues = self._validate_type_hints(component)
            issues.extend(type_issues)
        elif self._is_class(component):
            # For classes, check on_configure and key methods
            class_issues = self._validate_class_documentation(component)
            issues.extend(class_issues)

        # Determine quality grade
        quality_grade = self._determine_quality_grade(docstring, issues)

        return ValidationResult(component=full_name, quality_grade=quality_grade, issues=issues)

    def _load_component(self, import_path: str) -> Any:
        """
        Load a component using griffe for static analysis to avoid OS-specific import issues.

        Args:
            import_path: Import path in format "module.path:object_name"

        Returns:
            Griffe object for static analysis
        """
        # Parse import path
        if ":" not in import_path:
            raise ValueError(f"Invalid import path format: {import_path}")

        module_path, object_name = import_path.split(":", 1)

        # Load the module using griffe (static analysis only)
        module = griffe.load(module_path, allow_inspection=False)

        # Navigate to the specific object
        if "." in object_name:
            # Handle nested objects like "ClassName.method_name"
            parts = object_name.split(".")
            obj = module
            for part in parts:
                obj = obj[part]
            return obj
        else:
            # Direct object access
            return module[object_name]

    def _get_docstring(self, component: Any) -> str:
        """Get docstring from griffe component object."""
        if hasattr(component, "docstring") and component.docstring:
            return component.docstring.value if hasattr(component.docstring, "value") else str(component.docstring)
        return ""

    def _is_function_or_method(self, component: Any) -> bool:
        """Check if component is a function or method using griffe object."""
        if hasattr(component, "kind"):
            kind_value = component.kind.value if hasattr(component.kind, "value") else str(component.kind)
            return kind_value in ("function", "method")
        return False

    def _is_class(self, component: Any) -> bool:
        """Check if component is a class using griffe object."""
        if hasattr(component, "kind"):
            kind_value = component.kind.value if hasattr(component.kind, "value") else str(component.kind)
            return kind_value == "class"
        return False

    def _validate_docstring_quality(self, docstring: str) -> List[str]:
        """Validate the quality of a docstring."""
        issues = []

        # Check for summary (first line should be a summary)
        lines = docstring.strip().split("\n")
        if not lines or not lines[0].strip():
            issues.append("Missing summary in docstring")
        elif len(lines[0].strip()) < 50:
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

    def _validate_type_hints(self, func: Any, validate_return: bool = True) -> List[str]:
        """Validate type hints for a function using griffe object."""
        issues = []

        try:
            if hasattr(func, "parameters"):
                # Griffe object - use griffe's parameter information
                for param in func.parameters:
                    if param.name in ("self", "cls"):
                        continue
                    if not param.annotation:
                        issues.append(f"Parameter '{param.name}' missing type hint")

                # Check return type hint if validate_return is True
                if validate_return and not func.annotation:
                    issues.append("Missing return type hint")
            else:
                issues.append("Unable to inspect function signature - not a griffe function object")

        except (ValueError, TypeError, AttributeError):
            # Can't inspect signature
            issues.append("Unable to inspect function signature")

        return issues

    def _validate_class_documentation(self, cls: Any) -> List[str]:
        """Validate documentation for a class using griffe object."""
        issues = []

        # Check on_configure method if it exists
        try:
            if hasattr(cls, "members") and "on_configure" in cls.members:
                on_configure = cls.members["on_configure"]
                # Don't check return type hint for on_configure
                init_issues = self._validate_type_hints(on_configure, validate_return=False)
                issues.extend([f"on_configure {issue}" for issue in init_issues])
        except (AttributeError, KeyError):
            # Method doesn't exist or can't be accessed
            pass

        return issues

    def _check_skip_directive(self, docstring: str) -> str:
        """Check if docstring contains @skip-quality-check directive."""

        # Look for @skip-quality-check: reason pattern
        pattern = r"@skip-quality-check:\s*([a-zA-Z-]+)"
        match = re.search(pattern, docstring)

        if match:
            reason = match.group(1)
            valid_reasons = {"legacy-code", "internal-api", "experimental", "deprecated", "third-party"}
            if reason in valid_reasons:
                return reason
            else:
                # Invalid reason, treat as not skipped
                return ""

        return ""

    def _determine_quality_grade(self, docstring: str, issues: List[str]) -> str:
        """Determine quality grade based on docstring content and issues."""
        # GOOD: Has examples AND type hints AND comprehensive description
        has_examples = "Example" in docstring or "Examples" in docstring
        has_comprehensive_desc = len(docstring.strip()) > 50
        has_type_hints = not any("missing type hint" in issue.lower() for issue in issues)

        if has_examples and has_type_hints and has_comprehensive_desc:
            return "good"
        else:
            return "acceptable"
