"""
Documentation validation system for OWA plugins.

This module implements the core validation logic for OEP-0004,
providing comprehensive documentation quality checks for plugin components.
"""

import inspect
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from ..plugin_discovery import get_plugin_discovery
from ..registry import CALLABLES, LISTENERS, RUNNABLES


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
        for component_type, components in plugin_spec.components.items():
            for component_name in components.keys():
                full_name = f"{plugin_spec.namespace}/{component_name}"

                try:
                    # Load the component to inspect it
                    component = self._load_component(component_type, full_name)
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
            component: The component object to validate
            full_name: Full name of the component (namespace/name)

        Returns:
            Validation result for the component
        """
        issues = []

        # Check docstring presence
        docstring = inspect.getdoc(component)
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
        if inspect.isfunction(component) or inspect.ismethod(component):
            type_issues = self._validate_type_hints(component)
            issues.extend(type_issues)
        elif inspect.isclass(component):
            # For classes, check __init__ and key methods
            class_issues = self._validate_class_documentation(component)
            issues.extend(class_issues)

        # Determine quality grade
        quality_grade = self._determine_quality_grade(docstring, issues)

        return ValidationResult(component=full_name, quality_grade=quality_grade, issues=issues)

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
        # GOOD: Has examples OR type hints OR comprehensive description
        has_examples = "Example" in docstring or "Examples" in docstring
        has_comprehensive_desc = len(docstring.strip()) > 100
        has_type_hints = not any("missing type hint" in issue.lower() for issue in issues)

        if has_examples or has_type_hints or has_comprehensive_desc:
            return "good"
        else:
            return "acceptable"
