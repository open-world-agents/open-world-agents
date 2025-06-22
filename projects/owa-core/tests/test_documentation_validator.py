"""
Tests for owa.core.documentation.validator module.

This module tests the documentation validation system for OWA plugins,
including ValidationResult, PluginValidationResult, and DocumentationValidator classes.
"""

from unittest.mock import Mock, patch

import pytest

from owa.core.documentation.validator import (
    DocumentationValidator,
    PluginValidationResult,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult instances with and without skip reason."""
        # Basic result
        result = ValidationResult(component="test/component", quality_grade="good", issues=["issue1"])
        assert result.component == "test/component"
        assert result.quality_grade == "good"
        assert result.issues == ["issue1"]
        assert result.skip_reason == ""

        # Result with skip reason
        result_skipped = ValidationResult(
            component="test/component", quality_grade="skipped", issues=[], skip_reason="legacy-code"
        )
        assert result_skipped.quality_grade == "skipped"
        assert result_skipped.skip_reason == "legacy-code"


class TestPluginValidationResult:
    """Test PluginValidationResult dataclass."""

    def test_plugin_validation_result_creation_and_calculations(self):
        """Test creating PluginValidationResult and property calculations."""
        components = [
            ValidationResult("test/comp1", "good", []),
            ValidationResult("test/comp2", "acceptable", ["minor issue"]),
        ]
        result = PluginValidationResult(
            plugin_name="test_plugin", documented=8, total=10, good_quality=6, skipped=2, components=components
        )

        # Basic properties
        assert result.plugin_name == "test_plugin"
        assert result.documented == 8
        assert result.total == 10
        assert result.good_quality == 6
        assert result.skipped == 2
        assert len(result.components) == 2

        # Calculated properties
        assert result.coverage == 0.8
        assert result.quality_ratio == 0.6

        # Edge case: zero total
        zero_result = PluginValidationResult(
            plugin_name="empty", documented=0, total=0, good_quality=0, skipped=0, components=[]
        )
        assert zero_result.coverage == 0.0
        assert zero_result.quality_ratio == 0.0

    def test_get_status_scenarios(self):
        """Test get_status method with various scenarios."""
        # Pass: high coverage and quality
        pass_result = PluginValidationResult(
            plugin_name="test", documented=9, total=10, good_quality=7, skipped=0, components=[]
        )
        assert pass_result.get_status() == "pass"
        assert pass_result.status == "pass"  # Test property too

        # Fail: low coverage
        fail_result = PluginValidationResult(
            plugin_name="test", documented=5, total=10, good_quality=7, skipped=0, components=[]
        )
        assert fail_result.get_status() == "fail"

        # Warning: medium coverage/quality
        warning_result = PluginValidationResult(
            plugin_name="test", documented=7, total=10, good_quality=5, skipped=0, components=[]
        )
        assert warning_result.get_status() == "warning"

        # Custom thresholds
        assert (
            warning_result.get_status(
                min_coverage_pass=0.7, min_coverage_fail=0.5, min_quality_pass=0.5, min_quality_fail=0.3
            )
            == "pass"
        )


class TestDocumentationValidator:
    """Test DocumentationValidator class."""

    @pytest.fixture
    def mock_plugin_discovery(self):
        """Create a mock plugin discovery instance."""
        mock_discovery = Mock()
        mock_discovery.discovered_plugins = {}
        return mock_discovery

    @pytest.fixture
    def validator(self, mock_plugin_discovery):
        """Create a DocumentationValidator with mocked dependencies."""
        with patch("owa.core.documentation.validator.get_plugin_discovery", return_value=mock_plugin_discovery):
            return DocumentationValidator()

    def test_validate_all_plugins(self, validator, mock_plugin_discovery):
        """Test validate_all_plugins with empty and populated plugin lists."""
        # Test empty plugins
        mock_plugin_discovery.discovered_plugins = {}
        results = validator.validate_all_plugins()
        assert results == {}

        # Test with plugins
        mock_plugin_spec = Mock()
        mock_plugin_spec.namespace = "test"
        mock_plugin_spec.components = {"callables": {"test_func": "test.module:test_func"}}
        mock_plugin_discovery.discovered_plugins = {"test_plugin": mock_plugin_spec}

        with patch.object(validator, "validate_plugin") as mock_validate:
            mock_result = PluginValidationResult(
                plugin_name="test_plugin", documented=1, total=1, good_quality=1, skipped=0, components=[]
            )
            mock_validate.return_value = mock_result
            results = validator.validate_all_plugins()

            assert "test_plugin" in results
            assert results["test_plugin"] == mock_result
            mock_validate.assert_called_once_with("test_plugin")

    def test_validate_plugin_not_found(self, validator, mock_plugin_discovery):
        """Test validate_plugin with non-existent plugin."""
        mock_plugin_discovery.discovered_plugins = {}
        with pytest.raises(KeyError, match="Plugin 'nonexistent' not found"):
            validator.validate_plugin("nonexistent")

    def test_validate_plugin_success(self, validator, mock_plugin_discovery):
        """Test validate_plugin with successful validation."""
        # Create mock plugin spec
        mock_plugin_spec = Mock()
        mock_plugin_spec.namespace = "test"
        mock_plugin_spec.components = {"callables": {"test_func": "test.module:test_func"}}

        mock_plugin_discovery.discovered_plugins = {"test_plugin": mock_plugin_spec}

        # Mock component loading and validation
        mock_component = Mock()
        mock_validation_result = ValidationResult(component="test/test_func", quality_grade="good", issues=[])

        with (
            patch.object(validator, "_load_component", return_value=mock_component),
            patch.object(validator, "validate_component", return_value=mock_validation_result),
        ):
            result = validator.validate_plugin("test_plugin")

            assert result.plugin_name == "test_plugin"
            assert result.documented == 1
            assert result.total == 1
            assert result.good_quality == 1
            assert result.skipped == 0
            assert len(result.components) == 1
            assert result.components[0] == mock_validation_result

    def test_validate_plugin_with_component_load_error(self, validator, mock_plugin_discovery):
        """Test validate_plugin when component loading fails."""
        # Create mock plugin spec
        mock_plugin_spec = Mock()
        mock_plugin_spec.namespace = "test"
        mock_plugin_spec.components = {"callables": {"test_func": "test.module:test_func"}}

        mock_plugin_discovery.discovered_plugins = {"test_plugin": mock_plugin_spec}

        # Mock component loading to raise an exception
        with patch.object(validator, "_load_component", side_effect=ImportError("Module not found")):
            result = validator.validate_plugin("test_plugin")

            assert result.plugin_name == "test_plugin"
            assert result.documented == 0
            assert result.total == 1
            assert result.good_quality == 0
            assert result.skipped == 0
            assert len(result.components) == 1
            assert result.components[0].quality_grade == "poor"
            assert "Failed to load component: Module not found" in result.components[0].issues

    def test_validate_plugin_mixed_results(self, validator, mock_plugin_discovery):
        """Test validate_plugin with mixed component results."""
        # Create mock plugin spec with multiple components
        mock_plugin_spec = Mock()
        mock_plugin_spec.namespace = "test"
        mock_plugin_spec.components = {
            "callables": {
                "good_func": "test.module:good_func",
                "poor_func": "test.module:poor_func",
                "skipped_func": "test.module:skipped_func",
            }
        }

        mock_plugin_discovery.discovered_plugins = {"test_plugin": mock_plugin_spec}

        # Mock different validation results
        validation_results = [
            ValidationResult("test/good_func", "good", []),
            ValidationResult("test/poor_func", "poor", ["Missing docstring"]),
            ValidationResult("test/skipped_func", "skipped", [], "legacy-code"),
        ]

        with (
            patch.object(validator, "_load_component", return_value=Mock()),
            patch.object(validator, "validate_component", side_effect=validation_results),
        ):
            result = validator.validate_plugin("test_plugin")

            assert result.plugin_name == "test_plugin"
            assert result.documented == 1  # only good
            assert result.total == 2  # good + poor (excluding skipped)
            assert result.good_quality == 1  # only good
            assert result.skipped == 1  # skipped_func
            assert len(result.components) == 3


class TestDocumentationValidatorComponentMethods:
    """Test DocumentationValidator component validation methods."""

    @pytest.fixture
    def validator(self):
        """Create a DocumentationValidator with mocked dependencies."""
        with patch("owa.core.documentation.validator.get_plugin_discovery"):
            return DocumentationValidator()

    def test_validate_component_missing_docstring(self, validator):
        """Test validate_component with missing docstring."""
        mock_component = Mock()

        with patch.object(validator, "_get_docstring", return_value=""):
            result = validator.validate_component(mock_component, "test/component")

            assert result.component == "test/component"
            assert result.quality_grade == "poor"
            assert result.issues == ["Missing docstring"]
            assert result.skip_reason == ""

    def test_validate_component_skipped(self, validator):
        """Test validate_component with skip directive."""
        mock_component = Mock()
        docstring = "Test docstring\n@skip-quality-check: legacy-code"

        with (
            patch.object(validator, "_get_docstring", return_value=docstring),
            patch.object(validator, "_check_skip_directive", return_value="legacy-code"),
        ):
            result = validator.validate_component(mock_component, "test/component")

            assert result.component == "test/component"
            assert result.quality_grade == "skipped"
            assert result.issues == []
            assert result.skip_reason == "legacy-code"

    def test_validate_component_function_good_quality(self, validator):
        """Test validate_component for function with good quality."""
        mock_component = Mock()
        docstring = "Comprehensive function description with examples.\n\nExample:\n    >>> func()\n    'result'"

        with (
            patch.object(validator, "_get_docstring", return_value=docstring),
            patch.object(validator, "_check_skip_directive", return_value=""),
            patch.object(validator, "_validate_docstring_quality", return_value=[]),
            patch.object(validator, "_is_function_or_method", return_value=True),
            patch.object(validator, "_validate_type_hints", return_value=[]),
            patch.object(validator, "_determine_quality_grade", return_value="good"),
        ):
            result = validator.validate_component(mock_component, "test/component")

            assert result.component == "test/component"
            assert result.quality_grade == "good"
            assert result.issues == []

    def test_validate_component_class_with_issues(self, validator):
        """Test validate_component for class with documentation issues."""
        mock_component = Mock()
        docstring = "Short description"
        quality_issues = ["Missing usage examples"]
        class_issues = ["on_configure Missing return type hint"]

        with (
            patch.object(validator, "_get_docstring", return_value=docstring),
            patch.object(validator, "_check_skip_directive", return_value=""),
            patch.object(validator, "_validate_docstring_quality", return_value=quality_issues),
            patch.object(validator, "_is_function_or_method", return_value=False),
            patch.object(validator, "_is_class", return_value=True),
            patch.object(validator, "_validate_class_documentation", return_value=class_issues),
            patch.object(validator, "_determine_quality_grade", return_value="acceptable"),
        ):
            result = validator.validate_component(mock_component, "test/component")

            assert result.component == "test/component"
            assert result.quality_grade == "acceptable"
            assert result.issues == quality_issues + class_issues

    def test_load_component_invalid_format(self, validator):
        """Test _load_component with invalid import path format."""
        with pytest.raises(ValueError, match="Invalid import path format"):
            validator._load_component("invalid_path_without_colon")

    def test_load_component_success(self, validator):
        """Test _load_component with valid import path."""
        mock_module = Mock()
        mock_object = Mock()
        mock_module.__getitem__ = Mock(return_value=mock_object)

        with patch("owa.core.documentation.validator.griffe.load", return_value=mock_module):
            result = validator._load_component("test.module:test_object")

            assert result == mock_object
            mock_module.__getitem__.assert_called_once_with("test_object")

    def test_load_component_nested_object(self, validator):
        """Test _load_component with nested object path."""
        mock_module = Mock()
        mock_class = Mock()
        mock_method = Mock()

        # Setup nested access: module["ClassName"]["method_name"]
        mock_module.__getitem__ = Mock(return_value=mock_class)
        mock_class.__getitem__ = Mock(return_value=mock_method)

        with patch("owa.core.documentation.validator.griffe.load", return_value=mock_module):
            result = validator._load_component("test.module:ClassName.method_name")

            assert result == mock_method
            mock_module.__getitem__.assert_called_once_with("ClassName")
            mock_class.__getitem__.assert_called_once_with("method_name")

    def test_get_docstring_scenarios(self, validator):
        """Test _get_docstring with various docstring formats."""
        # With value attribute (griffe object)
        mock_component1 = Mock()
        mock_docstring = Mock()
        mock_docstring.value = "Test docstring content"
        mock_component1.docstring = mock_docstring
        assert validator._get_docstring(mock_component1) == "Test docstring content"

        # Direct string docstring
        mock_component2 = Mock()
        mock_component2.docstring = "Direct docstring content"
        assert validator._get_docstring(mock_component2) == "Direct docstring content"

        # No docstring
        mock_component3 = Mock()
        mock_component3.docstring = None
        assert validator._get_docstring(mock_component3) == ""

        # No docstring attribute
        mock_component4 = Mock(spec=[])
        assert validator._get_docstring(mock_component4) == ""


class TestDocumentationValidatorHelperMethods:
    """Test DocumentationValidator helper methods."""

    @pytest.fixture
    def validator(self):
        """Create a DocumentationValidator with mocked dependencies."""
        with patch("owa.core.documentation.validator.get_plugin_discovery"):
            return DocumentationValidator()

    def test_check_skip_directive(self, validator):
        """Test _check_skip_directive with various scenarios."""
        # Valid reason
        docstring = "Test docstring\n@skip-quality-check: legacy-code\nMore content"
        assert validator._check_skip_directive(docstring) == "legacy-code"

        # Invalid reason
        docstring = "Test docstring\n@skip-quality-check: invalid-reason\nMore content"
        assert validator._check_skip_directive(docstring) == ""

        # No directive
        docstring = "Test docstring without skip directive"
        assert validator._check_skip_directive(docstring) == ""

        # Test all valid reasons
        valid_reasons = ["legacy-code", "internal-api", "experimental", "deprecated", "third-party"]
        for reason in valid_reasons:
            docstring = f"Test docstring\n@skip-quality-check: {reason}\nMore content"
            assert validator._check_skip_directive(docstring) == reason

    def test_validate_docstring_quality(self, validator):
        """Test _validate_docstring_quality with various docstring scenarios."""
        # Comprehensive docstring - should have no issues
        comprehensive = """
        Comprehensive function description that is long enough to be descriptive.

        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2

        Returns:
            Description of return value

        Example:
            >>> func(1, 2)
            3
        """
        assert validator._validate_docstring_quality(comprehensive) == []

        # Empty docstring
        assert "Missing summary in docstring" in validator._validate_docstring_quality("")

        # Short summary
        assert "Summary too short (should be descriptive)" in validator._validate_docstring_quality("Short")

        # Missing examples
        no_examples = "Good description that is long enough to be descriptive."
        assert "Missing usage examples" in validator._validate_docstring_quality(no_examples)

        # Missing return docs
        no_returns = "Good description that mentions return value but has no Returns section."
        assert "Missing return value documentation" in validator._validate_docstring_quality(no_returns)

    def test_determine_quality_grade(self, validator):
        """Test _determine_quality_grade with various scenarios."""
        # Good quality: comprehensive with examples
        comprehensive = "Comprehensive description with examples.\n\nExample:\n    >>> func()\n    'result'"
        assert validator._determine_quality_grade(comprehensive, []) == "good"

        # Acceptable: has issues or missing examples
        short = "Short description"
        assert validator._determine_quality_grade(short, ["Parameter 'x' missing type hint"]) == "acceptable"

        long_no_examples = "Comprehensive description that is long enough but has no examples."
        assert validator._determine_quality_grade(long_no_examples, []) == "acceptable"
