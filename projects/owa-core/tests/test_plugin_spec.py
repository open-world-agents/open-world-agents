"""
Tests for the plugin specification system (owa.core.plugin_spec).
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from owa.core.plugin_spec import PluginSpec


class TestPluginSpec:
    """Test cases for PluginSpec class."""

    def test_plugin_spec_creation(self):
        """Test PluginSpec creation and validation."""
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={
                "callables": {
                    "add": "test.module:add_function",
                    "multiply": "test.module:multiply_function",
                },
                "listeners": {
                    "timer": "test.module:TimerListener",
                },
            },
        )

        assert plugin_spec.namespace == "test"
        assert plugin_spec.version == "1.0.0"
        assert "callables" in plugin_spec.components
        assert "listeners" in plugin_spec.components

        # Test component name generation
        callable_names = plugin_spec.get_component_names("callables")
        assert "test/add" in callable_names
        assert "test/multiply" in callable_names

        # Test import path retrieval
        add_path = plugin_spec.get_import_path("callables", "add")
        assert add_path == "test.module:add_function"

    def test_plugin_spec_validation(self):
        """Test PluginSpec validation for unsupported component types."""
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={
                "callables": {"test": "test.module:test"},
                "invalid_type": {"test": "test.module:test"},
            },
        )

        try:
            plugin_spec.validate_components()
            assert False, "Should have raised ValueError for invalid component type"
        except ValueError as e:
            assert "invalid_type" in str(e)

    def test_minimal_plugin_spec(self):
        """Test creating a minimal plugin spec."""
        plugin_spec = PluginSpec(
            namespace="minimal",
            version="0.1.0",
            description="Minimal plugin",
            components={},
        )

        assert plugin_spec.namespace == "minimal"
        assert plugin_spec.version == "0.1.0"
        assert plugin_spec.components == {}

        # Should validate successfully even with no components
        plugin_spec.validate_components()

    def test_get_component_names_empty(self):
        """Test get_component_names with empty component type."""
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={"callables": {}},
        )

        callable_names = plugin_spec.get_component_names("callables")
        assert callable_names == []

    def test_get_import_path_nonexistent(self):
        """Test get_import_path with non-existent component."""
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={"callables": {"existing": "test.module:function"}},
        )

        # Should return None for non-existent component
        path = plugin_spec.get_import_path("callables", "nonexistent")
        assert path is None

        # Should return None for non-existent component type
        path = plugin_spec.get_import_path("nonexistent", "existing")
        assert path is None


class TestPluginSpecValidation:
    """Test PluginSpec validation functionality."""

    def test_namespace_validation(self):
        """Test namespace validation with valid and invalid names."""
        # Valid namespaces
        valid_namespaces = ["test", "my_plugin", "plugin-name", "test123", "a1_b2-c3"]
        for namespace in valid_namespaces:
            plugin_spec = PluginSpec(namespace=namespace, version="1.0.0", description="Test plugin", components={})
            assert plugin_spec.namespace == namespace

        # Invalid namespaces
        invalid_namespaces = ["test.plugin", "test plugin", "test@plugin", "test/plugin", ""]
        for namespace in invalid_namespaces:
            with pytest.raises(ValueError, match="Namespace .* is invalid"):
                PluginSpec(namespace=namespace, version="1.0.0", description="Test plugin", components={})

    def test_component_name_validation(self):
        """Test component name validation with valid and invalid names."""
        # Valid names
        valid_names = ["test", "my_component", "component.name", "test123", "a1_b2.c3"]
        components = {"callables": {name: "test.module:function" for name in valid_names}}
        plugin_spec = PluginSpec(namespace="test", version="1.0.0", description="Test plugin", components=components)
        assert len(plugin_spec.components["callables"]) == len(valid_names)

        # Invalid names
        invalid_names = ["test-component", "test component", "test@component", "test/component"]
        for name in invalid_names:
            components = {"callables": {name: "test.module:function"}}
            with pytest.raises(ValueError, match="Component name .* is invalid"):
                PluginSpec(namespace="test", version="1.0.0", description="Test plugin", components=components)

    def test_get_component_names_nonexistent_type(self):
        """Test get_component_names with non-existent component type."""
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={"callables": {"test": "test.module:function"}},
        )

        # Should return empty list for non-existent type
        names = plugin_spec.get_component_names("nonexistent")
        assert names == []


class TestPluginSpecYAML:
    """Test PluginSpec YAML functionality."""

    def test_from_yaml_success(self):
        """Test loading PluginSpec from YAML file."""
        yaml_content = {
            "namespace": "test",
            "version": "1.0.0",
            "description": "Test plugin",
            "author": "Test Author",
            "components": {"callables": {"test_func": "test.module:test_function"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name

        try:
            plugin_spec = PluginSpec.from_yaml(yaml_path)

            assert plugin_spec.namespace == "test"
            assert plugin_spec.version == "1.0.0"
            assert plugin_spec.description == "Test plugin"
            assert plugin_spec.author == "Test Author"
            assert "callables" in plugin_spec.components
            assert plugin_spec.components["callables"]["test_func"] == "test.module:test_function"
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_error_cases(self):
        """Test from_yaml with various error scenarios."""
        # File not found
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            PluginSpec.from_yaml("nonexistent.yaml")

        # Invalid YAML content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            yaml_path = f.name
        try:
            with pytest.raises(yaml.YAMLError, match="Invalid YAML"):
                PluginSpec.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

        # Non-dictionary YAML content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "dictionary"], f)
            yaml_path = f.name
        try:
            with pytest.raises(ValueError, match="YAML file must contain a dictionary"):
                PluginSpec.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_to_yaml_success(self):
        """Test saving PluginSpec to YAML file."""
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"callables": {"test_func": "test.module:test_function"}},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "test_plugin.yaml"

            plugin_spec.to_yaml(yaml_path)

            assert yaml_path.exists()

            # Verify content by loading it back
            with open(yaml_path, "r") as f:
                loaded_data = yaml.safe_load(f)

            assert loaded_data["namespace"] == "test"
            assert loaded_data["version"] == "1.0.0"
            assert loaded_data["description"] == "Test plugin"
            assert loaded_data["author"] == "Test Author"
            assert "callables" in loaded_data["components"]

    def test_to_yaml_creates_directories(self):
        """Test that to_yaml creates parent directories."""
        plugin_spec = PluginSpec(namespace="test", version="1.0.0", description="Test plugin", components={})

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "subdir" / "test_plugin.yaml"

            plugin_spec.to_yaml(yaml_path)

            assert yaml_path.exists()
            assert yaml_path.parent.exists()


class TestPluginSpecEntryPoint:
    """Test PluginSpec entry point functionality."""

    def test_from_entry_point_success(self):
        """Test loading PluginSpec from entry point."""
        # Create a mock plugin spec
        mock_plugin_spec = PluginSpec(namespace="test", version="1.0.0", description="Test plugin", components={})

        # Mock the module and object
        mock_module = Mock()
        mock_module.plugin_spec = mock_plugin_spec

        with patch("importlib.import_module", return_value=mock_module):
            result = PluginSpec.from_entry_point("test.module:plugin_spec")

            assert result == mock_plugin_spec

    def test_from_entry_point_error_cases(self):
        """Test from_entry_point with various error scenarios."""
        # Invalid format
        with pytest.raises(ValueError, match="Invalid entry point format"):
            PluginSpec.from_entry_point("invalid_format_without_colon")

        # Import error
        with patch("importlib.import_module", side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Cannot import module"):
                PluginSpec.from_entry_point("nonexistent.module:plugin_spec")

        # Missing attribute
        mock_module = Mock()
        del mock_module.plugin_spec
        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(AttributeError, match="Object .* not found in module"):
                PluginSpec.from_entry_point("test.module:plugin_spec")

        # Wrong object type
        mock_module = Mock()
        mock_module.plugin_spec = "not a PluginSpec instance"
        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(TypeError, match="Object .* must be a PluginSpec instance"):
                PluginSpec.from_entry_point("test.module:plugin_spec")
