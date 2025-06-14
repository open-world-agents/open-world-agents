"""
Tests for the plugin specification system (owa.core.plugin_spec).
"""


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
