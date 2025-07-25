"""
Tests for owa.core.plugin_discovery module.

This module tests the plugin discovery functionality including entry point
discovery, plugin registration, and error handling.
"""

from unittest.mock import Mock, patch

from owa.core.plugin_discovery import PluginDiscovery, discover_and_register_plugins, get_plugin_discovery
from owa.core.plugin_spec import PluginSpec


class TestPluginDiscovery:
    """Test PluginDiscovery class functionality."""

    def test_plugin_discovery_initialization(self):
        """Test PluginDiscovery initialization."""
        discovery = PluginDiscovery()

        assert discovery.discovered_plugins == {}
        assert discovery.failed_plugins == {}
        assert discovery.ENTRY_POINT_GROUP == "owa.env.plugins"

    def test_discover_plugins_no_entry_points(self):
        """Test discover_plugins when no entry points are found."""
        discovery = PluginDiscovery()

        # Mock entry_points to return empty
        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.return_value = []

            discovery.discover_plugins()

            assert discovery.discovered_plugins == {}
            assert discovery.failed_plugins == {}

    def test_discover_plugins_with_valid_entry_points(self):
        """Test discover_plugins with valid entry points."""
        discovery = PluginDiscovery()

        # Create mock entry point
        mock_entry_point = Mock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.load.return_value = PluginSpec(
            namespace="test", version="1.0.0", description="Test plugin", author="Test Author", components={}
        )

        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_entry_point]

            discovery.discover_plugins()

            assert "test_plugin" in discovery.discovered_plugins
            assert discovery.failed_plugins == {}

            plugin_spec = discovery.discovered_plugins["test_plugin"]
            assert plugin_spec.namespace == "test"
            assert plugin_spec.version == "1.0.0"

    def test_discover_plugins_with_invalid_entry_points(self):
        """Test discover_plugins with entry points that fail to load."""
        discovery = PluginDiscovery()

        # Create mock entry point that raises exception
        mock_entry_point = Mock()
        mock_entry_point.name = "failing_plugin"
        mock_entry_point.load.side_effect = ImportError("Module not found")

        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_entry_point]

            discovery.discover_plugins()

            assert discovery.discovered_plugins == {}
            assert "failing_plugin" in discovery.failed_plugins
            assert "Module not found" in discovery.failed_plugins["failing_plugin"]

    def test_discover_plugins_with_invalid_plugin_spec(self):
        """Test discover_plugins with entry points that return invalid plugin specs."""
        discovery = PluginDiscovery()

        # Create mock entry point that returns non-PluginSpec object
        mock_entry_point = Mock()
        mock_entry_point.name = "invalid_plugin"
        mock_entry_point.load.return_value = "not a plugin spec"

        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_entry_point]

            discovery.discover_plugins()

            assert discovery.discovered_plugins == {}
            assert "invalid_plugin" in discovery.failed_plugins
            assert "must point to a PluginSpec instance" in discovery.failed_plugins["invalid_plugin"]

    def test_register_plugin_components_callables(self, isolated_registries):
        """Test registering plugin components for callables."""
        discovery = PluginDiscovery()

        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"callables": {"add": "operator:add", "mul": "operator:mul"}},
        )

        with patch("owa.core.plugin_discovery.CALLABLES", isolated_registries["callables"]):
            discovery._register_plugin_components("test_plugin", plugin_spec)

            # Check that components were registered
            assert "test/add" in isolated_registries["callables"]._import_paths
            assert "test/mul" in isolated_registries["callables"]._import_paths
            assert isolated_registries["callables"]._import_paths["test/add"] == "operator:add"

    def test_register_plugin_components_listeners(self, isolated_registries):
        """Test registering plugin components for listeners."""
        discovery = PluginDiscovery()

        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"listeners": {"timer": "time:sleep"}},
        )

        with patch("owa.core.plugin_discovery.LISTENERS", isolated_registries["listeners"]):
            discovery._register_plugin_components("test_plugin", plugin_spec)

            # Check that components were registered
            assert "test/timer" in isolated_registries["listeners"]._import_paths
            assert isolated_registries["listeners"]._import_paths["test/timer"] == "time:sleep"

    def test_register_plugin_components_runnables(self, isolated_registries):
        """Test registering plugin components for runnables."""
        discovery = PluginDiscovery()

        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"runnables": {"worker": "threading:Thread"}},
        )

        with patch("owa.core.plugin_discovery.RUNNABLES", isolated_registries["runnables"]):
            discovery._register_plugin_components("test_plugin", plugin_spec)

            # Check that components were registered
            assert "test/worker" in isolated_registries["runnables"]._import_paths
            assert isolated_registries["runnables"]._import_paths["test/worker"] == "threading:Thread"

    def test_register_plugin_components_all_types(self, isolated_registries):
        """Test registering plugin components for all component types."""
        discovery = PluginDiscovery()

        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={
                "callables": {"add": "operator:add"},
                "listeners": {"timer": "time:sleep"},
                "runnables": {"worker": "threading:Thread"},
            },
        )

        with patch("owa.core.plugin_discovery.CALLABLES", isolated_registries["callables"]):
            with patch("owa.core.plugin_discovery.LISTENERS", isolated_registries["listeners"]):
                with patch("owa.core.plugin_discovery.RUNNABLES", isolated_registries["runnables"]):
                    discovery._register_plugin_components("test_plugin", plugin_spec)

                    # Check that all components were registered
                    assert "test/add" in isolated_registries["callables"]._import_paths
                    assert "test/timer" in isolated_registries["listeners"]._import_paths
                    assert "test/worker" in isolated_registries["runnables"]._import_paths

    def test_register_plugin_components_empty_components(self, isolated_registries):
        """Test registering plugin with empty components."""
        discovery = PluginDiscovery()

        plugin_spec = PluginSpec(
            namespace="test", version="1.0.0", description="Test plugin", author="Test Author", components={}
        )

        with patch("owa.core.plugin_discovery.CALLABLES", isolated_registries["callables"]):
            # Should not raise any errors
            discovery._register_plugin_components("test_plugin", plugin_spec)

            # No components should be registered
            assert len(isolated_registries["callables"]._import_paths) == 0

    def test_discover_and_register_integration(self, isolated_registries):
        """Test full discover and register integration."""
        discovery = PluginDiscovery()

        # Create mock entry point with valid plugin spec
        mock_entry_point = Mock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.load.return_value = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"callables": {"add": "operator:add"}},
        )

        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_entry_point]

            with patch("owa.core.plugin_discovery.CALLABLES", isolated_registries["callables"]):
                discovery.discover_plugins()
                discovery.register_all_components()

                # Check that plugin was discovered
                assert "test_plugin" in discovery.discovered_plugins

                # Check that components were registered
                assert "test/add" in isolated_registries["callables"]._import_paths


class TestModuleFunctions:
    """Test module-level functions."""

    def test_discover_and_register_plugins(self, isolated_registries):
        """Test discover_and_register_plugins function."""
        # Create mock entry point
        mock_entry_point = Mock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.load.return_value = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"callables": {"add": "operator:add"}},
        )

        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.return_value = [mock_entry_point]

            with patch("owa.core.plugin_discovery.CALLABLES", isolated_registries["callables"]):
                discover_and_register_plugins()

                # Check that components were registered
                assert "test/add" in isolated_registries["callables"]._import_paths

    def test_get_plugin_discovery_singleton(self):
        """Test get_plugin_discovery returns singleton instance."""
        discovery1 = get_plugin_discovery()
        discovery2 = get_plugin_discovery()

        assert discovery1 is discovery2
        assert isinstance(discovery1, PluginDiscovery)


class TestPluginDiscoveryErrorHandling:
    """Test error handling in PluginDiscovery."""

    def test_discover_plugins_entry_points_exception(self):
        """Test discover_plugins when entry_points() raises an exception."""
        discovery = PluginDiscovery()

        with patch("owa.core.plugin_discovery.entry_points") as mock_entry_points:
            mock_entry_points.side_effect = Exception("Entry points discovery failed")

            discovery.discover_plugins()

            # Should handle exception gracefully
            assert discovery.discovered_plugins == {}
            assert discovery.failed_plugins == {}

    def test_register_all_components_exception(self, isolated_registries):
        """Test register_all_components when component registration fails."""
        discovery = PluginDiscovery()

        # Add a plugin to discovered_plugins
        plugin_spec = PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            components={"callables": {"add": "operator:add"}},
        )
        discovery.discovered_plugins["test_plugin"] = plugin_spec

        # Mock _register_plugin_components to raise an exception
        with patch.object(discovery, "_register_plugin_components", side_effect=Exception("Registration failed")):
            discovery.register_all_components()

            # Should handle exception and add to failed_plugins
            assert "test_plugin" in discovery.failed_plugins
            assert "Registration failed" in discovery.failed_plugins["test_plugin"]

    def test_get_plugin_info_all_plugins(self):
        """Test get_plugin_info with no specific plugin name (returns all)."""
        discovery = PluginDiscovery()

        # Add some test data
        plugin_spec = PluginSpec(
            namespace="test", version="1.0.0", description="Test plugin", author="Test Author", components={}
        )
        discovery.discovered_plugins["test_plugin"] = plugin_spec
        discovery.failed_plugins["failed_plugin"] = "Some error"

        discovered, failed = discovery.get_plugin_info()

        assert "test_plugin" in discovered
        assert discovered["test_plugin"]["namespace"] == "test"
        assert "failed_plugin" in failed
        assert failed["failed_plugin"] == "Some error"

    def test_get_plugin_info_specific_plugin_string(self):
        """Test get_plugin_info with specific plugin name as string."""
        discovery = PluginDiscovery()

        # Add some test data
        plugin_spec = PluginSpec(
            namespace="test", version="1.0.0", description="Test plugin", author="Test Author", components={}
        )
        discovery.discovered_plugins["test_plugin"] = plugin_spec
        discovery.failed_plugins["failed_plugin"] = "Some error"

        # Test with discovered plugin
        discovered, failed = discovery.get_plugin_info("test_plugin")
        assert "test_plugin" in discovered
        assert discovered["test_plugin"]["namespace"] == "test"
        assert failed == {}

        # Test with failed plugin
        discovered, failed = discovery.get_plugin_info("failed_plugin")
        assert discovered == {}
        assert "failed_plugin" in failed
        assert failed["failed_plugin"] == "Some error"

    def test_get_plugin_info_specific_plugin_list(self):
        """Test get_plugin_info with specific plugin names as list."""
        discovery = PluginDiscovery()

        # Add some test data
        plugin_spec1 = PluginSpec(
            namespace="test1", version="1.0.0", description="Test plugin 1", author="Test Author", components={}
        )
        plugin_spec2 = PluginSpec(
            namespace="test2", version="2.0.0", description="Test plugin 2", author="Test Author", components={}
        )
        discovery.discovered_plugins["plugin1"] = plugin_spec1
        discovery.discovered_plugins["plugin2"] = plugin_spec2
        discovery.failed_plugins["failed_plugin"] = "Some error"

        # Test with list of plugin names
        discovered, failed = discovery.get_plugin_info(["plugin1", "failed_plugin"])

        assert "plugin1" in discovered
        assert discovered["plugin1"]["namespace"] == "test1"
        assert "plugin2" not in discovered  # Not requested
        assert "failed_plugin" in failed
        assert failed["failed_plugin"] == "Some error"

    def test_get_plugin_info_nonexistent_plugin(self):
        """Test get_plugin_info with non-existent plugin name."""
        discovery = PluginDiscovery()

        discovered, failed = discovery.get_plugin_info("nonexistent")

        assert discovered == {}
        assert failed == {}

    def test_importlib_metadata_fallback(self):
        """Test fallback to importlib_metadata for Python < 3.10."""
        # This test verifies the import fallback works
        # We can't easily test the actual version check without complex mocking
        # But we can verify the module structure supports both imports

        # Test that the module can be imported (this exercises the version check)
        import owa.core.plugin_discovery

        assert hasattr(owa.core.plugin_discovery, "entry_points")
        assert hasattr(owa.core.plugin_discovery, "PluginDiscovery")
