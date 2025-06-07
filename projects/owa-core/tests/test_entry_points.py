"""
Tests for the entry points-based plugin discovery system.

This module tests the new OEP-0003 implementation including:
- PluginSpec validation
- Entry points discovery
- Unified component naming
- Enhanced component access API
"""

import pytest

from owa.core.plugin_spec import PluginSpec
from owa.core.registry import (
    CALLABLES,
    LISTENERS,
    RUNNABLES,
    entry_point_registry,
    get_component,
    list_components,
)


def test_plugin_spec_validation():
    """Test PluginSpec validation."""
    # Valid plugin spec
    spec = PluginSpec(
        namespace="test",
        version="1.0.0",
        description="Test plugin",
        author="Test Author",
        components={"callables": {"func": "test.module:test_func"}},
    )
    assert spec.namespace == "test"
    assert spec.version == "1.0.0"

    # Invalid namespace
    with pytest.raises(ValueError, match="Plugin namespace cannot be empty"):
        PluginSpec(namespace="", version="1.0.0", description="Test", author="Test", components={})

    # Invalid component type
    with pytest.raises(ValueError, match="Invalid component type"):
        PluginSpec(
            namespace="test", version="1.0.0", description="Test", author="Test", components={"invalid_type": {}}
        )

    # Invalid module path
    with pytest.raises(ValueError, match="Invalid module path"):
        PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test",
            author="Test",
            components={"callables": {"func": "invalid_path_without_colon"}},
        )


def test_unified_naming_std_plugin():
    """Test that std plugin components are available with unified naming."""
    # Test new unified naming
    assert "std/time_ns" in CALLABLES
    assert "std/tick" in LISTENERS

    # Test backwards compatibility
    assert "clock.time_ns" in CALLABLES
    assert "clock/tick" in LISTENERS

    # Test that both point to working components
    time_func_new = CALLABLES["std/time_ns"]
    time_func_old = CALLABLES["clock.time_ns"]

    # Both should return valid timestamps
    assert isinstance(time_func_new(), int)
    assert isinstance(time_func_old(), int)


def test_get_component_api():
    """Test the enhanced get_component API."""
    # Get specific component
    time_func = get_component("callables", namespace="std", name="time_ns")
    assert time_func is not None
    assert callable(time_func)

    # Get all components in namespace
    std_callables = get_component("callables", namespace="std")
    assert isinstance(std_callables, dict)
    assert "time_ns" in std_callables

    # Get all callables
    all_callables = get_component("callables")
    assert isinstance(all_callables, dict)
    assert "std/time_ns" in all_callables

    # Test non-existent component
    result = get_component("callables", namespace="nonexistent", name="func")
    assert result is None

    # Test invalid component type
    with pytest.raises(ValueError, match="Unknown component type"):
        get_component("invalid_type")


def test_list_components_api():
    """Test the list_components API."""
    # List all components
    all_components = list_components()
    assert isinstance(all_components, dict)
    assert "callables" in all_components
    assert "listeners" in all_components
    assert "runnables" in all_components

    # List specific component type
    callables = list_components("callables")
    assert isinstance(callables, dict)
    assert "callables" in callables
    assert "std/time_ns" in callables["callables"]

    # List components in namespace
    std_components = list_components(namespace="std")
    assert isinstance(std_components, dict)
    for component_type, components in std_components.items():
        for component_name in components:
            assert component_name.startswith("std/")


def test_entry_point_registry_initialization():
    """Test that entry point registry initializes correctly."""
    assert entry_point_registry is not None
    assert hasattr(entry_point_registry, "discovered_plugins")
    assert isinstance(entry_point_registry.discovered_plugins, dict)

    # Should have discovered the std plugin
    assert "std" in entry_point_registry.discovered_plugins
    std_plugin = entry_point_registry.discovered_plugins["std"]
    assert std_plugin.namespace == "std"
    assert std_plugin.version == "0.3.9"


def test_backwards_compatibility():
    """Test that old activation method still works."""
    from owa.core.registry import activate_module

    # This should work without errors (already activated by entry points)
    module = activate_module("owa.env.std")
    assert module is not None
    assert hasattr(module, "activate")

    # Components should still be available
    assert "clock.time_ns" in CALLABLES
    assert "clock/tick" in LISTENERS


def test_component_functionality():
    """Test that components actually work correctly."""
    # Test std/time_ns callable
    time_func = CALLABLES["std/time_ns"]
    timestamp = time_func()
    assert isinstance(timestamp, int)
    assert timestamp > 0

    # Test std/tick listener
    tick_listener_cls = LISTENERS["std/tick"]
    assert tick_listener_cls is not None

    # Should be able to instantiate and configure
    listener = tick_listener_cls()

    def dummy_callback():
        pass

    configured_listener = listener.configure(callback=dummy_callback, interval=0.1)
    assert configured_listener is not None
