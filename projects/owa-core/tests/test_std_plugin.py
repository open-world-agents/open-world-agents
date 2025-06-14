"""
Tests for the owa.env.std plugin.

This module tests the built-in std plugin components including:
- std/time_ns callable
- std/tick listener
- Plugin registration and discovery

These tests focus only on owa-core's built-in std plugin functionality.
"""

import time

from owa.core.component_access import get_component, list_components
from owa.core.registry import CALLABLES, LISTENERS


def test_std_plugin_registration():
    """Test that the std plugin components are properly registered."""
    # Test that std plugin components are registered in global registries
    assert "std/time_ns" in CALLABLES
    assert "std/tick" in LISTENERS


def test_std_time_ns_callable():
    """Test the std/time_ns callable component."""
    # Test that std/time_ns is available
    assert "std/time_ns" in CALLABLES

    # Get the time function
    time_func = CALLABLES["std/time_ns"]
    assert callable(time_func)

    # Test that it returns a valid timestamp
    timestamp1 = time_func()
    assert isinstance(timestamp1, int)
    assert timestamp1 > 0

    # Test that time progresses
    time.sleep(0.001)  # Sleep 1ms
    timestamp2 = time_func()
    assert timestamp2 > timestamp1

    # Test that timestamps are in nanoseconds (should be very large numbers)
    # Should be > 1 second in nanoseconds since epoch (roughly year 2001+)
    assert timestamp1 > 1_000_000_000_000_000_000


def test_std_tick_listener():
    """Test the std/tick listener component."""
    # Test that std/tick listener is available
    assert "std/tick" in LISTENERS

    # Get the listener class
    listener_cls = LISTENERS["std/tick"]
    listener = listener_cls()

    # Test configuration with a dummy callback
    def dummy_callback():
        pass

    configured = listener.configure(callback=dummy_callback, interval=0.05)  # 50ms interval
    assert configured.interval == 0.05 * 1_000_000_000  # Should be converted to nanoseconds

    # Test that the listener can be instantiated and configured
    # (We won't run it in the test to avoid timing issues, but we verify the setup works)
    assert hasattr(configured, "loop")
    assert hasattr(configured, "interval")


def test_std_plugin_via_enhanced_api():
    """Test accessing std plugin components via the enhanced API."""
    # Test get_component with std namespace and specific name
    time_func = get_component("callables", namespace="std", name="time_ns")
    assert callable(time_func)

    # Test that it works
    result = time_func()
    assert isinstance(result, int)
    assert result > 0

    # Test get_component with namespace only
    std_callables = get_component("callables", namespace="std")
    assert "time_ns" in std_callables
    assert callable(std_callables["time_ns"])

    # Test list_components for std namespace
    std_components = list_components(namespace="std")
    assert "callables" in std_components
    assert "listeners" in std_components
    assert "std/time_ns" in std_components["callables"]
    assert "std/tick" in std_components["listeners"]


def test_std_plugin_spec():
    """Test that the std plugin spec is properly defined."""
    # Import the plugin spec
    from owa.env.plugins.std import plugin_spec

    # Test basic properties
    assert plugin_spec.namespace == "std"
    assert plugin_spec.version == "0.1.0"
    assert plugin_spec.description == "Standard system components for OWA"
    assert plugin_spec.author == "OWA Development Team"

    # Test components structure
    assert "callables" in plugin_spec.components
    assert "listeners" in plugin_spec.components

    # Test specific components
    assert "time_ns" in plugin_spec.components["callables"]
    assert "tick" in plugin_spec.components["listeners"]

    # Test import paths
    assert plugin_spec.components["callables"]["time_ns"] == "owa.env.std.clock:time_ns"
    assert plugin_spec.components["listeners"]["tick"] == "owa.env.std.clock:ClockTickListener"
