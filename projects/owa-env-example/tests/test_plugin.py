"""Tests for the owa-env-example plugin."""

import tempfile
import threading
import time
from pathlib import Path
from threading import Event

import pytest

from owa.core.registry import (
    CALLABLES,
    LISTENERS,
    RUNNABLES,
    entry_point_registry,
    get_component,
    list_components,
)


@pytest.fixture(scope="session", autouse=True)
def register_example_components():
    """Manually register example components for testing since plugin isn't installed."""
    from owa.env.example.example_callable import example_add, example_callable, example_print
    from owa.env.example.example_listener import ExampleListener, ExampleTimerListener
    from owa.env.example.example_runnable import ExampleCounterRunnable, ExampleRunnable

    # Register components manually for testing
    CALLABLES.register("example/callable")(example_callable)
    CALLABLES.register("example/print")(example_print)
    CALLABLES.register("example/add")(example_add)

    LISTENERS.register("example/listener")(ExampleListener)
    LISTENERS.register("example/timer")(ExampleTimerListener)

    RUNNABLES.register("example/runnable")(ExampleRunnable)
    RUNNABLES.register("example/counter")(ExampleCounterRunnable)


def test_plugin_components_registered():
    """Test that the plugin components are registered."""
    # Check expected components are registered
    expected_callables = ["example/callable", "example/print", "example/add"]
    expected_listeners = ["example/listener", "example/timer"]
    expected_runnables = ["example/runnable", "example/counter"]

    for name in expected_callables:
        assert name in CALLABLES._registry

    for name in expected_listeners:
        assert name in LISTENERS._registry

    for name in expected_runnables:
        assert name in RUNNABLES._registry


def test_callable_components():
    """Test callable components."""
    # Test example/callable
    callable_func = CALLABLES["example/callable"]
    result = callable_func()
    assert isinstance(result, str)
    assert "Hello from example_callable" in result

    # Test example/print
    print_func = CALLABLES["example/print"]
    assert print_func("Test") == "Test"
    assert print_func() == "Hello, World!"

    # Test example/add
    add_func = CALLABLES["example/add"]
    assert add_func(5, 3) == 8
    assert add_func(10, -5) == 5

    # Test error handling
    with pytest.raises(TypeError):
        add_func("not", "numbers")


def test_listener_components():
    """Test listener components."""
    # Test example/listener
    listener_cls = LISTENERS["example/listener"]
    listener = listener_cls()

    events = []

    def callback(event_data):
        events.append(event_data)

    configured = listener.configure(callback=callback, interval=0.05, message="Test")
    assert configured.interval == 0.05
    assert configured.message == "Test"

    # Run briefly
    stop_event = Event()
    thread = threading.Thread(target=lambda: configured.loop(stop_event=stop_event, callback=callback))
    thread.start()
    time.sleep(0.15)
    stop_event.set()
    thread.join(timeout=1.0)

    assert len(events) >= 2
    assert all("Test" in event for event in events)

    # Test example/timer
    timer_cls = LISTENERS["example/timer"]
    timer = timer_cls()

    triggered = []

    def timer_callback():
        triggered.append(True)

    configured_timer = timer.configure(callback=timer_callback, delay=0.1)
    assert configured_timer.delay == 0.1

    stop_event = Event()
    thread = threading.Thread(target=lambda: configured_timer.loop(stop_event=stop_event, callback=timer_callback))
    thread.start()
    time.sleep(0.15)
    thread.join(timeout=1.0)

    assert len(triggered) == 1


def test_runnable_components():
    """Test runnable components."""
    # Test example/runnable
    runnable_cls = RUNNABLES["example/runnable"]
    runnable = runnable_cls()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "test.txt"
        configured = runnable.configure(interval=0.05, output_file=str(output_file))
        assert configured.interval == 0.05
        assert configured.output_file == output_file

        with configured.session:
            time.sleep(0.15)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Example Runnable Output" in content
        assert "Task #" in content

    # Test example/counter
    counter_cls = RUNNABLES["example/counter"]
    counter = counter_cls()

    configured_counter = counter.configure(max_count=3, interval=0.02)
    assert configured_counter.max_count == 3
    assert configured_counter.interval == 0.02

    start_time = time.time()
    with configured_counter.session:
        time.sleep(0.1)

    elapsed = time.time() - start_time
    expected_time = 3 * 0.02
    assert elapsed >= expected_time
    assert elapsed <= expected_time + 0.1


def test_entry_points_discovery():
    """Test that the entry point registry works and std plugin is discovered."""
    # Check that entry point registry is initialized
    assert entry_point_registry is not None

    # The std plugin should be discovered since owa-core is installed
    assert "std" in entry_point_registry.discovered_plugins

    std_plugin = entry_point_registry.discovered_plugins["std"]
    assert std_plugin.namespace == "std"
    assert std_plugin.version == "0.3.9"
    assert std_plugin.description == "Standard environment plugin with time utilities"

    # Note: example plugin won't be discovered via entry points in tests
    # because it's not installed as a package, but activated manually


def test_enhanced_component_access():
    """Test the enhanced component access API with example plugin."""
    # Test get_component with example namespace
    example_callables = get_component("callables", namespace="example")
    assert isinstance(example_callables, dict)
    assert "callable" in example_callables
    assert "print" in example_callables
    assert "add" in example_callables

    # Test get specific component
    add_func = get_component("callables", namespace="example", name="add")
    assert add_func is not None
    assert add_func(2, 3) == 5

    # Test list_components with example namespace
    example_components = list_components(namespace="example")
    assert isinstance(example_components, dict)

    # Should have all component types
    assert "callables" in example_components
    assert "listeners" in example_components
    assert "runnables" in example_components

    # All component names should start with "example/"
    for component_type, components in example_components.items():
        for component_name in components:
            assert component_name.startswith("example/")
