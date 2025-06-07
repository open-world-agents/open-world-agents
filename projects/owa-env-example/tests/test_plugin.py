"""Tests for the owa-env-example plugin."""

import tempfile
import threading
import time
from pathlib import Path
from threading import Event

import pytest


def test_plugin_registration(example_registries):
    """Test that the plugin components are registered properly."""
    callables = example_registries["callables"]
    listeners = example_registries["listeners"]
    runnables = example_registries["runnables"]

    # Check expected components are registered
    expected_callables = ["example/callable", "example/print", "example/add"]
    expected_listeners = ["example/listener", "example/timer"]
    expected_runnables = ["example/runnable", "example/counter"]

    for name in expected_callables:
        assert name in callables

    for name in expected_listeners:
        assert name in listeners

    for name in expected_runnables:
        assert name in runnables


def test_callable_components(example_registries):
    """Test callable components."""
    callables = example_registries["callables"]

    # Test example/callable
    callable_cls = callables["example/callable"]
    result = callable_cls()()
    assert isinstance(result, str)
    assert "Hello from ExampleCallable" in result

    # Test example/print
    print_func = callables["example/print"]
    assert print_func("Test") == "Test"
    assert print_func() == "Hello, World!"

    # Test example/add
    add_func = callables["example/add"]
    assert add_func(5, 3) == 8
    assert add_func(10, -5) == 5

    # Test error handling
    with pytest.raises(TypeError):
        add_func("not", "numbers")


def test_listener_components(example_registries):
    """Test listener components."""
    listeners = example_registries["listeners"]

    # Test example/listener
    listener_cls = listeners["example/listener"]
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
    timer_cls = listeners["example/timer"]
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


def test_runnable_components(example_registries):
    """Test runnable components."""
    runnables = example_registries["runnables"]

    # Test example/runnable
    runnable_cls = runnables["example/runnable"]
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
    counter_cls = runnables["example/counter"]
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
