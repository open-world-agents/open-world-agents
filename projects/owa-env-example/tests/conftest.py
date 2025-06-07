"""Pytest configuration for owa-env-example tests."""

import pytest

from owa.core.registry import LazyImportRegistry, RegistryType


@pytest.fixture(scope="session")
def example_registries():
    """Create test registries with example components."""
    # Create test registries
    callables = LazyImportRegistry(RegistryType.CALLABLES)
    listeners = LazyImportRegistry(RegistryType.LISTENERS)
    runnables = LazyImportRegistry(RegistryType.RUNNABLES)

    # Register example components manually
    callables.register("example/callable", obj_or_import_path="owa.env.example.example_callable:ExampleCallable")
    callables.register("example/print", obj_or_import_path="owa.env.example.example_callable:example_print")
    callables.register("example/add", obj_or_import_path="owa.env.example.example_callable:example_add")

    listeners.register("example/listener", obj_or_import_path="owa.env.example.example_listener:ExampleListener")
    listeners.register("example/timer", obj_or_import_path="owa.env.example.example_listener:ExampleTimerListener")

    runnables.register("example/runnable", obj_or_import_path="owa.env.example.example_runnable:ExampleRunnable")
    runnables.register("example/counter", obj_or_import_path="owa.env.example.example_runnable:ExampleCounterRunnable")

    return {
        "callables": callables,
        "listeners": listeners,
        "runnables": runnables,
    }
