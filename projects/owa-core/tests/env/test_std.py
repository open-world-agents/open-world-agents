from owa.core.registry import LazyImportRegistry, RegistryType


def test_lazy_loading():
    """Test lazy loading functionality."""
    # Create a test registry
    test_registry = LazyImportRegistry(RegistryType.CALLABLES)

    # Register time_ns function with lazy loading
    test_registry.register(name="std/time_ns", obj_or_import_path="time:time_ns")

    # Test that component is registered but not loaded
    assert "std/time_ns" in test_registry
    assert len(test_registry._registry) == 0  # Not loaded yet
    assert len(test_registry._import_paths) == 1  # Import path stored

    # Access the component to trigger lazy loading
    time_func = test_registry["std/time_ns"]

    # Test that component is now loaded
    assert len(test_registry._registry) == 1  # Now loaded
    assert callable(time_func)

    # Test that the function works
    result = time_func()
    assert isinstance(result, int)
    assert result > 0


def test_pre_loaded_instance():
    """Test pre-loaded instance functionality."""
    # Create a test registry
    test_registry = LazyImportRegistry(RegistryType.CALLABLES)

    # Register a pre-loaded instance
    def test_func(x, y):
        return x + y

    test_registry.register(name="test/add", obj_or_import_path=test_func, is_instance=True)

    # Test that component is immediately available
    assert "test/add" in test_registry
    assert len(test_registry._registry) == 1  # Immediately loaded
    assert len(test_registry._import_paths) == 0  # No import path needed

    # Test that the function works
    add_func = test_registry["test/add"]
    result = add_func(5, 3)
    assert result == 8


def test_eager_loading():
    """Test eager loading functionality."""
    # Create a test registry
    test_registry = LazyImportRegistry(RegistryType.CALLABLES)

    # Register with eager loading
    test_registry.register(name="std/time_ns_eager", obj_or_import_path="time:time_ns", eager_load=True)

    # Test that component is immediately loaded
    assert "std/time_ns_eager" in test_registry
    assert len(test_registry._registry) == 1  # Immediately loaded
    assert len(test_registry._import_paths) == 1  # Import path still stored

    # Test that the function works
    time_func = test_registry["std/time_ns_eager"]
    result = time_func()
    assert isinstance(result, int)
    assert result > 0
