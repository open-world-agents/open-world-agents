"""
Tests for the registry system (owa.core.registry).
"""

import pytest

from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES, LazyImportRegistry, Registry, RegistryType


class TestRegistry:
    """Test cases for basic Registry class."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = Registry(RegistryType.CALLABLES)
        assert len(registry._registry) == 0
        assert registry.registry_type == RegistryType.CALLABLES

    def test_register_and_access(self):
        """Test registering and accessing objects."""
        registry = Registry()

        def test_func():
            return "test"

        registry.register("test", test_func)
        assert "test" in registry
        assert registry["test"] is test_func
        assert registry.get("test") is test_func

    def test_get_with_default(self):
        """Test get() method with default values."""
        registry = Registry()

        def default_func():
            return "default"

        # Non-existent with default
        result = registry.get("nonexistent", default_func)
        assert result is default_func

        # Non-existent without default
        result = registry.get("nonexistent")
        assert result is None

    def test_extend(self):
        """Test extending registry with another registry."""
        registry1 = Registry()
        registry2 = Registry()

        def func1():
            return "func1"

        def func2():
            return "func2"

        registry1.register("func1", func1)
        registry2.register("func2", func2)

        registry1.extend(registry2)

        assert "func1" in registry1
        assert "func2" in registry1
        assert registry1["func1"] is func1
        assert registry1["func2"] is func2


class TestLazyImportRegistry:
    """Test cases for LazyImportRegistry class."""

    def test_lazy_import_registry_inheritance(self):
        """Test that LazyImportRegistry properly inherits from Registry."""
        registry = LazyImportRegistry(RegistryType.CALLABLES)

        # Test that it has Registry methods
        assert hasattr(registry, "register")
        assert hasattr(registry, "__getitem__")
        assert hasattr(registry, "__contains__")
        assert hasattr(registry, "get")

        # Test that it has LazyImportRegistry-specific attributes
        assert hasattr(registry, "_import_paths")
        assert hasattr(registry, "_load_component")

        # Test registry type
        assert registry.registry_type == RegistryType.CALLABLES

    def test_register_instance(self):
        """Test registering pre-loaded instances."""
        registry = LazyImportRegistry(RegistryType.CALLABLES)

        def test_func():
            return "test"

        registry.register("test", obj_or_import_path=test_func, is_instance=True)
        assert "test" in registry
        assert registry["test"] is test_func

    def test_register_import_path(self):
        """Test registering import paths for lazy loading."""
        registry = LazyImportRegistry(RegistryType.CALLABLES)

        # Register with import path
        registry.register("operator_add", obj_or_import_path="operator:add")
        assert "operator_add" in registry._import_paths
        assert "operator_add" not in registry._registry  # Not loaded yet

        # Access should trigger loading
        add_func = registry["operator_add"]
        import operator

        assert add_func is operator.add
        assert "operator_add" in registry._registry  # Now loaded

    def test_eager_loading(self):
        """Test eager loading at registration time."""
        registry = LazyImportRegistry(RegistryType.CALLABLES)

        # Register with eager loading
        registry.register("operator_sub", obj_or_import_path="operator:sub", eager_load=True)

        # Should be loaded immediately
        assert "operator_sub" in registry._registry
        import operator

        assert registry["operator_sub"] is operator.sub

    def test_load_component_error_handling(self):
        """Test error handling during component loading."""
        registry = LazyImportRegistry(RegistryType.CALLABLES)

        # Register invalid import path
        registry.register("invalid", obj_or_import_path="nonexistent.module:function")

        # Should raise ImportError when accessed
        with pytest.raises(ImportError, match="Failed to load component 'invalid'"):
            registry["invalid"]

    def test_namespace_name_pattern(self):
        """Test that the namespace/name pattern works correctly."""
        registry = LazyImportRegistry(RegistryType.CALLABLES)

        # Register components with namespace/name pattern
        def test_func():
            return "test"

        registry.register("example/test", obj_or_import_path=test_func, is_instance=True)
        registry.register("other/test", obj_or_import_path=test_func, is_instance=True)
        registry.register("example/other", obj_or_import_path=test_func, is_instance=True)

        # Test that components are properly separated by namespace
        assert "example/test" in registry
        assert "other/test" in registry
        assert "example/other" in registry

        # Test that they don't conflict
        assert registry["example/test"] is test_func
        assert registry["other/test"] is test_func
        assert registry["example/other"] is test_func

        # Test that partial names don't match
        assert "example" not in registry
        assert "test" not in registry


class TestGlobalRegistries:
    """Test cases for global registry instances."""

    def test_global_registries_exist(self):
        """Test that global registries are properly initialized."""
        assert isinstance(CALLABLES, LazyImportRegistry)
        assert isinstance(LISTENERS, LazyImportRegistry)
        assert isinstance(RUNNABLES, LazyImportRegistry)

        assert CALLABLES.registry_type == RegistryType.CALLABLES
        assert LISTENERS.registry_type == RegistryType.LISTENERS
        assert RUNNABLES.registry_type == RegistryType.RUNNABLES
