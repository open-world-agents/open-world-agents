# ================ Definition of the Registry class ================================
# references:
# - https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
# - https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

import importlib
from enum import StrEnum
from typing import Dict, Generic, Optional, TypeVar

from .callable import Callable as CallableCls
from .listener import Listener as ListenerCls
from .runnable import Runnable


class RegistryType(StrEnum):
    CALLABLES = "callables"
    LISTENERS = "listeners"
    RUNNABLES = "runnables"
    UNKNOWN = "unknown"


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, registry_type: RegistryType = RegistryType.UNKNOWN):
        self._registry: Dict[str, T] = {}
        self.registry_type = registry_type

    def register(self, name: str, obj: T) -> None:
        """Register an object directly."""
        self._registry[name] = obj

    def extend(self, other: "Registry[T]") -> None:
        self._registry.update(other._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __getitem__(self, name: str) -> T:
        return self._registry[name]

    def get(self, name: str, default: Optional[T] = None) -> Optional[T]:
        return self._registry.get(name, default)

    def __repr__(self) -> str:
        return repr(self._registry)


class LazyImportRegistry(Registry[T]):
    """
    Registry with lazy loading support.
    Inherits all Registry functionality and adds lazy import capabilities.
    """

    def __init__(self, registry_type: RegistryType = RegistryType.UNKNOWN):
        super().__init__(registry_type)
        self._import_paths: Dict[str, str] = {}  # name -> import_path

    def register(
        self,
        name: str,
        obj_or_import_path: T | str,
        is_instance: bool = False,
        eager_load: bool = False,
    ) -> None:
        """
        Register a component with lazy loading support.

        Args:
            name: Component name
            obj_or_import_path: Object to register (when is_instance=True) or import path for lazy loading
            is_instance: True if obj_or_import_path is a pre-loaded instance, False for lazy loading
            eager_load: Load immediately at registration time
        """
        if is_instance:
            # Register pre-loaded instance using base registry method
            super().register(name, obj_or_import_path)
        else:
            # Register for lazy loading
            self._import_paths[name] = obj_or_import_path
            if eager_load:
                # Load immediately and store using base registry method
                component = self._load_component(name)
                super().register(name, component)

    def _load_component(self, name: str) -> T:
        """Load a component from its import path."""
        if name not in self._import_paths:
            raise KeyError(f"Component '{name}' not found in registry")

        import_path = self._import_paths[name]
        try:
            # Parse import path: "module.path:object_name"
            if ":" in import_path:
                module_path, object_name = import_path.split(":", 1)
            elif "." in import_path:
                # Fallback: assume last part is the object name
                module_path, object_name = import_path.rsplit(".", 1)
            else:
                raise ValueError(f"Invalid import path format: {import_path}. Must contain ':' or '.'")

            module = importlib.import_module(module_path)
            component = getattr(module, object_name)
            return component

        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Failed to load component '{name}' from '{import_path}': {e}") from e

    def __getitem__(self, name: str) -> T:
        """Get a component, loading it lazily if necessary."""
        # Check if already loaded in base registry
        if name in self._registry:
            return self._registry[name]

        # Load lazily and cache using base registry method
        if name in self._import_paths:
            component = self._load_component(name)
            super().register(name, component)
            return component

        raise KeyError(f"'{name}' not found in registry")

    def get(self, name: str, default: Optional[T] = None) -> Optional[T]:
        """Get a component with lazy loading support, returning default if not found."""
        try:
            return self[name]  # Use __getitem__ to trigger lazy loading
        except KeyError:
            return default

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._registry or name in self._import_paths

    def __repr__(self) -> str:
        """Show both loaded and unloaded components."""
        loaded = list(self._registry.keys())
        unloaded = list(self._import_paths.keys())
        all_components = sorted(set(loaded + unloaded))
        return f"LazyImportRegistry({all_components})"


# Now specify the types of the registries
CALLABLES: LazyImportRegistry[CallableCls] = LazyImportRegistry(registry_type=RegistryType.CALLABLES)
LISTENERS: LazyImportRegistry[ListenerCls] = LazyImportRegistry(registry_type=RegistryType.LISTENERS)
RUNNABLES: LazyImportRegistry[Runnable] = LazyImportRegistry(registry_type=RegistryType.RUNNABLES)
