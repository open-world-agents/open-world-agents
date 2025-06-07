# ================ Definition of the Registry class ================================
# references:
# - https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
# - https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

import importlib
from enum import StrEnum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .callable import Callable as CallableCls
from .listener import Listener as ListenerCls
from .owa_env_interface import OwaEnvInterface
from .runnable import Runnable


class RegistryType(StrEnum):
    CALLABLES = "callables"
    LISTENERS = "listeners"
    RUNNABLES = "runnables"
    MODULES = "modules"
    UNKNOWN = "unknown"


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, registry_type: RegistryType = RegistryType.UNKNOWN):
        self._registry: Dict[str, T] = {}
        self.registry_type = registry_type

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            self._registry[name] = obj
            return obj

        return decorator

    def extend(self, other: "Registry[T]") -> None:
        self._registry.update(other._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __getitem__(self, name: str) -> T:
        return self._registry[name]

    def get(self, name: str) -> Optional[T]:
        return self._registry.get(name)

    # List all the registered items
    def __repr__(self) -> str:
        return repr(self._registry)


# Now specify the types of the registries
CALLABLES: Registry[CallableCls] = Registry(registry_type=RegistryType.CALLABLES)
LISTENERS: Registry[ListenerCls] = Registry(registry_type=RegistryType.LISTENERS)
RUNNABLES: Registry[Runnable] = Registry(registry_type=RegistryType.RUNNABLES)

# _MODULES is managed by the activate_module function
_MODULES: Registry[OwaEnvInterface] = Registry(registry_type=RegistryType.MODULES)


def activate_module(entrypoint: str) -> OwaEnvInterface:
    """
    Activate a module by its entrypoint.

    Modules are expected to have an `activate` function, following OwaEnvInterface.

    Args:
        entrypoint: The module entrypoint to activate (e.g., "owa.env.desktop")

    Returns:
        The activated module instance
    """
    if entrypoint in _MODULES:
        return _MODULES[entrypoint]

    try:
        entrypoint_module: OwaEnvInterface = importlib.import_module(entrypoint)
    except ModuleNotFoundError as e:
        if e.name == entrypoint:
            raise Exception(f"Module '{entrypoint}' not found. Please ensure it is installed.") from e
        raise e

    try:
        # Check if the module satisfies the OwaEnvInterface
        entrypoint_module.activate()
    except AttributeError as e:
        if e.args[0] == f"module '{entrypoint}' has no attribute 'activate'":
            raise Exception(f"Module '{entrypoint}' has no attribute 'activate'. Please define it.") from e
        raise e

    _MODULES.register(entrypoint)(entrypoint_module)
    return entrypoint_module


# ================ Entry Points-Based Plugin Discovery ========================


def get_registry(component_type: str) -> Optional[Registry]:
    """Get the appropriate registry for component type."""
    registries = {
        "callables": CALLABLES,
        "listeners": LISTENERS,
        "runnables": RUNNABLES,
    }
    return registries.get(component_type)


def get_component(component_type: str, namespace: str = None, name: str = None) -> Any:
    """
    Flexible component access with multiple patterns.

    Usage patterns:
    1. get_component("callables", namespace="example", name="add")
    2. get_component("callables", namespace="example")  # Returns all in namespace
    3. get_component("callables")  # Returns all callables

    Args:
        component_type: Type of component ("callables", "listeners", "runnables")
        namespace: Optional namespace filter
        name: Optional component name (requires namespace)

    Returns:
        Component, dictionary of components, or None if not found
    """
    registry = get_registry(component_type)
    if not registry:
        raise ValueError(f"Unknown component type: {component_type}")

    if namespace and name:
        # Get specific component: namespace/name
        full_name = f"{namespace}/{name}"
        return registry.get(full_name)

    elif namespace:
        # Get all components in namespace
        components = {}
        for full_name in registry._registry:
            if full_name.startswith(f"{namespace}/"):
                component_name = full_name.split("/", 1)[1]
                components[component_name] = registry._registry[full_name]
        return components

    else:
        # Get all components
        return dict(registry._registry)


def list_components(component_type: str = None, namespace: str = None) -> Dict[str, List[str]]:
    """
    List available components with optional filtering.

    Args:
        component_type: Optional component type filter
        namespace: Optional namespace filter

    Returns:
        Dictionary mapping component types to lists of component names
    """
    if component_type:
        registries = {component_type: get_registry(component_type)}
    else:
        registries = {
            "callables": CALLABLES,
            "listeners": LISTENERS,
            "runnables": RUNNABLES,
        }

    result = {}
    for reg_type, registry in registries.items():
        if not registry:
            continue

        components = []
        for full_name in registry._registry:
            if namespace:
                # Filter by namespace
                if full_name.startswith(f"{namespace}/"):
                    components.append(full_name)
            else:
                components.append(full_name)

        result[reg_type] = components

    return result


class EntryPointPluginRegistry:
    """Discover and register plugins using Entry Points."""

    def __init__(self):
        self.discovered_plugins = {}
        # Auto-discover plugins on initialization
        self.auto_discover()

    def auto_discover(self):
        """Automatically discover and register all plugins via Entry Points."""
        try:
            plugins = self.discover_plugins()

            for plugin_name, spec in plugins.items():
                self.register_plugin_components(spec)
                self.discovered_plugins[plugin_name] = spec

        except Exception as e:
            # Don't crash on plugin discovery errors, just log them
            print(f"Warning: Plugin discovery failed: {e}")

    def discover_plugins(self) -> Dict[str, Any]:
        """Discover all plugins via Entry Points."""
        discovered = {}

        try:
            # Try using importlib.metadata first (Python 3.8+)
            try:
                from importlib.metadata import entry_points

                eps = entry_points(group="owa.env.plugins")
            except ImportError:
                # Fallback to pkg_resources for older Python versions
                import pkg_resources

                eps = pkg_resources.iter_entry_points("owa.env.plugins")

            for entry_point in eps:
                try:
                    # Load the plugin specification
                    plugin_spec = entry_point.load()

                    # Import PluginSpec here to avoid circular imports
                    from .plugin_spec import PluginSpec

                    if isinstance(plugin_spec, PluginSpec):
                        discovered[entry_point.name] = plugin_spec
                        print(f"Discovered plugin: {entry_point.name} v{plugin_spec.version}")
                    else:
                        print(f"Warning: {entry_point.name} does not provide a valid PluginSpec")

                except Exception as e:
                    print(f"Warning: Could not load plugin {entry_point.name}: {e}")

        except Exception as e:
            print(f"Warning: Entry points discovery failed: {e}")

        return discovered

    def register_plugin_components(self, spec):
        """Register all components from a plugin specification."""
        for component_type, components in spec.components.items():
            registry = get_registry(component_type)
            if not registry:
                continue

            for name, module_path in components.items():
                # Use unified namespace/name pattern
                full_name = f"{spec.namespace}/{name}"

                try:
                    # Import and register the component
                    component = self.import_component(module_path)
                    registry.register(full_name)(component)
                    print(f"Registered {component_type}: {full_name}")

                except Exception as e:
                    print(f"Warning: Could not register {full_name}: {e}")

    def import_component(self, module_path: str):
        """Import component from module path."""
        try:
            module_name, component_name = module_path.split(":", 1)
        except ValueError:
            raise ValueError(
                f"Invalid module path format: '{module_path}'. Expected format: 'module_name:component_name'"
            )

        module = importlib.import_module(module_name)
        return getattr(module, component_name)


# Global entry point registry instance - initialized on import
try:
    entry_point_registry = EntryPointPluginRegistry()
except Exception as e:
    print(f"Warning: Failed to initialize entry point registry: {e}")
    entry_point_registry = None
