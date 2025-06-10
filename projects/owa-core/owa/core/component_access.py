# ================ Enhanced Component Access API ================================
# Provides flexible component access functions with namespace support

from typing import Any, Dict, List, Optional

from .registry import CALLABLES, LISTENERS, RUNNABLES, LazyImportRegistry


def get_component(component_type: str, namespace: Optional[str] = None, name: Optional[str] = None) -> Any:
    """
    Flexible component access with multiple patterns.

    Usage patterns:
    1. get_component("callables", namespace="example", name="add")
    2. get_component("callables", namespace="example")  # Returns all in namespace
    3. get_component("callables")  # Returns all callables

    Args:
        component_type: Type of component ("callables", "listeners", "runnables")
        namespace: Optional namespace filter
        name: Optional specific component name

    Returns:
        Single component, dictionary of components, or all components

    Raises:
        ValueError: If component_type is not supported
        KeyError: If specific component is not found
    """
    registry = get_registry(component_type)
    if not registry:
        raise ValueError(f"Unknown component type: {component_type}")

    if namespace and name:
        # Get specific component: namespace/name
        full_name = f"{namespace}/{name}"
        component = registry.get(full_name)
        if component is None:
            raise KeyError(f"Component '{full_name}' not found")
        return component

    elif namespace:
        # Get all components in namespace
        return get_namespace_components(registry, namespace)

    else:
        # Get all components
        all_components = {}
        for component_name in _get_all_component_names(registry):
            component = registry.get(component_name)
            if component is not None:
                all_components[component_name] = component
        return all_components


def get_namespace_components(registry: LazyImportRegistry, namespace: str) -> Dict[str, Any]:
    """
    Get all components in a namespace as a dictionary.

    Args:
        registry: The registry to search
        namespace: Namespace to filter by

    Returns:
        Dictionary mapping component names (without namespace) to components
    """
    prefix = f"{namespace}/"
    components = {}

    for full_name in _get_all_component_names(registry):
        if full_name.startswith(prefix):
            component_name = full_name[len(prefix) :]  # Remove namespace prefix
            components[component_name] = registry[full_name]

    return components


def list_components(component_type: Optional[str] = None, namespace: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available components with optional filtering.

    This function returns metadata only - no imports occur during listing.

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

        components = _get_all_component_names(registry)

        if namespace:
            # Filter by namespace
            prefix = f"{namespace}/"
            components = [name for name in components if name.startswith(prefix)]

        result[reg_type] = components

    return result


def get_registry(component_type: str) -> Optional[LazyImportRegistry]:
    """
    Get the appropriate registry for component type.

    Args:
        component_type: Type of component

    Returns:
        Registry instance or None if not found
    """
    registries = {
        "callables": CALLABLES,
        "listeners": LISTENERS,
        "runnables": RUNNABLES,
    }
    return registries.get(component_type)


def _get_all_component_names(registry: LazyImportRegistry) -> List[str]:
    """
    Get all component names from a registry (both loaded and unloaded).

    Args:
        registry: Registry to get names from

    Returns:
        List of all component names
    """
    # Get names from both loaded components and import paths
    loaded_names = set(registry._registry.keys())
    import_names = set(registry._import_paths.keys())
    return list(loaded_names | import_names)


def get_component_info(component_type: str, namespace: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about components without loading them.

    Args:
        component_type: Type of component
        namespace: Optional namespace filter

    Returns:
        Dictionary mapping component names to their info
    """
    registry = get_registry(component_type)
    if not registry:
        return {}

    info = {}
    for name in _get_all_component_names(registry):
        if namespace and not name.startswith(f"{namespace}/"):
            continue

        component_info = {
            "name": name,
            "loaded": name in registry._registry,
            "import_path": registry._import_paths.get(name),
        }
        info[name] = component_info

    return info
