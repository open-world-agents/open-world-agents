# ================ Enhanced Component Access API ================================
# Provides flexible component access functions with namespace support

from typing import Any, Dict, List, Optional

from lazyregistry import ImportString, Registry

from .registry import CALLABLES, LISTENERS, RUNNABLES


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
    if registry is None:
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
        for component_name in registry.keys():
            component = registry.get(component_name)
            if component is not None:
                all_components[component_name] = component
        return all_components


def get_namespace_components(registry: Registry, namespace: str) -> Dict[str, Any]:
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

    for full_name in registry.keys():
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
        if registry is None:
            continue

        components = list(registry.keys())

        if namespace:
            # Filter by namespace
            prefix = f"{namespace}/"
            components = [name for name in components if name.startswith(prefix)]

        result[reg_type] = components

    return result


def get_registry(component_type: str) -> Optional[Registry]:
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
    if registry is None:
        return {}

    info = {}
    for name in registry.keys():
        if namespace and not name.startswith(f"{namespace}/"):
            continue

        value = registry.data.get(name)
        is_lazy = isinstance(value, ImportString)

        component_info = {
            "name": name,
            "loaded": not is_lazy,
            "import_path": str(value) if is_lazy else None,
        }
        info[name] = component_info

    return info
