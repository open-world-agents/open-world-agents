# ================ Definition of the Registry class ================================
# references:
# - https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
# - https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

import importlib
from enum import StrEnum
from typing import Callable, Dict, Generic, Optional, Type, TypeVar

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

    def __getitem__(self, name: str) -> Type[T]:
        return self._registry[name]

    def get(self, name: str) -> Optional[Type[T]]:
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


def activate_module(entrypoint):
    """
    Activate a module by its entrypoint. Modules are expected to have an `activate` function, following OwaEnvInterface.
    """
    if entrypoint in _MODULES:
        return _MODULES[entrypoint]

    try:
        entrypoint_module = importlib.import_module(entrypoint)
    except ModuleNotFoundError as e:
        if e.name == entrypoint:
            print(f"Module '{entrypoint}' not found.")
        else:
            raise e

    try:
        entrypoint_module.activate()
    except AttributeError as e:
        if e.args[0] == f"module '{entrypoint}' has no attribute 'activate'":
            print(f"Module '{entrypoint}' has no attribute 'activate'. Please define it.")
        else:
            raise e

    _MODULES.register(entrypoint)(entrypoint_module)
    return entrypoint_module
