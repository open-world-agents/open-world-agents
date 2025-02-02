# ================ Definition of the Registry class ================================
# references:
# - https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
# - https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

from enum import StrEnum


class RegistryType(StrEnum):
    CALLABLES = "callables"
    LISTENERS = "listeners"
    PACKAGES = "packages"
    UNKNOWN = "unknown"


class Registry:
    def __init__(self, registry_type: RegistryType = RegistryType.UNKNOWN):
        self._registry = {}
        self.registry_type = registry_type

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls

        return decorator

    def extend(self, other: "Registry"):
        self._registry.update(other._registry)

    def __getitem__(self, name: str):
        return self._registry[name]

    def get(self, name: str):
        return self._registry.get(name)

    # list all the registered items
    def __repr__(self):
        return self._registry.__repr__()


CALLABLES = Registry()
LISTENERS = Registry()


def activate_module(entrypoint):
    import importlib

    try:
        entrypoint_module = importlib.import_module(entrypoint)
        entrypoint_module.activate()
    except ModuleNotFoundError:
        print(f"Module {entrypoint} not found.")
    except AttributeError:
        print(f"Module {entrypoint} does not have the `activate` function. You must define it.")
