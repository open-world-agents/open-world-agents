# ================ Definition of the Registry class ================================
# Using lazyregistry library for lazy-loading registries with namespace support and type safety
# references:
# - https://github.com/milkclouds/lazyregistry


from lazyregistry import Registry

from .callable import Callable as CallableCls
from .listener import Listener as ListenerCls
from .runnable import Runnable

# Global registries for each component type
CALLABLES: Registry[str, CallableCls] = Registry(name="callables")
LISTENERS: Registry[str, type[ListenerCls]] = Registry(name="listeners")
RUNNABLES: Registry[str, type[Runnable]] = Registry(name="runnables")

__all__ = ["CALLABLES", "LISTENERS", "RUNNABLES"]
