from .auto_discovery import auto_discover_plugins
from .callable import Callable
from .component_access import get_component, get_component_info, list_components
from .listener import Listener
from .message import BaseMessage, OWAMessage
from .messages import MESSAGES, MessageRegistry
from .plugin_discovery import discover_and_register_plugins, get_plugin_discovery
from .plugin_spec import PluginSpec
from .registry import CALLABLES, LISTENERS, RUNNABLES, LazyImportRegistry, Registry
from .runnable import Runnable

# Automatically discover and register plugins on import
auto_discover_plugins()

__all__ = [
    # Core components
    "Callable",
    "Listener",
    "Registry",
    "LazyImportRegistry",
    "Runnable",
    # Messages
    "BaseMessage",
    "OWAMessage",
    # Message registry (OEP-0006)
    "MESSAGES",
    "MessageRegistry",
    # Plugin system (OEP-0003)
    "PluginSpec",
    "discover_and_register_plugins",
    "get_plugin_discovery",
    # Component access API
    "get_component",
    "get_component_info",
    "list_components",
    # Global registries
    "CALLABLES",
    "LISTENERS",
    "RUNNABLES",
]
