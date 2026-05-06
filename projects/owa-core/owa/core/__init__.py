from loguru import logger

from .auto_discovery import auto_discover_plugins
from .callable import Callable
from .component_access import get_component, get_component_info, list_components
from .listener import Listener
from .message import BaseMessage, OWAMessage
from .messages import MESSAGES
from .plugin_discovery import get_plugin_discovery
from .registry import CALLABLES, LISTENERS, RUNNABLES
from .runnable import Runnable

# Disable logger by default for library usage (following loguru best practices)
# Reference: https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.core")

# Automatically discover and register plugins on import
auto_discover_plugins()

__all__ = [
    # Core components
    "Callable",
    "Listener",
    "Runnable",
    # Messages
    "BaseMessage",
    "OWAMessage",
    # Message registry
    "MESSAGES",
    # Plugin system
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
