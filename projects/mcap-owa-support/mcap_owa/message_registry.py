"""
Message type registry for automatic discovery and registration.

This module implements OEP-0005 by automatically discovering and registering
message types from installed plugins, making them available for OWAMcap operations.
"""

import importlib
from typing import Dict, Type, Optional
from loguru import logger

# Global message type registry
_message_registry: Dict[str, Type] = {}
_registry_initialized = False


def initialize_message_registry() -> None:
    """
    Initialize the message registry by discovering message types from installed plugins.
    
    This function is called automatically when mcap-owa-support is imported
    to ensure all message types are available for OWAMcap operations.
    """
    global _registry_initialized
    
    if _registry_initialized:
        return
    
    try:
        # Import owa.core to trigger plugin discovery
        from owa.core import discover_message_definitions
        
        # Get all message definitions from discovered plugins
        message_definitions = discover_message_definitions()
        
        # Register each message type
        for msg_type, import_path in message_definitions.items():
            try:
                register_message_type(msg_type, import_path)
            except Exception as e:
                logger.warning(f"Failed to register message type '{msg_type}': {e}")
        
        logger.info(f"Registered {len(_message_registry)} message types from plugins")
        
    except ImportError:
        # owa.core not available - this is fine for standalone usage
        logger.debug("owa.core not available - message auto-discovery disabled")
    except Exception as e:
        logger.warning(f"Message registry initialization failed: {e}")
    
    _registry_initialized = True


def register_message_type(msg_type: str, import_path_or_class) -> None:
    """
    Register a message type in the global registry.
    
    Args:
        msg_type: Message type identifier (e.g., "example/SensorData")
        import_path_or_class: Import path string or message class
    """
    if isinstance(import_path_or_class, str):
        # Import the message class from the path
        module_path, class_name = import_path_or_class.rsplit(":", 1)
        module = importlib.import_module(module_path)
        message_class = getattr(module, class_name)
    else:
        # Already a class
        message_class = import_path_or_class
    
    # Validate that it's a proper message class
    if not hasattr(message_class, '_type'):
        raise ValueError(f"Message class {message_class} must have a '_type' attribute")
    
    if not hasattr(message_class, 'get_schema'):
        raise ValueError(f"Message class {message_class} must have a 'get_schema' method")
    
    # Register the message type
    _message_registry[msg_type] = message_class
    logger.debug(f"Registered message type: {msg_type} -> {message_class}")


def get_message_class(msg_type: str) -> Optional[Type]:
    """
    Get a message class by its type identifier.
    
    Args:
        msg_type: Message type identifier
        
    Returns:
        Message class or None if not found
    """
    return _message_registry.get(msg_type)


def list_message_types() -> Dict[str, Type]:
    """
    Get all registered message types.
    
    Returns:
        Dictionary mapping message type identifiers to message classes
    """
    return _message_registry.copy()


def is_message_type_registered(msg_type: str) -> bool:
    """
    Check if a message type is registered.
    
    Args:
        msg_type: Message type identifier
        
    Returns:
        True if the message type is registered
    """
    return msg_type in _message_registry


def clear_registry() -> None:
    """Clear all registered message types (mainly for testing)."""
    global _registry_initialized
    _message_registry.clear()
    _registry_initialized = False


# Automatically initialize the registry when this module is imported
initialize_message_registry()
