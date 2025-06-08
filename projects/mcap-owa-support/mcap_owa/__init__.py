__version__ = "0.3.9"

# Initialize message registry for OEP-0005 support
from .message_registry import initialize_message_registry

# Ensure message types are discovered and registered
initialize_message_registry()
