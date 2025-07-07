"""
Message registry system for automatic message discovery via entry points.

This module implements the message registry system for centralized message management,
providing automatic discovery of message types through Python entry points.
"""

try:
    from importlib.metadata import entry_points
except ImportError:
    # Python < 3.10 fallback
    from importlib_metadata import entry_points

from typing import Dict, ItemsView, Iterator, KeysView, Type, ValuesView

from .message import BaseMessage


class MessageRegistry:
    """
    Registry for automatic message discovery via entry points.

    This registry discovers and manages message types registered through
    the 'owa.msgs' entry point group. It provides dict-like access
    to message classes by their type names.

    Example:
        >>> from owa.core import MESSAGES
        >>> KeyboardEvent = MESSAGES['desktop/KeyboardEvent']
        >>> event = KeyboardEvent(event_type="press", vk=65)
    """

    def __init__(self):
        self._messages: Dict[str, Type[BaseMessage]] = {}
        self._loaded = False

    def _load_messages(self) -> None:
        """Load all registered message types from entry points."""
        if self._loaded:
            return

        try:
            eps = entry_points(group="owa.msgs")
        except TypeError:
            # Python < 3.10 compatibility
            eps = entry_points().get("owa.msgs", [])

        for entry_point in eps:
            try:
                message_class = entry_point.load()
                if not issubclass(message_class, BaseMessage):
                    print(f"Warning: Message {entry_point.name} does not inherit from BaseMessage")
                    continue
                self._messages[entry_point.name] = message_class
            except Exception as e:
                # Log warning but continue loading other messages
                print(f"Warning: Failed to load message {entry_point.name}: {e}")

        self._loaded = True

    def __getitem__(self, key: str) -> Type[BaseMessage]:
        """Get a message class by its type name."""
        self._load_messages()
        return self._messages[key]

    def __contains__(self, key: str) -> bool:
        """Check if a message type is registered."""
        self._load_messages()
        return key in self._messages

    def get(self, key: str, default: Type[BaseMessage] | None = None) -> Type[BaseMessage] | None:
        """Get a message class by its type name, returning default if not found."""
        self._load_messages()
        return self._messages.get(key, default)

    def keys(self) -> KeysView[str]:
        """Get all registered message type names."""
        self._load_messages()
        return self._messages.keys()

    def values(self) -> ValuesView[Type[BaseMessage]]:
        """Get all registered message classes."""
        self._load_messages()
        return self._messages.values()

    def items(self) -> ItemsView[str, Type[BaseMessage]]:
        """Get all (name, class) pairs."""
        self._load_messages()
        return self._messages.items()

    def __iter__(self) -> Iterator[str]:
        """Iterate over message type names."""
        self._load_messages()
        return iter(self._messages)

    def __len__(self) -> int:
        """Get the number of registered message types."""
        self._load_messages()
        return len(self._messages)

    def reload(self) -> None:
        """Force reload of all message types from entry points."""
        self._loaded = False
        self._messages.clear()
        self._load_messages()

    def __repr__(self) -> str:
        """Show both loaded and unloaded messages."""
        loaded = list(self._messages.keys())
        return f"MessageRegistry({loaded})"


# Global message registry instance
MESSAGES = MessageRegistry()
