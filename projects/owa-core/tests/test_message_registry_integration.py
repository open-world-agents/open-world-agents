"""
Integration tests for the message registry system with owa-msgs package.

These tests verify that the message registry can discover and load
messages from the owa-msgs package via entry points.
"""


from owa.core import MESSAGES
from owa.core.message import BaseMessage


class TestMessageRegistryIntegration:
    """Integration tests for message registry with real entry points."""

    def test_messages_registry_discovery(self):
        """Test that MESSAGES registry can discover messages from owa-msgs package."""
        # Force reload to ensure we get fresh entry points
        MESSAGES.reload()

        # Check that desktop messages are discovered
        expected_messages = [
            "desktop/KeyboardEvent",
            "desktop/KeyboardState",
            "desktop/MouseEvent",
            "desktop/MouseState",
        ]

        available_messages = list(MESSAGES.keys())

        for message_type in expected_messages:
            assert message_type in available_messages, f"Message type {message_type} not found in registry"

    def test_message_class_access(self):
        """Test accessing message classes through the registry."""
        MESSAGES.reload()

        # Test KeyboardEvent
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        assert issubclass(KeyboardEvent, BaseMessage)

        # Extract the actual _type value (handle ModelPrivateAttr)
        type_attr = KeyboardEvent._type
        if hasattr(type_attr, "default"):
            type_value = type_attr.default
        else:
            type_value = type_attr
        assert type_value == "desktop/KeyboardEvent"

        # Test MouseEvent
        MouseEvent = MESSAGES["desktop/MouseEvent"]
        assert issubclass(MouseEvent, BaseMessage)

        type_attr = MouseEvent._type
        if hasattr(type_attr, "default"):
            type_value = type_attr.default
        else:
            type_value = type_attr
        assert type_value == "desktop/MouseEvent"

    def test_message_instantiation(self):
        """Test creating message instances from registry classes."""
        MESSAGES.reload()

        # Create KeyboardEvent instance
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        event = KeyboardEvent(event_type="press", vk=65)
        assert event.event_type == "press"
        assert event.vk == 65

        # Create MouseEvent instance
        MouseEvent = MESSAGES["desktop/MouseEvent"]
        mouse_event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True)
        assert mouse_event.event_type == "click"
        assert mouse_event.x == 100
        assert mouse_event.y == 200
        assert mouse_event.button == "left"
        assert mouse_event.pressed is True

    def test_message_serialization_deserialization(self):
        """Test that registry messages can be serialized and deserialized."""
        import io

        MESSAGES.reload()

        # Test KeyboardEvent serialization/deserialization
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        original_event = KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)

        # Serialize
        buffer = io.BytesIO()
        original_event.serialize(buffer)

        # Deserialize
        buffer.seek(0)
        deserialized_event = KeyboardEvent.deserialize(buffer)

        assert deserialized_event.event_type == original_event.event_type
        assert deserialized_event.vk == original_event.vk
        assert deserialized_event.timestamp == original_event.timestamp

    def test_backward_compatibility(self):
        """Test that direct imports still work alongside registry access."""
        from owa.msgs.desktop.keyboard import KeyboardEvent as DirectKeyboardEvent

        MESSAGES.reload()
        RegistryKeyboardEvent = MESSAGES["desktop/KeyboardEvent"]

        # Both should be the same class
        assert DirectKeyboardEvent is RegistryKeyboardEvent

        # Both should create equivalent instances
        direct_event = DirectKeyboardEvent(event_type="press", vk=65)
        registry_event = RegistryKeyboardEvent(event_type="press", vk=65)

        assert direct_event.event_type == registry_event.event_type
        assert direct_event.vk == registry_event.vk
        assert direct_event._type == registry_event._type

    def test_registry_dict_like_behavior(self):
        """Test that registry behaves like a dictionary."""
        MESSAGES.reload()

        # Test 'in' operator
        assert "desktop/KeyboardEvent" in MESSAGES
        assert "nonexistent/Message" not in MESSAGES

        # Test get() method
        KeyboardEvent = MESSAGES.get("desktop/KeyboardEvent")
        assert KeyboardEvent is not None

        # Extract the actual _type value (handle ModelPrivateAttr)
        type_attr = KeyboardEvent._type
        if hasattr(type_attr, "default"):
            type_value = type_attr.default
        else:
            type_value = type_attr
        assert type_value == "desktop/KeyboardEvent"

        # Test get() with default
        NonExistent = MESSAGES.get("nonexistent/Message", None)
        assert NonExistent is None

        # Test len()
        assert len(MESSAGES) >= 4  # At least the 4 desktop messages

        # Test iteration
        message_types = list(MESSAGES)
        assert "desktop/KeyboardEvent" in message_types
        assert "desktop/MouseEvent" in message_types

    def test_message_schema_access(self):
        """Test that message schemas can be accessed."""
        MESSAGES.reload()

        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        schema = KeyboardEvent.get_schema()

        assert schema is not None
        assert "properties" in schema
        assert "event_type" in schema["properties"]
        assert "vk" in schema["properties"]
