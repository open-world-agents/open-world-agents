"""
Integration tests for the complete message system.

This test suite validates the complete message system implementation including:
- Message registry functionality with real entry points
- Message package integration
- Backward compatibility
- Cross-package interoperability
"""

import warnings

from owa.core import MESSAGES


class TestMessageSystemIntegration:
    """Integration tests for the complete message system."""

    def test_message_registry_discovery(self):
        """Test that MESSAGES registry can discover messages from owa-msgs package."""
        # Force reload to ensure we get fresh entry points
        MESSAGES.reload()

        # Check that desktop messages are discovered
        expected_messages = [
            "desktop/KeyboardEvent",
            "desktop/MouseEvent",
            "desktop/WindowInfo",
            "desktop/ScreenCaptured",
        ]

        available_messages = list(MESSAGES.keys())

        for message_type in expected_messages:
            assert message_type in available_messages, f"Message type {message_type} not found in registry"

    def test_owa_msgs_package_integration(self):
        """Test owa-msgs package integration with registry."""
        # Test direct imports work
        from owa.msgs.desktop.keyboard import KeyboardEvent
        from owa.msgs.desktop.mouse import MouseEvent
        from owa.msgs.desktop.screen import ScreenCaptured
        from owa.msgs.desktop.window import WindowInfo

        # Test message creation
        kb_event = KeyboardEvent(event_type="press", vk=65)
        assert kb_event.event_type == "press"
        assert kb_event.vk == 65

        mouse_event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True)
        assert mouse_event.x == 100
        assert mouse_event.button == "left"

        # Test WindowInfo
        window = WindowInfo(title="Test Window", rect=(0, 0, 800, 600), hWnd=12345)
        assert window.title == "Test Window"
        assert window.width == 800
        assert window.height == 600

        # Test ScreenCaptured
        import numpy as np

        frame = np.zeros((100, 200, 4), dtype=np.uint8)
        screen = ScreenCaptured(utc_ns=1234567890, frame_arr=frame)
        assert screen.utc_ns == 1234567890
        assert screen.is_loaded()
        assert screen.shape == (200, 100)  # width, height

        # Test registry and direct import return same classes
        assert MESSAGES["desktop/KeyboardEvent"] is KeyboardEvent
        assert MESSAGES["desktop/MouseEvent"] is MouseEvent
        assert MESSAGES["desktop/WindowInfo"] is WindowInfo
        assert MESSAGES["desktop/ScreenCaptured"] is ScreenCaptured

    def test_backward_compatibility(self):
        """Test backward compatibility with legacy message imports."""
        # Test legacy imports work with deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import the class (this should trigger warning)
            from owa.env.desktop.msg import KeyboardEvent as LegacyKeyboardEvent

            # Create instance (this should also trigger warning)
            event = LegacyKeyboardEvent(event_type="press", vk=65)

            # Should have issued deprecation warning (at least one)
            assert len(w) >= 1
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message)

            # Should create new-style message
            assert event.event_type == "press"

            # Should be same class as registry version
            RegistryKeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
            assert type(event) is RegistryKeyboardEvent

    def test_message_schema_consistency(self):
        """Test that message schemas are consistent across access methods."""
        # Get message via registry
        RegistryKeyboardEvent = MESSAGES["desktop/KeyboardEvent"]

        # Get message via direct import
        from owa.msgs.desktop.keyboard import KeyboardEvent as DirectKeyboardEvent

        # Get message via legacy import (with warnings suppressed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from owa.env.desktop.msg import KeyboardEvent as LegacyKeyboardEvent

        # All should be the same class
        assert RegistryKeyboardEvent is DirectKeyboardEvent

        # Legacy should create instances of the same class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy_event = LegacyKeyboardEvent(event_type="press", vk=65)
        assert type(legacy_event) is RegistryKeyboardEvent

        # Schemas should be identical
        registry_schema = RegistryKeyboardEvent.get_schema()
        direct_schema = DirectKeyboardEvent.get_schema()
        assert registry_schema == direct_schema

    def test_domain_based_naming_convention(self):
        """Test that domain-based naming convention is properly implemented."""
        # All desktop messages should use domain/MessageType format
        desktop_messages = [name for name in MESSAGES.keys() if name.startswith("desktop/")]

        assert len(desktop_messages) >= 4

        expected_messages = [
            "desktop/KeyboardEvent",
            "desktop/MouseEvent",
            "desktop/WindowInfo",
            "desktop/ScreenCaptured",
        ]

        for expected in expected_messages:
            assert expected in desktop_messages

            # Verify _type attribute matches the registry name
            message_class = MESSAGES[expected]
            type_attr = message_class._type
            if hasattr(type_attr, "default"):
                type_value = type_attr.default
            else:
                type_value = type_attr
            assert type_value == expected

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
