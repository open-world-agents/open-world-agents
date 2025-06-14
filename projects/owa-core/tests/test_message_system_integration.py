"""
Comprehensive integration tests for the message system.

This test suite validates the complete message system implementation including:
- Message registry functionality
- Message package integration
- CLI tooling integration
- Backward compatibility
- Cross-package interoperability
"""

import warnings

import pytest

from owa.core import MESSAGES


class TestMessageSystemIntegration:
    """Integration tests for the complete message system."""

    def test_message_registry_core(self):
        """Test core message registry functionality."""
        # Test registry is available and working
        assert MESSAGES is not None
        assert len(MESSAGES) >= 6  # At least 6 desktop messages

        # Test domain-based message access
        assert "desktop/KeyboardEvent" in MESSAGES
        assert "desktop/MouseEvent" in MESSAGES
        assert "desktop/WindowInfo" in MESSAGES
        assert "desktop/ScreenEmitted" in MESSAGES

        # Test message class retrieval
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        assert KeyboardEvent is not None
        assert hasattr(KeyboardEvent, "_type")

    def test_owa_msgs_package_integration(self):
        """Test owa-msgs package integration with registry."""
        # Test direct imports work
        from owa.msgs.desktop.keyboard import KeyboardEvent
        from owa.msgs.desktop.mouse import MouseEvent
        from owa.msgs.desktop.screen import ScreenEmitted
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

        # Test ScreenEmitted
        import numpy as np

        frame = np.zeros((100, 200, 4), dtype=np.uint8)
        screen = ScreenEmitted(utc_ns=1234567890, frame_arr=frame)
        assert screen.utc_ns == 1234567890
        assert screen.is_loaded()
        assert screen.shape == (200, 100)  # width, height

        # Test registry and direct import return same classes
        assert MESSAGES["desktop/KeyboardEvent"] is KeyboardEvent
        assert MESSAGES["desktop/MouseEvent"] is MouseEvent
        assert MESSAGES["desktop/WindowInfo"] is WindowInfo
        assert MESSAGES["desktop/ScreenEmitted"] is ScreenEmitted

    def test_mcap_integration_availability(self):
        """Test that MCAP integration is available (detailed tests in mcap-owa-support)."""
        try:
            from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter  # noqa: F401  # noqa: F401
            # If import succeeds, MCAP integration is available
            # Detailed MCAP tests are in the mcap-owa-support package
        except ImportError:
            pytest.skip("mcap-owa-support not available")

    def test_cli_integration(self):
        """Test CLI tooling integration."""
        # Test that CLI modules can be imported
        from owa.cli.messages.list import list_messages
        from owa.cli.messages.show import show_message
        from owa.cli.messages.validate import validate_messages

        # These functions should be callable (we won't actually call them
        # to avoid CLI output in tests, but we verify they exist)
        assert callable(list_messages)
        assert callable(show_message)
        assert callable(validate_messages)

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

        assert len(desktop_messages) >= 6

        expected_messages = [
            "desktop/KeyboardEvent",
            "desktop/KeyboardState",
            "desktop/MouseEvent",
            "desktop/MouseState",
            "desktop/WindowInfo",
            "desktop/ScreenEmitted",
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

    def test_extensibility_model(self):
        """Test that the extensibility model works correctly."""
        # Test that registry can be reloaded
        original_count = len(MESSAGES)
        MESSAGES.reload()
        new_count = len(MESSAGES)

        # Should have same number of messages after reload
        assert new_count == original_count

        # Test registry dict-like behavior
        assert "desktop/KeyboardEvent" in MESSAGES
        assert "nonexistent/Message" not in MESSAGES

        # Test get method
        KeyboardEvent = MESSAGES.get("desktop/KeyboardEvent")
        assert KeyboardEvent is not None

        NonExistent = MESSAGES.get("nonexistent/Message")
        assert NonExistent is None

        # Test iteration
        message_types = list(MESSAGES)
        assert "desktop/KeyboardEvent" in message_types

    def test_complete_message_workflow(self):
        """Test complete workflow from message discovery to instantiation."""
        # 1. Discover messages via registry
        available_messages = list(MESSAGES.keys())
        assert "desktop/KeyboardEvent" in available_messages

        # 2. Create message instances
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        MouseEvent = MESSAGES["desktop/MouseEvent"]

        kb_event = KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)
        mouse_event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True, timestamp=1234567890)

        # Verify message creation works
        assert kb_event.event_type == "press"
        assert mouse_event.x == 100


if __name__ == "__main__":
    # Run a quick validation
    test = TestMessageSystemIntegration()
    test.test_message_registry_core()
    test.test_owa_msgs_package_integration()
    test.test_backward_compatibility()
    print("✅ Message system integration validation passed!")
