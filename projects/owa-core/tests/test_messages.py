"""
Tests for the message registry system (owa.core.messages).
"""

from unittest.mock import patch

import pytest

from owa.core.message import OWAMessage
from owa.core.messages import MESSAGES, MessageRegistry


class MockMessage(OWAMessage):
    """Test message for registry testing."""

    _type = "test/MockMessage"
    data: str


class TestMessageRegistry:
    """Test cases for MessageRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = MessageRegistry()
        assert len(registry._messages) == 0
        assert not registry._loaded

    def test_lazy_loading(self, mock_entry_points_factory, create_mock_entry_point):
        """Test that messages are loaded lazily."""
        registry = MessageRegistry()

        # Mock entry points
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            # First access should trigger loading
            assert not registry._loaded
            message_class = registry["test/MockMessage"]
            assert registry._loaded
            assert message_class is MockMessage

    def test_getitem_access(self, mock_entry_points_factory, create_mock_entry_point):
        """Test accessing messages via [] operator."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            message_class = registry["test/MockMessage"]
            assert message_class is MockMessage

    def test_getitem_keyerror(self, mock_entry_points_factory):
        """Test KeyError when accessing non-existent message."""
        registry = MessageRegistry()

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([])):
            with pytest.raises(KeyError):
                registry["nonexistent/Message"]

    def test_contains_operator(self, mock_entry_points_factory, create_mock_entry_point):
        """Test 'in' operator for checking message existence."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            assert "test/MockMessage" in registry
            assert "nonexistent/Message" not in registry

    def test_get_method(self, mock_entry_points_factory, create_mock_entry_point):
        """Test get() method with default values."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            # Existing message
            message_class = registry.get("test/MockMessage")
            assert message_class is MockMessage

            # Non-existent message with default
            default_class = registry.get("nonexistent/Message", MockMessage)
            assert default_class is MockMessage

            # Non-existent message without default
            result = registry.get("nonexistent/Message")
            assert result is None

    def test_reload(self, mock_entry_points_factory, create_mock_entry_point):
        """Test reload() method."""
        registry = MessageRegistry()

        # First load
        mock_entry_point1 = create_mock_entry_point("test/MockMessage1", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point1])):
            registry._load_messages()
            assert len(registry) == 1
            assert "test/MockMessage1" in registry

        # Reload with different messages
        mock_entry_point2 = create_mock_entry_point("test/MockMessage2", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point2])):
            registry.reload()
            assert len(registry) == 1
            assert "test/MockMessage2" in registry
            assert "test/MockMessage1" not in registry

    def test_global_messages_instance(self):
        """Test that MESSAGES is a MessageRegistry instance."""
        assert isinstance(MESSAGES, MessageRegistry)
