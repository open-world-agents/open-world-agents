"""
Tests for the message registry system.
"""

from unittest.mock import Mock, patch

import pytest

from owa.core.message import OWAMessage
from owa.core.messages import MESSAGES, MessageRegistry


class MockMessage(OWAMessage):
    """Test message for registry testing."""

    _type = "test/MockMessage"
    data: str


def create_mock_entry_point(name: str, load_return_value):
    """Create a mock entry point for testing."""
    mock_entry_point = Mock()
    mock_entry_point.name = name
    mock_entry_point.load.return_value = load_return_value
    return mock_entry_point


def mock_entry_points_factory(entry_points_list):
    """Create a mock entry points function for testing."""

    def mock_entry_points(group=None):
        return entry_points_list if group == "owa.msgs" else []

    return mock_entry_points


class InvalidMessage:
    """Invalid message that doesn't inherit from BaseMessage."""

    _type = "test/InvalidMessage"


class MockMessageRegistry:
    """Test cases for MessageRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = MessageRegistry()
        assert len(registry._messages) == 0
        assert not registry._loaded

    def test_lazy_loading(self):
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

    def test_getitem_access(self):
        """Test accessing messages via [] operator."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            message_class = registry["test/MockMessage"]
            assert message_class is MockMessage

    def test_getitem_keyerror(self):
        """Test KeyError when accessing non-existent message."""
        registry = MessageRegistry()

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([])):
            with pytest.raises(KeyError):
                registry["nonexistent/Message"]

    def test_contains_operator(self):
        """Test 'in' operator for checking message existence."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            assert "test/MockMessage" in registry
            assert "nonexistent/Message" not in registry

    def test_get_method(self):
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

    def test_keys_values_items(self):
        """Test dict-like methods: keys(), values(), items()."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            keys = list(registry.keys())
            values = list(registry.values())
            items = list(registry.items())

            assert keys == ["test/MockMessage"]
            assert values == [MockMessage]
            assert items == [("test/MockMessage", MockMessage)]

    def test_iteration(self):
        """Test iteration over registry."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            message_names = list(registry)
            assert message_names == ["test/MockMessage"]

    def test_len(self):
        """Test len() function on registry."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            assert len(registry) == 1

    def test_reload(self):
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

    def test_invalid_message_handling(self):
        """Test handling of invalid messages that don't inherit from BaseMessage."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/InvalidMessage", InvalidMessage)

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            with patch("builtins.print") as mock_print:
                registry._load_messages()
                # Should print warning and not include the invalid message
                mock_print.assert_called_once()
                assert "test/InvalidMessage" not in registry

    def test_loading_error_handling(self):
        """Test handling of errors during message loading."""
        from unittest.mock import Mock

        registry = MessageRegistry()
        mock_entry_point = Mock()
        mock_entry_point.name = "test/ErrorMessage"
        mock_entry_point.load.side_effect = ImportError("Module not found")

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_factory([mock_entry_point])):
            with patch("builtins.print") as mock_print:
                registry._load_messages()
                # Should print warning and continue
                mock_print.assert_called_once()
                assert "test/ErrorMessage" not in registry

    def test_global_messages_instance(self):
        """Test that MESSAGES is a MessageRegistry instance."""
        assert isinstance(MESSAGES, MessageRegistry)
