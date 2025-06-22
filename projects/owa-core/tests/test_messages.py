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

    def test_dict_like_methods(self, mock_entry_points_factory, create_mock_entry_point):
        """Test dict-like methods: keys(), values(), items(), __iter__, __len__."""
        registry = MessageRegistry()

        # Create multiple mock messages
        mock_entry_point1 = create_mock_entry_point("test/Message1", MockMessage)
        mock_entry_point2 = create_mock_entry_point("test/Message2", MockMessage)

        with patch(
            "owa.core.messages.entry_points",
            side_effect=mock_entry_points_factory([mock_entry_point1, mock_entry_point2]),
        ):
            # Test keys()
            keys = list(registry.keys())
            assert "test/Message1" in keys
            assert "test/Message2" in keys
            assert len(keys) == 2

            # Test values()
            values = list(registry.values())
            assert MockMessage in values
            assert len(values) == 2

            # Test items()
            items = list(registry.items())
            assert ("test/Message1", MockMessage) in items
            assert ("test/Message2", MockMessage) in items
            assert len(items) == 2

            # Test __iter__
            names = list(registry)
            assert "test/Message1" in names
            assert "test/Message2" in names
            assert len(names) == 2

            # Test __len__
            assert len(registry) == 2

    def test_entry_points_type_error_fallback(self, create_mock_entry_point):
        """Test fallback when entry_points() raises TypeError (Python < 3.10)."""
        registry = MessageRegistry()
        mock_entry_point = create_mock_entry_point("test/MockMessage", MockMessage)

        # Mock entry_points to raise TypeError on first call, then return fallback
        def mock_entry_points_with_fallback(*args, **kwargs):
            if "group" in kwargs:
                raise TypeError("entry_points() got an unexpected keyword argument 'group'")
            else:
                # Fallback call - return mock entry points
                mock_eps = type("MockEntryPoints", (), {})()
                mock_eps.get = lambda group, default=None: [mock_entry_point] if group == "owa.msgs" else default
                return mock_eps

        with patch("owa.core.messages.entry_points", side_effect=mock_entry_points_with_fallback):
            message_class = registry["test/MockMessage"]
            assert message_class is MockMessage

    def test_load_message_with_non_basemessage_class(self, create_mock_entry_point, capsys):
        """Test loading a message that doesn't inherit from BaseMessage."""
        registry = MessageRegistry()

        # Create a class that doesn't inherit from BaseMessage
        class NotAMessage:
            pass

        mock_entry_point = create_mock_entry_point("test/NotAMessage", NotAMessage)

        with patch("owa.core.messages.entry_points", return_value=[mock_entry_point]):
            registry._load_messages()

            # Should not be in registry
            assert "test/NotAMessage" not in registry

            # Should print warning
            captured = capsys.readouterr()
            assert "Warning: Message test/NotAMessage does not inherit from BaseMessage" in captured.out

    def test_load_message_with_exception(self, create_mock_entry_point, capsys):
        """Test loading a message that raises an exception."""
        registry = MessageRegistry()

        # Create a mock entry point that raises an exception when loaded
        mock_entry_point = create_mock_entry_point("test/FailingMessage", None)
        mock_entry_point.load.side_effect = ImportError("Module not found")

        with patch("owa.core.messages.entry_points", return_value=[mock_entry_point]):
            registry._load_messages()

            # Should not be in registry
            assert "test/FailingMessage" not in registry

            # Should print warning
            captured = capsys.readouterr()
            assert "Warning: Failed to load message test/FailingMessage: Module not found" in captured.out

    def test_importlib_metadata_fallback(self):
        """Test fallback to importlib_metadata for Python < 3.10."""
        # This test verifies the import fallback works
        # We can't easily test the actual import error scenario without complex mocking
        # But we can verify the module structure supports both imports

        # Test that the module can be imported (this exercises the try/except block)
        import owa.core.messages

        assert hasattr(owa.core.messages, "entry_points")
        assert hasattr(owa.core.messages, "MessageRegistry")
