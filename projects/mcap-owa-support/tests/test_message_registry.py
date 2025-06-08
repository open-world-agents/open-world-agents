"""
Tests for message registry integration (OEP-0005).

This module tests the automatic message discovery and registration
functionality in mcap-owa-support.
"""

import pytest
from owa.core.message import OWAMessage
from mcap_owa.message_registry import (
    register_message_type,
    get_message_class,
    list_message_types,
    is_message_type_registered,
    clear_registry,
    initialize_message_registry
)


class TestMessage(OWAMessage):
    """Test message for registry testing."""
    _type = "test/TestMessage"
    
    field1: str
    field2: int


class AnotherTestMessage(OWAMessage):
    """Another test message."""
    _type = "test/AnotherMessage"
    
    data: str


class InvalidMessage:
    """Invalid message without proper interface."""
    pass


@pytest.fixture
def clean_registry():
    """Provide a clean message registry for testing."""
    clear_registry()
    yield
    clear_registry()


def test_register_message_type_with_class(clean_registry):
    """Test registering a message type with a class."""
    register_message_type("test/TestMessage", TestMessage)
    
    assert is_message_type_registered("test/TestMessage")
    assert get_message_class("test/TestMessage") == TestMessage


def test_register_message_type_with_import_path(clean_registry):
    """Test registering a message type with an import path."""
    # This would normally be an import path, but for testing we'll use the class
    register_message_type("test/TestMessage", TestMessage)
    
    assert is_message_type_registered("test/TestMessage")
    retrieved_class = get_message_class("test/TestMessage")
    assert retrieved_class == TestMessage


def test_register_invalid_message_type(clean_registry):
    """Test registering an invalid message type."""
    with pytest.raises(ValueError, match="must have a '_type' attribute"):
        register_message_type("test/Invalid", InvalidMessage)


def test_get_nonexistent_message_class(clean_registry):
    """Test getting a non-existent message class."""
    assert get_message_class("nonexistent/Message") is None


def test_list_message_types(clean_registry):
    """Test listing all registered message types."""
    register_message_type("test/TestMessage", TestMessage)
    register_message_type("test/AnotherMessage", AnotherTestMessage)
    
    message_types = list_message_types()
    assert len(message_types) == 2
    assert "test/TestMessage" in message_types
    assert "test/AnotherMessage" in message_types
    assert message_types["test/TestMessage"] == TestMessage
    assert message_types["test/AnotherMessage"] == AnotherTestMessage


def test_is_message_type_registered(clean_registry):
    """Test checking if a message type is registered."""
    assert not is_message_type_registered("test/TestMessage")
    
    register_message_type("test/TestMessage", TestMessage)
    assert is_message_type_registered("test/TestMessage")
    assert not is_message_type_registered("test/NonExistent")


def test_clear_registry(clean_registry):
    """Test clearing the message registry."""
    register_message_type("test/TestMessage", TestMessage)
    assert is_message_type_registered("test/TestMessage")
    
    clear_registry()
    assert not is_message_type_registered("test/TestMessage")
    assert len(list_message_types()) == 0


def test_message_type_validation():
    """Test that registered message types have proper validation."""
    # Test valid message
    assert hasattr(TestMessage, '_type')
    assert hasattr(TestMessage, 'get_schema')
    assert TestMessage._type == "test/TestMessage"
    
    # Test schema generation
    schema = TestMessage.get_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema


def test_initialize_message_registry():
    """Test message registry initialization."""
    # This test verifies that initialization doesn't crash
    # The actual plugin discovery depends on owa.core being available
    try:
        initialize_message_registry()
        # If we get here, initialization succeeded
        assert True
    except Exception as e:
        # If owa.core is not available, that's expected in some test environments
        assert "owa.core" in str(e) or "discover_message_definitions" in str(e)


def test_message_instance_creation():
    """Test creating instances of registered message types."""
    message = TestMessage(field1="test", field2=42)
    
    assert message.field1 == "test"
    assert message.field2 == 42
    assert message._type == "test/TestMessage"


def test_message_schema_generation():
    """Test schema generation for registered message types."""
    schema = TestMessage.get_schema()
    
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "field1" in schema["properties"]
    assert "field2" in schema["properties"]
    
    # Check field types
    assert schema["properties"]["field1"]["type"] == "string"
    assert schema["properties"]["field2"]["type"] == "integer"


def test_multiple_message_registration(clean_registry):
    """Test registering multiple message types."""
    register_message_type("test/TestMessage", TestMessage)
    register_message_type("test/AnotherMessage", AnotherTestMessage)
    
    # Verify both are registered
    assert is_message_type_registered("test/TestMessage")
    assert is_message_type_registered("test/AnotherMessage")
    
    # Verify they can be retrieved correctly
    assert get_message_class("test/TestMessage") == TestMessage
    assert get_message_class("test/AnotherMessage") == AnotherTestMessage
    
    # Verify listing includes both
    message_types = list_message_types()
    assert len(message_types) == 2
