"""
Tests for the message system (OEP-0005).

This module tests the message registration, discovery, and validation
functionality introduced in OEP-0005.
"""

import pytest
from owa.core.message import OWAMessage
from owa.core.plugin_spec import PluginSpec
from owa.core.registry import MESSAGES
from owa.core.component_access import get_component, list_components


class TestMessage(OWAMessage):
    """Test message for validation."""
    _type = "test/TestMessage"
    
    field1: str
    field2: int
    optional_field: str = None


class InvalidMessage:
    """Invalid message class without proper interface."""
    pass


def test_plugin_spec_with_messages():
    """Test PluginSpec creation with message definitions."""
    plugin_spec = PluginSpec(
        namespace="test",
        version="1.0.0",
        description="Test plugin with messages",
        components={
            "callables": {
                "test_func": "test.module:test_function",
            }
        },
        messages={
            "TestMessage": "test.module:TestMessage",
            "AnotherMessage": "test.module:AnotherMessage",
        }
    )
    
    assert plugin_spec.messages is not None
    assert "TestMessage" in plugin_spec.messages
    assert "AnotherMessage" in plugin_spec.messages
    
    # Test message name generation
    message_names = plugin_spec.get_message_names()
    assert "test/TestMessage" in message_names
    assert "test/AnotherMessage" in message_names
    
    # Test import path retrieval
    import_path = plugin_spec.get_message_import_path("TestMessage")
    assert import_path == "test.module:TestMessage"


def test_plugin_spec_without_messages():
    """Test PluginSpec creation without message definitions."""
    plugin_spec = PluginSpec(
        namespace="test",
        version="1.0.0",
        description="Test plugin without messages",
        components={
            "callables": {
                "test_func": "test.module:test_function",
            }
        }
    )
    
    assert plugin_spec.messages is None
    assert plugin_spec.get_message_names() == []
    assert plugin_spec.get_message_import_path("NonExistent") is None


def test_message_name_validation():
    """Test message name validation rules."""
    # Valid PascalCase names should pass
    valid_spec = PluginSpec(
        namespace="test",
        version="1.0.0",
        description="Test plugin",
        components={},
        messages={
            "TestMessage": "test:TestMessage",
            "SensorData": "test:SensorData",
            "ProcessingResult": "test:ProcessingResult",
            "A": "test:A",  # Single letter is valid
            "Test123": "test:Test123",  # Numbers are allowed
        }
    )
    assert valid_spec.messages is not None
    
    # Invalid names should raise ValueError
    with pytest.raises(ValueError, match="Message name .* is invalid"):
        PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={},
            messages={
                "invalidName": "test:InvalidName",  # lowercase start
            }
        )
    
    with pytest.raises(ValueError, match="Message name .* is invalid"):
        PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={},
            messages={
                "test_message": "test:TestMessage",  # underscore
            }
        )
    
    with pytest.raises(ValueError, match="Message name .* is invalid"):
        PluginSpec(
            namespace="test",
            version="1.0.0",
            description="Test plugin",
            components={},
            messages={
                "Test-Message": "test:TestMessage",  # hyphen
            }
        )


def test_message_registry():
    """Test message registration and retrieval."""
    # Clear registry for clean test
    original_registry = MESSAGES._registry.copy()
    original_import_paths = MESSAGES._import_paths.copy()
    
    try:
        MESSAGES._registry.clear()
        MESSAGES._import_paths.clear()
        
        # Register a message type
        MESSAGES.register("test/TestMessage", obj_or_import_path=TestMessage, is_instance=True)
        
        # Test retrieval
        retrieved_class = MESSAGES["test/TestMessage"]
        assert retrieved_class == TestMessage
        
        # Test message type attribute
        assert retrieved_class._type == "test/TestMessage"
        
        # Test schema generation
        schema = retrieved_class.get_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        
    finally:
        # Restore original registry
        MESSAGES._registry.clear()
        MESSAGES._import_paths.clear()
        MESSAGES._registry.update(original_registry)
        MESSAGES._import_paths.update(original_import_paths)


def test_message_component_access():
    """Test message access through component access API."""
    # Clear registry for clean test
    original_registry = MESSAGES._registry.copy()
    original_import_paths = MESSAGES._import_paths.copy()
    
    try:
        MESSAGES._registry.clear()
        MESSAGES._import_paths.clear()
        
        # Register test messages
        MESSAGES.register("test/TestMessage", obj_or_import_path=TestMessage, is_instance=True)
        
        # Test component access
        message_class = get_component("messages", namespace="test", name="TestMessage")
        assert message_class == TestMessage
        
        # Test listing messages
        message_list = list_components("messages", namespace="test")
        assert "messages" in message_list
        assert "test/TestMessage" in message_list["messages"]
        
        # Test namespace access
        test_messages = get_component("messages", namespace="test")
        assert "TestMessage" in test_messages
        assert test_messages["TestMessage"] == TestMessage
        
    finally:
        # Restore original registry
        MESSAGES._registry.clear()
        MESSAGES._import_paths.clear()
        MESSAGES._registry.update(original_registry)
        MESSAGES._import_paths.update(original_import_paths)


def test_message_instance_creation():
    """Test creating instances of registered message types."""
    # Create a test message instance
    message = TestMessage(
        field1="test_value",
        field2=42,
        optional_field="optional"
    )
    
    assert message.field1 == "test_value"
    assert message.field2 == 42
    assert message.optional_field == "optional"
    assert message._type == "test/TestMessage"
    
    # Test schema generation
    schema = message.get_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "field1" in schema["properties"]
    assert "field2" in schema["properties"]
    assert "optional_field" in schema["properties"]


def test_message_serialization():
    """Test message serialization and deserialization."""
    import io
    
    # Create a test message
    original_message = TestMessage(
        field1="test_value",
        field2=42,
        optional_field="optional"
    )
    
    # Serialize
    buffer = io.BytesIO()
    original_message.serialize(buffer)
    
    # Deserialize
    buffer.seek(0)
    deserialized_message = TestMessage.deserialize(buffer)
    
    # Verify
    assert deserialized_message.field1 == original_message.field1
    assert deserialized_message.field2 == original_message.field2
    assert deserialized_message.optional_field == original_message.optional_field
    assert deserialized_message._type == original_message._type
