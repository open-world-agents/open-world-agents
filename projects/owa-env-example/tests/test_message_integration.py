"""
Integration tests for OEP-0005 message system.

This module tests the complete workflow of message registration,
discovery, and usage in OWAMcap files.
"""

import tempfile
import pytest
from pathlib import Path

from owa.core import MESSAGES, get_component, list_components
from owa.env.example.example_messages import (
    ExampleSensorData,
    ExampleEvent,
    ExampleProcessingResult,
    ExampleConfiguration,
    ExampleLogEntry
)


def test_example_message_registration():
    """Test that example messages are properly registered."""
    # Check that example messages are available in the registry
    message_list = list_components("messages", namespace="example")
    
    assert "messages" in message_list
    example_messages = message_list["messages"]
    
    # Verify all example messages are registered
    expected_messages = [
        "example/SensorData",
        "example/Event", 
        "example/ProcessingResult",
        "example/Configuration",
        "example/LogEntry"
    ]
    
    for msg_type in expected_messages:
        assert msg_type in example_messages


def test_example_message_access():
    """Test accessing example messages through the component API."""
    # Test direct access
    sensor_data_class = get_component("messages", namespace="example", name="SensorData")
    assert sensor_data_class == ExampleSensorData
    
    # Test namespace access
    example_messages = get_component("messages", namespace="example")
    assert "SensorData" in example_messages
    assert example_messages["SensorData"] == ExampleSensorData


def test_example_message_instances():
    """Test creating instances of example messages."""
    # Test SensorData
    sensor_data = ExampleSensorData(
        timestamp=1234567890,
        sensor_id="temp_01",
        values=[23.5, 24.1, 23.8],
        metadata={"location": "room_a"}
    )
    
    assert sensor_data.timestamp == 1234567890
    assert sensor_data.sensor_id == "temp_01"
    assert len(sensor_data.values) == 3
    assert sensor_data.metadata["location"] == "room_a"
    assert sensor_data._type == "example/SensorData"
    
    # Test Event
    event = ExampleEvent(
        event_type="sensor_reading",
        source="temperature_sensor",
        timestamp=1234567890,
        payload={"value": 23.5}
    )
    
    assert event.event_type == "sensor_reading"
    assert event.source == "temperature_sensor"
    assert event.payload["value"] == 23.5
    assert event._type == "example/Event"
    
    # Test ProcessingResult
    result = ExampleProcessingResult(
        operation_id="op_123",
        status="success",
        processing_time_ms=150.5,
        input_count=100,
        output_count=95,
        metrics={"accuracy": 0.95}
    )
    
    assert result.operation_id == "op_123"
    assert result.status == "success"
    assert result.metrics["accuracy"] == 0.95
    assert result._type == "example/ProcessingResult"


def test_example_message_schemas():
    """Test schema generation for example messages."""
    # Test SensorData schema
    schema = ExampleSensorData.get_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "timestamp" in schema["properties"]
    assert "sensor_id" in schema["properties"]
    assert "values" in schema["properties"]
    assert "metadata" in schema["properties"]
    
    # Check required fields
    assert "required" in schema
    assert "timestamp" in schema["required"]
    assert "sensor_id" in schema["required"]
    assert "values" in schema["required"]
    assert "metadata" not in schema["required"]  # Optional field
    
    # Test Event schema
    event_schema = ExampleEvent.get_schema()
    assert "event_type" in event_schema["properties"]
    assert "source" in event_schema["properties"]
    assert "timestamp" in event_schema["properties"]
    assert "payload" in event_schema["properties"]


def test_example_message_serialization():
    """Test serialization and deserialization of example messages."""
    import io
    
    # Create a test message
    original_message = ExampleSensorData(
        timestamp=1234567890,
        sensor_id="temp_01",
        values=[23.5, 24.1, 23.8],
        metadata={"location": "room_a"}
    )
    
    # Serialize
    buffer = io.BytesIO()
    original_message.serialize(buffer)
    
    # Deserialize
    buffer.seek(0)
    deserialized_message = ExampleSensorData.deserialize(buffer)
    
    # Verify
    assert deserialized_message.timestamp == original_message.timestamp
    assert deserialized_message.sensor_id == original_message.sensor_id
    assert deserialized_message.values == original_message.values
    assert deserialized_message.metadata == original_message.metadata
    assert deserialized_message._type == original_message._type


@pytest.mark.skipif(
    not pytest.importorskip("mcap_owa", reason="mcap-owa-support not available"),
    reason="mcap-owa-support not available"
)
def test_example_messages_in_mcap():
    """Test using example messages in OWAMcap files."""
    try:
        from mcap_owa.highlevel import OWAMcapWriter, OWAMcapReader
    except ImportError:
        pytest.skip("mcap-owa-support not available")
    
    # Create test messages
    sensor_data = ExampleSensorData(
        timestamp=1234567890,
        sensor_id="temp_01",
        values=[23.5, 24.1, 23.8],
        metadata={"location": "room_a"}
    )
    
    event = ExampleEvent(
        event_type="sensor_reading",
        source="temperature_sensor",
        timestamp=1234567890,
        payload={"value": 23.5}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_messages.mcap"
        
        # Write messages to MCAP file
        with OWAMcapWriter(str(file_path)) as writer:
            writer.write_message("sensors", sensor_data, log_time=1234567890)
            writer.write_message("events", event, log_time=1234567891)
        
        # Read messages back
        with OWAMcapReader(str(file_path)) as reader:
            messages = list(reader.iter_decoded_messages())
            
            assert len(messages) == 2
            
            # Check first message (sensor data)
            topic1, timestamp1, msg1 = messages[0]
            assert topic1 == "sensors"
            assert timestamp1 == 1234567890
            assert msg1.sensor_id == "temp_01"
            assert len(msg1.values) == 3
            
            # Check second message (event)
            topic2, timestamp2, msg2 = messages[1]
            assert topic2 == "events"
            assert timestamp2 == 1234567891
            assert msg2.event_type == "sensor_reading"
            assert msg2.source == "temperature_sensor"


def test_message_type_naming_convention():
    """Test that example messages follow the correct naming convention."""
    # All example messages should follow namespace/MessageType pattern
    expected_types = {
        ExampleSensorData: "example/SensorData",
        ExampleEvent: "example/Event",
        ExampleProcessingResult: "example/ProcessingResult",
        ExampleConfiguration: "example/Configuration",
        ExampleLogEntry: "example/LogEntry"
    }
    
    for message_class, expected_type in expected_types.items():
        assert message_class._type == expected_type
        
        # Verify namespace matches
        namespace, message_name = expected_type.split("/", 1)
        assert namespace == "example"
        
        # Verify message name is PascalCase
        assert message_name[0].isupper()
        assert "_" not in message_name  # No underscores
        assert "-" not in message_name  # No hyphens
