"""
Example message definitions for the example plugin.

This module demonstrates how to create custom OWAMessage types
for use in OWAMcap files and plugin systems.
"""

from typing import Optional, List, Dict, Any
from owa.core.message import OWAMessage


class ExampleSensorData(OWAMessage):
    """
    Example sensor data message.

    This message represents sensor readings with timestamp,
    sensor identification, and measurement values.
    """

    timestamp: int
    sensor_id: str
    values: List[float]
    metadata: Optional[Dict[str, Any]] = None

# Set _type as class attribute after class definition
ExampleSensorData._type = "example/SensorData"


class ExampleEvent(OWAMessage):
    """
    Example event message.

    This message represents a generic event with type,
    source, and optional payload data.
    """

    event_type: str
    source: str
    timestamp: int
    payload: Optional[Dict[str, Any]] = None

# Set _type as class attribute after class definition
ExampleEvent._type = "example/Event"


class ExampleProcessingResult(OWAMessage):
    """
    Example processing result message.

    This message represents the result of data processing
    operations with status, metrics, and optional error information.
    """

    operation_id: str
    status: str  # "success", "error", "pending"
    processing_time_ms: float
    input_count: int
    output_count: int
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

# Set _type as class attribute after class definition
ExampleProcessingResult._type = "example/ProcessingResult"


class ExampleConfiguration(OWAMessage):
    """
    Example configuration message.

    This message represents configuration settings
    for the example plugin components.
    """

    component_name: str
    settings: Dict[str, Any]
    version: str = "1.0"
    enabled: bool = True

# Set _type as class attribute after class definition
ExampleConfiguration._type = "example/Configuration"


class ExampleLogEntry(OWAMessage):
    """
    Example log entry message.

    This message represents a structured log entry
    with level, message, and contextual information.
    """

    level: str  # "debug", "info", "warning", "error", "critical"
    message: str
    timestamp: int
    component: str
    context: Optional[Dict[str, Any]] = None

# Set _type as class attribute after class definition
ExampleLogEntry._type = "example/LogEntry"
