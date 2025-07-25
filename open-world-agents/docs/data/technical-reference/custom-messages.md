# Custom Message Types in OWAMcap

OWAMcap's extensible design allows you to define and register custom message types for domain-specific data while maintaining compatibility with the standard OWAMcap ecosystem.

!!! tip "When to Use Custom Messages"
    Custom messages are perfect for:
    
    - **Domain-specific data**: Sensor readings, game events, robotics telemetry
    - **Application-specific context**: Custom metadata, annotations, or derived data
    - **Research extensions**: Novel data types for experimental workflows
    
    Standard desktop messages (`mouse`, `keyboard`, `screen`, `window`) cover most desktop agent use cases.

## Creating Custom Messages

All custom messages must inherit from `OWAMessage` and follow the domain/MessageType naming convention:

```python
from owa.core.message import OWAMessage
from typing import Optional, List
from pydantic import Field, validator
import time

class TemperatureReading(OWAMessage):
    _type = "sensors/TemperatureReading"

    temperature: float          # Temperature in Celsius
    humidity: float = Field(..., ge=0, le=100)  # Relative humidity (0-100%)
    location: str              # Sensor location identifier
    timestamp: Optional[int] = Field(default_factory=time.time_ns)  # Unix timestamp in nanoseconds

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < -273.15:  # Absolute zero check
            raise ValueError('Temperature cannot be below absolute zero')
        return v

class GameEvent(OWAMessage):
    _type = "gaming/PlayerAction"

    action_type: str           # "move", "attack", "interact"
    player_id: str            # Unique player identifier
    coordinates: List[float] = Field(..., min_items=3, max_items=3)  # [x, y, z] world coordinates
    metadata: dict = {}        # Additional action-specific data

    @validator('action_type')
    def validate_action_type(cls, v):
        allowed_actions = {'move', 'attack', 'interact', 'idle'}
        if v not in allowed_actions:
            raise ValueError(f'action_type must be one of {allowed_actions}')
        return v
```

## Package Registration

Custom messages are registered through Python entry points in your package's `pyproject.toml`:

```toml
[project.entry-points."owa.msgs"]
"sensors/TemperatureReading" = "my_sensors.messages:TemperatureReading"
"gaming/PlayerAction" = "my_game.events:GameEvent"
"custom/MyMessage" = "my_package.messages:MyMessage"
```

**Important**: The package containing your custom messages must be installed in the same environment where you're using OWAMcap for the entry points to be discovered:

```bash
# Install your custom message package
pip install my-custom-messages

# Or install in development mode
pip install -e /path/to/my-custom-messages

# Now custom messages are available in the registry
python -c "from owa.core import MESSAGES; print('sensors/TemperatureReading' in MESSAGES)"
```

## Usage with OWAMcap

Once registered, custom messages work seamlessly with OWAMcap tools:

```python
from mcap_owa.highlevel import OWAMcapWriter, OWAMcapReader
from owa.core import MESSAGES

# Access your custom message through the registry
TemperatureReading = MESSAGES['sensors/TemperatureReading']

# Write custom messages to MCAP
with OWAMcapWriter("sensor_data.mcap") as writer:
    reading = TemperatureReading(
        temperature=23.5,
        humidity=65.2,
        location="office_desk"
    )
    writer.write_message(reading, topic="temperature", timestamp=reading.timestamp)

# Read custom messages from MCAP
with OWAMcapReader("sensor_data.mcap") as reader:
    for msg in reader.iter_messages(topics=["temperature"]):
        temp_data = msg.decoded
        print(f"Temperature: {temp_data.temperature}°C at {temp_data.location}")
```

## Best Practices

=== "Naming Conventions"
    - **Domain**: Use descriptive domain names (`sensors`, `gaming`, `robotics`)
    - **MessageType**: Use PascalCase (`TemperatureReading`, `PlayerAction`)
    - **Avoid conflicts**: Check existing message types before naming
    - **Be specific**: `sensors/TemperatureReading` vs generic `sensors/Reading`

=== "Schema Design"
    - **Use type hints**: Enable automatic JSON schema generation
    - **Leverage pydantic features**: See [Pydantic documentation](https://docs.pydantic.dev/) for validation, field constraints, and defaults
    - **Documentation**: Include docstrings for complex message types

=== "Package Structure"
    ```
    my_custom_package/
    ├── pyproject.toml              # Entry point registration
    ├── my_package/
    │   ├── __init__.py
    │   └── messages.py             # Message definitions
    └── tests/
        └── test_messages.py        # Message validation tests
    ```

## CLI Integration

Custom messages automatically work with OWA CLI tools:

```bash
# List all available message types (including custom)
owl messages list

# View custom message schema
owl messages show sensors/TemperatureReading

# View custom messages in MCAP files
owl mcap cat sensor_data.mcap --topics temperature
```

!!! info "Complete CLI Reference"
    For detailed information about all CLI commands and options:

    - **[CLI Tools](../../cli/index.md)** - Complete command overview
    - **[Message Commands](../../cli/messages.md)** - Detailed `owl messages` documentation
    - **[MCAP Commands](../../cli/mcap.md)** - Working with custom messages in MCAP files

## Next Steps

- **[OWAMcap Format Guide](format-guide.md)**: Return to the main format documentation
- **[Data Pipeline](data-pipeline.md)**: Learn how custom messages work in the training pipeline
- **[CLI Reference](../../cli/index.md)**: Complete CLI documentation for working with custom messages
