# OWAMcap Format Guide

!!! info "What is OWAMcap?"
    OWAMcap is a specification for using the open-source [MCAP](https://mcap.dev/) container format with Open World Agents (OWA) message definitions. It provides an efficient way to store and process multimodal desktop interaction data including screen captures, mouse events, keyboard events, and window information.

!!! tip "New to OWAMcap?"
    Start with **[Why OWAMcap?](why_owamcap.md)** to understand the problem it solves and why you should use it.

## Table of Contents

- [Quick Start](#quick-start) - Get started in 3 steps
- [Core Concepts](#core-concepts) - Essential message types and features
- [Working with OWAMcap](#working-with-owamcap) - Reading, writing, and media handling
- [Storage & Performance](#storage--performance) - Efficiency characteristics
- [Custom Message Types](#custom-message-types) - Extend OWAMcap with your own message types
- [Advanced Usage](#advanced-usage) - Integrations and best practices
- [Migration & Troubleshooting](#migration--troubleshooting) - Practical help
- [Technical Reference](#technical-reference) - Specifications and standards

## Quick Start

!!! example "Try OWAMcap in 3 Steps"

    **1. Install the packages:**
    ```bash
    pip install mcap-owa-support owa-msgs
    ```

    **2. Explore an example file with the `owl` CLI:**

    !!! info "What is `owl`?"
        `owl` is the command-line interface for OWA tools, installed with `owa-cli`. See the [CLI documentation](../owl_cli_reference.md) for complete usage.

    ```bash
    # Download example file
    wget https://github.com/open-world-agents/open-world-agents/raw/main/docs/data/example.mcap

    # View file info
    owl mcap info example.mcap

    # List first 5 messages
    owl mcap cat example.mcap --n 5
    ```

    **3. Load in Python:**
    ```python
    from mcap_owa.highlevel import OWAMcapReader

    with OWAMcapReader("example.mcap", decode_args={"return_dict": True}) as reader:
        for msg in reader.iter_messages(topics=["screen"]):
            screen_data = msg.decoded
            print(f"Frame: {screen_data.shape} at {screen_data.utc_ns}")
            break  # Just show first frame
    ```

## Core Concepts

OWAMcap combines the robustness of the MCAP container format with OWA's specialized message types for desktop environments, creating a powerful format for recording, analyzing, and training on human-computer interaction data.

### Key Terms

!!! info "Essential Terminology"
    - **MCAP**: A modular container file format for heterogeneous, timestamped data (like a ZIP file for time-series data)
    - **Topic**: A named channel in MCAP files (e.g., "screen", "mouse") that groups related messages
    - **Lazy Loading**: Loading data only when needed, crucial for memory efficiency with large datasets

### What Makes a File "OWAMcap"

=== "Technical Definition"
    - **Base Format**: Standard MCAP container format
    - **Profile**: `owa` designation in MCAP metadata
    - **Schema Encoding**: JSON Schema
    - **Message Interface**: All messages implement `BaseMessage` from `owa.core.message`
    - **Standard Messages**: Core message types from `owa-msgs` package

    **Why MCAP?** Efficient storage and retrieval for heterogeneous timestamped data with minimal dependencies. This is format after ROSBag, but designed for modern use cases and optimized for random access.

=== "Architecture Overview"
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    OWAMcap File (.mcap)                     │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
    │  │   Metadata      │  │   Timestamps    │  │  Messages   │  │
    │  │   - Profile     │  │   - Nanosecond  │  │  - Mouse    │  │
    │  │   - Topics      │  │     precision   │  │  - Keyboard │  │
    │  │   - Schemas     │  │   - Event sync  │  │  - Window   │  │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    │ References
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                External Media Files (.mkv, .png)           │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
    │  │  Video Frames   │  │  Screenshots    │  │   Audio     │  │
    │  │  - H.265 codec  │  │  - PNG/JPEG     │  │  - Optional │  │
    │  │  - Hardware acc │  │  - Lossless     │  │  - Sync'd   │  │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    ```

=== "Practical Example"
    ```bash
    $ owl mcap info example.mcap
    library:   mcap-owa-support 0.5.1; mcap 1.3.0
    profile:   owa
    messages:  864
    duration:  10.3574349s
    start:     2025-06-27T18:49:52.129876+09:00 (1751017792.129876000)   
    end:       2025-06-27T18:50:02.4873109+09:00 (1751017802.487310900)  
    compression:
            zstd: [1/1 chunks] [116.46 KiB/16.61 KiB (85.74%)] [1.60 KiB/sec] 
    channels:
            (1) window           11 msgs (1.06 Hz)    : desktop/WindowInfo [jsonschema]      
            (2) keyboard/state   11 msgs (1.06 Hz)    : desktop/KeyboardState [jsonschema]   
            (3) mouse/state      11 msgs (1.06 Hz)    : desktop/MouseState [jsonschema]      
            (4) screen          590 msgs (56.96 Hz)   : desktop/ScreenCaptured [jsonschema]
            (5) mouse           209 msgs (20.18 Hz)   : desktop/MouseEvent [jsonschema]
            (6) keyboard         32 msgs (3.09 Hz)    : desktop/KeyboardEvent [jsonschema]
    channels: 6
    attachments: 0
    metadata: 0
    ```
    <!-- TODO: example for modern version mcap -->

### Key Features

- **Efficient Storage**: External video file references keep MCAP files lightweight
- **Precise Synchronization**: Nanosecond-precision timestamps for perfect event alignment
- **Multimodal Data**: Unified storage for visual, input, and context data
- **Standard Format**: Built on the proven MCAP container format
- **Extensible**: Support for custom message types through entry points

### Core Message Types

| Message Type | Description |
|--------------|-------------|
| `desktop/KeyboardEvent` | Keyboard press/release events |
| `desktop/KeyboardState` | Current keyboard state |
| `desktop/MouseEvent` | Mouse movement, clicks, scrolls |
| `desktop/MouseState` | Current mouse position and buttons |
| `desktop/ScreenCaptured` | Screen capture frames with timestamps |
| `desktop/WindowInfo` | Active window information |

=== "KeyboardEvent"
    ```python
    class KeyboardEvent(OWAMessage):
        _type = "desktop/KeyboardEvent"

        event_type: str  # "press" or "release"
        vk: int         # Virtual key code (e.g., 65 for 'A')
        timestamp: int  # Event timestamp

    # Example: User presses the 'A' key
    KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)
    ```

    !!! tip "What's VK (Virtual Key Code)?"
        Operating systems don't directly use the physical keyboard input values (scan codes) but instead use virtualized keys called VKs. OWA's recorder uses VKs to record keyboard-agnostic data. If you're interested in more details, you can refer to the following resources:

        - [Keyboard Input Overview, Microsoft](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-keyboard-input)
        - [Virtual-Key Codes, Microsoft](https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes)

=== "MouseEvent"
    ```python
    class MouseEvent(OWAMessage):
        _type = "desktop/MouseEvent"

        event_type: str  # "move", "click", "scroll", "drag"
        x: int          # Screen X coordinate
        y: int          # Screen Y coordinate
        button: Optional[str] = None    # "left", "right", "middle"

    # Example: Mouse click at position (100, 200)
    MouseEvent(event_type="click", x=100, y=200, button="left")
    ```

=== "KeyboardState"
    ```python
    class KeyboardState(OWAMessage):
        _type = "desktop/KeyboardState"

        buttons: List[int]  # List of currently pressed virtual key codes

    # Example: No keys currently pressed
    KeyboardState(buttons=[])
    ```

=== "MouseState"
    ```python
    class MouseState(OWAMessage):
        _type = "desktop/MouseState"

        x: int                    # Current mouse X coordinate
        y: int                    # Current mouse Y coordinate
        buttons: List[str] = []   # Currently pressed mouse buttons

    # Example: Mouse at position with no buttons pressed
    MouseState(x=1594, y=1112, buttons=[])
    ```

=== "WindowInfo"
    ```python
    class WindowInfo(OWAMessage):
        _type = "desktop/WindowInfo"

        title: str              # Window title text
        rect: List[int]         # [x, y, width, height]
        hWnd: Optional[int]     # Windows handle (platform-specific)

    # Example: Browser window
    WindowInfo(
        title="GitHub - Open World Agents - Chrome",
        rect=[100, 50, 1200, 800]
    )
    ```

=== "ScreenCaptured ⭐"
    ```python
    class ScreenCaptured(OWAMessage):
        _type = "desktop/ScreenCaptured"

        utc_ns: Optional[int] = None                    # System timestamp (nanoseconds)
        source_shape: Optional[Tuple[int, int]] = None  # Original (width, height)
        shape: Optional[Tuple[int, int]] = None         # Current (width, height)
        media_ref: Optional[MediaRef] = None            # URI or file path reference
        frame_arr: Optional[np.ndarray] = None          # In-memory BGRA array (excluded from JSON)

    # MediaRef supports: URI(data:image/png;base64,... | file:///path | http[s]://...)
    # or file path(/absolute/path | relative/path) for both images and videos

    # Creation patterns:
    # - From raw image: ScreenCaptured(frame_arr=bgra_array).embed_as_data_uri()
    # - From image file: ScreenCaptured(media_ref={"uri": "/path/to/image.png"})
    # - From video frame: ScreenCaptured(media_ref={"uri": "/path/video.mkv", "pts_ns": 123456})
    # - From file URI: ScreenCaptured(media_ref={"uri": "file:///path/to/video.mp4", "pts_ns": 123456})
    # - From URL: ScreenCaptured(media_ref={"uri": "https://example.com/image.png"})
    # - From data URI: ScreenCaptured(media_ref={"uri": "data:image/png;base64,..."})
    ```

## Working with OWAMcap

### Media Handling

OWAMcap's key advantage is efficient media handling through external media references:

=== "Creating ScreenCaptured Messages"
    MediaRef supports: `URI(data:image/png;base64,... | file:///path | http[s]://...) or file path(/absolute/path | relative/path)` for both images and videos.

    ```python
    from owa.core import MESSAGES
    import numpy as np

    ScreenCaptured = MESSAGES['desktop/ScreenCaptured']

    # File paths (absolute/relative) - works for images and videos
    screen_msg = ScreenCaptured(media_ref={"uri": "/absolute/path/image.png"})
    screen_msg = ScreenCaptured(media_ref={"uri": "relative/video.mkv", "pts_ns": 123456})

    # File URIs - works for images and videos
    screen_msg = ScreenCaptured(media_ref={"uri": "file:///path/to/image.jpg"})
    screen_msg = ScreenCaptured(media_ref={"uri": "file:///path/to/video.mp4", "pts_ns": 123456})

    # HTTP/HTTPS URLs - works for images and videos
    screen_msg = ScreenCaptured(media_ref={"uri": "https://example.com/image.png"})
    screen_msg = ScreenCaptured(media_ref={"uri": "https://example.com/video.mp4", "pts_ns": 123456})

    # Data URIs (embedded base64) - typically for images
    screen_msg = ScreenCaptured(media_ref={"uri": "data:image/png;base64,iVBORw0KGgo..."})

    # From raw image array (BGRA format required)
    bgra_array = np.random.randint(0, 255, (1080, 1920, 4), dtype=np.uint8)
    screen_msg = ScreenCaptured(frame_arr=bgra_array)
    screen_msg.embed_as_data_uri(format="png")  # Required for serialization
    # Now screen_msg.media_ref contains: {"uri": "data:image/png;base64,..."}
    ```

=== "Loading and Accessing Frame Data"
    **Lazy Loading**: Frame data is not loaded until explicitly requested, providing enormous advantages when working with external references - you can iterate through thousands of messages without loading actual frame data.

    ```python
    # IMPORTANT: For MCAP files, resolve relative paths first
    # The OWA recorder saves media paths relative to the MCAP file location
    ScreenCaptured = MESSAGES['desktop/ScreenCaptured']
    screen_msg = ScreenCaptured(
        media_ref={"uri": "relative/video.mkv", "pts_ns": 123456789}
    )

    # Must resolve external paths before loading from MCAP files
    screen_msg.resolve_external_path("/path/to/data.mcap")

    # Lazy loading: Frame data is loaded on-demand when these methods are called
    rgb_array = screen_msg.to_rgb_array()        # RGB numpy array (most common)
    pil_image = screen_msg.to_pil_image()        # PIL Image object
    bgra_array = screen_msg.load_frame_array()   # Raw BGRA array (native format)

    # Check if frame data is loaded (lazy loading means it starts as None)
    if screen_msg.frame_arr is not None:
        height, width, channels = screen_msg.frame_arr.shape
        print(f"Frame: {width}x{height}, {channels} channels")
    else:
        print("Frame data not loaded - use load_frame_array() first")
    ```

### Reading and Writing

=== "Reading"
    ```python
    from mcap_owa.highlevel import OWAMcapReader

    with OWAMcapReader("session.mcap") as reader:
        # File metadata
        print(f"Topics: {reader.topics}")
        print(f"Duration: {(reader.end_time - reader.start_time) / 1e9:.2f}s")

        # Lazy loading advantage: Fast iteration without loading frame data
        for msg in reader.iter_messages(topics=["screen"]):
            screen_data = msg.decoded
            print(f"Frame metadata: {screen_data.shape} at {screen_data.utc_ns}")
            # No frame data loaded yet - extremely fast for large datasets

            # Only load frame data when actually needed
            if some_condition:  # e.g., every 10th frame
                frame = screen_data.to_rgb_array()  # Now frame is loaded
                break  # Just show first frame
    ```

=== "Writing"
    ```python
    from mcap_owa.highlevel import OWAMcapWriter
    from owa.core import MESSAGES

    ScreenCaptured = MESSAGES['desktop/ScreenCaptured']
    MouseEvent = MESSAGES['desktop/MouseEvent']

    with OWAMcapWriter("output.mcap") as writer:
        # Write screen capture
        screen_msg = ScreenCaptured(
            utc_ns=1234567890,
            media_ref={"uri": "video.mkv", "pts_ns": 1234567890},
            shape=(1920, 1080)
        )
        writer.write_message(screen_msg, topic="screen", timestamp=1234567890)

        # Write mouse event
        mouse_msg = MouseEvent(event_type="click", x=100, y=200)
        writer.write_message(mouse_msg, topic="mouse", timestamp=1234567891)
    ```

=== "Advanced"
    ```python
    # Time range filtering
    with OWAMcapReader("session.mcap") as reader:
        start_time = reader.start_time + 1_000_000_000  # Skip first second
        end_time = reader.start_time + 10_000_000_000   # First 10 seconds

        for msg in reader.iter_messages(start_time=start_time, end_time=end_time):
            print(f"Message in range: {msg.topic}")

    # Remote files
    with OWAMcapReader("https://example.com/data.mcap") as reader:
        for msg in reader.iter_messages(topics=["screen"]):
            print(f"Remote frame: {msg.decoded.shape}")
    ```

=== "CLI Tools"
    ```bash
    # File information
    owl mcap info session.mcap

    # List messages
    owl mcap cat session.mcap --n 10 --topics screen --topics mouse

    # Migrate between versions
    owl mcap migrate run session.mcap

    # Extract frames
    owl mcap extract-frames session.mcap --output frames/
    ```


## Storage & Performance

OWAMcap achieves remarkable storage efficiency through external video references and intelligent compression:

### Compression Benefits

!!! info "Understanding the Baseline"
    Raw screen capture data is enormous: a single 1920×1080 frame in BGRA format is 8.3 MB. At 60 FPS, this means 498 MB per second of recording. OWAMcap's hybrid storage makes this manageable.

Desktop screen capture at 600 × 800 resolution, 13 s @ 60 Hz:

| Format                               | Size per Frame | Whole Size | Compression Ratio   |
|--------------------------------------|---------------:|-----------:|---------------------|
| Raw BGRA                             | 1.28 MB        | 1.0 GB     | 1.0× (baseline)     |
| PNG                                  | 436 KB         | 333 MB     | 3.0×                |
| JPEG (Quality 85)                    | 59 KB          | 46 MB      | 21.7×               |
| H.265 (keyframe 0.5s, nvd3d11h265enc)| 14.5 KB avg    | 11.3 MB    | 91.7×               |

!!! note "H.265 Configuration"
    The H.265 settings shown above (keyframe 0.5s, nvd3d11h265enc) are the same as those used by [ocap](ocap.md) for efficient desktop recording.

**Key advantages:**

- **Lightweight MCAP:** very fast to parse, transfer, and back up  
- **Video Compression:** leverages hardware-accelerated codecs for extreme savings  
- **Selective Loading:** grab only the frames you need without full decompression  
- **Standard Tools:** preview in any video player and edit with off-the-shelf software  


## Custom Message Types

OWAMcap's extensible design allows you to define and register custom message types for domain-specific data while maintaining compatibility with the standard OWAMcap ecosystem.

### Creating Custom Messages

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

### Package Registration

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

### Usage with OWAMcap

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

### Best Practices

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

### CLI Integration

Custom messages automatically work with OWA CLI tools:

```bash
# List all available message types (including custom)
owl messages list

# View custom message schema
owl messages show sensors/TemperatureReading

# View custom messages in MCAP files
owl mcap cat sensor_data.mcap --topics temperature
```

## Advanced Usage

### Integration Examples

=== "Computer Vision"
    ```python
    from mcap_owa.highlevel import OWAMcapReader

    def detect_objects(mcap_path):
        with OWAMcapReader(mcap_path) as reader:
            for msg in reader.iter_messages(topics=["screen"]):
                frame = msg.decoded.to_rgb_array()
                results = model.detect(frame)
                yield msg.timestamp, results
    ```

=== "Behavior Analysis"
    ```python
    def analyze_user_behavior(mcap_path):
        events = {"mouse": [], "keyboard": [], "screen": []}
        with OWAMcapReader(mcap_path) as reader:
            for msg in reader.iter_messages():
                events[msg.topic].append({
                    "timestamp": msg.timestamp,
                    "data": msg.decoded
                })
        return compute_interaction_patterns(events)
    ```

### Best Practices

=== "Storage Strategy"
    **Decision Tree: Choose Your Storage Approach**

    ```
    Recording Length?
    ├─ < 30 seconds
    │  └─ Use embedded data URIs (self-contained)
    └─ > 30 seconds
       └─ File Size Priority?
          ├─ Minimize MCAP size
          │  └─ Use external video (.mkv)
          └─ Maximize quality
             └─ Use external images (.png)
    ```

    | Use Case | Strategy | Benefits | Trade-offs |
    |----------|----------|----------|------------|
    | **Long recordings** | External video | Minimal MCAP size, efficient | Requires external files |
    | **Short sessions** | Embedded data | Self-contained | Larger MCAP files |
    | **High-quality** | External images | Lossless compression | Many files to manage |
    | **Remote datasets** | Video + URLs | Bandwidth efficient | Network dependency |

=== "Performance"
    ```python
    # ✅ Good: Filter topics early
    with OWAMcapReader("file.mcap") as reader:
        for msg in reader.iter_messages(topics=["screen"]):
            process_frame(msg.decoded)

    # ✅ Good: Lazy loading
    for msg in reader.iter_messages(topics=["screen"]):
        if should_process_frame(msg.timestamp):
            frame = msg.decoded.load_frame_array()  # Only when needed

    # ❌ Avoid: Loading all frames
    frames = [msg.decoded.load_frame_array() for msg in reader.iter_messages()]
    ```

=== "File Organization"
    **Recommended structure:**
    ```
    /data/
    ├── mcaps/                          # Raw MCAP recordings
    │   ├── session_001.mcap
    │   ├── session_001.mkv             # External video files
    │   └── session_002.mcap
    ├── event-dataset/                  # Stage 1: Event Dataset
    │   ├── train/
    │   └── test/
    └── binned-dataset/                 # Stage 2: Binned Dataset
        ├── train/
        └── test/
    ```

    See [OWA Data Pipeline](owa_data_pipeline.md) for complete pipeline details.

## Migration & Troubleshooting

### File Migration

The `owl mcap migrate` command handles version upgrades automatically:

```bash
# Migrate to latest version
owl mcap migrate run old_file.mcap

# Migrate multiple files
owl mcap migrate run file1.mcap file2.mcap

# Dry run to see what would be migrated
owl mcap migrate run old_file.mcap --dry-run
```

### Common Issues

!!! warning "File Not Found Errors"
    When video files are missing:
    ```python
    # Resolve relative paths
    screen_msg.resolve_external_path("/path/to/mcap/file.mcap")
    # Check if external media exists
    screen_msg.media_ref.validate_uri()
    ```

!!! warning "Memory Usage"
    Large datasets can consume memory:
    ```python
    # Use lazy loading instead of loading all frames
    for msg in reader.iter_messages(topics=["screen"]):
        if should_process_frame(msg.timestamp):
            frame = msg.decoded.load_frame_array()  # Only when needed
    ```

## Technical Reference

For detailed technical specifications, see:

- **[OEP-0006: OWAMcap Profile Specification](../../oeps/oep-0006.md)** - Authoritative format specification
- **[MCAP Format](https://mcap.dev/)** - Base container format documentation
- **Message Registry** - See `projects/owa-core/owa/core/messages.py` for implementation

### Quick Reference

**OWAMcap Definition:**

- Base format: Standard MCAP container
- Profile: `owa` designation in MCAP metadata
- Schema encoding: JSON Schema
- Message interface: All messages implement `BaseMessage`

**Standard Topics:**

- `screen` → `desktop/ScreenCaptured`
- `mouse` → `desktop/MouseEvent`
- `keyboard` → `desktop/KeyboardEvent`
- `keyboard/state` → `desktop/KeyboardState`
- `mouse/state` → `desktop/MouseState`
- `window` → `desktop/WindowInfo`

## Next Steps

- **[Explore and Edit](how_to_explorer_and_edit.md)**: Learn to work with OWAMcap files
- **[Data Pipeline](owa_data_pipeline.md)**: Process OWAMcap for ML training
- **[Viewer](viewer.md)**: Visualize OWAMcap data interactively
- **[Comparison with LeRobot](comparison_with_lerobot.md)**: See how OWAMcap differs from other formats
