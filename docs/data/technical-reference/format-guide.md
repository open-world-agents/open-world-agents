# OWAMcap Format Guide

!!! info "What is OWAMcap?"
    OWAMcap is a specification for using the open-source [MCAP](https://mcap.dev/) container format with Open World Agents (OWA) message definitions. It provides an efficient way to store and process multimodal desktop interaction data including screen captures, mouse events, keyboard events, and window information.

!!! tip "New to OWAMcap?"
    Start with **[Why OWAMcap?](../getting-started/why-owamcap.md)** to understand the problem it solves and why you should use it.

## Table of Contents

- [Getting Started](#getting-started)
    - [Quick Start](#quick-start) - Get started in 3 steps
    - [Core Concepts](#core-concepts) - Essential message types and features
- [Working with OWAMcap](#working-with-owamcap)
    - [Media Handling](#media-handling) - External references and lazy loading
    - [Reading and Writing](#reading-and-writing) - File operations and CLI tools
    - [Storage & Performance](#storage-performance) - Efficiency characteristics
- [Advanced Topics](#advanced-topics)
    - [Extending OWAMcap](#extending-owamcap) - Custom message types and extensibility
    - [Data Pipeline Integration](#data-pipeline-integration) - Real-world integrations
    - [Best Practices](#best-practices) - Performance and organization guidelines
- [Reference](#reference)
    - [Migration & Troubleshooting](#migration-troubleshooting) - Practical help and common issues
    - [Technical Reference](#technical-reference) - Specifications and standards

## Getting Started

### Quick Start

!!! example "Try OWAMcap in 3 Steps"

    **1. Install the packages:**
    ```bash
    pip install mcap-owa-support owa-msgs
    ```

    **2. Explore an example file with the `owl` CLI:**

    !!! info "What is `owl`?"
        `owl` is the command-line interface for OWA tools, installed with `owa-cli`. See the [CLI documentation](../../cli/index.md) for complete usage.

    ```bash
    # Download example file
    wget https://github.com/open-world-agents/open-world-agents/raw/main/docs/data/examples/example.mcap

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

### Core Concepts

OWAMcap combines the robustness of the MCAP container format with OWA's specialized message types for desktop environments, creating a powerful format for recording, analyzing, and training on human-computer interaction data.

#### Key Terms

!!! info "Essential Terminology"
    - **MCAP**: A modular container file format for heterogeneous, timestamped data (like a ZIP file for time-series data). Developed by [Foxglove](https://mcap.dev/), MCAP provides efficient random access, compression, and self-describing schemas. Widely adopted in robotics (ROS ecosystem), autonomous vehicles, and IoT applications for its performance and interoperability.
    - **Topic**: A named channel in MCAP files (e.g., "screen", "mouse") that groups related messages
    - **Lazy Loading**: Loading data only when needed, crucial for memory efficiency with large datasets

#### What Makes a File "OWAMcap"

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
    │                External Media Files (.mkv, .png)            │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
    │  │  Video Frames   │  │  Screenshots    │  │   Audio     │  │
    │  │  - H.265 codec  │  │  - PNG/JPEG     │  │  - Optional │  │
    │  │  - Hardware acc │  │  - Lossless     │  │  - Sync'd   │  │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    ```

=== "Technical Definition"
    - **Base Format**: Standard [MCAP](https://mcap.dev/) container format
    - **Profile**: `owa` designation in MCAP metadata
    - **Schema Encoding**: JSON Schema
    - **Message Interface**: All messages implement `BaseMessage` from `owa.core.message`
    - **Standard Messages**: Core message types from `owa-msgs` package

    !!! info "Why MCAP?"
    
        Built as the successor to ROSBag, MCAP offers efficient storage and retrieval for heterogeneous timestamped data with minimal dependencies. It's designed for modern use cases with optimized random access, built-in compression, and language-agnostic schemas. The format has gained significant adoption across the robotics community, autonomous vehicle companies (Cruise, Waymo), and IoT platforms due to its performance advantages and excellent tooling ecosystem.


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

#### Key Features

<!-- SYNC-ID: owamcap-key-features -->
- 🔄 **Universal Standard**: Unlike fragmented formats, enables seamless dataset combination for large-scale foundation models *(OWAMcap)*
- 🎯 **High-Performance Multimodal Storage**: Lightweight [MCAP](https://mcap.dev/) container with nanosecond precision for synchronized data streams *(MCAP)*
- 🔗 **Flexible MediaRef**: Smart references to both external and embedded media (file paths, URLs, data URIs, video frames) with lazy loading - keeps metadata files small while supporting rich media *(OWAMcap)* → [Learn more](#media-handling)
- 🤗 **Training Pipeline Ready**: Native HuggingFace integration, seamless dataset loading, and direct compatibility with ML frameworks *(Ecosystem)* → [Browse datasets](https://huggingface.co/datasets?other=OWA) | [Data pipeline](https://open-world-agents.github.io/open-world-agents/data/technical-reference/data-pipeline/)
<!-- END-SYNC: owamcap-key-features -->

#### Core Message Types

OWA provides standardized message types through the `owa-msgs` package for consistent desktop interaction recording:

| Message Type | Description |
|--------------|-------------|
| `desktop/KeyboardEvent` | Keyboard press/release events |
| `desktop/KeyboardState` | Current keyboard state |
| `desktop/MouseEvent` | Mouse movement, clicks, scrolls |
| `desktop/MouseState` | Current mouse position and buttons |
| `desktop/RawMouseEvent` | High-definition raw mouse input data |
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

=== "KeyboardState"
    ```python
    class KeyboardState(OWAMessage):
        _type = "desktop/KeyboardState"

        buttons: List[int]  # List of currently pressed virtual key codes

    # Example: No keys currently pressed
    KeyboardState(buttons=[])
    ```

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

=== "RawMouseEvent"
    ```python
    class RawMouseEvent(OWAMessage):
        _type = "desktop/RawMouseEvent"

        us_flags: mouse state flags, containing movement data type (relative/absolute). Default is relative.
        last_x: can be relative or absolute, depends on us_flags
        last_y: can be relative or absolute, depends on us_flags
        button_flags: Raw button state flags from Windows RAWMOUSE structure
        button_data: Additional button data (wheel delta, etc.)
        device_handle: Raw input device handle (optional)
        timestamp: Optional timestamp in nanoseconds since epoch

    # Example: Raw mouse movement
    RawMouseEvent(us_flags=0x0000, last_x=15, last_y=-10, button_flags=0x0000, button_data=0)
    ```

=== "ScreenCaptured"
    ```python
    class ScreenCaptured(OWAMessage):
        _type = "desktop/ScreenCaptured"

        utc_ns: Optional[int] = None                    # System timestamp (nanoseconds)
        source_shape: Optional[Tuple[int, int]] = None  # Original (width, height)
        shape: Optional[Tuple[int, int]] = None         # Current (width, height)
        media_ref: Optional[MediaRef] = None            # URI or file path reference
        frame_arr: Optional[np.ndarray] = None          # In-memory BGRA array (excluded from JSON)
    ```

    !!! tip "Working with ScreenCaptured Messages"
        For detailed information on creating, loading, and working with ScreenCaptured messages, see the **[Media Handling](#media-handling)** section below. It covers MediaRef formats, lazy loading, and practical usage patterns.

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

## Working with OWAMcap

This section covers the essential operations for working with OWAMcap files in your applications. Whether you're processing recorded desktop sessions or creating new datasets, these patterns will help you work efficiently with the format.

### Media Handling

OWAMcap's key advantage is efficient media handling through external media references. Instead of storing large image/video data directly in the MCAP file, OWAMcap stores lightweight references to external media files, keeping the MCAP file small and fast to process.

=== "Creating ScreenCaptured Messages"

    !!! tip "Understanding MediaRef"
        MediaRef is OWAMcap's way of referencing media content. It supports multiple formats:

        - **File paths**: `/absolute/path` or `relative/path`
        - **File URIs**: `file:///path/to/file`
        - **HTTP URLs**: `https://example.com/image.png`
        - **Data URIs**: `data:image/png;base64,...` (embedded content)

        For videos, add `pts_ns` (presentation timestamp) to specify which frame.

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

    !!! info "Why Lazy Loading Matters"
        **Lazy Loading** means frame data is only loaded when you explicitly request it. This is crucial for performance:

        - ✅ **Fast**: Iterate through thousands of messages instantly
        - ✅ **Memory efficient**: Only load frames you actually need
        - ✅ **Scalable**: Work with datasets larger than your RAM

        Without lazy loading, opening a 1-hour recording would try to load ~200GB of frame data into memory!

    ```python
    # IMPORTANT: For MCAP files, resolve relative paths first
    # The OWA recorder saves media paths relative to the MCAP file location
    ScreenCaptured = MESSAGES['desktop/ScreenCaptured']
    screen_msg = ScreenCaptured(
        media_ref={"uri": "relative/video.mkv", "pts_ns": 123456789}
    )

    # Must resolve external paths before loading from MCAP files
    screen_msg.resolve_relative_path("/path/to/data.mcap")

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

        # Write standard mouse event
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


### Storage & Performance

OWAMcap achieves remarkable storage efficiency through external video references and intelligent compression:

#### Compression Benefits

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
    The H.265 settings shown above (keyframe 0.5s, nvd3d11h265enc) are the same as those used by [ocap](../getting-started/recording-data.md) for efficient desktop recording.

**Key advantages:**

- **Lightweight MCAP:** very fast to parse, transfer, and back up  
- **Video Compression:** leverages hardware-accelerated codecs for extreme savings  
- **Selective Loading:** grab only the frames you need without full decompression  
- **Standard Tools:** preview in any video player and edit with off-the-shelf software  


## Advanced Topics

### Extending OWAMcap

#### Custom Message Types

Need to store domain-specific data beyond standard desktop interactions? OWAMcap supports custom message types for sensors, gaming, robotics, and more.

!!! info "Custom Messages Documentation"
    **[📖 Custom Message Types Guide](custom-messages.md)** - Complete guide to creating, registering, and using custom message types in OWAMcap.

    Covers: message creation, package registration, best practices, and CLI integration.

### Data Pipeline Integration

<!-- termynal -->

```
$ # 1. Record desktop interaction
$ ocap my-session.mcap
🎥 Recording desktop interaction...
✓ Saved 1,247 events to my-session.mcap


$ # 2. Process to training format
$ python scripts/01_raw_events_to_event_dataset.py --train-dir ./
🔄 Raw Events to Event Dataset
📁 Loading from: ./
📊 Found 1 train files
---> 100%
✓ Created 1,247 train examples
💾 Saving to ./event-dataset
✓ Saved successfully


$ # 3. Train your model
$ python train.py --dataset ./event-dataset
🚀 Loading dataset...
🏋️ Training desktop agent...
📈 Epoch 1/10: loss=0.234
```

**Pipeline Benefits:**


<!-- SYNC-ID: data-pipeline-benefits -->
- **🔄 Flexible**: Skip binning and use Event Dataset directly, or use traditional Binned Dataset approach
- **💾 Storage Optimized**: Since event/binned dataset saves only reference to media, the entire pipeline is designed to be **space-efficient**.
```sh
/data/
├── mcaps/           # Raw recordings (400MB)
├── event-dataset/   # References only (20MB)
└── binned-dataset/  # Aggregated refs (2MB)
```
- **🤗 Native HuggingFace**: Event/binned dataset is a true HuggingFace `datasets.Dataset` with `set_transform()`, not wrappers.
```py
# Since event/binned datasets are true HuggingFace datasets,
# they can be loaded directly into training pipelines
from datasets import load_from_disk
dataset = load_from_disk("/data/event-dataset")

# Transform to VLA training format is applied on-the-fly during training
from owa.data import create_event_dataset_transform
transform = create_event_dataset_transform(
    encoder_type="hierarchical",
    load_images=True,
    encode_actions=True,
)
dataset.set_transform(transform)

# Use in training
for sample in dataset["train"].take(1):
    print(f"Images: {len(sample['images'])} frames")
    print(f"Actions: {sample['encoded_events'][:3]}...")
    print(f"Instruction: {sample['instruction']}")
```
- **⚡ Compute-optimized, On-the-Fly Processing**: During preprocess stage, media is not loaded. During training, only the required media is loaded on-demand.
```sh
$ python scripts/01_raw_events_to_event_dataset.py
🔄 Raw Events to Event Dataset
📁 Loading from: /data/mcaps/game-session
📊 Found 3 train, 1 test files
---> 100%
✓ Created 24,907 train, 20,471 test examples
💾 Saving to /data/event-dataset
✓ Saved successfully
🎉 Completed in 3.9s (0.1min)
```
<!-- END-SYNC: data-pipeline-benefits -->

!!! tip "Complete Pipeline Documentation"
    See **[🚀 Data Pipeline](data-pipeline.md)** for detailed documentation on each stage, configuration options, and integration with training frameworks.

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

    See [OWA Data Pipeline](data-pipeline.md) for complete pipeline details.

## Reference

### Migration & Troubleshooting

#### File Migration

OWAMcap format evolves over time. When you encounter older files that need updating, use the migration tool:

!!! info "When Do You Need Migration?"
    - **Error messages** about unsupported schema versions
    - **Missing fields** when loading older recordings
    - **Compatibility warnings** from OWA tools
    - **Performance issues** with legacy file formats

**Migration Commands:**

```bash
# Check if migration is needed
owl mcap info old_file.mcap  # Look for version warnings

# Preview what will change (safe, no modifications)
owl mcap migrate run old_file.mcap --dry-run

# Migrate single file (creates backup automatically)
owl mcap migrate run old_file.mcap

# Migrate multiple files in batch
owl mcap migrate run *.mcap

# Migrate with custom output location
owl mcap migrate run old_file.mcap --output new_file.mcap
```

!!! tip "Migration Safety"
    - **Automatic backups**: Original files are preserved as `.backup`
    - **Validation**: Migrated files are automatically validated
    - **Rollback**: Use backup files if migration causes issues

!!! info "Complete Migration Reference"
    For detailed information about all migration commands and options, see the [OWA CLI - MCAP Commands](../../cli/mcap.md) documentation.

#### Common Issues

!!! warning "File Not Found Errors"
    When video files are missing:
    ```python
    # Resolve relative paths
    screen_msg.resolve_relative_path("/path/to/mcap/file.mcap")
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

### Technical Reference

For detailed technical specifications, see:

- **[OEP-0006: OWAMcap Profile Specification](https://github.com/open-world-agents/open-world-agents/blob/main/oeps/oep-0006.md)** - Authoritative format specification
- **[MCAP Format](https://mcap.dev/)** - Base container format documentation
- **Message Registry** - See `projects/owa-core/owa/core/messages.py` for implementation

## Next Steps

- **[Explore and Edit](../getting-started/exploring-data.md)**: Learn to work with OWAMcap files
- **[Data Pipeline](data-pipeline.md)**: Process OWAMcap for ML training
- **[Viewer](../tools/viewer.md)**: Visualize OWAMcap data interactively
- **[Comparison with LeRobot](../tools/comparison-with-lerobot.md)**: See how OWAMcap differs from other formats
