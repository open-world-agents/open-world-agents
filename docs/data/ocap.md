# ocap

Desktop recorder for Windows that captures screen, audio, keyboard, mouse, and window events with high performance.

<!-- ![ocap recording demo](../images/ocap-demo.gif) -->

<video controls>
    <source src="../ocap.mkv" type="video/mp4">
</video>

## What is ocap?

**ocap** (Omnimodal CAPture) captures all essential desktop signals in a synchronized format - screen video, audio, keyboard/mouse input, and window events. Built for the _open-world-agents_ project but suitable for any desktop recording needs.

> **TL;DR**: The most complete, high-performance desktop recording tool for Windows that captures everything in one command.

## Key Features

- **Complete desktop recording**: Video, audio, keyboard/mouse events, window events
- **High performance**: Hardware-accelerated, low resource usage with Windows APIs and [GStreamer](https://gstreamer.freedesktop.org/). Video encoded by [H265/HEVC](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding) ensures both high-quality and low bitrate.
- **Single command operation**: `ocap FILE_LOCATION` (stop with Ctrl+C)
- **Simple architecture**: Core recording logic in a [single Python file with 250 lines](https://github.com/open-world-agents/open-world-agents/blob/main/projects/ocap/owa/ocap/record.py), making it easy to verify and customize
- **Modern data formats**: Video with embedded timestamps in MKV, events in [MCAP format](https://mcap.dev/) for efficient storage, fast querying, and cross-language compatibility

## Getting Started in 30 Seconds

```sh
# Install (Option 1): Download and unzip from releases
# Install (Option 2): conda install owa-ocap

# Start recording your desktop
ocap my-recording

# Stop with Ctrl+C when done
```

Your recording files will be ready to use immediately!

## Feature Comparison

| **Feature**                              | **ocap**                 | [OBS](https://obsproject.com/) | [wcap](https://github.com/mmozeiko/wcap) | [pillow](https://github.com/python-pillow/Pillow)/[mss](https://github.com/BoboTiG/python-mss) |
|------------------------------------------|--------------------------|--------------------------------|------------------------------------------|----------------------------------|
| Advanced data formats (MCAP/MKV)     | ✅ Yes                   | ❌ No                          | ❌ No                                    | ❌ No                            |
| Timestamp aligned logging                | ✅ Yes                   | ❌ No                          | ❌ No                                    | ❌ No                            |
| Customizable event definition & Listener | ✅ Yes                   | ❌ No                          | ❌ No                                    | ❌ No                            |
| Single python file                       | ✅ Yes                   | ❌ No                          | ❌ No                                    | ❌ No                            |
| Audio + Window + Keyboard + Mouse        | ✅ Yes                   | ⚠️ Partial                    | ❌ No                                    | ❌ No                            |
| Hardware-accelerated encoder             | ✅ Yes                   | ✅ Yes                         | ✅ Yes                                   | ❌ No                            |
| Supports latest Windows APIs             | ✅ Yes                   | ✅ Yes                         | ✅ Yes                                   | ❌ No (legacy APIs only)         |
| Optional mouse cursor capture            | ✅ Yes                   | ✅ Yes                         | ✅ Yes                                   | ❌ No                            |

## Technical Architecture

ocap is built on GStreamer for media processing with a clean, maintainable architecture:

```mermaid
flowchart TD
    %% Input Sources
    A[owa.env.desktop] --> B[Keyboard Events]
    A --> C[Mouse Events] 
    A --> D[Window Events]
    E[owa.env.gst] --> F[Screen Capture]
    E --> G[Audio Capture]
    
    %% Core Processing
    B --> H[Event Queue]
    C --> H
    D --> H
    F --> H
    F --> I[Video/Audio Pipeline]
    G --> I
    
    %% Outputs
    H --> J[MCAP Writer]
    I --> K[MKV Pipeline]
    
    %% Files
    J --> L[📄 events.mcap]
    K --> M[🎥 video.mkv]
    
    style A fill:#e1f5fe
    style E fill:#e1f5fe
    style H fill:#fff3e0
    style L fill:#e8f5e8
    style M fill:#e8f5e8
```

- **Easily verifiable and customizable**: Extensive [OWA's Env](../../env) design enables customizable, single [`record.py`](https://github.com/open-world-agents/open-world-agents/blob/main/projects/ocap/owa/ocap/record.py)
- **Native performance**: Direct integration with Windows APIs ([DXGI](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/d3d10-graphics-programming-guide-dxgi)/[WGC](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture?view=winrt-26100) for video, [WASAPI](https://learn.microsoft.com/en-us/windows/win32/coreaudio/wasapi) for audio)

## Installation & Usage

### Supported Systems
- **Windows 10+** (NVIDIA GPUs recommended)
- **Other OS:** macOS/Linux support in progress

### Quick Installation
1. Download `ocap.zip` from [releases](https://github.com/open-world-agents/open-world-agents/releases)
2. Unzip and run either:
   - Double-click `run.bat` (opens terminal with virtual environment)
   - Or in CLI: `run.bat --help`

### Basic Usage

```sh
ocap --help                         # See the help!
ocap FILENAME --window-name "App"   # Record specific window
ocap FILENAME --monitor-idx 1       # Record specific monitor
ocap FILENAME --fps 60              # Set framerate
ocap FILENAME --no-record-audio     # Disable audio
```

Press Ctrl+C to stop recording.

### Output Files
- `.mcap` — Event log (keyboard, mouse, windows)
- `.mkv`  — Video/audio with embedded timestamps

## Common Use Cases

- **AI Training Data Collection**: Record all inputs/outputs for training machine learning models
- **User Experience Research**: Capture detailed user interactions for UX studies
- **Automated Testing**: Record application behavior for regression testing
- **Tutorial Creation**: Create high-quality, synchronized tutorials with all interactions visible

## FAQ

### How much disk space do recordings use?
Approximately 100MB per minute for 1080p recording with H265 encoding.

### Can I customize what events are recorded?
Yes, you can enable/disable audio, keyboard, mouse and window event recording.

### Will ocap slow down my computer?
ocap is designed for minimal performance impact, using hardware acceleration when available.

## When to Use ocap

- **Agent training data collection**: Captures all inputs and outputs
- **Workflow reproducibility**: Records exact steps with timing
- **Performance testing**: Low-overhead recording even during intensive tasks
- **Simple screen recording**: When you need more than just video