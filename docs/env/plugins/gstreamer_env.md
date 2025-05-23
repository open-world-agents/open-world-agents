# 🎥 GStreamer Environment Plugin (`owa-env-gst`)

## 🔧 Technical Specifications

**`owa-env-gst`** provides high-performance screen capture and recording capabilities powered by:

- **🎬 H.265 (HEVC) Encoding**: Modern video codec for superior compression efficiency
- **⚡ DirectX 11 Acceleration**: Hardware-accelerated capture via `d3d11screencapturesrc`
- **🚀 NVIDIA GPU Optimization**: GPU-based encoding with `nvd3d11h265enc`
- **🎵 AAC Audio Codec**: High-quality audio compression with `avenc_aac`
- **📊 Ultra-Low Latency**: <30ms processing pipeline for real-time applications

### Performance Advantages

- **6x faster** than traditional screen capture methods
- **80%+ compression ratio** with ZSTD in data storage
- **60+ FPS** capture with hardware acceleration
- **Minimal CPU usage** through GPU-accelerated encoding

## 🖥️ Platform Requirements

!!! warning "Windows & NVIDIA Required"
Currently **Windows-only** with **NVIDIA GPU** requirement for optimal performance.

    - ✅ **Windows 10/11** with DirectX 11 support
    - ✅ **NVIDIA GPU** with D3D11 hardware encoding support
    - ✅ **GStreamer 1.24.11+** with GPU plugins
    - 🚧 **AMD/Intel GPU support** in development
    - 🚧 **macOS/Linux** planned for future releases

To see detailed implementation, skim over [owa_env_gst](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst). API Docs is being written WIP.

## Examples

- example of `screen` listener

  ```python
  from owa.core.registry import LISTENERS, activate_module
  import cv2
  import numpy as np

  # Activate the GStreamer module
  activate_module("owa.env.gst")

  # Define a callback to process frames
  def process_frame(frame):
      # Display the frame
      cv2.imshow("Screen Capture", frame.frame_arr)
      cv2.waitKey(1)

  # Create and configure the listener
  screen = LISTENERS["screen"]().configure(
      callback=process_frame,
      fps=30,
      show_cursor=True
  )

  # Run the screen capture
  with screen.session:
      input("Press Enter to stop")
  ```

  For performance metrics:

  ```python
  def process_with_metrics(frame, metrics):
      print(f"FPS: {metrics.fps:.2f}, Latency: {metrics.latency*1000:.2f} ms")
      cv2.imshow("Screen", frame.frame_arr)
      cv2.waitKey(1)

  screen.configure(callback=process_with_metrics)
  ```

- example of `screen_capture` runnable

  ```python
  from owa.core.registry import RUNNABLES, activate_module

  activate_module("owa.env.gst")
  screen_capture = RUNNABLES["screen_capture"]().configure(fps=60)

  with screen_capture.session:
      for _ in range(10):
          frame = screen_capture.grab()
          print(f"Shape: {frame.frame_arr.shape}")
  ```

## ⚠️ Known Issues & Limitations

### Platform Limitations

- **Windows Only**: Currently supports Windows 10/11 exclusively
- **NVIDIA GPU Required**: Optimal performance requires NVIDIA GPU with D3D11 support
- **DirectX 11 Dependency**: Requires DirectX 11 compatible graphics drivers

### Performance Considerations

- When capturing with **WGC** (Windows Graphics Capture API, activated when specifying window handle):
  - Maximum FPS limited by physical monitor refresh rate
  - **Discord & Windows Terminal**: FPS drops to 1-5 when idle, recovers to 60+ during activity
  - This behavior is inherent to WGC API design

### Hardware Requirements

- **Minimum**: NVIDIA GTX 900 series or newer for GPU encoding support
- **Recommended**: NVIDIA RTX series for optimal H.265 encoding performance
- **Memory**: 4GB+ VRAM recommended for high-resolution capture

!!! tip "Performance Optimization. For resource-constrained systems, consider using H.264 encoding instead of H.265 by modifying the pipeline configuration, though this will reduce compression efficiency."

## 🔬 Technical Deep Dive

### GStreamer Pipeline Architecture

The plugin utilizes a sophisticated GStreamer pipeline:

```
d3d11screencapturesrc → d3d11convert → nvd3d11h265enc → h265parse → matroskamux
```

- **`d3d11screencapturesrc`**: DirectX-based screen capture with WGC/DXGI APIs
- **`d3d11convert`**: Format conversion to NV12 for GPU encoding compatibility
- **`nvd3d11h265enc`**: NVIDIA hardware H.265 encoding
- **`h265parse`**: Stream parsing and metadata handling
- **`matroskamux`**: Container muxing for MKV output

### Latency Optimization

The framework maintains sub-30ms latency through:

- **Zero-copy GPU operations** where possible
- **Asynchronous processing** with event-driven architecture
- **Hardware-accelerated encoding** to minimize CPU overhead
- **Optimized buffer management** to reduce memory allocations
