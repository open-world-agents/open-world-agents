# GStreamer Environment

High-performance screen capture and multimedia processing (**6x faster** than alternatives).

!!! info "Installation"
    ```bash
    pip install owa-env-gst
    # Requires GStreamer dependencies - see installation guide
    ```

!!! warning "Requirements"
    - **OS**: Windows (Linux/macOS support planned)
    - **GPU**: NVIDIA GPU required (our GStreamer implementation is NVIDIA-specific)

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `gst/screen` | Listener | Real-time screen capture with callbacks |
| `gst/screen_capture` | Runnable | On-demand screen capture |
| `gst/omnimodal.appsink_recorder` | Listener | Omnimodal recording with appsink |
| `gst/omnimodal.subprocess_recorder` | Runnable | Omnimodal recording via subprocess |

## Performance

<!-- SYNC-ID: gst-performance-benchmark -->
Powered by GStreamer and Windows API, our implementation is **6x faster** than alternatives:

| **Library** | **Avg. Time per Frame** | **Relative Speed** |
|-------------|------------------------|--------------------|
| **owa.env.gst** | **5.7 ms** | ⚡ **1× (Fastest)** |
| `pyscreenshot` | 33 ms | 🚶‍♂️ 5.8× slower |
| `PIL` | 34 ms | 🚶‍♂️ 6.0× slower |
| `MSS` | 37 ms | 🚶‍♂️ 6.5× slower |
| `PyQt5` | 137 ms | 🐢 24× slower |

📌 **Tested on:** Intel i5-11400, GTX 1650
<!-- END-SYNC: gst-performance-benchmark -->

Not only does `owa.env.gst` **achieve higher FPS**, but it also maintains **lower CPU/GPU usage**, making it the ideal choice for screen recording. Same applies for `ocap`, since it internally imports `owa.env.gst`.

!!! info "Benchmark Details"
    These performance measurements were generated using our comprehensive benchmark script: [`benchmark_screen_captures.py`](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-env-gst/scripts/benchmark_screen_captures.py)

    The script tests multiple screen capture libraries under identical conditions to ensure fair comparison. You can run it yourself to verify performance on your hardware.

## Usage Examples

=== "Real-time Capture"
    ```python
    from owa.core import LISTENERS
    import cv2

    def process_frame(frame):
        cv2.imshow("Screen Capture", frame.frame_arr)
        cv2.waitKey(1)

    screen = LISTENERS["gst/screen"]().configure(
        callback=process_frame,
        fps=60,
        show_cursor=True
    )

    with screen.session:
        input("Press Enter to stop")
    ```

=== "Performance Monitoring"
    ```python
    def process_with_metrics(frame, metrics):
        print(f"FPS: {metrics.fps:.2f}, Latency: {metrics.latency*1000:.2f}ms")
        cv2.imshow("Screen", frame.frame_arr)
        cv2.waitKey(1)

    screen = LISTENERS["gst/screen"]().configure(callback=process_with_metrics)
    ```

=== "On-Demand Capture"
    ```python
    from owa.core import RUNNABLES

    capture = RUNNABLES["gst/screen_capture"]().configure(fps=60)

    with capture.session:
        for i in range(10):
            frame = capture.grab()
            print(f"Frame {i}: {frame.frame_arr.shape}")
    ```

## Known Limitations

!!! warning "Current Limitations"
    - **Windows only** (Linux/macOS support planned)
    - **NVIDIA GPU required** (our GStreamer implementation is NVIDIA-specific)

### Windows Graphics Capture API (WGC) Issues

When capturing some screen with `WGC` (Windows Graphics Capture API, activated when you specify window handle), the following issues are observed:

- **FPS Limitation**: Maximum FPS can't exceed maximum Hz of physical monitor
- **Variable FPS with specific applications**: When capturing `Windows Terminal` and `Discord`, the following behavior was reported:

    - When there's no change in window, FPS drops to 1-5 frames
    - When there's change (e.g. mouse movement) in window, FPS immediately recovers to 60+

This phenomenon is likely due to WGC's optimization behavior.

!!! info "Implementation"
    See [owa-env-gst source](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst) for detailed implementation.

## API Reference

::: gst
    handler: owa