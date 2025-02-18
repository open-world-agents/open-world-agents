## ⚡ Blazing Performance

OWA Recorder significantly outperforms other Python screen capture libraries in both speed and resource efficiency:

| **Library**         | **Avg. Time per Frame** | **Relative Speed**    |
|---------------------|-------------------------|-----------------------|
| **OWA Recorder**    | **5.7 ms**              | ⚡ **1× (Fastest)**    |
| `pyscreenshot`     | 33 ms                   | 🚶‍♂️ 5.8× slower       |
| `PIL`              | 34 ms                   | 🚶‍♂️ 6.0× slower       |
| `MSS`              | 37 ms                   | 🚶‍♂️ 6.5× slower       |
| `PyQt5`            | 137 ms                  | 🐢 24× slower          |

*Measurement Environment: i5-11400, GTX 1650.*  

OWA Recorder not only delivers higher FPS but also ensures **significantly lower CPU/GPU usage**, making it the optimal choice for high-performance screen recording.

## 🔑 Key Features

- ⚡ **Real-Time Performance**: Experience **sub-1ms latency** in screen capture for seamless real-time applications.
- 🎥 **High-Frequency Capture**: Record screen at **144+ FPS** with **minimal CPU/GPU usage**.
    - Powered by Windows APIs (`DXGI/WGC`) and the robust [GStreamer](https://gstreamer.freedesktop.org/) framework, ensuring superior performance compared to alternatives like `PIL.ImageGrab` and `mss`.

### Supported Desktop Events & Interfaces

- 📺 **Screen Capture**: 
    - Capture entire monitor or specific monitors by index.
    - Specify window name and adjust framerate as needed.
- ⌨️🖱️ **Input Handling**: 
    - Capture and inject keyboard and mouse events effortlessly.
- 🪟 **Window Management**: 
    - Retrieve active window details, including name, bounding box, and handle (`hWnd`).

### ✨ Supported Operating Systems

- **Windows**: Full-featured with optimized Direct3D11 integration.
- **macOS**: Comprehensive support utilizing AVFoundation for efficient screen capture.
- **Linux**: Basic functionality available, with ongoing enhancements.

## 🚀 Why Choose OWA Recorder?

OWA Recorder stands out as the most efficient Python-based screen recorder, offering unmatched performance and ease of use:

- **Simplicity**: 
    - Start recording with a single command: `recorder FILE_LOCATION`.
    - Stop seamlessly using `Ctrl+C`.
- **Optimized Performance**: 
    - Near-zero CPU/GPU load, comparable to commercial screen recording and broadcasting solutions.
    - Leverages advanced Windows APIs and the GStreamer framework for superior efficiency.
- **Comprehensive Recording**:
    - Simultaneously records screen, audio, and timestamps in a single Matroska (`.mkv`) file.
      - Timestamps are embedded as video subtitles.
    - Captures keyboard, mouse, and window events in an `event.jsonl` file for detailed event logging.

For detailed options and configurations, run `recorder --help`!

### 📋 Usage

```bash
Usage: recorder.py [OPTIONS] FILE_LOCATION

Arguments:
  FILE_LOCATION        The output file path with `.mkv` extension. [required]

Options:
    --record-audio / --no-record-audio        Enable or disable audio recording. [default: record-audio]
    --record-video / --no-record-video        Enable or disable video recording. [default: record-video]
    --record-timestamp / --no-record-timestamp  Enable or disable timestamp recording. [default: record-timestamp]
    --window-name TEXT                        Capture a specific window by name (supports substring matches).
    --monitor-idx INTEGER                     Specify the monitor index to capture.
    --help                                    Show this message and exit.
```

---



