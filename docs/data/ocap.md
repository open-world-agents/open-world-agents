# ocap

High-performance, omnimodal desktop recorder for Windows — capture screen, audio, keyboard, mouse, and window events, all at once.

---

## 🚀 What is ocap?

**ocap** (Omnimodal CAPture) is a powerful, efficient, and easy-to-use recording tool built for the _open-world-agents_ project. It captures every important desktop signal — **screen video, audio, keyboard/mouse input, and window events** — and saves them in a synchronized, analysis-ready format.  

Whether you need precise workflow reproducibility, training data for agents, or simple screen+event capture, `ocap` delivers seamless performance in a lightweight package.

---

## ✅ Key Features

- **🔹 One Command, Everything Logged**
    - Quick-start:  
      ```sh
      ocap record FILE_LOCATION
      ```
      Stop anytime: <kbd>Ctrl</kbd>+<kbd>C</kbd>

- **🖥️ Omnimodal Desktop Recording**
    - Simultaneously saves:
        - **Video, Audio, & Timestamps** in a single `.mkv` file (timestamps embedded as subtitles)
        - **Keyboard, Mouse, Window events** in [mcap format](https://mcap.dev/)
    - [Data format details](data_format.md)

- **🎛️ Highly Flexible Options**
    - Control framerate, select specific windows/monitors, toggle cursor display, etc.

- **⚡ Native Performance**
    - Hardware-accelerated pipelines via DXGI/WGC for video & WASAPI for audio
    - Low CPU/GPU usage, high FPS; ideal even for intensive workflows or games
    - Built on the powerful [GStreamer](https://gstreamer.freedesktop.org/) multimedia framework, enabling both high performance and exceptional flexibility for advanced users and developers.

---

## 📊 Feature Comparison

| **Feature**                           | **ocap** | **[wcap](https://github.com/mmozeiko/wcap)** | **[pillow](https://github.com/python-pillow/Pillow)/[mss](https://github.com/BoboTiG/python-mss)** |
|---------------------------------------|----------|-----------------------------------------------|--------------------------------------------------------------------|
| Screen+audio+events                   | ✅ Yes   | ❌ No                                        | ❌ No                                                             |
| Keyboard/mouse logging                | ✅ Yes   | ❌ No                                        | ❌ No                                                             |
| Window event logging                  | ✅ Yes   | ❌ No                                        | ❌ No                                                             |
| Timestamp (as subtitles)              | ✅ Yes   | ❌ No                                        | ❌ No                                                             |
| Python API support                    | ✅ Yes   | ❌ No                                        | ❌ No                                                             |
| Latest Windows API support            | ✅ Yes   | ✅ Yes                                        | ❌ No (legacy APIs only)                                          |
| Hardware-accelerated encoding         | ✅ Yes   | ✅ Yes                                        | ❌ No                                                             |
| High FPS (>100 FPS)                   | ✅ Yes   | ✅ Yes                                        | ❌ No                                                             |
| Optional mouse cursor                 | ✅ Yes   | ✅ Yes                                        | ❌ No                                                             |


---

## ⚡ Performance Benchmark

ocap consistently **outperforms other screen recorders** written in Python:

| **Library**    | **Avg. Time/Frame** | **Relative Speed**  |
|----------------|---------------------|---------------------|
| **ocap**       | **5.7 ms**          | ⚡ **1× (Fastest)**  |
| pyscreenshot   | 33 ms               | 🚶‍♂️ 5.8× slower     |
| PIL            | 34 ms               | 🚶‍♂️ 6.0× slower     |
| MSS            | 37 ms               | 🚶‍♂️ 6.5× slower     |
| PyQt5          | 137 ms              | 🐢 24× slower       |

📌 **Tested on:** Intel i5-11400, GTX 1650  
ocap achieves **higher FPS and lower CPU/GPU usage**, even in high-load scenarios.

---

## 🖥️ Installation & Usage

### Supported OS & Hardware

- **Windows 10+** (Tier 1): Direct3D 11 integration, fully optimized
    - **GPU:** NVIDIA (other GPU support in progress)
- **Other OS:** _macOS_/**Linux**: Work in progress

**⚠️ System Requirements:**  
Performance demands are similar to [OBS](https://obsproject.com/) — to run ocap alongside games or heavy apps, similar hardware is recommended.

### Quick Installation

1. Download `ocap.zip` from [OWA releases](https://github.com/open-world-agents/open-world-agents/releases)
2. Unzip `ocap.zip`
3. Run in one of two ways:
    - Double-click `run.bat` in Windows Explorer (opens a terminal with virtual environment)
        - Use: `ocap --help`
    - Or in CLI: `run.bat --help` (equivalent to `ocap --help`)
4. Ready!

#### Manual Install

See [Installation Guide](../install.md) — **GStreamer** must be installed (see guide).

---

### Basic Usage

Start screen, audio, and desktop event recording:

```sh
ocap output  # writes output.mcap and output.mkv
```

**Common Options:**
```sh
ocap output.mkv --window-name "App"      # a window by name
ocap output.mkv --monitor-idx 1          # Record a specific monitor
ocap output.mkv --no-record-audio        # Disable audio
ocap output.mkv --fps 144                # Specify frame rate
```

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop.

> See all options:
```sh
ocap --help
```

- `--record-audio` / `--no-record-audio`
- `--record-video` / `--no-record-video`
- `--window-name TEXT`
- `--monitor-idx INTEGER`
- `--show-cursor` / `--no-show-cursor`
- `--width INTEGER`, `--height INTEGER`
- `--additional-args TEXT`
- ...and more!

#### Output

- `.mcap` — Machine-readable event log (keyboard, mouse, windows, etc.)
- `.mkv`  — Video/audio file (with embedded timestamp subtitles!)

For format specs, see [Data Format Guide](data_format.md).

---

## 💡 Tips & Additional Info

- **Performance:** Hardware-accelerated, supports >144 FPS and <1ms event latency for responsive capture.
- **Latency:** Warnings appear if latency exceeds 20–30ms when writing or capturing (should rarely appear unless under very high load).
---

**Made for OWA (open-world-agents), but open for any workflow seeking full-fidelity, synchronized desktop capture.**

---