# Installation & Usage

This guide will help you install and use the **owl**(Open World agent cLi) for high-performance screen recording and event capturing.

## 🖥 Supported OS & HW

- **Windows 10+** (Tier 1): Fully optimized with Direct3D 11 integration.  
    - **GPU:** NVIDIA (supports for w/o NVIDIA GPU is in TODO)  
- **macOS**: Work in progress.  
- **Linux**: Work in progress.

- **⚠️ Recommended Setup:** The load from the recorder is similar to [OBS](https://obsproject.com/) recording. To run games and recording simultaneously, you'll need hardware specifications similar to what would be required when streaming the same game using OBS.

## Installation

### Quick-Start Guide

1. Download `owl.zip` in [OWA releases](https://github.com/open-world-agents/open-world-agents/releases)
2. unzip `owl.zip`
3. You may choose among 2 options:
    1. double-click `run.bat` on Windows Explorer. It opens up terminal(`cmd`) with virtual environment activated. Run `owl mcap --help` on terminal.
    2. on CLI(`cmd/powershell`), run `run.bat mcap --help`. Note that `run.bat (args)` is equivalent to `owl (args)`.
4. Done!

### Manual Installation Guide

Follow the [OWA Installation Guide](../../install.md), make sure to install GStreamer using conda.

## Usage

The OWA Recorder can be used to capture screen, audio, and various desktop events. Below are the basic usage instructions.

### Basic Command

To start recording, use the following command:
```sh
owl mcap record --help
                                                                                                                                                
 Usage: owl mcap record [OPTIONS] FILE_LOCATION                                                                                                       

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    file_location      TEXT  The location of the output file, use `.mkv` extension. [default: None] [required]                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --record-audio          --no-record-audio                 Whether to record audio [default: record-audio]                                    │
│ --record-video          --no-record-video                 Whether to record video [default: record-video]                                    │
│ --record-timestamp      --no-record-timestamp             Whether to record timestamp [default: record-timestamp]                            │
│ --window-name                                    TEXT     The name of the window to capture, substring of window name is supported           │
│                                                           [default: None]                                                                    │
│ --monitor-idx                                    INTEGER  The index of the monitor to capture [default: None]                                │
│ --install-completion                                      Install completion for the current shell.                                          │
│ --show-completion                                         Show completion for the current shell, to copy it or customize the installation.   │
│ --help                                                    Show this message and exit.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

### Example Usage

1. **Record screen and audio**:
    ```sh
    owl mcap record output.mkv --record-audio --record-video
    ```

2. **Record a specific window**:
    ```sh
    owl mcap record output.mkv --window-name "My Application"
    ```

3. **Record a specific monitor**:
    ```sh
    owl mcap record output.mkv --monitor-idx 1
    ```

4. **Disable audio recording**:
    ```sh
    owl mcap record output.mkv --no-record-audio
    ```

### Stopping the Recording

To stop the recording, simply press `Ctrl+C`.


## Additional Information

- **Output Files**:
    - For the format of output file, see [Data Format Guide](../data_format.md)

- **Performance**:
    - OWA Recorder is optimized for high performance with minimal CPU/GPU usage.
    - It supports high-frequency capture (144+ FPS) and real-time performance with sub-1ms latency.

For more details on the features and performance of OWA Recorder, refer to the [Why use OWA Recorder](why.md) section.

### Real-time latency information

- Currently, `probe` in `appsink_recorder` and appsink callback in screen listener automatically warns you if `latency > 30ms`. And in `recorder` it warns you if it takes `> 20ms` to write a queued events. In common case **you would not see** this warning but if you so, be take care of the system's performance and loads.
- Normally, it takes some time(e.g. `170ms`) for screen record to be started. In other words, first screen record event is recorded after some time.