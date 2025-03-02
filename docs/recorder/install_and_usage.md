# Installation & Usage

This guide will help you install and use the OWA Recorder for high-performance screen recording and event capturing.

## 🖥 Supported OS & HW

- **Windows 10+** (Tier 1): Fully optimized with Direct3D 11 integration.  
    - **GPU:** NVIDIA (supports for w/o NVIDIA GPU is in TODO)  
- **macOS**: Work in progress.  
- **Linux**: Work in progress.

- **⚠️ Recommended Setup:** The load from the recorder is similar to [OBS](https://obsproject.com/) recording. To run games and recording simultaneously, you'll need hardware specifications similar to what would be required when streaming the same game using OBS.

## Installation

### Quick-Start Guide

1. Download `recorder.zip` in [OWA releases](https://github.com/open-world-agents/open-world-agents/releases)
2. unzip `recorder.zip`
3. on `cmd` or `PowerShell`, run `entrypoint.bat --help`. `entrypoint.bat` unzip `env.tar.gz`, which is `conda` env, and run `recorder.py` with given arguments.
    - e.g. `entrypoint.bat output.mkv` is equivalent to `recorder.py output.mkv`
4. It's all!

### Manual Installation Guide

If you have followed [OWA Installation Guide](../install.md), you can install `recorder` very easily by simply running:

```sh
uv pip install -e projects/data_collection
# `pip install -e projects/data_collection` also work, but slower
```

## Usage

The OWA Recorder can be used to capture screen, audio, and various desktop events. Below are the basic usage instructions.

### Basic Command

To start recording, use the following command:
```sh
recorder --help
                                                                                                                                                
 Usage: recorder [OPTIONS] FILE_LOCATION                                                                                                        

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
    recorder output.mkv --record-audio --record-video
    ```

2. **Record a specific window**:
    ```sh
    recorder output.mkv --window-name "My Application"
    ```

3. **Record a specific monitor**:
    ```sh
    recorder output.mkv --monitor-idx 1
    ```

4. **Disable audio recording**:
    ```sh
    recorder output.mkv --no-record-audio
    ```

### Stopping the Recording

To stop the recording, simply press `Ctrl+C`.


## Additional Information

- **Output Files**:
    - For the format of output file, see [Data Format Guide](data_format.md)

- **Performance**:
    - OWA Recorder is optimized for high performance with minimal CPU/GPU usage.
    - It supports high-frequency capture (144+ FPS) and real-time performance with sub-1ms latency.

For more details on the features and performance of OWA Recorder, refer to the [Why use OWA Recorder](why.md) section.

