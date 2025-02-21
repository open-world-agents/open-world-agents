# Installation & Usage

This guide will help you install and use the OWA Recorder for high-performance screen recording and event capturing.

## Installation

1. **Install `owa`**: following [OWA's Installation Guide](../install.md), install `owa`.

2. **Install data_collection**:
    ```sh
    python vuv.py pip install -e projects/data_collection
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

