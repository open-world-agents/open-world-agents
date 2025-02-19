# Installation & Usage

This guide will help you install and use the OWA Recorder for high-performance screen recording and event capturing.

## Installation

1. **Install core**:
    - see [here](../index.md)

2. **Install data_collection**:
    ```sh
    python vuv.py pip install -e projects/data_collection
    ```

## Usage

The OWA Recorder can be used to capture screen, audio, and various desktop events. Below are the basic usage instructions.

### Basic Command

To start recording, use the following command:
```sh
recorder output.mkv
```

### Options

- **--record-audio / --no-record-audio**: Enable or disable audio recording. Default is to record audio.
- **--record-video / --no-record-video**: Enable or disable video recording. Default is to record video.
- **--record-timestamp / --no-record-timestamp**: Enable or disable timestamp recording. Default is to record timestamps.
- **--window-name TEXT**: Capture a specific window by name (supports substring matches).
- **--monitor-idx INTEGER**: Specify the monitor index to capture.

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

## Advanced Usage

For more advanced usage and options, refer to the help command:
```sh
recorder --help
```

This will display all available options and their descriptions.

## Additional Information

- **Output Files**:
  - The main recording will be saved as a Matroska (`.mkv`) file.
  - Events such as keyboard, mouse, and window events will be logged in an `event.jsonl` file.

- **Performance**:
  - OWA Recorder is optimized for high performance with minimal CPU/GPU usage.
  - It supports high-frequency capture (144+ FPS) and real-time performance with sub-1ms latency.

For more details on the features and performance of OWA Recorder, refer to the Why use OWA Recorder section.

---

Happy recording!