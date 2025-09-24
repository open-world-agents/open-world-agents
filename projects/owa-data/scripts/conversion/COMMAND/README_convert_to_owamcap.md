# COMMAND Dataset to OWAMcap Conversion

This script converts the COMMAND dataset from the Convert format (video.mp4 + joystick.json) to OWAMcap format using gamepad events.

## Overview

The COMMAND dataset uses gamepad controls instead of keyboard/mouse. This script:

1. Reads video.mp4 and joystick.json from the Convert dataset
2. Converts joystick data to OWA GamepadEvent messages
3. Creates ScreenCaptured events that reference the video
4. Writes everything to OWAMcap format

## Input Format (Convert Dataset)

Each dataset directory should contain:
- `video.mp4`: Front-view video (required)
- `joystick.json`: Gamepad control data with timestamps (optional)
- `metadata.json`: Metadata about the conversion (required)

## Output Format (OWAMcap)

The script generates `.mcap` files with:
- **Window messages**: Window information
- **Screen messages**: Video frame references with timestamps
- **Gamepad messages**: Gamepad axis events (if joystick data available)

## Usage

### Basic Usage

```bash
python convert_to_owamcap.py INPUT_DIR OUTPUT_DIR
```

### Examples

```bash
# Convert all datasets
python convert_to_owamcap.py /mnt/raid12/datasets/sketchdrive/COMMAND/Convert /mnt/raid12/datasets/sketchdrive/COMMAND/OWAMcap

# Convert specific datasets with pattern matching
python convert_to_owamcap.py /path/to/Convert /path/to/OWAMcap --pattern "94f410*"

# Convert only first 10 datasets for testing
python convert_to_owamcap.py /path/to/Convert /path/to/OWAMcap --max-datasets 10
```

### Command Line Options

- `--pattern PATTERN`: Pattern to match dataset directories (default: "*")
- `--max-datasets N`: Maximum number of datasets to process
- `--help`: Show help message

## Gamepad Mapping

The script converts joystick axes to standard gamepad format:

- **Left Stick X** (index 0): Angular control (turning left/right)
- **Left Stick Y** (index 1): Unused
- **Right Stick X** (index 2): Unused  
- **Right Stick Y** (index 3): Linear control (forward/backward)

## Handling Missing Data

The script gracefully handles datasets without joystick data:
- If `joystick.json` is missing, creates video-only OWAMcap files
- Still includes window and screen capture events
- No gamepad events are generated

## Output Verification

You can verify the generated OWAMcap files using:

```bash
# Check file info
owl mcap info output.mcap

# View first few messages
owl mcap cat output.mcap --n 5
```

## Dependencies

- opencv-python (cv2)
- mcap-owa-support
- owa-msgs (gamepad, screen, window messages)

## Notes

- The script preserves original video files by referencing them in MediaRef
- Timestamps from joystick data are used for gamepad events
- Video frame timestamps are calculated based on video FPS
- Only axis changes above deadzone (0.01) generate gamepad events
