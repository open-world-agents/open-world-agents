# Data Conversion Guide : VPT to OWAMcap

This document explains how to convert [Video PreTraining (VPT)](https://github.com/openai/Video-Pre-Training) data format to the Open World Agents MCAP format (OWAMcap).

The full example script can be found [here](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-data/scripts/00_vpt_to_owamcap.py).
The converted dataset is uploaded [in the OWA huggingface repo](https://huggingface.co/datasets/open-world-agents/vpt-owamcap).

## Overview

The VPT dataset consists of paired MP4 video files and JSONL files containing keyboard and mouse actions. The conversion process transforms these into OWAMcap format, which is used for storing multimodal interaction data in Open World Agents.

The conversion script handles:
- Filter validation for 5-minute VPT recordings
- Mapping VPT keyboard actions to OWA virtual key codes
- Converting mouse movements to OWA mouse events
- Synchronizing video frames with input events
- Creating proper timestamps for all events

## Requirements

- VPT dataset with paired MP4 and JSONL files
- OWA environment with `mcap_owa` package installed

## Conversion Process

The conversion involves these key steps:

1. **Validation**: Only JSONL files with exactly 6000 lines (5 minutes of 50ms ticks) are processed
2. **Window Setup**: A virtual window is created with 1280x720 resolution
3. **Input Handling**:
   - Mouse is pinned to center of screen, with relative movements recorded
   - Only navigation-related keyboard inputs are mapped (WASD, Space, Shift, Ctrl)
4. **Timing**: Events are spaced at 50ms intervals, with precise timing for mouse movements

### Key Mapping

The script maps VPT keyboard inputs to OWA virtual key codes:

| VPT Key | OWA Virtual Key |
|---------|----------------|
| key.keyboard.w | VK.KEY_W |
| key.keyboard.a | VK.KEY_A |
| key.keyboard.s | VK.KEY_S |
| key.keyboard.d | VK.KEY_D |
| key.keyboard.space | VK.SPACE |
| key.keyboard.left.shift | VK.LSHIFT |
| key.keyboard.left.control | VK.LCONTROL |

## Usage

1. Set the `VPT_FOLDER_PATH` variable to the location of your VPT dataset
2. Run the script to generate a list of valid VPT files for conversion
3. The script will convert each valid file to OWAMcap format

```python
# Example configuration
VPT_FOLDER_PATH = Path("~/data/Video-Pre-Training/data/").expanduser()
VPT_TARGET_LIST_FILE = "./vpt_target_files.txt"
```

## Example Command

```bash
cd projects/owa-data
python scripts/00_vpt_to_owamcap.py
```

## Output

For each valid VPT file pair (MP4 + JSONL), the script generates a corresponding `.mcap` file containing:
- Window information
- Keyboard events (press/release)
- Mouse events (movement)
- Screen events linking to the original MP4 file

## Limitations

- Only navigation-related keys are mapped (not inventory, hotbar, etc.)
- Mouse is assumed to be pinned to center of screen
- Original VPT timestamps are not used; instead, events are spaced at fixed 50ms intervals

## Technical Details

### Event Timing

- Each tick is 50ms (50,000,000 nanoseconds)
- Mouse pin movements are assumed to take 1ms
- Timestamps start from Unix epoch (0) and increment by tick duration

### Resolution

The VPT dataset uses 1280x720 resolution, which is maintained in the conversion.

## Implementation Details

The conversion script (`00_vpt_to_owamcap.py`) performs the following steps:

1. Generates a list of valid target VPT files that have both MP4 and JSONL components
2. For each file pair, creates an OWAMcap file with proper event timing
3. Converts keyboard and mouse events from VPT format to OWA format
4. Links the video frames to the original MP4 file

### Key Components

```python
# Key constants
VPT_INTERVAL_TICK_NS = 50_000_000  # 50 ms interval per tick
VPT_EXPECTED_TICKS = 6000  # 5 minutes of 50ms ticks
VPT_MOUSE_PIN_NS = 1_000_000  # 1 ms for mouse pin movement
VPT_X_RESOLUTION = 1280
VPT_Y_RESOLUTION = 720
```

The script maintains the keyboard state between ticks to properly generate press and release events, simulating continuous interaction.