# Extract Video and Actions from Bagfiles

This script (`extract_video_actions.py`) extracts RGB_FRONT camera video and CMD_VEL actions from ROS bagfiles with configurable frame rates.

## Features

- **RGB_FRONT Video Extraction**: Extracts front camera images as MP4 video or individual frames
- **CMD_VEL Action Extraction**: Extracts robot control commands (linear.x, angular.z) as numpy arrays
- **Joystick Conversion**: Automatically converts CMD_VEL back to joystick input format
- **Isaac_Houndbot Teleop Integration**: Based on actual Isaac_Houndbot teleop implementation
- **Simplified Joystick Format**: Clean 4-axis gamepad layout without unused buttons
- **Configurable Rates**: Different frame rates for video and actions
- **Smart Topic Detection**: Automatically finds correct image topics (excludes camera_info)
- **Auto-detection**: Automatically detects simulation vs real robot bagfiles
- **Parallel Processing**: Uses Ray for efficient multi-core processing
- **Resume Support**: Skips already processed files
- **Robust Error Handling**: Gracefully handles missing topics and corrupted data
- **Movement Pattern Analysis**: Analyzes human control patterns from joystick data
- **Comprehensive Output**: Saves video, actions, joystick data, and processing logs

## Usage

### Basic Usage

```bash
# Extract video at 10 FPS and actions at 20 FPS
python extract_video_actions.py /path/to/bagfiles /path/to/output

# Custom frame rates
python extract_video_actions.py /path/to/bagfiles /path/to/output --video-fps 15 --action-fps 30

# Save as individual frames instead of MP4
python extract_video_actions.py /path/to/bagfiles /path/to/output --save-frames

# Use more parallel workers
python extract_video_actions.py /path/to/bagfiles /path/to/output --max-workers 8

# Process specific bagfile pattern
python extract_video_actions.py /path/to/bagfiles /path/to/output --pattern "*_episode_*.bag"
```

### Command Line Arguments

- `bagfiles_dir`: Directory containing bagfiles
- `output_dir`: Output directory for processed data
- `--video-fps`: Video frame rate (default: 10.0)
- `--action-fps`: Action sampling rate (default: 20.0) 
- `--save-frames`: Save as individual frames instead of MP4
- `--no-resume`: Don't skip already processed files
- `--max-workers`: Maximum number of parallel workers (default: 4)
- `--pattern`: Bagfile pattern to match (default: "*.bag")

## Output Structure

For each bagfile, the script creates a directory with the following structure:

```
output_dir/
├── bagfile1_name/
│   ├── video.mp4              # RGB_FRONT video (or frames/ directory)
│   ├── actions.npy            # Actions as numpy array [timestamp_ns, linear_x, angular_z]
│   ├── actions.csv            # Actions as CSV for easy inspection
│   ├── joystick.json          # Converted joystick commands (NEW!)
│   ├── joystick_stats.json    # Movement pattern analysis (NEW!)
│   └── metadata.json          # Processing metadata
├── bagfile2_name/
│   └── ...
├── extract_video_actions.log  # Processing log
└── processing_summary.json    # Overall processing summary
```

## Supported Bagfile Types

The script automatically detects and supports:

- **Simulation bagfiles**: Topics like `/rgb_front/compressed`, `/cmd_vel`
- **Real robot bagfiles**: Topics like `/usb_cam_front/image_raw/compressed`, `/cmd_vel`

### Smart Topic Detection

The script intelligently finds the correct image topics by:
- **Filtering out camera_info**: Ignores calibration data, only processes actual images
- **Prioritizing front camera**: Looks for topics with "front" in the name first
- **Supporting multiple formats**: Works with both compressed and raw image topics
- **Fallback logic**: Uses any available image topic if front camera not found

### Common Topic Patterns

| Bagfile Type | RGB Topics | CMD Topics |
|--------------|------------|------------|
| Simulation | `/rgb_front/compressed`<br>`/rgb_left/compressed`<br>`/rgb_right/compressed` | `/cmd_vel` |
| Real Robot | `/usb_cam_front/image_raw/compressed`<br>`/usb_cam_left/image_raw/compressed` | `/cmd_vel` |
| Alternative | `/rgb_front/image_raw`<br>`/camera/image_raw/compressed` | `/cmd_vel`<br>`/mobile_base/commands/velocity` |

## Data Formats

### Video Output
- **MP4 format**: H.264 encoded, configurable FPS, optimized for random access
- **Frame format**: Individual PNG files with timestamps
- **Resolution**: Resized to 672x378 pixels (configurable in code)

### Action Output
- **NumPy array**: Shape (N, 3) with columns [timestamp_ns, linear_x, angular_z]
- **CSV format**: Human-readable version with same data
- **Timestamps**: Nanosecond precision ROS timestamps

### Joystick Output (NEW!)
- **JSON format**: Converted joystick commands with simplified 4-axis layout
- **Isaac_Houndbot teleop-based**: Uses actual teleop implementation parameters
- **Realistic velocity scaling**: 2.5 m/s forward, 0.625 m/s backward, 1.4 rad/s angular
- **Movement analysis**: Statistics on control patterns and joystick usage
- **Standard gamepad layout**: Left stick for turning, right stick for movement
- **No buttons**: Buttons removed since they're not used in teleop
- **Status tracking**: STOP/ROTATION/RUN modes matching teleop behavior

### Metadata
Each processed bagfile includes metadata with:
- Bagfile type (sim/real)
- Processing parameters (FPS, etc.)
- Number of frames and actions extracted
- Topics used for extraction
- Joystick conversion status and parameters

### Joystick Data Format (Updated!)
```json
{
  "timestamp": 2245008450420,
  "axes": [0.228, 0.0, 0.0, 0.943],    // [left_x, left_y, right_x, right_y]
  "linear_axis_value": 0.943,          // Forward/backward (-1 to 1)
  "angular_axis_value": 0.228,         // Left/right (-1 to 1)
  "original_linear_x": 2.358,          // Original CMD_VEL linear.x
  "original_angular_z": 0.387,         // Original CMD_VEL angular.z
  "robot_status": "RUN"                // STOP/ROTATION/RUN
}
```

#### Axis Mapping (Standard 4-Axis Gamepad)
- **axes[0]**: Left stick X-axis (angular control - turning left/right)
- **axes[1]**: Left stick Y-axis (unused - always 0.0)
- **axes[2]**: Right stick X-axis (unused - always 0.0)
- **axes[3]**: Right stick Y-axis (linear control - forward/backward)

#### Robot Status
- **STOP**: Robot stationary (both velocities near zero)
- **ROTATION**: Pure turning (only angular velocity)
- **RUN**: Moving forward/backward (with optional turning)

### Movement Pattern Analysis
The joystick statistics include analysis of:
- **Pure movements**: Forward/backward only, turn left/right only
- **Combined movements**: Forward+turn, backward+turn combinations
- **Stationary periods**: Times when robot was not moving
- **Usage statistics**: Min/max/mean joystick values and percentages

## Performance

- **Parallel processing**: Uses Ray for multi-core utilization
- **Memory efficient**: Processes one bagfile at a time per worker
- **Resume capability**: Skips already processed files for interrupted runs
- **Progress tracking**: Real-time progress bars and logging

## Dependencies

Required packages (should be available in your environment):
- `ray` - Parallel processing
- `av` - Video encoding/decoding
- `opencv-python` - Image processing
- `pillow` - Image handling
- `rosbags` - ROS bagfile reading
- `numpy`, `pandas` - Data handling
- `tqdm` - Progress bars
- `loguru` - Logging

## Examples

### Process all bagfiles with high quality video
```bash
python extract_video_actions.py \
    /raid/datasets/sketchdrive/bag_files \
    /output/video_actions \
    --video-fps 30 \
    --action-fps 50 \
    --max-workers 8
```

### Extract frames for computer vision training
```bash
python extract_video_actions.py \
    /path/to/bagfiles \
    /output/frames_and_actions \
    --save-frames \
    --video-fps 20
```

### Quick processing for analysis
```bash
python extract_video_actions.py \
    /path/to/bagfiles \
    /output/quick_analysis \
    --video-fps 5 \
    --action-fps 10 \
    --max-workers 16
```

### Real-world example (COMMAND dataset with Isaac_Houndbot)
```bash
# Convert COMMAND dataset bagfiles (includes automatic joystick conversion)
python extract_video_actions.py \
    "/mnt/raid12/datasets/sketchdrive/COMMAND/COMMAND/bagfiles" \
    "/mnt/raid12/datasets/sketchdrive/COMMAND/Convert" \
    --video-fps 10 \
    --action-fps 20 \
    --max-workers 4

# If video encoding issues occur, use frames instead
python extract_video_actions.py \
    "/mnt/raid12/datasets/sketchdrive/COMMAND/COMMAND/bagfiles" \
    "/mnt/raid12/datasets/sketchdrive/COMMAND/Convert" \
    --video-fps 10 \
    --action-fps 20 \
    --max-workers 4 \
    --save-frames

# Test joystick conversion on existing data
python test_joystick_conversion.py

# Debug individual bagfiles
python debug_bagfile.py
```

## Expected Warnings and Behaviors

### Normal Warnings (These are OK!)

- **"Warning: No CMD_VEL topic found"**: Some bagfiles may not contain robot control commands
  - This is normal for observation-only data or stationary periods
  - Images will still be extracted successfully
  - An empty actions array will be saved
  - No joystick conversion will be performed

- **"Warning: No RGB_FRONT topic found"**: Rare, but some bagfiles may lack camera data
  - Actions will still be extracted if available
  - Check the metadata.json for available topics

- **"Isaac_Houndbot teleop configuration loaded"**: Confirms proper robot configuration
  - Uses actual teleop parameters: 2.5 m/s forward, 0.625 m/s backward, 1.4 rad/s angular
  - Ensures accurate scaling of CMD_VEL to joystick values matching real teleop behavior

### Processing Status

- **"Using RGB topic: /rgb_front/compressed"**: Shows which image topic is being used
- **"Extracting images from bagfile.bag: 785it [00:05, 162.37it/s]"**: Normal extraction progress
- **"Saved video with X frames"**: Successful video creation
- **"Saved X actions"**: Successful action extraction
- **"Converted X actions to joystick format"**: Successful joystick conversion
- **"Joystick conversion not available"**: Missing cmd_vel_to_joystick.py module
- **"Robot status: RUN/ROTATION/STOP"**: Shows movement mode during conversion

## Troubleshooting

### Common Issues

1. **"'float' object has no attribute 'numerator'"**: Video encoding error
   - **Solution**: Use `--save-frames` to save individual images instead of MP4
   - Or restart the script (it will resume from where it left off)

2. **No images/actions found**: Check if bagfile contains expected topics
   - **Solution**: Check metadata.json to see what topics were found
   - The script will list available topics in warnings

3. **Memory issues**: Ray workers consuming too much memory
   - **Solution**: Reduce `--max-workers` (try 2 or 1)
   - Or process smaller batches of bagfiles

4. **Disk space**: Video files can be large, ensure sufficient storage
   - **Solution**: Use `--save-frames` for smaller output size
   - Or reduce `--video-fps` to create smaller videos

5. **Permission errors**: Cannot write to output directory
   - **Solution**: Ensure write access to target directory
   - Check disk space and permissions

### Debugging Tools

- **Check processing summary**: `output_dir/processing_summary.json`
- **Examine individual metadata**: `bagfile_dir/metadata.json` shows:
  - Topics found in bagfile
  - Number of frames/actions extracted
  - Any errors encountered
- **Debug script**: Use `debug_bagfile.py` to inspect individual bagfiles
- **Minimal test**: Use `extract_minimal.py` to test with fewer bagfiles

## Data Quality and Statistics

### Typical Results (COMMAND Dataset with Isaac_Houndbot)
- **Total bagfiles**: 1,388 bagfiles processed
- **Success rate**: ~95% (some bagfiles may lack certain topics)
- **Images per bagfile**: 500-1000 frames (varies by duration and FPS)
- **Actions per bagfile**: 1,000-3,000 commands (varies by robot activity)
- **Joystick conversion**: Automatic for all bagfiles with actions
- **Processing speed**: ~2-4 bagfiles per minute (depends on hardware)
- **Velocity ranges**: Linear 0-2.5 m/s forward, 0-0.625 m/s backward, Angular 0-1.4 rad/s

### Data Quality Indicators
- **Bagfiles with no actions**: Normal for observation-only or stationary periods
- **Bagfiles with no images**: Rare, usually indicates sensor issues
- **Variable frame counts**: Normal, depends on recording duration
- **Empty directories**: Indicates processing errors, check metadata.json
- **Joystick values all zero**: Check deadzone settings or CMD_VEL ranges
- **Movement pattern analysis**: Helps identify control behavior and data quality

## Integration

This script can be easily integrated into larger data processing pipelines:

```python
# Use as a module
from extract_video_actions import process_single_bagfile, extract_rgb_front_images

# Process single bagfile (without Ray)
result = process_single_bagfile._function(
    bagfile=bagfile_path,
    output_dir=output_dir,
    video_fps=10,
    action_fps=20,
    save_as_video=True,
    resume=True
)

# Extract just images
SimTopics, RealTopics = get_topics_config()
topics_config = SimTopics  # or RealTopics
images = extract_rgb_front_images(bagfile_path, topics_config, fps=15)

# Extract just actions
actions = extract_cmd_vel_actions(bagfile_path, topics_config)
```

### Batch Processing Script
```python
# Custom batch processing
import ray
from pathlib import Path
from extract_video_actions import process_single_bagfile

ray.init()

bagfiles = list(Path("/path/to/bagfiles").glob("*.bag"))
tasks = [process_single_bagfile.remote(bf, output_dir) for bf in bagfiles]
results = ray.get(tasks)

successful = [r for r in results if r["status"] == "success"]
print(f"Processed {len(successful)}/{len(bagfiles)} successfully")
```

### Loading and Analyzing Extracted Data
```python
# Load extracted data in Python
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Load actions (CMD_VEL)
actions = np.load('output_dir/bagfile_uuid/actions.npy')
# Shape: (N, 3) - [timestamp_ns, linear_x, angular_z]

# Load as DataFrame for analysis
actions_df = pd.read_csv('output_dir/bagfile_uuid/actions.csv')

# Load joystick data (NEW!)
with open('output_dir/bagfile_uuid/joystick.json') as f:
    joystick_data = json.load(f)

# Load joystick statistics (NEW!)
with open('output_dir/bagfile_uuid/joystick_stats.json') as f:
    joystick_stats = json.load(f)

# Load metadata
with open('output_dir/bagfile_uuid/metadata.json') as f:
    metadata = json.load(f)

print(f"Extracted {metadata['num_frames']} frames and {metadata['num_actions']} actions")
print(f"Joystick conversion: {metadata['has_joystick_conversion']}")

# Analyze movement patterns
if joystick_stats:
    patterns = joystick_stats['movement_patterns']
    total = joystick_stats['total_commands']
    print(f"Movement patterns:")
    for pattern, count in patterns.items():
        if count > 0:
            print(f"  {pattern}: {count} ({count/total*100:.1f}%)")

# Video can be loaded with OpenCV, PIL, or video processing libraries
import cv2
cap = cv2.VideoCapture('output_dir/bagfile_uuid/video.mp4')

# Access joystick axes for each command
for joy_cmd in joystick_data[:5]:  # First 5 commands
    axes = joy_cmd['axes']
    status = joy_cmd['robot_status']
    print(f"Joystick: Turn={axes[0]:.2f}, Move={axes[3]:.2f}, Status={status}")
    print(f"Original CMD_VEL: {joy_cmd['original_linear_x']:.2f}, {joy_cmd['original_angular_z']:.2f}")

# Analyze joystick usage patterns
left_stick_usage = [abs(cmd['axes'][0]) for cmd in joystick_data if abs(cmd['axes'][0]) > 0.01]
right_stick_usage = [abs(cmd['axes'][3]) for cmd in joystick_data if abs(cmd['axes'][3]) > 0.01]

print(f"Left stick (turning) usage: {len(left_stick_usage)} commands")
print(f"Right stick (movement) usage: {len(right_stick_usage)} commands")
```

## Isaac_Houndbot Teleop Integration

The joystick conversion is based on the actual Isaac_Houndbot teleop implementation from:
`/mnt/home/jyjung/sketchdrive/projects/data_processing/issac_sim_teleop_node.py`

### Teleop Parameters (From Actual Implementation):
- **max_forward_lin_vel: 2.5 m/s** - Maximum forward velocity
- **max_backward_lin_vel: 0.625 m/s** - Maximum backward velocity (2.5 * 0.25)
- **max_ang_vel: 1.4 rad/s** - Maximum angular velocity
- **Angular correction: 1.8/1.4** - Applied when moving forward/backward
- **Deadzone: 0.01** - Minimal joystick deadzone for precise control

### Joystick Mapping (Standard 4-Axis Gamepad):
- **Left stick X-axis (axes[0])**: Angular velocity (turning left/right)
- **Left stick Y-axis (axes[1])**: Unused (always 0.0)
- **Right stick X-axis (axes[2])**: Unused (always 0.0)
- **Right stick Y-axis (axes[3])**: Linear velocity (forward/backward)
- **Buttons**: Removed entirely (not used in teleop)

### Status-Based Conversion:
The conversion tracks robot movement status just like the real teleop:
- **STOP → RUN**: When linear velocity becomes non-zero
- **STOP → ROTATION**: When angular velocity becomes non-zero
- **RUN/ROTATION → STOP**: When both velocities return to zero

This ensures that joystick values **exactly match** the human control inputs that would produce the recorded CMD_VEL commands using the actual Isaac_Houndbot teleop system.
