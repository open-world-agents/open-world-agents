# Extract Video and Actions from Bagfiles

This script (`extract_video_actions.py`) extracts RGB_FRONT camera video and CMD_VEL actions from ROS bagfiles with configurable frame rates.
For dataset conversion, we recommand to use "Sketchdrive" docker.

## Features

- **RGB_FRONT Video Extraction**: Extracts front camera images as MP4 video or individual frames
- **CMD_VEL Action Extraction**: Extracts robot control commands (linear.x, angular.z) as numpy arrays
- **Configurable Rates**: Different frame rates for video and actions
- **Auto-detection**: Automatically detects simulation vs real robot bagfiles
- **Parallel Processing**: Uses Ray for efficient multi-core processing
- **Resume Support**: Skips already processed files
- **Comprehensive Output**: Saves video, actions, metadata, and processing logs

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

## Data Formats

### Video Output
- **MP4 format**: H.264 encoded, configurable FPS, optimized for random access
- **Frame format**: Individual PNG files with timestamps
- **Resolution**: Resized to 672x378 pixels (configurable in code)

### Action Output
- **NumPy array**: Shape (N, 3) with columns [timestamp_ns, linear_x, angular_z]
- **CSV format**: Human-readable version with same data
- **Timestamps**: Nanosecond precision ROS timestamps

### Metadata
Each processed bagfile includes metadata with:
- Bagfile type (sim/real)
- Processing parameters (FPS, etc.)
- Number of frames and actions extracted
- Topics used for extraction

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

## Real Example
```bash
python extract_video_actions.py \
    "/mnt/raid12/datasets/sketchdrive/COMMAND/COMMAND/bagfiles" \
    "/mnt/raid12/datasets/sketchdrive/COMMAND/Convert" \
    --video-fps 60 \
    --action-fps 60 \
    --max-workers 64
```

## Troubleshooting

### Common Issues

1. **No images/actions found**: Check if bagfile contains expected topics
2. **Memory issues**: Reduce `--max-workers` or process fewer files at once
3. **Disk space**: Video files can be large, ensure sufficient storage
4. **Permission errors**: Ensure write access to output directory

### Debugging

- Check the log file: `output_dir/extract_video_actions.log`
- Review processing summary: `output_dir/processing_summary.json`
- Examine individual metadata files for detailed information

## Integration

This script can be easily integrated into larger data processing pipelines:

```python
# Use as a module
from extract_video_actions import process_single_bagfile, extract_rgb_front_images

# Process single bagfile
result = process_single_bagfile.remote(bagfile_path, output_dir)

# Extract just images
images = extract_rgb_front_images(bagfile_path, topics_config, fps=15)
```
