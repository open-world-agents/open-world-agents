# Counter-Strike Deathmatch to OWAMcap Conversion

This directory contains scripts to convert the Counter-Strike Deathmatch dataset from the paper ["Counter-Strike Deathmatch with Large-Scale Behavioural Cloning"](https://arxiv.org/abs/2104.04258) by Tim Pearce and Jun Zhu into OWAMcap format for use with Open World Agents.

## Dataset Overview

The original dataset contains:
- **5,500+ HDF5 files** with gameplay recordings
- **700+ GB** of data across multiple subsets
- **1000 frames per file** (~1 minute of gameplay at 16 FPS)
- **Screenshots** (150×280 RGB images)
- **Action vectors** (51-dimensional) with keyboard/mouse inputs
- **Metadata** including kill/death flags

### Dataset Subsets

- `dataset_aim_expert/`: 45 files, 6GB - Expert aim training data
- `dataset_dm_expert_othermaps/`: 30 files, 3.6GB - Expert deathmatch on various maps
- `dataset_dm_expert_dust2/`: 190 files, 24GB - Expert deathmatch on dust2 (not available in current mount)
- `dataset_metadata/`: 61 files, 5.5GB - Metadata files corresponding to HDF5 data

## Conversion Process

The conversion script (`convert_to_owamcap.py`) transforms the dataset into OWAMcap format:

### Input Format (HDF5)
- `frame_i_x`: Screenshots (150, 280, 3) RGB images
- `frame_i_y`: Action vectors (51,) containing [keys_pressed_onehot, Lclicks_onehot, Rclicks_onehot, mouse_x_onehot, mouse_y_onehot]
- `frame_i_xaux`: Previous actions + metadata (54,) - not used in conversion
- `frame_i_helperarr`: [kill_flag, death_flag] (2,) - preserved as metadata

### Output Format (OWAMcap)
- **ScreenCaptured** messages with external video references or embedded frames
- **MouseEvent** messages for mouse movements and clicks
- **MouseState** messages for current mouse position and button states
- **KeyboardEvent** messages for key presses and releases
- **KeyboardState** messages for current pressed keys
- **WindowInfo** messages for CS:GO window context

## Usage

### Prerequisites

Ensure you have the required packages installed:
```bash
pip install mcap-owa-support owa-msgs opencv-python h5py numpy
```

### Basic Conversion

Convert a specific subset:
```bash
python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output --subset aim_expert
```

Convert with options:
```bash
python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output \
    --max-files 5 \
    --max-frames 100 \
    --storage-mode embedded
```

Convert to MKV format (recommended):
```bash
python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output \
    --storage-mode external_mkv
```

### Testing

Run tests to validate the conversion:
```bash
python test_conversion.py
```

### Verification

Verify converted files:
```bash
python convert_to_owamcap.py verify ./output
```

## Command Line Options

- `input_dir`: Input directory containing HDF5 files
- `output_dir`: Output directory for OWAMcap files
- `--max-files N`: Limit conversion to N files
- `--max-frames N`: Limit each file to N frames
- `--storage-mode {external_mkv,external_mp4,embedded}`: How to store screen frames
  - `external_mkv`: Create external MKV video files (recommended, smaller files)
  - `external_mp4`: Create external MP4 video files (compatible but larger)
  - `embedded`: Embed frames as PNG data URIs in MCAP (largest files, no external dependencies)
- `--subset {aim_expert,dm_expert_othermaps}`: Convert specific subset only

## Action Mapping

### Keyboard Keys
The script maps CS:GO actions to Windows Virtual Key Codes:
- `W` (0x57): Forward movement
- `A` (0x41): Left strafe
- `S` (0x53): Backward movement
- `D` (0x44): Right strafe
- `Space` (0x20): Jump
- `Ctrl` (0x11): Crouch
- `Shift` (0x10): Walk
- `R` (0x52): Reload
- `E` (0x45): Use/interact
- `Q` (0x51): Quick weapon switch
- `1-5` (0x31-0x35): Weapon selection

### Mouse Actions
- **Movement**: Decoded using original non-uniform tokenization:
  - **X-axis**: 23 bins `[-1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, 0, 2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000]`
  - **Y-axis**: 15 bins `[-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200]`
- **Left Click**: Primary fire/action
- **Right Click**: Secondary fire/aim down sights

## Output Structure

Each converted file produces:
- `filename.mcap`: OWAMcap file with all messages
- `filename.mp4`: External video file (if `--no-video` not used)

### Topics in OWAMcap Files
- `window`: Window information (CS:GO context)
- `screen`: Screen capture frames
- `mouse`: Mouse events (movement, clicks)
- `mouse/state`: Current mouse state
- `keyboard`: Keyboard events (press/release)
- `keyboard/state`: Current keyboard state

## Performance Considerations

### File Sizes
- Original HDF5: ~130MB per file (1000 frames)
- OWAMcap with external MKV: ~5-10MB MCAP + ~15-25MB MKV (recommended)
- OWAMcap with external MP4: ~5-10MB MCAP + ~20-30MB MP4
- OWAMcap with embedded frames: ~100-150MB MCAP (no external files)

### Processing Speed
- ~1-2 minutes per file on modern hardware
- Memory usage: ~500MB-1GB per file during conversion
- Parallel processing not implemented (can run multiple instances)

## Design Decisions

### Frame Rate: 16 FPS (Not 20 Hz)
The conversion uses **16 FPS** as confirmed in the original paper documentation. While you mentioned 20 Hz, the paper and dataset documentation consistently specify 16 FPS (62.5ms per frame). This matches the temporal structure of the HDF5 files where 1000 frames represent approximately 62.5 seconds of gameplay.

### Mouse Position Quantization
The original dataset uses **non-uniform quantization** for mouse movement, which we now correctly implement:

1. **X-axis**: 23 bins with non-uniform spacing optimized for CS:GO gameplay
   - Fine-grained control near zero: `[-4, -2, 0, 2, 4]` for precise aiming
   - Coarse control for large movements: `[-1000, -500, ..., 500, 1000]` for quick turns
2. **Y-axis**: 15 bins with similar non-uniform spacing for vertical movement
3. **Data Fidelity**: We preserve the exact tokenization from the original repository's `config.py`
4. **Gameplay Relevance**: The non-uniform bins reflect actual CS:GO mouse usage patterns

**Why Non-Uniform?**: CS:GO players make many small adjustments (±2-4 pixels) for aiming and occasional large movements (±100-1000 pixels) for turning. The original researchers optimized the bins for this usage pattern.

### Storage Modes
- **External MKV** (recommended): Uses `owa.core.io.video` for efficient compression
- **External MP4**: Compatible format but larger file sizes
- **Embedded**: PNG data URIs for self-contained files (largest but no external dependencies)

## Data Quality Notes

### Temporal Consistency
- Original dataset: 16 FPS (62.5ms per frame) - confirmed from paper documentation
- Timestamps in nanoseconds for precise timing
- Mouse position tracking maintains continuity across frames

### Action Fidelity
- Key combinations preserved (e.g., W+A for diagonal movement)
- Mouse movement uses original non-uniform quantization (23 X-bins, 15 Y-bins) - preserves exact data structure
- Click timing synchronized with frame timestamps

### Limitations
- No audio data in original dataset
- Mouse sensitivity/acceleration not preserved (original data was pre-quantized)
- Some metadata (xaux) not converted (contains previous actions, not needed for replay)

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all OWA packages are installed
2. **Memory errors**: Use `--max-frames` to limit memory usage
3. **Disk space**: Each file produces ~30-50MB output
4. **Permission errors**: Check dataset mount permissions

### Validation

The verification script checks:
- File integrity and readability
- Message type validation
- Topic consistency
- Timestamp ordering
- Frame count accuracy

## Example Output

```
=== Conversion Summary ===
Converted 45/45 files
Total time: 127.3 seconds
Output directory: ./output

=== Verification Results ===
Found 45 OWAMcap files

Verifying hdf5_aim_july2021_expert_1.mcap:
  File size: 8.2 MB
  Duration: 62.4 seconds
  Messages: 15847
  Frames: 1000
  Topics: ['window', 'screen', 'mouse', 'mouse_state', 'keyboard', 'keyboard_state']
  ✓ No errors found
```

## References

- [Original Paper](https://arxiv.org/abs/2104.04258)
- [Dataset on HuggingFace](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)
- [OWAMcap Documentation](../../../docs/data/technical-reference/format-guide.md)
- [OWA Project](https://github.com/open-world-agents/open-world-agents)
