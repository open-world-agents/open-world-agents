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

- `dataset_dm_july2021/`: 5501 files, 658GB - scraped from online servers
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
- **RawMouseEvent** messages for mouse movements and clicks (state-change based)
- **KeyboardEvent** messages for key presses and releases (state-change based)
- **WindowInfo** messages for CS:GO window context

**Event Handling**: Uses state-change approach - events only generated when input state changes, not every frame.

## Usage

The script uses Typer CLI with multiple commands. Ensure you have `uv` installed.

### Convert Command
```bash
# Basic conversion
uv run convert_to_owamcap.py convert INPUT_DIR OUTPUT_DIR

# With options
uv run convert_to_owamcap.py convert INPUT_DIR OUTPUT_DIR \
  --max-files 100 \
  --storage-mode external_mkv \
  --workers 8 \
  --subset dm_july2021
```

### Verify Command
```bash
uv run convert_to_owamcap.py verify OUTPUT_DIR
```

### Help
```bash
uv run convert_to_owamcap.py --help
uv run convert_to_owamcap.py convert --help
uv run convert_to_owamcap.py verify --help
```

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
- `keyboard`: Keyboard events (press/release)
- `keyboard/state`: Current keyboard state

## Performance Considerations

### File Sizes
- Original HDF5: ~130MB per file (1000 frames)
- OWAMcap with external MKV: ~5-10MB MCAP + ~15-25MB MKV (recommended)
- OWAMcap with external MP4: ~5-10MB MCAP + ~20-30MB MP4
- OWAMcap with embedded frames: ~100-150MB MCAP (no external files)

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

## Execution Output

```sh
$ uv run convert_to_owamcap.py convert /mnt/raid12/datasets/CounterStrike_Deathmatch /mnt/raid12/datasets/owa/mcaps/csgo --workers 24
Converting 5765 files using 24 workers
Converting files: 100%|████████████████████████████████████████████| 5765/5765 [31:20<00:00, 3.07file/s]

✅ Converted: 5763/5765 files
❌ Failed: 2 files
⏱️  Time: 1880.3s

❌ Failed files:
  hdf5_dm_july2021_expert_90.hdf5: Failed to read HDF5 file (truncated file)
  hdf5_dm_july2021_expert_96.hdf5: Failed to read HDF5 file (truncated file)
```

## References

- [Original Paper](https://arxiv.org/abs/2104.04258)
- [Dataset on HuggingFace](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)
- [OWAMcap Documentation](../../../docs/data/technical-reference/format-guide.md)
- [OWA Project](https://github.com/open-world-agents/open-world-agents)
