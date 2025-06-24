# OWA Data Pipeline

A streamlined 3-stage data processing pipeline for Vision-Language-Action (VLA) model training.

## Pipeline Overview

```
Raw MCAP Data → Event Dataset → Binned Dataset → VLA Training Ready
     (1)            (2)            (3)
```

**Key Changes:**
- **Stage 2** now includes `filter_empty_actions` option for better efficiency
- **Stage 3** simplified to use unified `VLADataset` class
- **VLADataset** provides both on-the-fly conversion and pre-converted dataset support

## Stage 1: Raw MCAP Data → Event Dataset

**Script**: `scripts/01_raw_events_to_event_dataset.py`

**Purpose**: Extract and downsample raw events from MCAP files

**Usage**:
```bash
# Filter only screen and keyboard
python scripts/01_raw_events_to_event_dataset.py \
  --train_dir /mnt/raid12/datasets/owa/mcaps/super-hexagon \
  --output-dir /mnt/raid12/datasets/owa/data/super-hexagon-event \
  --rate mouse=60 --rate screen=20 \
  --keep_topic screen --keep_topic keyboard  # Only screen and keyboard
```

**Output Schema**:
```python
{
    "file_path": Value("string"),      # Source MCAP file path
    "topic": Value("string"),          # Event topic (keyboard, mouse, screen)
    "timestamp_ns": Value("int64"),    # Timestamp in nanoseconds
    "message_type": Value("string"),   # Full message type identifier
    "mcap_message": Value("binary"),   # Serialized McapMessage bytes (topic/timestamp_ns/message_type duplicated for preview)
}
```

**Key Features**:
- Rate-limiting per topic (e.g., mouse=60Hz, screen=20Hz)
- Topic filtering (defaults to screen, keyboard, mouse events)
- Automatic train/test splitting
- Preserves raw event data for downstream processing

## Stage 2: Event Dataset → Binned Dataset

**Script**: `scripts/02_event_dataset_to_binned_dataset.py`

**Purpose**: Aggregate events into fixed-rate time bins for uniform temporal sampling

**Usage**:
```bash
python scripts/02_event_dataset_to_binned_dataset.py \
  --input_dir /mnt/raid12/datasets/owa/data/super-hexagon-event \
  --output_dir /mnt/raid12/datasets/owa/data/super-hexagon-bin \
  --fps 10 \
  --filter-empty-actions  # NEW: Filter out bins with no actions
```

**Output Schema**:
```python
{
    "file_path": Value("string"),      # Source MCAP file path
    "bin_idx": Value("int32"),         # Time bin index
    "timestamp_ns": Value("int64"),    # Bin start timestamp
    "state": Sequence(feature=Value("binary"), length=-1),    # Sequence of serialized McapMessage bytes (screen events)
    "actions": Sequence(feature=Value("binary"), length=-1),  # Sequence of serialized McapMessage bytes (action events)
}
```

**Key Features**:
- Fixed-rate temporal binning (e.g., 10 FPS = 100ms bins)
- State-action separation (screen = state, keyboard/mouse = actions)
- **NEW**: Optional filtering of bins with no actions for efficiency
- Preserves temporal structure for sequence modeling

## Stage 3: Binned Dataset → VLA Training Ready

**Script**: `scripts/03_binned_dataset_to_mllm_dataset.py` (simplified)
**Class**: `owa.data.VLADataset` (unified interface)

**Purpose**: Provide unified interface for VLA training with on-the-fly or pre-converted data

**Usage (CLI - for pre-conversion)**:
```bash
python scripts/03_binned_dataset_to_mllm_dataset.py \
  --input_dir /mnt/raid12/datasets/owa/data/super-hexagon-bin \
  --output_dir /mnt/raid12/datasets/owa/data/super-hexagon-mllm \
  --instruction "Complete the computer task" \
  --encoder-type hierarchical
```

**Usage (Python - direct training)**:
```python
from datasets import load_from_disk
from owa.data import VLADataset

# Load binned dataset
binned_dataset = load_from_disk("/path/to/binned/dataset")

# Create VLADataset with on-the-fly conversion
vla_dataset = VLADataset(
    dataset=binned_dataset["train"],
    instruction="Complete the computer task",
    encoder_type="hierarchical",
    cache_samples=True  # Cache for performance
)

# Use directly for training
sample = vla_dataset[0]
print(f"Instruction: {sample['instruction']}")
print(f"Images: {len(sample['images'])} loaded")
print(f"Actions: {len(sample['encoded_events'])} encoded")
```

**Output Schema**:
```python
{
    "instruction": Value("string"),           # Task instruction
    "image_refs": Sequence(feature=Value("binary"), length=-1),  # Sequence of serialized ScreenCaptured bytes
    "encoded_events": Sequence(Value("string")),  # EventEncoder outputs for actions
    "metadata": {                            # Sample metadata
        "file_path": Value("string"),        # Source file
        "bin_idx": Value("int32"),           # Bin index
        "timestamp_ns": Value("int64"),      # Timestamp
        "num_actions": Value("int32"),       # Number of actions
    }
}
```

**Key Features**:
- **Unified Interface**: Single `VLADataset` class handles both binned and pre-converted data
- **On-the-fly Conversion**: No need to pre-convert datasets - convert during training
- **Configurable Encoders**: Support for hierarchical, JSON, and flat event encoders
- **Lazy Image Loading**: Efficient memory usage with on-demand image loading
- **Sample Caching**: Optional caching for improved training performance

**Encoder Types**:
- `hierarchical`: Compositional token structure (default, most efficient)
- `json`: JSON string format with event tokens
- `flat`: Traditional flat token-based encoding

**Migration Notes**:
- `filter_empty_actions` moved to Stage 2 for better efficiency
- On-the-fly conversion eliminates need for Stage 3 file conversion in many cases
- Use `VLADataset` for all new projects - it's the unified solution

## EventEncoder

Converts raw events to text representations for LLM training using `<EVENT_START>` and `<EVENT_END>` tokens.

