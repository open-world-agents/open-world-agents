# OWA Data Pipeline

A streamlined 2-stage data processing pipeline for Vision-Language-Action (VLA) model training.

## Pipeline Overview

```
Raw MCAP Data → Event Dataset → Binned Dataset → VLA Training Ready
     (1)            (2)            VLADataset
                                (on-the-fly conversion)
```

**Key Features:**
- **Stage 2** includes `filter_empty_actions` option for better efficiency
- **VLADataset** provides on-the-fly conversion from binned datasets to training format

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
  --filter-empty-actions  # Filter out bins with no actions
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
- Optional filtering of bins with no actions for efficiency
- Preserves temporal structure for sequence modeling

## VLA Training with VLADataset

**Class**: `owa.data.VLADataset`

**Purpose**: Direct training interface with on-the-fly conversion from binned datasets

**Usage**:
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

**Key Features**:
- **On-the-fly Conversion**: Convert binned datasets to training format during data loading
- **Configurable Encoders**: Support for hierarchical, JSON, and flat event encoders
- **Lazy Image Loading**: Efficient memory usage with on-demand image loading
- **Sample Caching**: Optional caching for improved training performance

**Encoder Types**:
- `hierarchical`: Compositional token structure (default, most efficient)
- `json`: JSON string format with event tokens
- `flat`: Traditional flat token-based encoding

## EventEncoder

Converts raw events to text representations for LLM training using `<EVENT_START>` and `<EVENT_END>` tokens.

