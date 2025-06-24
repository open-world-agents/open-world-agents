# OWA Data Pipeline

A 4-stage data processing pipeline for Vision-Language-Action (VLA) model training.

## Pipeline Overview

```
Raw MCAP Data → Event Dataset → Binned Dataset → MLLM Dataset → Training Ready
     (1)            (2)            (3)             (4)
```

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
  --fps 10
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
- Preserves temporal structure for sequence modeling

## Stage 3: Binned Dataset → MLLM Dataset

**Script**: `scripts/03_binned_dataset_to_mllm_dataset.py`

**Purpose**: Convert each bin into one training sample (1:1 conversion)

**Usage**:
```bash
python scripts/03_binned_dataset_to_mllm_dataset.py \
  --input_dir /mnt/raid12/datasets/owa/data/super-hexagon-bin \
  --output_dir /mnt/raid12/datasets/owa/data/super-hexagon-mllm \
  --instruction "Complete the computer task" \
  --filter-empty-actions  # Filter out samples with no actions (default: enabled)
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
- Simple 1:1 bin-to-sample conversion (each bin = one training sample)
- Single state image per sample for clear state-action pairing
- EventEncoder integration for action text serialization
- Configurable filtering of samples with no actions (no-ops)
- Efficient format for VLA training: instruction + state_image → target_actions

## Stage 4: MLLM Dataset → Training Ready

**Class**: `owa.data.owa_dataset.OWADataset`

**Purpose**: PyTorch Dataset interface with lazy image loading for efficient training

**Usage**:
```python
from datasets import load_from_disk

from owa.data import OWADataset

# Load MLLM dataset
dataset_path = "/mnt/raid12/datasets/owa/data/super-hexagon-mllm"
mllm_dataset = load_from_disk(dataset_path)

# Create OWADataset
owa_dataset = OWADataset(mllm_dataset["train"])
print(f"Dataset length: {len(owa_dataset)}")

# Get a sample
sample = owa_dataset[0]
print(f"Instruction: {sample['instruction']}")
print(f"Images: {len(sample['images'])} loaded")
print(f"Encoded events: {len(sample['encoded_events'])} events")

# Show image details
for i, image in enumerate(sample["images"]):
    print(f"  Image {i}: {image=}")

# Show first few events
for i, event in enumerate(sample["encoded_events"][:3]):
    print(f"  Event {i}: {event}")

"""
Dataset length: 3189
Instruction: Complete the computer task
Images: 1 loaded
Encoded events: 1 events
  Image 0: image=<PIL.Image.Image image mode=RGB size=768x480 at 0x7F2F995F9C50>
  Event 0: <EVENT_START><TIMESTAMP><111><KEYBOARD><27><press><EVENT_END>
"""
```

## EventEncoder

Converts raw events to text representations for LLM training using `<EVENT_START>` and `<EVENT_END>` tokens.

