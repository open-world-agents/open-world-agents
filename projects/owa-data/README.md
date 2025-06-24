# OWA Data Pipeline

A streamlined 2-stage data processing pipeline for Vision-Language-Action (VLA) model training.

## Pipeline Overview

```
Raw MCAP Data → Event Dataset ────────────→ VLA Training Ready
     (1)            (2)      Dataset Transforms
                              (on-the-fly conversion)
                    ↓
               Binned Dataset ────────────→ VLA Training Ready
                              Dataset Transforms
                              (on-the-fly conversion)
```

**Key Features:**
- **Dataset Transforms** work with both Event Dataset and Binned Dataset
- On-the-fly conversion to VLA training format during data loading
- Direct HuggingFace datasets integration with `set_transform()`
- Flexible pipeline: skip binning step and use Event Dataset directly if preferred

## Stage 1: Raw MCAP Data → Event Dataset

**Script**: `scripts/01_raw_events_to_event_dataset.py`

**Purpose**: Extract and downsample raw events from MCAP files

**Usage**:
```bash
# Filter only screen and keyboard
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir /mnt/raid12/datasets/owa/mcaps/super-hexagon \
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
  --input-dir /mnt/raid12/datasets/owa/data/super-hexagon-event \
  --output-dir /mnt/raid12/datasets/owa/data/super-hexagon-bin \
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

## Dataset Transforms

**Purpose**: Apply encoding and image loading transforms directly to HuggingFace datasets using `set_transform`

Dataset transforms provide a unified interface for both Event Dataset and Binned Dataset, allowing you to:
- Use Event Dataset directly for training (skip binning step)
- Use Binned Dataset for training (traditional approach)
- Switch between approaches without changing training code

**Key Benefits**:
- **Unified Interface**: Works with both Event Dataset and Binned Dataset
- **Flexible Pipeline**: Choose your preferred dataset format
- **Better Integration**: Direct HuggingFace datasets integration with training pipelines
- **Efficient**: On-demand image loading and action encoding
- **Configurable**: Support for multiple encoder types (hierarchical, JSON, flat)

### Event Dataset Transform

**Function**: `create_event_dataset_transform()`

**Purpose**: Transform Event Dataset to add encoded events and loaded images

**Usage**:
```python
from datasets import load_from_disk
from owa.data import create_event_dataset_transform

# Load event dataset
event_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")

# Create and apply transform
transform = create_event_dataset_transform(
    encoder_type="hierarchical",
    load_images=True,  # Load images for screen events
    encode_actions=True,  # Encode keyboard/mouse events
)
event_dataset.set_transform(transform)

# Use transformed dataset
for sample in event_dataset["train"].take(10):
    print(f"{sample=}")
```

### Binned Dataset Transform

**Function**: `create_binned_dataset_transform()`

**Purpose**: Transform Binned Dataset to VLA training format for vision-language-action models

**Usage**:
```python
from datasets import load_from_disk
from owa.data import create_binned_dataset_transform

# Load binned dataset
binned_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-bin")

# Create and apply transform
transform = create_binned_dataset_transform(
    encoder_type="hierarchical",
    instruction="Complete the computer task",
    load_images=True,
    encode_actions=True,
)
binned_dataset.set_transform(transform)

# Use transformed dataset for VLA training
for sample in binned_dataset["train"].take(10):
    print(f"{sample=}")
```

### Training Pipeline Integration

**Usage with PyTorch DataLoader**:
```python
from datasets import load_from_disk
from torch.utils.data import DataLoader
from owa.data import create_binned_dataset_transform

# Transform and use with DataLoader
dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-binned")["train"]
transform = create_binned_dataset_transform()
dataset.set_transform(transform)

def collate_fn(examples):
    ret = {
        "images": [example["images"] for example in examples],
        "encoded_events": [example["encoded_events"] for example in examples],
        "instruction": [example["instruction"] for example in examples],
    }
    return ret

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
for batch in dataloader:
    images = batch['images']        # List of List[PIL.Image]
    actions = batch['encoded_events']  # List of List[str]
    instructions = batch['instruction']  # List[str]
```

## EventEncoder

Converts raw events to text representations for LLM training using `<EVENT_START>` and `<EVENT_END>` tokens.

**Encoder Types**:
- `hierarchical`: Compositional token structure (default, most efficient)
- `json`: JSON string format with event tokens
- `flat`: Traditional flat token-based encoding

