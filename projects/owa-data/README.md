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
    "msg": Value("string"),            # Serialized message content (JSON string)
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
    "state": Value("binary"),          # Screen event data (latest in bin)
    "actions": Value("binary"),        # List of action events in bin
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
    "state_image_ref": {                     # Single state image reference
        "path": Value("string"),             # MKV file path
        "pts": Value("int64"),               # Presentation timestamp
        "utc_ns": Value("int64"),            # UTC timestamp
        "timestamp_ns": Value("int64"),      # Event timestamp
        "bin_idx": Value("int32"),           # Bin index
    },
    "target_actions": Sequence(Value("string")),  # EventEncoder outputs for actions
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

**Class**: `owa.data.vlm_dataset_builder.VLMDatasetBuilder`

**Purpose**: PyTorch Dataset interface with lazy image loading for efficient training

**Usage**:
```python
from datasets import load_from_disk
from owa.data.vlm_dataset_builder import VLMDatasetBuilder

# Load MLLM dataset
mllm_dataset = load_from_disk('/mnt/raid12/datasets/owa/data/super-hexagon-mllm')

# Create PyTorch dataset with lazy image loading
vlm_dataset = VLMDatasetBuilder(
    mllm_dataset['train'],
    image_format='pil',
    cache_images=True,
    max_cache_size=1000
)

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(vlm_dataset, batch_size=4)
```

**Output Format**:
```python
{
    "instruction": str,                    # Task instruction
    "target_actions": List[str],           # EventEncoder outputs for actions
    "state_image": PIL.Image,              # Lazy-loaded state image from MKV file
    "metadata": Dict                       # Sample metadata
}
```

**Key Features**:
- Lazy image loading from MKV files using single image reference per sample
- Multiple image formats (PIL, tensor, numpy)
- Optional LRU caching for performance
- Proper PyTorch Dataset interface for VLA training

## EventEncoder

Converts raw events to text representations for LLM training using `<EVENT_START>` and `<EVENT_END>` tokens.

## nanoVLM Integration

```python
from data.datasets import OWADataset

owa_dataset = OWADataset(vlm_dataset, tokenizer, image_processor, mp_image_token_length)
dataloader = DataLoader(owa_dataset, batch_size=32, collate_fn=vqa_collator)
```

## Usage

```bash
# Stage 1: Extract events (uses default topics: screen, keyboard, mouse)
python scripts/01_raw_events_to_event_dataset.py \
    --train_dir /data/mcap_files \
    --output-dir /data/event_dataset

# Stage 2: Create bins
python scripts/02_event_dataset_to_binned_dataset.py \
    --input_dir /data/event_dataset \
    --output_dir /data/binned_dataset \
    --fps 10

# Stage 3: Create MLLM dataset
python scripts/03_binned_dataset_to_mllm_dataset.py \
    --input_dir /data/binned_dataset \
    --output_dir /data/mllm_dataset

# Stage 4: Use in training
python load_owa_for_nanovlm.py
```