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
python scripts/01_raw_events_to_event_dataset.py \
  --train_dir /mnt/raid11/datasets/owa/mcaps/super-hexagon \
  --test_dir /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s \
  --output-dir /mnt/raid11/datasets/owa/data/super-hexagon-event \
  --rate mouse=60 --rate screen=20
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
- Automatic train/test splitting
- Preserves raw event data for downstream processing

## Stage 2: Event Dataset → Binned Dataset

**Script**: `scripts/02_event_dataset_to_binned_dataset.py`

**Purpose**: Aggregate events into fixed-rate time bins for uniform temporal sampling

**Usage**:
```bash
python scripts/02_event_dataset_to_binned_dataset.py \
  --input_dir /mnt/raid11/datasets/owa/data/super-hexagon-event \
  --output_dir /mnt/raid11/datasets/owa/data/super-hexagon-bin \
  --fps 10 \
  --keep_topic screen --keep_topic keyboard --keep_topic mouse
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

**Purpose**: Create training sequences with image references and encoded events

**Usage**:
```bash
python scripts/03_binned_dataset_to_mllm_dataset.py \
  --input_dir /mnt/raid11/datasets/owa/data/super-hexagon-bin \
  --output_dir /mnt/raid11/datasets/owa/data/super-hexagon-mllm \
  --sequence_length 32 \
  --instruction "Complete the computer task" \
  --overlap_ratio 0.5
```

**Output Schema**:
```python
{
    "instruction": Value("string"),           # Task instruction
    "encoded_events": Sequence(Value("string")),  # EventEncoder outputs
    "image_refs": Sequence({                 # Image references for lazy loading
        "path": Value("string"),             # MKV file path
        "pts": Value("int64"),               # Presentation timestamp
        "utc_ns": Value("int64"),            # UTC timestamp
        "timestamp_ns": Value("int64"),      # Sequence timestamp
        "bin_idx": Value("int32"),           # Bin index
    }),
    "metadata": {                            # Sequence metadata
        "file_path": Value("string"),
        "sequence_idx": Value("int32"),
        "start_bin_idx": Value("int32"),
        "end_bin_idx": Value("int32"),
        "start_timestamp_ns": Value("int64"),
        "end_timestamp_ns": Value("int64"),
        "num_bins": Value("int32"),
        "num_images": Value("int32"),
        "num_actions": Value("int32"),
    }
}
```

**Key Features**:
- Configurable sequence length and overlap
- Image references for memory-efficient lazy loading
- EventEncoder integration for action text serialization
- Rich metadata for analysis and debugging

## Stage 4: MLLM Dataset → Training Ready

**Class**: `owa.data.vlm_dataset_builder.VLMDatasetBuilder`

**Purpose**: PyTorch Dataset interface with lazy image loading for efficient training

**Usage**:
```python
from datasets import load_from_disk
from owa.data.vlm_dataset_builder import VLMDatasetBuilder

# Load MLLM dataset
mllm_dataset = load_from_disk('/mnt/raid11/datasets/owa/data/super-hexagon-mllm')

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
    "encoded_events": List[str],           # EventEncoder outputs
    "images": List[PIL.Image],             # Lazy-loaded images from MKV files
    "metadata": Dict                       # Sequence metadata
}
```

**Key Features**:
- Lazy image loading from MKV files using image references
- Multiple image formats (PIL, tensor, numpy)
- Optional LRU caching for performance
- Proper PyTorch Dataset interface

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
# Stage 1: Extract events
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
    --output_dir /data/mllm_dataset \
    --sequence_length 32

# Stage 4: Use in training
python load_owa_for_nanovlm.py
```