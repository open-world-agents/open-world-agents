# OWA Data Pipeline for VLA Training

This document describes the complete 4-stage OWA data pipeline designed for Vision-Language-Action (VLA) model training with nanoVLM integration.

## Pipeline Overview

```
Raw MCAP Data → Event Dataset → Binned Dataset → MLLM Dataset → Training Ready
     (1)            (2)            (3)             (4)
```

The pipeline transforms raw MCAP recordings into training-ready datasets through four distinct stages, each with clear separation of concerns and optimized data formats.

## Stage 1: Raw MCAP Data → Event Dataset

**Script**: `scripts/01_raw_events_to_event_dataset.py`

**Purpose**: Extract and downsample raw events from MCAP files

**Usage**:
```bash
python scripts/01_raw_events_to_event_dataset.py \
  --train_dir /mnt/raid11/datasets/owa/mcaps/super-hexagon \
  --test_dir /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s \
  --output_dir /mnt/raid11/datasets/owa/data/super-hexagon-event \
  --rate mouse=60 screen=20
```

**Output Schema**:
```python
{
    "file_path": Value("string"),      # Source MCAP file path
    "topic": Value("string"),          # Event topic (keyboard, mouse, screen)
    "timestamp_ns": Value("int64"),    # Timestamp in nanoseconds
    "message_type": Value("string"),   # Full message type identifier
    "msg": Value("binary"),            # Serialized message content
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

## EventEncoder Design

**Class**: `owa.data.event_encoder.EventEncoder`

**Purpose**: Convert raw events to text representations for LLM training

**Current Implementation**:
```python
# Raw format (Phase 1)
<EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 1745362786814673800, 'message_type': 'owa.env.desktop.msg.KeyboardEvent', 'msg': b'{"event_type":"press","vk":37}'}<EVENT_END>

# Optimized format (Future Phase 2)
<EVENT_START>08.490,KEYBOARD,(press,37)<EVENT_END>
```

**Features**:
- ✅ Raw format encoding/decoding
- ✅ Multimodal support for ScreenEmitted events
- ✅ Type-safe handling of KeyboardEvent, MouseEvent, ScreenEmitted
- ✅ Integration with HuggingFace datasets
- 🔄 Token-efficient format (planned optimization)

## nanoVLM Integration

The pipeline integrates seamlessly with nanoVLM's OWADataset:

```python
from data.datasets import OWADataset

# VLMDatasetBuilder output is compatible with OWADataset
owa_dataset = OWADataset(
    vlm_dataset,
    tokenizer,
    image_processor,
    mp_image_token_length
)

# Use in training
dataloader = DataLoader(owa_dataset, batch_size=32, collate_fn=vqa_collator)
```

## Design Principles

### 1. **Clear Separation of Concerns**
- Each stage has a single responsibility
- Easy to debug and optimize individual stages
- Modular design allows stage replacement

### 2. **Memory Efficiency**
- Images stored as references, not loaded data
- Lazy loading only when needed for training
- Optional caching with LRU eviction

### 3. **HuggingFace Native**
- All intermediate datasets are HuggingFace compatible
- Easy to save, load, and share datasets
- Built-in support for train/test splits

### 4. **PyTorch Integration**
- VLMDatasetBuilder is a proper PyTorch Dataset
- Works with DataLoader, DistributedSampler, etc.
- Supports multiple image formats

### 5. **Scalability**
- Configurable sequence parameters
- Efficient processing of large datasets
- Parallel processing support

## Implementation Status

### ✅ Completed
- **Stage 1**: Event Dataset extraction (existing)
- **Stage 2**: Binned Dataset creation (redesigned)
- **Stage 3**: MLLM Dataset generation (new)
- **Stage 4**: VLMDatasetBuilder (completely redesigned)
- **EventEncoder**: Raw format implementation
- **Tests**: Updated test suite
- **Documentation**: Comprehensive pipeline documentation

### 🔄 Future Optimizations
- **EventEncoder Phase 2**: Token-efficient serialization
- **Performance**: Benchmarking and optimization
- **Advanced Features**: Custom instruction generation, data augmentation

## File Structure

```
projects/owa-data/
├── scripts/
│   ├── 01_raw_events_to_event_dataset.py    # Stage 1: MCAP → Event Dataset
│   ├── 02_event_dataset_to_binned_dataset.py # Stage 2: Event → Binned Dataset
│   └── 03_binned_dataset_to_mllm_dataset.py  # Stage 3: Binned → MLLM Dataset
├── owa/data/
│   ├── __init__.py                           # Package exports
│   ├── event_encoder.py                     # EventEncoder for text serialization
│   ├── vlm_dataset_builder.py              # Stage 4: PyTorch Dataset interface
│   ├── load_dataset.py                     # HuggingFace dataset utilities
│   └── interval/                            # Time interval utilities
│       ├── __init__.py
│       ├── interval.py
│       └── selector.py
├── tests/
│   ├── test_event_encoder.py               # EventEncoder tests
│   ├── test_vlm_dataset_builder.py         # VLMDatasetBuilder tests
│   ├── test_load_dataset.py                # LoadDataset tests
│   └── test_intervals.py                   # Interval utilities tests
├── example_new_pipeline.py                 # Pipeline demonstration
├── goal.md                                 # This document
├── README.md                               # Project overview
├── README_NEW_PIPELINE.md                  # Detailed pipeline documentation
└── pyproject.toml                          # Project configuration
```

The pipeline is production-ready for VLA model training with OWA data! 🚀