# New OWA Data Pipeline

This document describes the redesigned 4-stage OWA data pipeline that replaces the previous VLMDatasetBuilder implementation.

## Overview

The new pipeline provides clear separation of concerns, memory-efficient processing, and direct integration with nanoVLM training frameworks.

### Pipeline Stages

```
Raw MCAP Data → Event Dataset → Binned Dataset → MLLM Dataset → Training Ready
     (1)            (2)            (3)             (4)
```

## Stage 1: Raw MCAP Data → Event Dataset

**Script**: `01_raw_events_to_event_dataset.py` (existing)

**Purpose**: Extract and downsample raw events from MCAP files

**Input**: MCAP + MKV files
**Output**: HuggingFace Dataset with raw events

**Schema**:
```python
{
    "file_path": str,
    "topic": str, 
    "timestamp_ns": int64,
    "message_type": str,
    "msg": bytes
}
```

**Usage**:
```bash
python scripts/01_raw_events_to_event_dataset.py \
    --train_dir /path/to/mcap_files \
    --output_dir /path/to/event_dataset \
    --rate mouse=60 screen=20
```

## Stage 2: Event Dataset → Binned Dataset

**Script**: `02_event_dataset_to_binned_dataset.py` (redesigned)

**Purpose**: Aggregate events into fixed-rate time bins

**Input**: Event Dataset from Stage 1
**Output**: HuggingFace Dataset with time-binned data

**Schema**:
```python
{
    "file_path": str,
    "bin_idx": int32,
    "timestamp_ns": int64,
    "state": bytes,      # Screen event data
    "actions": bytes     # List of action events in this bin
}
```

**Usage**:
```bash
python scripts/02_event_dataset_to_binned_dataset.py \
    --input_dir /path/to/event_dataset \
    --output_dir /path/to/binned_dataset \
    --fps 10 \
    --keep_topic screen --keep_topic keyboard --keep_topic mouse
```

## Stage 3: Binned Dataset → MLLM Dataset

**Script**: `03_binned_dataset_to_mllm_dataset.py` (new)

**Purpose**: Create training sequences with image references and encoded events

**Input**: Binned Dataset from Stage 2
**Output**: HuggingFace Dataset with MLLM sequences

**Schema**:
```python
{
    "instruction": str,
    "encoded_events": List[str],    # EventEncoder outputs
    "image_refs": List[Dict],       # Image references for lazy loading
    "metadata": Dict                # Sequence metadata
}
```

**Usage**:
```bash
python scripts/03_binned_dataset_to_mllm_dataset.py \
    --input_dir /path/to/binned_dataset \
    --output_dir /path/to/mllm_dataset \
    --sequence_length 32 \
    --instruction "Complete the computer task" \
    --overlap_ratio 0.5
```

## Stage 4: MLLM Dataset → Training Ready

**Class**: `VLMDatasetBuilder` (redesigned)

**Purpose**: PyTorch Dataset with lazy image loading for efficient training

**Input**: MLLM Dataset from Stage 3
**Output**: PyTorch Dataset with loaded images

**Schema**:
```python
{
    "instruction": str,
    "encoded_events": List[str],
    "images": List[PIL.Image],      # Lazy loaded from MKV files
    "metadata": Dict
}
```

**Usage**:
```python
from datasets import load_from_disk
from owa.data import VLMDatasetBuilder

# Load MLLM dataset
mllm_dataset = load_from_disk('/path/to/mllm_dataset')

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

## Integration with nanoVLM

The new VLMDatasetBuilder is fully compatible with the existing nanoVLM OWADataset:

```python
from data.datasets import OWADataset

# The VLMDatasetBuilder output works directly with OWADataset
owa_dataset = OWADataset(
    vlm_dataset,
    tokenizer,
    image_processor,
    mp_image_token_length
)

# Use in training
dataloader = DataLoader(owa_dataset, batch_size=32, collate_fn=vqa_collator)
```

## Key Benefits

### ✅ Clear Separation of Concerns
- Each stage has a single responsibility
- Easy to debug and optimize individual stages
- Modular design allows stage replacement

### ✅ Memory Efficiency
- Images stored as references, not loaded data
- Lazy loading only when needed for training
- Optional caching with LRU eviction

### ✅ Flexible Sequence Generation
- Configurable sequence length and overlap
- Support for different instruction types
- Metadata preservation for analysis

### ✅ HuggingFace Native
- All intermediate datasets are HuggingFace compatible
- Easy to save, load, and share datasets
- Built-in support for train/test splits

### ✅ PyTorch Integration
- VLMDatasetBuilder is a proper PyTorch Dataset
- Works with DataLoader, DistributedSampler, etc.
- Supports multiple image formats (PIL, tensor, numpy)

### ✅ nanoVLM Compatibility
- Direct integration with existing OWADataset
- No changes needed to training scripts
- Maintains conversation format for VLA training

## Migration from Old VLMDatasetBuilder

The old VLMDatasetBuilder has been completely redesigned. Key changes:

1. **No longer processes raw events** - Use the 3-stage pipeline instead
2. **Lazy image loading** - Images loaded on-demand during training
3. **PyTorch Dataset interface** - Proper integration with PyTorch ecosystem
4. **HuggingFace datasets throughout** - Better data management and sharing

## Example Workflow

```bash
# Stage 1: Extract events
python scripts/01_raw_events_to_event_dataset.py \
    --train_dir /data/mcap_files \
    --output_dir /data/event_dataset

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
python train_vlm.py --mllm_dataset /data/mllm_dataset
```

## Files Changed

### New Files
- `scripts/03_binned_dataset_to_mllm_dataset.py` - Stage 3 script
- `example_new_pipeline.py` - Demonstration script
- `README_NEW_PIPELINE.md` - This documentation

### Modified Files
- `scripts/02_event_dataset_to_trajectory_dataset.py` → `scripts/02_event_dataset_to_binned_dataset.py`
- `owa/data/vlm_dataset_builder.py` - Complete redesign
- `tests/test_vlm_dataset_builder.py` - Updated tests

### Removed Files
- `README_VLM_Integration.md` - Replaced by this document
- `example_vlm_integration.py` - Replaced by `example_new_pipeline.py`

The new pipeline is ready for VLA model training! 🚀
