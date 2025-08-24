# OWA Data Pipeline

Streamlined data processing pipeline for Vision-Language-Action (VLA) model training with 3x training acceleration.

```
Raw MCAP Data → Event Dataset → [Path A] FSL Dataset → VLA Training Ready
     (1)            (2)           (02A)      (3)        (tokenization-aware packing)
                               → [Path B] Binned Dataset → Traditional Training
                                 (02B)      (3)           (state-action format)
```

## Quick Start
```bash
# Set variables
export MCAP_TRAIN_DIR="/mnt/raid12/datasets/owa/mcaps/vpt"
export EVENT_DATASET_DIR="/mnt/raid12/datasets/owa/data/vpt-event"
export FSL_DATASET_DIR="/mnt/raid12/datasets/owa/data/vpt-fsl-internvl3"
export BINNED_DATASET_DIR="/mnt/raid12/datasets/owa/data/vpt-bin"

# 1. Process MCAP → Event Dataset
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir $MCAP_TRAIN_DIR \
  --output-dir $EVENT_DATASET_DIR \
  --rate screen=10 --rate mouse/raw=20 \
  --keep-topic screen --keep-topic keyboard --keep-topic mouse/raw \
  --num-workers 4

# 2A. Path A: Event Dataset → FSL Dataset (for transformer training)
python scripts/02A_event_to_fsl.py \
  --config configs/internvl3_example.yaml \
  --input_dir $EVENT_DATASET_DIR \
  --output_dir $FSL_DATASET_DIR \
  --fsl_dataset.max_sequence_length 4096 \
  --fsl_workers 16

# 2B. Path B: Event Dataset → Binned Dataset (for traditional training)
python scripts/02B_event_dataset_to_binned_dataset.py \
  --input-dir $EVENT_DATASET_DIR \
  --output-dir $BINNED_DATASET_DIR \
  --fps 10 \
  --filter-empty-actions

# 3. Use the processed datasets
python -c "
from owa.data.datasets import load_from_disk

# Original: Use Event Dataset (with transforms)
event_dataset = load_from_disk('$EVENT_DATASET_DIR')
print(f'Event Dataset stage: {event_dataset.stage}')  # EVENT

# Apply event transform for on-the-fly processing
event_dataset.auto_set_transform(stage='event', encoder_type='hierarchical', load_images=True)
for sample in event_dataset['train'].take(10):
    print(f'{sample=}')
"

python -c "
from owa.data.datasets import load_from_disk

# Path A: Use FSL Dataset
fsl_dataset = load_from_disk('$FSL_DATASET_DIR')
print(f'FSL Dataset stage: {fsl_dataset.stage}')  # FSL

# Apply FSL transform for on-the-fly processing
fsl_dataset.auto_set_transform(stage='fsl', load_images=True)
for sample in fsl_dataset['train'].take(3):
    print(f'{sample=}')
"

python -c "
from owa.data.datasets import load_from_disk

# Path B: Use Binned Dataset
binned_dataset = load_from_disk('$BINNED_DATASET_DIR')
print(f'Binned Dataset stage: {binned_dataset.stage}')  # BINNED

# Apply stage-specific transform
binned_dataset.auto_set_transform(stage='binned', instruction='Complete the computer task')
for sample in binned_dataset['train'].take(3):
    print(f'{sample=}')
"
```

## Data Processing

### Workflow Overview

After creating event datasets from raw MCAP files, you have two processing paths:

- **Path A (02A)**: Event → FSL Dataset - Recommended for transformer-based VLA training
  - Pre-computes tokenization for 3x training acceleration
  - Handles sequence packing and padding automatically
  - Optimized for modern transformer architectures

- **Path B (02B)**: Event → Binned Dataset - For traditional robotics training
  - Time-binned state-action format
  - Compatible with existing robotics frameworks
  - Similar to OpenX, LeRobot, RLDS formats

### Stage 1: Raw MCAP → Event Dataset

```bash
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir $MCAP_TRAIN_DIR \
  --test-dir $MCAP_TEST_DIR \
  --output-dir $EVENT_DATASET_DIR \
  --rate screen=20 --rate mouse=60 \
  --keep-topic screen --keep-topic keyboard
```

**Schema**: `episode_path` (string), `topic` (string), `timestamp_ns` (int64), `message_type` (string), `mcap_message` (binary)

**Features**: Rate limiting per topic, topic filtering, train/test splitting, preserves raw event data

**Note**: Brand-new, event-oriented format where each row represents a single event

### Stage 2A: Event Dataset → FSL Dataset (Path A)

```bash
python scripts/02A_event_to_fsl.py \
  --input-dir $EVENT_DATASET_DIR \
  --output-dir $FSL_DATASET_DIR \
  --tokenizer "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" \
  --max-sequence-length 1024 \
  --num-proc 32
```

**Schema**: `input_ids` (sequence), `attention_mask` (sequence), `texts` (string), `images` (sequence), `episode_path` (string)

**Features**: Pre-computed tokenization, fixed sequence length padding, episode boundary handling, efficient training

**Note**: Recommended for transformer-based VLA training with 3x acceleration through sequence packing

### Stage 2B: Event Dataset → Binned Dataset (Path B)

```bash
python scripts/02B_event_dataset_to_binned_dataset.py \
  --input-dir $EVENT_DATASET_DIR \
  --output-dir $BINNED_DATASET_DIR \
  --fps 10 \
  --filter-empty-actions
```

**Schema**: `episode_path` (string), `bin_idx` (int32), `timestamp_ns` (int64), `state` (sequence), `actions` (sequence)

**Features**: Fixed-rate binning, state-action separation, empty action filtering, preserves temporal structure

**Note**: Legacy, state-action oriented format similar to conventional datasets like [OpenX](https://robotics-transformer-x.github.io/), [LeRobotDataset](https://github.com/huggingface/lerobot), [RLDS](https://github.com/google-research/rlds)

## Dataset Transforms

Raw datasets contain binary MCAP messages that need conversion to training-ready format (text + images). Transforms apply on-the-fly conversion using HuggingFace's `set_transform()`.

```python
from owa.data.datasets import load_from_disk

# Event Dataset Transform
dataset = load_from_disk("/path/to/event/dataset")
dataset["train"].auto_set_transform(stage="event", encoder_type="hierarchical", load_images=True)

# FSL Dataset Transform (recommended)
dataset = load_from_disk("/path/to/fsl/dataset")
dataset["train"].auto_set_transform(stage="fsl", load_images=True)

# Binned Dataset Transform
dataset = load_from_disk("/path/to/binned/dataset")
dataset["train"].auto_set_transform(stage="binned", instruction="Complete the computer task")
```

## FSL (Fixed Sequence Length) Processing

Core component for Fixed Sequence Length processing that prepares tokenized event data for training with sequence handling, padding, and image loading.

**Quick Start**: Use `scripts/02A_event_to_fsl.py` to convert event datasets directly to FSL format with pre-computed tokenization.

### Goals

1. **Accelerate training**: Packing events into fixed-length sequences for efficient training (3x acceleration, reported in [nanoVLM](https://github.com/huggingface/nanoVLM/pull/115))
2. **Context-aware learning**: Provide full context for each event in the sequence

### Design Principles

1. **Tokenization-aware packing**: Uses actual tokenizer to calculate sequence lengths
2. **Lazy image loading**: Images loaded on-the-fly for memory efficiency
3. **Automatic sequence splitting**: Long episodes split across multiple sequences
4. **Episode boundary tokens**: Configurable `<EPISODE_START>` and `<EPISODE_END>` tokens
5. **Enable random access**: Allow starting iteration from any position for sequence packing
6. **Simple implementation**: Clean, readable code with minimal complexity

### Complete Examples

For complete FSL usage examples, see:

- **Single GPU**: [`scripts/single_shuffle_loader.py`](scripts/single_shuffle_loader.py) - Basic FSL dataset usage with single GPU training
- **Multi GPU**: [`scripts/multi_gpu_loader.py`](scripts/multi_gpu_loader.py) - Distributed FSL dataset usage with multi-GPU training

These scripts demonstrate the full pipeline from event dataset → tokenization → FSL transforms → training-ready data.

### Performance Metrics

To enable logging, set `logger.enable("owa.data.datasets.transforms")` for loguru logger.

```
FSL[30] | Total: 3.2s/s, 3,274t/s, 44.8i/s, 49.5Mb/s | EMA: 3.0s/s, 3,073t/s, 42.0i/s, 46.5Mb/s
```

**Metrics explanation:**
- **s/s**: Samples per second
- **t/s**: Tokens per second
- **i/s**: Images per second
- **Mb/s**: Megabits per second
- **EMA**: Exponential Moving Average

## References

1. **[olmo-core FSLDataset](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/fsl_dataset.py)** - Original FSL implementation for language model training
2. **[nanoVLM Sequence Packing](https://github.com/huggingface/nanoVLM/pull/115)** - 3x training acceleration through sequence packing
3. **[HuggingFace Datasets](https://huggingface.co/docs/datasets/)** - Foundation for dataset handling and transforms
4. **[OpenX Embodied](https://robotics-transformer-x.github.io/)** - Large-scale robotics dataset format
5. **[LeRobot Dataset](https://github.com/huggingface/lerobot)** - Robotics dataset processing pipeline

