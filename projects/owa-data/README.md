# OWA Data Pipeline

Streamlined data processing pipeline for Vision-Language-Action (VLA) model training with 3x training acceleration.

```
Raw MCAP Data → Event Dataset → FSLDataset → VLA Training Ready
     (1)            (2)           (3)        (tokenization-aware packing)
```

## Quick Start
```bash
# Set variables
export MCAP_TRAIN_DIR="/mnt/raid12/datasets/owa/mcaps/super-hexagon"
export MCAP_TEST_DIR="/mnt/raid12/datasets/owa/mcaps/super-hexagon-30s"
export EVENT_DATASET_DIR="/mnt/raid12/datasets/owa/data/super-hexagon-event-tmp"
export BINNED_DATASET_DIR="/mnt/raid12/datasets/owa/data/super-hexagon-bin-tmp"

# 1. Process MCAP → Event Dataset
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir $MCAP_TRAIN_DIR \
  --test-dir $MCAP_TEST_DIR \
  --output-dir $EVENT_DATASET_DIR \
  --rate screen=20 --rate mouse=60 \
  --keep_topic screen --keep_topic keyboard

# 2. (Optional) Event Dataset → Binned Dataset
python scripts/02_event_dataset_to_binned_dataset.py \
  --input-dir $EVENT_DATASET_DIR \
  --output-dir $BINNED_DATASET_DIR \
  --fps 10 \
  --filter-empty-actions

# 3. Clean Dataset with separate transforms
python -c "
from owa.data.datasets import load_from_disk
dataset = load_from_disk('$EVENT_DATASET_DIR')
print(f'Dataset stage: {dataset.stage}')  # EVENT, BINNED, TOKENIZED, or FSL

# Apply stage-specific transform
dataset.auto_set_transform(stage='event', encoder_type='hierarchical', load_images=True)
for sample in dataset['train'].take(3):
    print(f'{sample=}')
"
python -c "
from owa.data.datasets import load_from_disk
dataset = load_from_disk('$BINNED_DATASET_DIR')
print(f'Dataset stage: {dataset.stage}')  # EVENT, BINNED, TOKENIZED, or FSL

# Apply stage-specific transform
dataset.auto_set_transform(stage='binned', instruction='Complete the computer task')
for sample in dataset['train'].take(3):
    print(f'{sample=}')
"

# 4. FSL (Fixed Sequence Length) approach
python -c "
from owa.data.datasets import load_from_disk, prepare_fsl
from transformers import AutoTokenizer
from owa.data.episode_tokenizer import EpisodeTokenizer

# Load event dataset and tokenize
event_dataset = load_from_disk('$EVENT_DATASET_DIR')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolVLM2-2.2B-Base')
event_tokenizer = EpisodeTokenizer(image_token='<image>')
event_tokenizer.prepare_model(tokenizer=tokenizer)

# Tokenize each split to create TOKENIZED stage datasets
tokenized_train = event_tokenizer.tokenize_event_dataset(event_dataset['train'])

# Create FSL stage dataset
fsl_dataset = prepare_fsl(tokenized_train, max_sequence_length=1024, pad_token_id=tokenizer.pad_token_id)

for sample in fsl_dataset.take(3):
    print(f'{sample=}')
"
```

## Data Processing

### Stage 1: Raw MCAP → Event Dataset

```bash
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir $MCAP_TRAIN_DIR \
  --test-dir $MCAP_TEST_DIR \
  --output-dir $EVENT_DATASET_DIR \
  --rate screen=20 --rate mouse=60 \
  --keep_topic screen --keep_topic keyboard
```

**Schema**: `episode_path` (string), `topic` (string), `timestamp_ns` (int64), `message_type` (string), `mcap_message` (binary)

**Features**: Rate limiting per topic, topic filtering, train/test splitting, preserves raw event data

**Note**: Brand-new, event-oriented format where each row represents a single event

### Stage 2: Event Dataset → Binned Dataset

```bash
python scripts/02_event_dataset_to_binned_dataset.py \
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
from owa.data.datasets import load_from_disk, create_event_transform, create_binned_transform

# Event Dataset Transform
dataset = load_from_disk("/path/to/event/dataset")
transform = create_event_transform(encoder_type="hierarchical", load_images=False)
dataset["train"].set_transform(transform)

# Binned Dataset Transform
dataset = load_from_disk("/path/to/binned/dataset")
transform = create_binned_transform(instruction="Complete the computer task")
dataset["train"].set_transform(transform)
```

## FSL (Fixed Sequence Length) Processing

Core component for Fixed Sequence Length processing that prepares tokenized event data for training with sequence handling, padding, and image loading.

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

To enable logging, set `logger.enable("owa.data.fsl_dataset")` for loguru logger.

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

