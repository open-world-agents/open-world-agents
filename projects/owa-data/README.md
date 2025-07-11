# OWA Data Pipeline

A streamlined data processing pipeline for Vision-Language-Action (VLA) model training.

## Quick Start

```bash
# Set variables
export MCAP_TRAIN_DIR="/mnt/raid12/datasets/owa/mcaps/super-hexagon"
export MCAP_TEST_DIR="/mnt/raid12/datasets/owa/mcaps/super-hexagon-30s"
export EVENT_DATASET_DIR="/mnt/raid12/datasets/owa/data/super-hexagon-event"
export BINNED_DATASET_DIR="/mnt/raid12/datasets/owa/data/super-hexagon-bin"

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

# 3. Dataset Transform approach
python -c "
from datasets import load_from_disk
from owa.data import create_event_dataset_transform
dataset = load_from_disk('$EVENT_DATASET_DIR')
transform = create_event_dataset_transform()
dataset.set_transform(transform)
for sample in dataset['train'].take(2):
    print(f'{sample=}')
"

# 4. FSLDataset approach
python -c "
from datasets import load_from_disk
from transformers import AutoTokenizer
from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset

event_dataset = load_from_disk('$EVENT_DATASET_DIR')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolVLM2-2.2B-Base')
event_tokenizer = EpisodeTokenizer(image_token='<image>')
event_tokenizer.prepare_model(tokenizer=tokenizer)

for split, ds in event_dataset.items():
    event_dataset[split] = event_tokenizer.tokenize_event_dataset(ds)

fsl_dataset = FSLDataset(event_dataset['train'], pad_token_id=tokenizer.pad_token_id, max_sequence_length=1024)
fsl_dataset.prepare()

for sample in fsl_dataset.take(1):
    print(f'{sample=}')
"
```

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

### Key Features

- **Flexible Architecture**: Use Event Dataset directly or create Binned Dataset for uniform sampling
- **On-the-fly Processing**: Dataset transforms apply encoding and image loading during training
- **HuggingFace Integration**: Direct compatibility with `datasets.Dataset.set_transform()`
- **Efficient**: Rate-limiting, topic filtering, and memory-optimized processing
- **Scalable**: Handles large-scale datasets with progress tracking and performance metrics

## Data Processing Stages

### Stage 1: Raw MCAP Data → Event Dataset

Extracts and downsamples raw events from MCAP files into a structured dataset format.

**Script**: `scripts/01_raw_events_to_event_dataset.py`

#### Usage

```bash
# Using the variables from Quick Start section
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir $MCAP_TRAIN_DIR \
  --test-dir $MCAP_TEST_DIR \
  --output-dir $EVENT_DATASET_DIR \
  --rate mouse=60 --rate screen=20 \
  --keep_topic screen --keep_topic keyboard
```

#### Output Schema

```python
{
    "episode_path": Value("string"),      # Source MCAP file path
    "topic": Value("string"),             # Event topic (keyboard, mouse, screen)
    "timestamp_ns": Value("int64"),       # Timestamp in nanoseconds
    "message_type": Value("string"),      # Full message type identifier
    "mcap_message": Value("binary"),      # Serialized McapMessage bytes
}
```

#### Features

- **Rate Limiting**: Control sampling frequency per topic (e.g., `mouse=60Hz`, `screen=20Hz`)
- **Topic Filtering**: Select specific event types (`screen`, `keyboard`, `mouse`)
- **Train/Test Splitting**: Automatic dataset splitting with separate directories
- **Raw Data Preservation**: Maintains original event data for flexible downstream processing

### Stage 2: Event Dataset → Binned Dataset

Aggregates events into fixed-rate time bins for uniform temporal sampling and state-action separation.

**Script**: `scripts/02_event_dataset_to_binned_dataset.py`

#### Usage

```bash
# Using the variables from Quick Start section
python scripts/02_event_dataset_to_binned_dataset.py \
  --input-dir $EVENT_DATASET_DIR \
  --output-dir $BINNED_DATASET_DIR \
  --fps 10 \
  --filter-empty-actions
```

#### Output Schema

```python
{
    "episode_path": Value("string"),      # Source MCAP file path
    "bin_idx": Value("int32"),            # Time bin index
    "timestamp_ns": Value("int64"),       # Bin start timestamp
    "state": Sequence(feature=Value("binary"), length=-1),    # Screen events
    "actions": Sequence(feature=Value("binary"), length=-1),  # Action events
}
```

#### Features

- **Fixed-Rate Binning**: Uniform temporal sampling (e.g., `10 FPS = 100ms bins`)
- **State-Action Separation**: Screen events as state, keyboard/mouse as actions
- **Empty Action Filtering**: Optional removal of bins with no actions for efficiency
- **Temporal Structure**: Preserves sequence relationships for time-series modeling

## Dataset Transforms

Dataset transforms provide on-the-fly conversion to VLA training format using HuggingFace's `set_transform()` method.

### Overview

Transforms work with both Event Dataset and Binned Dataset, providing:

- **Unified Interface**: Same API for both dataset types
- **Flexible Pipeline**: Choose Event Dataset (direct) or Binned Dataset (uniform sampling)
- **On-demand Processing**: Image loading and action encoding during training
- **HuggingFace Integration**: Direct compatibility with training pipelines
- **Multiple Encoders**: Support for hierarchical and JSON encoding formats

### Benefits

| Feature | Event Dataset | Binned Dataset |
|---------|---------------|----------------|
| **Use Case** | Direct event processing | Uniform temporal sampling |
| **Memory** | Lower (event-by-event) | Higher (batched bins) |
| **Temporal Structure** | Variable timing | Fixed intervals |
| **Best For** | Real-time applications | Traditional RL training |

### Event Dataset Transform

Transforms Event Dataset for direct training without binning.

**Function**: `create_event_dataset_transform()`

```python
from datasets import load_from_disk
from owa.data import create_event_dataset_transform

# Load and transform event dataset
event_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")
transform = create_event_dataset_transform(
    encoder_type="hierarchical",    # or "json"
    load_images=True,               # Load PIL images for screen events
    encode_actions=True,            # Encode keyboard/mouse events
)
event_dataset.set_transform(transform)

# Ready for training
for sample in event_dataset["train"].take(5):
    print(f"{sample=}")
```

### Binned Dataset Transform

Transforms Binned Dataset for VLA training with uniform temporal sampling.

**Function**: `create_binned_dataset_transform()`

```python
from datasets import load_from_disk
from owa.data import create_binned_dataset_transform

# Load and transform binned dataset
binned_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-bin")
transform = create_binned_dataset_transform(
    encoder_type="hierarchical",           # or "json"
    instruction="Complete the computer task",
    load_images=True,                      # Load PIL images
    encode_actions=True,                   # Encode actions
)
binned_dataset.set_transform(transform)

# Ready for VLA training
for sample in binned_dataset["train"].take(5):
    print(f"{sample=}")
```

### PyTorch Integration

```python
from datasets import load_from_disk
from torch.utils.data import DataLoader
from owa.data import create_binned_dataset_transform

# Setup dataset with transform
dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-bin")["train"]
transform = create_binned_dataset_transform(
    instruction="Complete the computer task",
    load_images=True,
    encode_actions=True
)
dataset.set_transform(transform)

# Custom collate function for batching
def collate_fn(examples):
    return {
        "images": [ex["images"] for ex in examples],
        "encoded_events": [ex["encoded_events"] for ex in examples],
        "instruction": [ex["instruction"] for ex in examples],
    }

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Training loop
for batch in dataloader:
    images = batch['images']           # List[List[PIL.Image]]
    actions = batch['encoded_events']  # List[List[str]]
    instructions = batch['instruction'] # List[str]
    # ... training code ...
    break
```

## FSLDataset

Core component for Few-Shot Learning that prepares tokenized event data for training with sequence handling, padding, and image loading.

### Goals

1. **Accelerate training**: Packing events into fixed-length sequences for efficient training (maybe 3x acceleration, reported on https://github.com/huggingface/nanoVLM/pull/115)
2. **Context-aware learning**: Provide full context for each event in the sequence

### Design Principles

1. **Tokenization-aware packing**: Uses actual tokenizer to calculate sequence lengths
2. **Lazy image loading**: Images loaded on-the-fly for memory efficiency
3. **Automatic sequence splitting**: Long episodes split across multiple sequences
4. **Episode boundary tokens**: Configurable `<EPISODE_START>` and `<EPISODE_END>` tokens
5. **Enable random access**: Allow starting iteration from any position for sequence packing
6. **Simple implementation**: Clean, readable code with minimal complexity

**Key Insight:** Simply preprocess the event dataset to add tokenization and image processing columns. This enables efficient random access and flexible sequence construction during training.

### Added Columns

FSLDataset adds the following columns to the original event dataset:

| Column | Type | Description |
|--------|------|-------------|
| `token_ids` | `List[int]` | Padded token sequences (length = `max_sequence_length`) |
| `attention_mask` | `List[int]` | Attention masks for padded sequences (1 = real token, 0 = padding) |
| `total_token_count` | `int` | Total number of tokens in the sequence (before padding) |
| `images` | `List[ScreenCaptured \| PIL.Image]` | Images corresponding to `<image>` tokens (type depends on `load_images` config) |

### Complete Example

```python
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset

# Load event dataset
event_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Base")

print("[!] Printing raw event dataset...")
for sample in event_dataset["train"].take(5):
    print(f"{sample=}")

event_tokenizer = EpisodeTokenizer(image_token="<image>")
event_tokenizer.prepare_model(tokenizer=tokenizer)

for split, dataset in event_dataset.items():
    tokenized = event_tokenizer.tokenize_event_dataset(dataset)
    event_dataset[split] = tokenized

print("[!] Printing tokenized event dataset...")
for sample in event_dataset["train"].take(5):
    print(f"{sample=}")


dataset = FSLDataset(event_dataset["train"], pad_token_id=tokenizer.pad_token_id, max_sequence_length=1024)
dataset.prepare()

print("[!] Printing FSL dataset...")
for sample in dataset.take(1):
    print(f"{sample=}")

for sample in tqdm(dataset.take(30)):
    ...
```

### Performance Metrics

```
FSL[30] | Total: 3.2s/s, 3,274t/s, 44.8i/s, 49.5Mb/s | EMA: 3.0s/s, 3,073t/s, 42.0i/s, 46.5Mb/s
```

- **s/s**: Samples per second | **t/s**: Tokens per second | **i/s**: Images per second | **Mb/s**: Megabytes per second | **EMA**: Exponential Moving Average

## API Reference

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **EpisodeTokenizer** | Event → Token conversion | Special tokens, timestamp encoding |
| **FSLDataset** | Training preparation | Padding, attention masks, image loading |
| **EventEncoder** | Event → Text encoding | Hierarchical/JSON formats |



### Encoder Types

| Type | Format | Use Case |
|------|--------|----------|
| **hierarchical** | `<EVENT_START><TIMESTAMP><2><0><4><image>...<EVENT_END>` | Most efficient, compositional |
| **json** | `<EVENT_START>{"type":"screen","timestamp":...}<EVENT_END>` | Human-readable, debugging |