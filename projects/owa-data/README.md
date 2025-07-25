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

**Why needed**: Raw datasets contain binary MCAP messages that need to be converted to training-ready format (text + images).

**What they do**: Apply on-the-fly conversion using HuggingFace's `set_transform()` - decode binary messages, encode events as text, load images as PIL objects.

**Output**: `encoded_event` (text), `image` (PIL.Image or None) for Event Dataset | `images` (List[PIL.Image]), `encoded_events` (List[str]), `instruction` (str) for Binned Dataset

### Event Dataset Transform

```python
from datasets import load_from_disk
from owa.data import create_event_dataset_transform

event_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")
transform = create_event_dataset_transform(encoder_type="hierarchical", load_images=True)
event_dataset.set_transform(transform)

for sample in event_dataset["train"].take(5):
    print(f"{sample=}")
```

### Binned Dataset Transform

```python
from datasets import load_from_disk
from owa.data import create_binned_dataset_transform

binned_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-bin")
transform = create_binned_dataset_transform(instruction="Complete the computer task")
binned_dataset.set_transform(transform)

for sample in binned_dataset["train"].take(5):
    print(f"{sample=}")
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
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.fsl_dataset import FSLDataset

# This line is to enable throughput logging from FSLDataset
logger.enable("owa.data.fsl_dataset")

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

for sample in tqdm(dataset.take(50)):
    ...
```

### Performance Metrics

To enable logging, set `logger.enable("owa.data.fsl_dataset")` for loguru logger.

```
FSL[30] | Total: 3.2s/s, 3,274t/s, 44.8i/s, 49.5Mb/s | EMA: 3.0s/s, 3,073t/s, 42.0i/s, 46.5Mb/s
```

- **s/s**: Samples per second | **t/s**: Tokens per second | **i/s**: Images per second | **Mb/s**: Megabits per second | **EMA**: Exponential Moving Average

## API Reference

| Component | Purpose | Encoder Types |
|-----------|---------|---------------|
| **EpisodeTokenizer** | Event → Token conversion | `hierarchical` (efficient), `json` (readable) |
| **FSLDataset** | Training preparation | Padding, attention masks, image loading |
| **create_event_dataset_transform** | On-the-fly processing | Event dataset → VLA format |
| **create_binned_dataset_transform** | On-the-fly processing | Binned dataset → VLA format |