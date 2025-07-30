# OWA Data Pipeline: From Raw MCAP to VLA Training

## Quick Demo: 3 Commands to VLA Training

**Step 1: Process raw MCAP files**

<!-- termynal -->

```
$ python scripts/01_raw_events_to_event_dataset.py \
  --train-dir /data/mcaps/game-session \
  --output-dir /data/event-dataset \
  --rate mouse=60 --rate screen=20 \
  --keep-topic screen --keep-topic keyboard
🔄 Raw Events to Event Dataset
📁 Loading from: /data/mcaps/game-session
📊 Found 3 train, 1 test files
---> 100%
✓ Created 24,907 train, 20,471 test examples
💾 Saving to /data/event-dataset
✓ Saved successfully
🎉 Completed in 3.9s (0.1min)
```

**Step 2: Create time bins (optional)**

<!-- termynal -->

```
$ python scripts/02_event_dataset_to_binned_dataset.py \
  --input-dir /data/event-dataset \
  --output-dir /data/binned-dataset \
  --fps 10 \
  --filter-empty-actions
🗂️ Event Dataset to Binned Dataset
📁 Loading from: /data/event-dataset
📊 Found 3 files to process
---> 100%
✓ Created 2,235 binned entries for train split
✓ Created 1,772 binned entries for test split
💾 Saving to /data/binned-dataset
✓ Saved 4,007 total binned entries
🎉 Completed in 4.0s (0.1min)
```

**Step 3: Train your model**

<!-- termynal -->

```
$ python
>>> from datasets import load_from_disk
>>> from owa.data import create_binned_dataset_transform
>>>
>>> # Load and transform dataset
>>> dataset = load_from_disk("/data/binned-dataset")
>>> transform = create_binned_dataset_transform(
...     encoder_type="hierarchical",
...     instruction="Complete the computer task"
... )
>>> dataset.set_transform(transform)
>>>
>>> # Use in training
>>> for sample in dataset["train"].take(1):
...     print(f"Images: {len(sample['images'])} frames")
...     print(f"Actions: {sample['encoded_events'][:3]}...")
...     print(f"Instruction: {sample['instruction']}")
Images: 12 frames
Actions: ['<EVENT_START>mouse_move<EVENT_END>', '<EVENT_START>key_press:w<EVENT_END>', '<EVENT_START>mouse_click:left<EVENT_END>']...
Instruction: Complete the computer task
```

That's it! Your MCAP recordings are now ready for VLA training.

---

The **OWA Data Pipeline** is a streamlined 2-stage processing system that transforms raw MCAP recordings into training-ready datasets for Vision-Language-Action (VLA) models. This pipeline bridges the gap between desktop interaction capture and foundation model training.

## Pipeline Architecture

```mermaid
graph LR
    A[Raw MCAP Files] --> B[Stage 1: Event Dataset]
    B --> C[Stage 2: Binned Dataset]
    B --> D[Dataset Transforms]
    C --> D
    D --> E[VLA Training Ready]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#ffebee
```

**Key Features:**

<!-- SYNC-ID: data-pipeline-benefits -->
- **🔄 Flexible**: Skip binning and use Event Dataset directly, or use traditional Binned Dataset approach
- **💾 Storage Optimized**: Since event/binned dataset saves only reference to media, the entire pipeline is designed to be **space-efficient**.
```sh
/data/
├── mcaps/           # Raw recordings (400MB)
├── event-dataset/   # References only (20MB)
└── binned-dataset/  # Aggregated refs (2MB)
```
- **🤗 Native HuggingFace**: Event/binned dataset is a true HuggingFace `datasets.Dataset` with `set_transform()`, not wrappers.
```py
# Since event/binned datasets are true HuggingFace datasets,
# they can be loaded directly into training pipelines
from datasets import load_from_disk
dataset = load_from_disk("/data/event-dataset")
dataset = load_from_disk("/data/binned-dataset")

# Transform to VLA training format is applied on-the-fly during training
from owa.data import create_binned_dataset_transform
transform = create_binned_dataset_transform(
    encoder_type="hierarchical",
    instruction="Complete the computer task",
)
dataset.set_transform(transform)

# Use in training
for sample in dataset["train"].take(1):
    print(f"Images: {len(sample['images'])} frames")
    print(f"Actions: {sample['encoded_events'][:3]}...")
    print(f"Instruction: {sample['instruction']}")
```
- **⚡ Compute-optimized, On-the-Fly Processing**: During preprocess stage, media is not loaded. During training, only the required media is loaded on-demand.
<!-- termynal -->
```sh
$ python scripts/01_raw_events_to_event_dataset.py
🔄 Raw Events to Event Dataset
📁 Loading from: /data/mcaps/game-session
📊 Found 3 train, 1 test files
---> 100%
✓ Created 24,907 train, 20,471 test examples
💾 Saving to /data/event-dataset
✓ Saved successfully
🎉 Completed in **3.9s** (0.1min)
```
<!-- END-SYNC: data-pipeline-benefits -->

## Stage 1: Raw MCAP → Event Dataset

!!! info "Purpose"
    Extract and downsample raw events from MCAP files while preserving temporal precision and event context.

### Script Usage

```bash
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir /path/to/mcap/files \
  --output-dir /path/to/event/dataset \
  --rate mouse=60 --rate screen=20 \
  --keep-topic screen --keep-topic keyboard
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--train-dir` | Directory containing MCAP files | `/data/recordings/` |
| `--output-dir` | Output directory for Event Dataset | `/data/event-dataset/` |
| `--rate` | Rate limiting per topic (Hz) | `mouse=60 screen=20` |
| `--keep-topic` | Topics to include in dataset | `screen keyboard mouse` |

### Output Schema

The Event Dataset uses a flat structure optimized for temporal queries:

```python
{
    "file_path": Value("string"),      # Source MCAP file path
    "topic": Value("string"),          # Event topic (keyboard, mouse, screen)
    "timestamp_ns": Value("int64"),    # Timestamp in nanoseconds
    "message_type": Value("string"),   # Full message type identifier
    "mcap_message": Value("binary"),   # Serialized McapMessage bytes
}
```

!!! tip "When to Use Event Dataset"
    - **High-frequency training**: When you need precise temporal resolution
    - **Custom binning**: When you want to implement your own temporal aggregation
    - **Event-level analysis**: When studying individual interaction patterns

## Stage 2: Event Dataset → Binned Dataset

!!! info "Purpose"
    Aggregate events into fixed-rate time bins for uniform temporal sampling, separating state (screen) from actions (keyboard/mouse). This format is equivalent to most existing VLA datasets, such as [LeRobotDataset](https://github.com/huggingface/lerobot)

### Script Usage

```bash
python scripts/02_event_dataset_to_binned_dataset.py \
  --input-dir /path/to/event/dataset \
  --output-dir /path/to/binned/dataset \
  --fps 10 \
  --filter-empty-actions
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fps` | Binning frequency (frames per second) | `10` |
| `--filter-empty-actions` | Remove bins with no actions | `False` |
| `--input-dir` | Event Dataset directory | Required |
| `--output-dir` | Output directory for Binned Dataset | Required |

### Output Schema

The Binned Dataset organizes events into temporal bins with state-action separation:

```python
{
    "file_path": Value("string"),      # Source MCAP file path
    "bin_idx": Value("int32"),         # Time bin index
    "timestamp_ns": Value("int64"),    # Bin start timestamp
    "state": Sequence(feature=Value("binary"), length=-1),    # Screen events
    "actions": Sequence(feature=Value("binary"), length=-1),  # Action events
}
```

!!! tip "When to Use Binned Dataset"
    - **Traditional VLA training**: When following established vision-language-action patterns
    - **Fixed-rate processing**: When you need consistent temporal sampling
    - **State-action separation**: When your model expects distinct state and action inputs
    - **Efficient filtering**: When you want to remove inactive periods

## Dataset Transforms: The Magic Layer

Dataset transforms provide the crucial bridge between stored data and training-ready format. They apply **on-demand** during data loading, not during preprocessing.

### Unified Transform Interface

Both Event Dataset and Binned Dataset support the same transform interface:

=== "Event Dataset Transform"

    ```python
    from datasets import load_from_disk
    from owa.data import create_event_dataset_transform
    
    # Load dataset
    dataset = load_from_disk("/path/to/event-dataset")
    
    # Create transform
    transform = create_event_dataset_transform(
        encoder_type="hierarchical",
        load_images=True,
        encode_actions=True,
    )
    
    # Apply transform
    dataset.set_transform(transform)
    
    # Use in training
    for sample in dataset["train"]:
        images = sample["images"]          # List[PIL.Image]
        events = sample["encoded_events"]  # List[str]
    ```

=== "Binned Dataset Transform"

    ```python
    from datasets import load_from_disk
    from owa.data import create_binned_dataset_transform
    
    # Load dataset
    dataset = load_from_disk("/path/to/binned-dataset")
    
    # Create transform
    transform = create_binned_dataset_transform(
        encoder_type="hierarchical",
        instruction="Complete the computer task",
        load_images=True,
        encode_actions=True,
    )
    
    # Apply transform
    dataset.set_transform(transform)
    
    # Use in training
    for sample in dataset["train"]:
        images = sample["images"]          # List[PIL.Image]
        actions = sample["encoded_events"] # List[str]
        instruction = sample["instruction"] # str
    ```

### Transform Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `encoder_type` | Event encoding strategy | `hierarchical`, `json` | `hierarchical` |
| `load_images` | Load screen images | `True`, `False` | `True` |
| `encode_actions` | Encode action events | `True`, `False` | `True` |
| `instruction` | Task instruction (Binned only) | Any string | `"Complete the task"` |

<!-- TODO: fill out encoder types and details -->
<!-- TODO: introduce entire training pipeline somewhere -->

## References

- [Format Guide](format-guide.md) - OWAMcap details
- [Recording Data](../getting-started/recording-data.md) - Create with `ocap`
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) - `datasets` library
