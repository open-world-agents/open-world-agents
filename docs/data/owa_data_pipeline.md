# OWA Data Pipeline: From Raw MCAP to VLA Training

## Quick Demo: 3 Commands to VLA Training

**Step 1: Process raw MCAP files**

<!-- termynal -->

```
$ python scripts/01_raw_events_to_event_dataset.py \
  --train-dir /data/mcaps/game-session \
  --output-dir /data/event-dataset \
  --rate mouse=60 --rate screen=20 \
  --keep_topic screen --keep_topic keyboard
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

**Key Design Principles:**

- **🔄 Flexible Pipeline**: Skip binning and use Event Dataset directly, or use traditional Binned Dataset approach
- **🚀 On-the-fly Processing**: Dataset transforms apply encoding and image loading during training, not preprocessing
- **🤗 HuggingFace Integration**: Direct compatibility with `datasets.Dataset.set_transform()`
- **⚡ Performance Optimized**: Efficient data loading with lazy image loading and configurable encoding

## Stage 1: Raw MCAP → Event Dataset

!!! info "Purpose"
    Extract and downsample raw events from MCAP files while preserving temporal precision and event context.

### Script Usage

```bash
python scripts/01_raw_events_to_event_dataset.py \
  --train-dir /path/to/mcap/files \
  --output-dir /path/to/event/dataset \
  --rate mouse=60 --rate screen=20 \
  --keep_topic screen --keep_topic keyboard
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--train-dir` | Directory containing MCAP files | `/data/recordings/` |
| `--output-dir` | Output directory for Event Dataset | `/data/event-dataset/` |
| `--rate` | Rate limiting per topic (Hz) | `mouse=60 screen=20` |
| `--keep_topic` | Topics to include in dataset | `screen keyboard mouse` |

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
    - **Minimal preprocessing**: When you prefer on-the-fly processing over pre-computed bins

## Stage 2: Event Dataset → Binned Dataset

!!! info "Purpose"
    Aggregate events into fixed-rate time bins for uniform temporal sampling, separating state (screen) from actions (keyboard/mouse).

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
| `encoder_type` | Event encoding strategy | `hierarchical`, `json`, `flat` | `hierarchical` |
| `load_images` | Load screen images | `True`, `False` | `True` |
| `encode_actions` | Encode action events | `True`, `False` | `True` |
| `instruction` | Task instruction (Binned only) | Any string | `"Complete the task"` |

## EventEncoder: From Events to Text

The EventEncoder converts raw desktop events into text representations suitable for language model training.

### Encoding Strategies

=== "Hierarchical (Recommended)"

    ```
    <EVENT_START>mouse_move<EVENT_END>
    <EVENT_START>key_press:a<EVENT_END>
    <EVENT_START>screen_capture<EVENT_END>
    ```
    
    **Benefits**: Compositional structure, efficient tokenization, clear event boundaries

=== "JSON Format"

    ```
    <EVENT_START>{"type": "mouse_move", "x": 450, "y": 300}<EVENT_END>
    <EVENT_START>{"type": "key_press", "key": "a"}<EVENT_END>
    ```
    
    **Benefits**: Structured data, easy parsing, familiar format

=== "Flat Tokens"

    ```
    MOUSE_MOVE_450_300 KEY_PRESS_A SCREEN_CAPTURE
    ```
    
    **Benefits**: Simple tokenization, compact representation

!!! tip "Choosing an Encoder"
    - **Hierarchical**: Best for most use cases, balances structure and efficiency
    - **JSON**: When you need structured event data for analysis
    - **Flat**: When working with token-limited models or need maximum compression

## Training Pipeline Integration

### PyTorch DataLoader Integration

```python
from datasets import load_from_disk
from torch.utils.data import DataLoader
from owa.data import create_binned_dataset_transform

# Setup dataset with transform
dataset = load_from_disk("/path/to/dataset")["train"]
transform = create_binned_dataset_transform(
    encoder_type="hierarchical",
    instruction="Complete the computer task"
)
dataset.set_transform(transform)

# Custom collate function
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
    images = batch["images"]        # List[List[PIL.Image]]
    actions = batch["encoded_events"]  # List[List[str]]
    instructions = batch["instruction"] # List[str]
    
    # Your training logic here
    loss = model(images, actions, instructions)
    loss.backward()
```

### HuggingFace Transformers Integration

```python
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from owa.data import create_binned_dataset_transform

# Prepare dataset
dataset = load_from_disk("/path/to/dataset")
transform = create_binned_dataset_transform()
dataset.set_transform(transform)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
)

# Initialize trainer
trainer = Trainer(
    model=your_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=your_collate_fn,
)

# Start training
trainer.train()
```

## Performance Considerations

### Memory Optimization

- **Lazy Loading**: Images loaded only when accessed
- **Transform Caching**: Encoded events cached per sample
- **Batch Processing**: Efficient batch transforms for training

### Storage Efficiency

```python
# Example dataset sizes (45-minute session)
Raw MCAP:           5.4 GiB
Event Dataset:      ~2.1 GiB  (metadata + references)
Binned Dataset:     ~1.8 GiB  (aggregated + filtered)
```

### Processing Speed

| Operation | Event Dataset | Binned Dataset |
|-----------|---------------|----------------|
| **Initial Processing** | ~2-3 min/hour | ~1-2 min/hour |
| **Transform Application** | Real-time | Real-time |
| **Training Data Loading** | ~50-100 samples/sec | ~100-200 samples/sec |

## Best Practices

### Choosing Your Pipeline Path

!!! success "Use Event Dataset When"
    - You need maximum temporal precision
    - You want to implement custom binning strategies
    - You're doing event-level analysis or research
    - You prefer minimal preprocessing overhead

!!! success "Use Binned Dataset When"
    - You're following traditional VLA training patterns
    - You want consistent temporal sampling rates
    - You need state-action separation
    - You want to filter out inactive periods

### Optimization Tips

1. **Rate Limiting**: Use appropriate rates for each topic (screen=20Hz, mouse=60Hz)
2. **Topic Filtering**: Only include necessary topics to reduce dataset size
3. **Empty Action Filtering**: Remove bins with no actions for efficiency
4. **Transform Caching**: Let transforms cache encoded events automatically
5. **Batch Size Tuning**: Larger batches improve GPU utilization but increase memory usage

### Common Pitfalls

!!! warning "Avoid These Mistakes"
    - **Over-sampling**: Don't use unnecessarily high rates (screen >30Hz rarely needed)
    - **Under-filtering**: Include only relevant topics to avoid noise
    - **Memory Issues**: Monitor memory usage with large batch sizes
    - **Transform Conflicts**: Don't apply multiple transforms to the same dataset

## Next Steps

Ready to start processing your MCAP data? Here's your action plan:

1. **📁 Organize Your Data**: Place MCAP files in a structured directory
2. **⚙️ Configure Processing**: Choose appropriate rates and topics for your use case
3. **🚀 Run Stage 1**: Create your Event Dataset with `01_raw_events_to_event_dataset.py`
4. **🎯 Choose Your Path**: Use Event Dataset directly or proceed to Stage 2
5. **🔄 Apply Transforms**: Set up dataset transforms for your training pipeline
6. **🏋️ Start Training**: Integrate with your favorite ML framework

For more details on specific components:

- [OWAMcap Format Overview](index.md) - Understanding the underlying data format
- [Desktop Recording with ocap](ocap.md) - Creating your own MCAP datasets
- [Data Exploration Tools](how_to_explorer_and_edit.md) - Analyzing your processed datasets
