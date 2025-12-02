# Demo

This guide walks you through the full OWA data pipeline, from downloading example data to loading the processed dataset for training.

## Prerequisites

First, install the `owa-data` package and HuggingFace CLI:

```bash
pip install -e projects/owa-data
pip install huggingface_hub[cli]
```

## Step 1: Download Example Data

We'll download a single OWAMcap recording (`.mcap` + `.mkv`) from the D2E-480p dataset.

```bash
hf download open-world-agents/D2E-480p --repo-type dataset --include "Apex_Legends/0805_01.*" --local-dir ./data/D2E-480p
```

## Step 2: Set Environment Variables

These paths will be used throughout the demo. Adjust them to your preference.

```bash
export MCAP_DIR="./data/D2E-480p/Apex_Legends"
export EVENT_DATASET_DIR="./data/event-dataset"
export FSL_DATASET_DIR="./data/fsl-dataset"
```

## Step 3: Convert MCAP to Event Dataset

This step reads the raw MCAP files and creates a HuggingFace Dataset where each row is a single event (screen frame, key press, mouse move, etc.) with nanosecond timestamps.

```bash
python scripts/01_raw_events_to_event_dataset.py \
  --config configs/mcap_to_event_example.yaml \
  --input_dir $MCAP_DIR \
  --output_dir $EVENT_DATASET_DIR \
  --mcap_to_event_config.num_workers 4
```

## Step 4: Convert Event Dataset to FSL Dataset

This step packs events into fixed-length sequences and pre-tokenizes them. The result is ready for transformer training with zero runtime overhead.

```bash
python scripts/02_event_to_fsl.py \
  --config configs/internvl3_example.yaml \
  --input_dir $EVENT_DATASET_DIR \
  --output_dir $FSL_DATASET_DIR \
  --event_to_fsl_config.num_proc 32 \
  --event_to_fsl_config.fsl_workers 4
```

## Step 5: Load and Verify

Now let's load the processed datasets and verify everything works.

### Load Event Dataset

```python
from owa.data.datasets import load_from_disk

event_dataset = load_from_disk('./data/event-dataset')
print(f'Event Dataset stage: {event_dataset.stage}')

event_dataset.auto_set_transform(stage='event', encoder_type='hierarchical', load_images=True)
for sample in event_dataset.take(3):
    print(sample)
```

### Load FSL Dataset

```python
from owa.data.datasets import load_from_disk

fsl_dataset = load_from_disk('./data/fsl-dataset')
print(f'FSL Dataset stage: {fsl_dataset.stage}')

fsl_dataset.auto_set_transform(stage='fsl', load_images=True)
for sample in fsl_dataset.take(3):
    print(sample)
```

That's it! You now have a training-ready FSL dataset.
