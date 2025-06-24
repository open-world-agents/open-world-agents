#!/usr/bin/env python3
"""
Minimal demo for OWADataset usage.
"""

from datasets import load_from_disk

from owa.data import OWADataset

# Load MLLM dataset
dataset_path = "/mnt/raid12/datasets/owa/data/super-hexagon-mllm"
mllm_dataset = load_from_disk(dataset_path)

# Create OWADataset
owa_dataset = OWADataset(mllm_dataset["train"])
print(f"Dataset length: {len(owa_dataset)}")

# Get a sample
sample = owa_dataset[0]
print(f"Instruction: {sample['instruction']}")
print(f"Images: {len(sample['images'])} loaded")
print(f"Encoded events: {len(sample['encoded_events'])} events")

# Show image details
for i, image in enumerate(sample["images"]):
    print(f"  Image {i}: {image.size=}, {image.mode=}")

# Show first few events
for i, event in enumerate(sample["encoded_events"][:3]):
    print(f"  Event {i}: {event}")

"""
Dataset length: 3189
Instruction: Complete the computer task
Images: 1 loaded
Encoded events: 1 events
  Image 0: image.size=(768, 480), image.mode='RGB'
  Event 0: <EVENT_START><TIMESTAMP><111><KEYBOARD><27><press><EVENT_END>
"""
