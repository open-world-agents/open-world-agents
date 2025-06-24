#!/usr/bin/env python3
"""
Minimal demo for VLADataset usage.
"""

from datasets import load_from_disk

from owa.data import VLADataset

# Load MLLM dataset
dataset_path = "/mnt/raid12/datasets/owa/data/super-hexagon-mllm"
mllm_dataset = load_from_disk(dataset_path)

# Create VLADataset
vla_dataset = VLADataset(mllm_dataset["train"])
print(f"Dataset length: {len(vla_dataset)}")

# Get a sample
sample = vla_dataset[0]
print(f"Instruction: {sample['instruction']}")
print(f"Images: {len(sample['images'])} loaded")
print(f"Encoded events: {len(sample['encoded_events'])} events")

# Show image details
for i, image in enumerate(sample["images"]):
    print(f"  Image {i}: {image=}")

# Show first few events
for i, event in enumerate(sample["encoded_events"][:3]):
    print(f"  Event {i}: {event}")
