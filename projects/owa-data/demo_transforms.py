#!/usr/bin/env python3
"""
Minimal demo for dataset transforms usage.

This script demonstrates how to use the dataset transforms to convert
OWA datasets for training vision-language-action models.
"""

from datasets import load_from_disk

from owa.data import create_binned_dataset_transform, create_event_dataset_transform

def demo_binned_dataset_transform():
    """Demo binned dataset transform for VLA training."""
    print("=== Binned Dataset Transform Demo ===")
    
    # Load binned dataset (replace with your dataset path)
    dataset_path = "/mnt/raid12/datasets/owa/data/super-hexagon-bin"
    try:
        binned_dataset = load_from_disk(dataset_path)
        print(f"Loaded binned dataset from {dataset_path}")
        
        # Create and apply transform
        transform = create_binned_dataset_transform(
            instruction="Complete the computer task",
            encoder_type="hierarchical",
            load_images=True,
            encode_actions=True,
        )
        
        # Apply transform to dataset
        train_dataset = binned_dataset["train"]
        train_dataset.set_transform(transform)
        
        print(f"Dataset length: {len(train_dataset)}")
        
        # Get a sample
        sample = train_dataset[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Images: {len(sample['images'])} loaded")
        print(f"Encoded events: {len(sample['encoded_events'])} events")
        
        # Show image details
        for i, image in enumerate(sample["images"][:3]):  # Show first 3 images
            print(f"  Image {i}: {image}")
            
    except Exception as e:
        print(f"Could not load dataset from {dataset_path}: {e}")
        print("Please update the dataset_path to point to your binned dataset")


def demo_event_dataset_transform():
    """Demo event dataset transform."""
    print("\n=== Event Dataset Transform Demo ===")
    
    # Load event dataset (replace with your dataset path)
    dataset_path = "/mnt/raid12/datasets/owa/data/super-hexagon-event"
    try:
        event_dataset = load_from_disk(dataset_path)
        print(f"Loaded event dataset from {dataset_path}")
        
        # Create and apply transform
        transform = create_event_dataset_transform(
            encoder_type="hierarchical",
            load_images=True,
            encode_actions=True,
        )
        
        # Apply transform to dataset
        train_dataset = event_dataset["train"]
        train_dataset.set_transform(transform)
        
        print(f"Dataset length: {len(train_dataset)}")
        
        # Get a sample
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        if "images" in sample:
            print(f"Images: {len(sample['images'])} loaded")
        if "encoded_events" in sample:
            print(f"Encoded events: {len(sample['encoded_events'])} events")
            
    except Exception as e:
        print(f"Could not load dataset from {dataset_path}: {e}")
        print("Please update the dataset_path to point to your event dataset")


def demo_training_pipeline():
    """Demo integration with PyTorch DataLoader."""
    print("\n=== Training Pipeline Integration Demo ===")
    
    try:
        from torch.utils.data import DataLoader
        
        # Load dataset
        dataset_path = "/mnt/raid12/datasets/owa/data/super-hexagon-bin"
        binned_dataset = load_from_disk(dataset_path)
        
        # Create transform
        transform = create_binned_dataset_transform(
            instruction="Complete the computer task",
            encoder_type="hierarchical",
            load_images=True,
            encode_actions=True,
        )
        
        # Apply transform
        train_dataset = binned_dataset["train"]
        train_dataset.set_transform(transform)
        
        # Create collate function
        def collate_fn(examples):
            return {
                "images": [example["images"] for example in examples],
                "encoded_events": [example["encoded_events"] for example in examples],
                "instruction": [example["instruction"] for example in examples],
            }
        
        # Create DataLoader
        dataloader = DataLoader(
            train_dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch size: {len(batch['instruction'])}")
        print(f"Images per sample: {[len(imgs) for imgs in batch['images']]}")
        print(f"Events per sample: {[len(events) for events in batch['encoded_events']]}")
        
    except ImportError:
        print("PyTorch not available, skipping DataLoader demo")
    except Exception as e:
        print(f"Could not run training pipeline demo: {e}")


if __name__ == "__main__":
    print("OWA Dataset Transforms Demo")
    print("===========================")
    
    demo_binned_dataset_transform()
    demo_event_dataset_transform()
    demo_training_pipeline()
    
    print("\n=== Summary ===")
    print("Dataset transforms provide a flexible way to:")
    print("- Convert datasets on-the-fly during training")
    print("- Integrate directly with HuggingFace datasets")
    print("- Work with PyTorch DataLoaders and training pipelines")
    print("- Support both event and binned dataset formats")
