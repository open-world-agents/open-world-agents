#!/usr/bin/env python3
"""
Load OWA MLLM Dataset for nanoVLM Training

This script demonstrates how to load the OWA MLLM dataset created by the 4-stage pipeline
and integrate it with nanoVLM's training framework.

Usage:
    python load_owa_for_nanovlm.py
"""

import sys

# Add nanoVLM to path
sys.path.append("/mnt/home/claude/GitHub/open-world-agents/projects/nanoVLM")

from data.collators import VQACollator

# Import nanoVLM components
from data.datasets import OWADataset
from datasets import load_from_disk
from torch.utils.data import DataLoader

# Import OWA components
from owa.data.vlm_dataset_builder import VLMDatasetBuilder


def load_owa_mllm_dataset(dataset_path: str, split: str = "train") -> VLMDatasetBuilder:
    """
    Load OWA MLLM dataset and create VLMDatasetBuilder.

    Args:
        dataset_path: Path to the MLLM dataset directory
        split: Dataset split to load ('train' or 'test')

    Returns:
        VLMDatasetBuilder instance ready for training
    """
    print(f"Loading OWA MLLM dataset from {dataset_path}...")

    # Load the MLLM dataset
    mllm_dataset = load_from_disk(dataset_path)
    print(f"Available splits: {list(mllm_dataset.keys())}")

    # Create VLMDatasetBuilder with lazy image loading
    vlm_dataset = VLMDatasetBuilder(
        mllm_dataset[split],
        image_format="pil",  # or 'tensor' for direct PyTorch tensors
        cache_images=True,  # Enable caching for better performance
        max_cache_size=1000,  # Adjust based on available memory
    )

    print(f"Created VLMDatasetBuilder for {split} split with {len(vlm_dataset)} samples")
    return vlm_dataset


def create_nanovlm_dataset(vlm_dataset: VLMDatasetBuilder, tokenizer, image_processor, mp_image_token_length: int):
    """
    Create nanoVLM-compatible dataset from VLMDatasetBuilder.

    Args:
        vlm_dataset: VLMDatasetBuilder instance
        tokenizer: HuggingFace tokenizer
        image_processor: Image processor for the model
        mp_image_token_length: Number of image tokens

    Returns:
        OWADataset instance ready for nanoVLM training
    """
    # Create OWADataset
    owa_dataset = OWADataset(vlm_dataset, tokenizer, image_processor, mp_image_token_length)

    print(f"Created OWADataset with {len(owa_dataset)} samples")
    return owa_dataset


def create_dataloader(dataset, tokenizer=None, max_length=2048, batch_size: int = 4, num_workers: int = 4):
    """
    Create DataLoader for training.

    Args:
        dataset: Dataset instance (OWADataset or VLMDatasetBuilder)
        tokenizer: Tokenizer for VQACollator (required for OWADataset)
        max_length: Maximum sequence length for VQACollator
        batch_size: Batch size for training
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    # Use VQACollator for OWADataset, default collation for VLMDatasetBuilder
    if hasattr(dataset, "tokenizer") and tokenizer is not None:
        # This is likely an OWADataset, use VQACollator
        collate_fn = VQACollator(tokenizer, max_length)
        print("Using VQACollator for OWADataset")
    else:
        # This is likely a VLMDatasetBuilder, use default collation
        collate_fn = None
        print("Using default collation for VLMDatasetBuilder")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Created DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    return dataloader


def inspect_sample(vlm_dataset: VLMDatasetBuilder, sample_idx: int = 0):
    """
    Inspect a sample from the dataset.

    Args:
        vlm_dataset: VLMDatasetBuilder instance
        sample_idx: Index of sample to inspect
    """
    print(f"\n=== SAMPLE INSPECTION (index {sample_idx}) ===")

    # Mock image loading for inspection
    original_load_images = vlm_dataset._load_images
    vlm_dataset._load_images = lambda refs: [
        f"MockImage_{i}" for i in range(len(refs["path"]) if isinstance(refs, dict) else len(refs))
    ]

    try:
        sample = vlm_dataset[sample_idx]

        print(f"Sample keys: {list(sample.keys())}")
        print(f"Instruction: {sample['instruction']}")
        print(f"Encoded events count: {len(sample['encoded_events'])}")
        print(f"Images count: {len(sample['images'])}")
        print(f"Metadata: {sample['metadata']}")

        print("\nFirst few encoded events:")
        for i, event in enumerate(sample["encoded_events"][:3]):
            print(f"  {i + 1}: {event[:100]}...")

        print("\nImage references (first 3):")
        raw_sample = vlm_dataset.dataset[sample_idx]
        if isinstance(raw_sample["image_refs"], dict):
            for i in range(min(3, len(raw_sample["image_refs"]["path"]))):
                print(f"  {i + 1}: {raw_sample['image_refs']['path'][i]} @ {raw_sample['image_refs']['pts'][i]}ns")

    finally:
        # Restore original image loading
        vlm_dataset._load_images = original_load_images


def main():
    """Main demonstration function."""
    print("=== OWA MLLM Dataset Loader for nanoVLM ===")

    # Configuration
    dataset_path = "/mnt/raid11/datasets/owa/data/super-hexagon-mllm"

    # Load train dataset
    print("\n1. Loading train dataset...")
    train_dataset = load_owa_mllm_dataset(dataset_path, split="train")

    # Load test dataset
    print("\n2. Loading test dataset...")
    test_dataset = load_owa_mllm_dataset(dataset_path, split="test")

    # Inspect a sample
    print("\n3. Inspecting sample...")
    inspect_sample(train_dataset, sample_idx=0)

    # Create DataLoader (without nanoVLM dependencies)
    print("\n4. Creating DataLoader...")
    train_dataloader = create_dataloader(train_dataset, batch_size=4)

    print("\n5. Dataset Statistics:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Train batches: {len(train_dataloader)}")

    # Example of how to integrate with nanoVLM
    print("\n6. nanoVLM Integration Example:")
    print("   # To create OWADataset for actual training:")
    print("   # from transformers import AutoTokenizer, AutoImageProcessor")
    print("   # tokenizer = AutoTokenizer.from_pretrained('your-model')")
    print("   # image_processor = AutoImageProcessor.from_pretrained('your-model')")
    print("   # owa_dataset = create_nanovlm_dataset(train_dataset, tokenizer, image_processor, 576)")
    print("   # train_dataloader = create_dataloader(owa_dataset, tokenizer=tokenizer, batch_size=32)")
    print("")
    print("   # For actual training, you would then use:")
    print("   # for batch in train_dataloader:")
    print("   #     images = batch['image']")
    print("   #     input_ids = batch['input_ids']")
    print("   #     attention_mask = batch['attention_mask']")
    print("   #     labels = batch['labels']")
    print("   #     # ... train your model ...")

    print("\n✅ OWA dataset successfully loaded and ready for nanoVLM training!")

    return train_dataset, test_dataset, train_dataloader


def demo_full_nanovlm_integration(train_dataset: VLMDatasetBuilder):
    """
    Demonstrate full nanoVLM integration with proper mock tokenizer and image processor.
    This shows how the complete pipeline would work in practice.
    """
    print("\n=== Full nanoVLM Integration Demo ===")

    try:
        import numpy as np
        import torch
        from PIL import Image

        # Create a proper mock tokenizer that works with VQACollator
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.image_token = "<image>"
                self.vocab_size = 1000

            def encode(self, text):
                # Return mock token IDs based on text length
                return list(range(1, min(len(text.split()) + 1, 20)))

            def apply_chat_template(self, messages, tokenize=False, add_special_tokens=False, return_dict=False):
                # Create realistic chat template
                if isinstance(messages, list) and len(messages) > 0:
                    if tokenize:
                        # Combine all message content and tokenize
                        full_text = ""
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            full_text += f"<{role}>{content}</{role}>"

                        tokens = self.encode(full_text)
                        if return_dict:
                            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
                        return tokens
                    else:
                        # Return formatted text
                        result = ""
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            result += f"<{role}>{content}</{role}>"
                        return result
                else:
                    # Handle single message case
                    if tokenize:
                        tokens = [1, 2, 3]
                        if return_dict:
                            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
                        return tokens
                    return "empty"

        class MockImageProcessor:
            def __call__(self, image):
                # Return mock tensor for image - handle both PIL and mock images
                if isinstance(image, str):
                    # Mock image string - create a fake PIL image first
                    fake_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    image = Image.fromarray(fake_array)

                if hasattr(image, "size"):  # PIL Image
                    return torch.randn(3, 224, 224)  # Mock processed image tensor
                else:
                    return torch.randn(3, 224, 224)  # Fallback

        # Create mock components
        tokenizer = MockTokenizer()
        image_processor = MockImageProcessor()

        # First, let's create a smaller subset for testing to avoid the mock image issue
        print("Creating subset of dataset for testing...")

        # Mock the image loading to return actual PIL Images instead of strings
        def mock_load_images(image_refs):
            """Mock image loading that returns actual PIL Images"""
            images = []
            if isinstance(image_refs, dict) and "path" in image_refs:
                num_refs = len(image_refs["path"])
                for i in range(min(num_refs, 3)):  # Limit to 3 images for testing
                    # Create a fake PIL image
                    fake_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    fake_image = Image.fromarray(fake_array, mode="RGB")
                    images.append(fake_image)
            return images

        # Temporarily replace the image loading function
        original_load_images = train_dataset._load_images
        train_dataset._load_images = mock_load_images

        # Create OWADataset with a small subset
        print("Creating OWADataset with mock tokenizer and image processor...")

        # Create a smaller dataset for testing
        class SubsetDataset:
            def __init__(self, original_dataset, size=5):
                self.original_dataset = original_dataset
                self.size = min(size, len(original_dataset))

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.original_dataset[idx]

        subset_vlm_dataset = SubsetDataset(train_dataset, size=5)
        owa_dataset = create_nanovlm_dataset(subset_vlm_dataset, tokenizer, image_processor, mp_image_token_length=576)

        # Create DataLoader with VQACollator
        print("Creating DataLoader with VQACollator...")
        train_dataloader = create_dataloader(
            owa_dataset, tokenizer=tokenizer, max_length=512, batch_size=2, num_workers=0
        )

        # Test one batch
        print("Testing batch processing...")
        for i, batch in enumerate(train_dataloader):
            print(f"Batch {i + 1}:")
            print(f"  Images shape: {batch['image'].shape}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Sample input IDs: {batch['input_ids'][0][:10].tolist()}...")
            if i >= 0:  # Test just one batch
                break

        # Restore original image loading
        train_dataset._load_images = original_load_images

        print("✅ Full nanoVLM integration demo successful!")
        return True

    except Exception as e:
        print(f"❌ Full integration demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    train_dataset, test_dataset, train_dataloader = main()

    # Optional: Test iteration with custom collator that can handle variable sequences
    print("\n=== Testing DataLoader Iteration ===")
    try:
        import numpy as np
        from PIL import Image

        # Custom collator for VLMDatasetBuilder that can handle variable-length sequences
        def vlm_collator(batch):
            """Custom collator that handles variable-length sequences in VLMDatasetBuilder"""
            # Extract all fields
            instructions = [item["instruction"] for item in batch]
            encoded_events = [item["encoded_events"] for item in batch]
            images = [item["images"] for item in batch]
            metadata = [item["metadata"] for item in batch]

            # For images, we'll just return them as nested lists since they're variable length
            # In practice, you'd want to pad or truncate to fixed size

            return {
                "instruction": instructions,
                "encoded_events": encoded_events,  # Keep as list of lists
                "images": images,  # Keep as list of lists
                "metadata": metadata,
            }

        # Mock image loading for testing - return FIXED number of PIL Images per sample
        def mock_load_images_for_test(refs):
            # Always return exactly 2 images per sample to make sequences uniform
            images = []
            for i in range(2):  # Fixed number of images
                # Create a fake PIL image
                fake_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                fake_image = Image.fromarray(fake_array, mode="RGB")
                images.append(fake_image)
            return images

        # Temporarily replace image loading
        original_load_images = train_dataset._load_images
        train_dataset._load_images = mock_load_images_for_test

        # Create a new DataLoader with custom collator
        from torch.utils.data import DataLoader

        test_dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=vlm_collator,
            pin_memory=False,  # Disable pin_memory to avoid warnings
        )

        batch_count = 0
        for i, batch in enumerate(test_dataloader):
            print(f"Batch {i + 1}:")
            print(f"  Instructions: {len(batch['instruction'])} samples")
            print(f"  Encoded events: {len(batch['encoded_events'])} samples")
            print(f"    First sample events: {len(batch['encoded_events'][0])} events")
            print(f"  Images: {len(batch['images'])} samples")
            print(f"    First sample images: {len(batch['images'][0])} images")
            if len(batch["images"][0]) > 0:
                print(f"    First image type: {type(batch['images'][0][0])}")
            print(f"  Metadata: {len(batch['metadata'])} samples")

            batch_count += 1
            if batch_count >= 2:  # Test first 2 batches
                break

        # Restore original image loading
        train_dataset._load_images = original_load_images
        print("✅ DataLoader iteration successful!")

    except Exception as e:
        print(f"DataLoader iteration test failed: {e}")
        import traceback

        traceback.print_exc()

    # Demo full nanoVLM integration
    demo_full_nanovlm_integration(train_dataset)
