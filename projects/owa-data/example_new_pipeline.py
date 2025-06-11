#!/usr/bin/env python3
"""
Example: New OWA Data Pipeline

This script demonstrates the new 4-stage OWA data pipeline:
1. Raw MCAP Data → Event Dataset (01_raw_events_to_event_dataset.py)
2. Event Dataset → Binned Dataset (02_event_dataset_to_binned_dataset.py)  
3. Binned Dataset → MLLM Dataset (03_binned_dataset_to_mllm_dataset.py)
4. MLLM Dataset → Training Ready (VLMDatasetBuilder)

The new pipeline provides:
- Clear separation of concerns
- Memory-efficient lazy image loading
- Flexible sequence generation
- Direct integration with nanoVLM
"""


# Import the new VLMDatasetBuilder


def demonstrate_pipeline_stages():
    """Demonstrate each stage of the new pipeline."""
    print("=" * 60)
    print("NEW OWA DATA PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    print("\n🔄 STAGE 1: Raw MCAP Data → Event Dataset")
    print("Script: 01_raw_events_to_event_dataset.py")
    print("Input:  MCAP + MKV files")
    print("Output: HuggingFace Dataset with raw events")
    print("Schema: {file_path, topic, timestamp_ns, message_type, msg}")
    print("Purpose: Extract and downsample raw events from MCAP files")
    
    print("\n🔄 STAGE 2: Event Dataset → Binned Dataset")
    print("Script: 02_event_dataset_to_binned_dataset.py")
    print("Input:  Event Dataset from Stage 1")
    print("Output: HuggingFace Dataset with time-binned data")
    print("Schema: {file_path, bin_idx, timestamp_ns, state, actions}")
    print("Purpose: Aggregate events into fixed-rate time bins (e.g., 10 FPS)")
    
    print("\n🔄 STAGE 3: Binned Dataset → MLLM Dataset")
    print("Script: 03_binned_dataset_to_mllm_dataset.py")
    print("Input:  Binned Dataset from Stage 2")
    print("Output: HuggingFace Dataset with MLLM sequences")
    print("Schema: {instruction, encoded_events, image_refs, metadata}")
    print("Purpose: Create training sequences with image references")
    
    print("\n🔄 STAGE 4: MLLM Dataset → Training Ready")
    print("Class:  VLMDatasetBuilder (PyTorch Dataset)")
    print("Input:  MLLM Dataset from Stage 3")
    print("Output: PyTorch Dataset with lazy-loaded images")
    print("Schema: {instruction, encoded_events, images, metadata}")
    print("Purpose: Lazy image loading for efficient training")


def demonstrate_usage_example():
    """Show how to use the new pipeline."""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60)
    
    print("\n1. Run the pipeline scripts:")
    print("   # Stage 1: Extract events")
    print("   python scripts/01_raw_events_to_event_dataset.py \\")
    print("       --train_dir /path/to/mcap_files \\")
    print("       --output_dir /path/to/event_dataset")
    
    print("\n   # Stage 2: Create bins")
    print("   python scripts/02_event_dataset_to_binned_dataset.py \\")
    print("       --input_dir /path/to/event_dataset \\")
    print("       --output_dir /path/to/binned_dataset \\")
    print("       --fps 10")
    
    print("\n   # Stage 3: Create MLLM dataset")
    print("   python scripts/03_binned_dataset_to_mllm_dataset.py \\")
    print("       --input_dir /path/to/binned_dataset \\")
    print("       --output_dir /path/to/mllm_dataset \\")
    print("       --sequence_length 32 \\")
    print("       --instruction 'Complete the computer task'")
    
    print("\n2. Use with VLMDatasetBuilder:")
    print("   from datasets import load_from_disk")
    print("   from owa.data import VLMDatasetBuilder")
    print("   ")
    print("   # Load MLLM dataset")
    print("   mllm_dataset = load_from_disk('/path/to/mllm_dataset')")
    print("   ")
    print("   # Create PyTorch dataset with lazy image loading")
    print("   vlm_dataset = VLMDatasetBuilder(")
    print("       mllm_dataset['train'],")
    print("       image_format='pil',")
    print("       cache_images=True")
    print("   )")
    print("   ")
    print("   # Use with DataLoader")
    print("   from torch.utils.data import DataLoader")
    print("   dataloader = DataLoader(vlm_dataset, batch_size=4)")
    
    print("\n3. Integrate with nanoVLM:")
    print("   from data.datasets import OWADataset")
    print("   ")
    print("   # The VLMDatasetBuilder output is compatible with OWADataset")
    print("   owa_dataset = OWADataset(")
    print("       vlm_dataset,")
    print("       tokenizer,")
    print("       image_processor,")
    print("       mp_image_token_length")
    print("   )")


def demonstrate_benefits():
    """Highlight the benefits of the new design."""
    print("\n" + "=" * 60)
    print("BENEFITS OF NEW PIPELINE")
    print("=" * 60)
    
    print("\n✅ CLEAR SEPARATION OF CONCERNS")
    print("   - Each stage has a single responsibility")
    print("   - Easy to debug and optimize individual stages")
    print("   - Modular design allows stage replacement")
    
    print("\n✅ MEMORY EFFICIENCY")
    print("   - Images stored as references, not loaded data")
    print("   - Lazy loading only when needed for training")
    print("   - Optional caching with LRU eviction")
    
    print("\n✅ FLEXIBLE SEQUENCE GENERATION")
    print("   - Configurable sequence length and overlap")
    print("   - Support for different instruction types")
    print("   - Metadata preservation for analysis")
    
    print("\n✅ HUGGINGFACE NATIVE")
    print("   - All intermediate datasets are HuggingFace compatible")
    print("   - Easy to save, load, and share datasets")
    print("   - Built-in support for train/test splits")
    
    print("\n✅ PYTORCH INTEGRATION")
    print("   - VLMDatasetBuilder is a proper PyTorch Dataset")
    print("   - Works with DataLoader, DistributedSampler, etc.")
    print("   - Supports multiple image formats (PIL, tensor, numpy)")
    
    print("\n✅ NANOVLM COMPATIBILITY")
    print("   - Direct integration with existing OWADataset")
    print("   - No changes needed to training scripts")
    print("   - Maintains conversation format for VLA training")


def demonstrate_data_flow():
    """Show the data transformation at each stage."""
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION FLOW")
    print("=" * 60)
    
    print("\n📁 STAGE 1 OUTPUT (Event Dataset):")
    print("   {")
    print("     'file_path': '/path/to/session.mcap',")
    print("     'topic': 'screen',")
    print("     'timestamp_ns': 1745362786814673800,")
    print("     'message_type': 'owa.env.gst.msg.ScreenEmitted',")
    print("     'msg': b'{\"path\":\"video.mkv\",\"pts\":1000000000}'")
    print("   }")
    
    print("\n📦 STAGE 2 OUTPUT (Binned Dataset):")
    print("   {")
    print("     'file_path': '/path/to/session.mcap',")
    print("     'bin_idx': 42,")
    print("     'timestamp_ns': 1745362786814673800,")
    print("     'state': b'{\"path\":\"video.mkv\",\"pts\":1000000000}',")
    print("     'actions': b'[{\"event_type\":\"press\",\"vk\":65}]'")
    print("   }")
    
    print("\n🧠 STAGE 3 OUTPUT (MLLM Dataset):")
    print("   {")
    print("     'instruction': 'Complete the computer task',")
    print("     'encoded_events': ['<EVENT_START>...', '<EVENT_START>...'],")
    print("     'image_refs': [")
    print("       {'path': 'video.mkv', 'pts': 1000000000, 'timestamp_ns': ...},")
    print("       {'path': 'video.mkv', 'pts': 1100000000, 'timestamp_ns': ...}")
    print("     ],")
    print("     'metadata': {'sequence_idx': 0, 'num_images': 2, ...}")
    print("   }")
    
    print("\n🎯 STAGE 4 OUTPUT (Training Ready):")
    print("   {")
    print("     'instruction': 'Complete the computer task',")
    print("     'encoded_events': ['<EVENT_START>...', '<EVENT_START>...'],")
    print("     'images': [PIL.Image(...), PIL.Image(...)],  # Lazy loaded!")
    print("     'metadata': {'sequence_idx': 0, 'num_images': 2, ...}")
    print("   }")


def main():
    """Run all demonstrations."""
    print("NEW OWA DATA PIPELINE")
    print("Demonstrating the improved 4-stage design")
    
    demonstrate_pipeline_stages()
    demonstrate_usage_example()
    demonstrate_benefits()
    demonstrate_data_flow()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ 4-stage pipeline with clear separation of concerns")
    print("✅ Memory-efficient lazy image loading")
    print("✅ HuggingFace native datasets throughout")
    print("✅ PyTorch Dataset integration")
    print("✅ Direct nanoVLM compatibility")
    print("✅ Flexible and maintainable design")
    print("\nThe new pipeline is ready for VLA model training! 🚀")


if __name__ == "__main__":
    main()
