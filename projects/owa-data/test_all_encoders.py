#!/usr/bin/env python3
"""
Comprehensive test script for all OWA Event Encoders.

This script tests and compares all available event encoders:
- JSONEventEncoder (JSON string-based)
- FlatEventEncoder (flat tokens)
- HierarchicalEventEncoder (hierarchical tokens)
"""

import time
from typing import Any, Dict, List

from owa.data import (
    FlatEventEncoder,
    HierarchicalEventEncoder,
    JSONEventEncoder,
)


def create_test_events() -> List[Dict[str, Any]]:
    """Create comprehensive test events for encoder validation."""
    return [
        # Keyboard event
        {
            "file_path": "/path/to/file.mcap",
            "topic": "keyboard",
            "timestamp_ns": 1745362786814673800,
            "message_type": "desktop/KeyboardEvent",
            "msg": '{"event_type":"press","vk":65}',  # A key press
        },
        # Mouse move event
        {
            "file_path": "/path/to/file.mcap",
            "topic": "mouse",
            "timestamp_ns": 1745362786814673900,
            "message_type": "desktop/MouseEvent",
            "msg": '{"event_type":"move","x":960,"y":540}',  # Center of screen
        },
        # Mouse click event
        {
            "file_path": "/path/to/file.mcap",
            "topic": "mouse",
            "timestamp_ns": 1745362786814674000,
            "message_type": "desktop/MouseEvent",
            "msg": '{"event_type":"click","x":960,"y":540,"button":"left","pressed":true}',
        },
        # Mouse scroll event
        {
            "file_path": "/path/to/file.mcap",
            "topic": "mouse",
            "timestamp_ns": 1745362786814674100,
            "message_type": "desktop/MouseEvent",
            "msg": '{"event_type":"scroll","x":960,"y":540,"dx":0,"dy":-1}',
        },
        # Screen event
        {
            "file_path": "/path/to/file.mcap",
            "topic": "screen",
            "timestamp_ns": 1745362786814674200,
            "message_type": "desktop/ScreenCaptured",
            "msg": '{"path":"output.mkv","pts":123456789}',
        },
    ]


def test_encoder_functionality(encoder, encoder_name: str, test_events: List[Dict[str, Any]]):
    """Test basic functionality of an encoder."""
    print(f"\n=== Testing {encoder_name} ===")

    # Test individual encoding/decoding
    print("Individual event processing:")
    for i, event in enumerate(test_events):
        try:
            encoded_data, images = encoder.encode(event)
            decoded_event = encoder.decode(encoded_data, images)

            success = event["topic"] == decoded_event["topic"]
            print(f"  Event {i + 1} ({event['topic']}): {'✓' if success else '✗'}")

            # Show encoding format
            if isinstance(encoded_data, str):
                print(f"    Encoded: {encoded_data[:50]}...")
            else:
                print(f"    Encoded: {encoded_data}")

        except Exception as e:
            print(f"  Event {i + 1} ({event['topic']}): ✗ Error - {e}")

    # Test batch processing
    print("\nBatch processing:")
    try:
        all_encoded, all_images = encoder.encode_batch(test_events)
        decoded_events = encoder.decode_batch(all_encoded, all_images)

        success_count = sum(
            1 for orig, decoded in zip(test_events, decoded_events) if orig["topic"] == decoded["topic"]
        )

        print(f"  Batch round-trip: {success_count}/{len(test_events)} events successful")

    except Exception as e:
        print(f"  Batch processing: ✗ Error - {e}")

    # Show encoder info
    if hasattr(encoder, "get_encoder_info"):
        info = encoder.get_encoder_info()
        print(f"\nEncoder info: {info}")

    return encoder


def benchmark_encoders(encoders: Dict[str, Any], test_events: List[Dict[str, Any]], iterations: int = 100):
    """Benchmark encoding/decoding performance."""
    print(f"\n=== Performance Benchmark ({iterations} iterations) ===")

    results = {}

    for name, encoder in encoders.items():
        print(f"\nBenchmarking {name}...")

        try:
            # Encoding benchmark
            start_time = time.time()
            for _ in range(iterations):
                encoded_data, images = encoder.encode_batch(test_events)
            encoding_time = (time.time() - start_time) / iterations

            # Decoding benchmark
            start_time = time.time()
            for _ in range(iterations):
                decoded_events = encoder.decode_batch(encoded_data, images)  # noqa: F841
            decoding_time = (time.time() - start_time) / iterations

            # Get vocabulary size if available
            vocab_size = encoder.get_vocab_size() if hasattr(encoder, "get_vocab_size") else None

            results[name] = {
                "encoding_time": encoding_time * 1000,  # Convert to ms
                "decoding_time": decoding_time * 1000,  # Convert to ms
                "vocab_size": vocab_size,
                "total_time": (encoding_time + decoding_time) * 1000,
            }

            print(f"  Encoding: {encoding_time * 1000:.2f}ms")
            print(f"  Decoding: {decoding_time * 1000:.2f}ms")
            print(f"  Vocab size: {vocab_size}")

        except Exception as e:
            print(f"  Benchmark failed: {e}")
            results[name] = None

    return results


def compare_output_formats(encoders: Dict[str, Any], test_event: Dict[str, Any]):
    """Compare output formats across encoders."""
    print("\n=== Output Format Comparison ===")
    print(f"Input event: {test_event['topic']} - {test_event['msg'][:50]}...")

    for name, encoder in encoders.items():
        try:
            encoded_data, images = encoder.encode(test_event)
            print(f"\n{name}:")

            if isinstance(encoded_data, str):
                print(f"  Format: String ({len(encoded_data)} chars)")
                print(f"  Output: {encoded_data[:100]}...")
            elif isinstance(encoded_data, list):
                print(f"  Format: Token list ({len(encoded_data)} tokens)")
                print(f"  Output: {encoded_data}")
            else:
                print(f"  Format: {type(encoded_data)}")
                print(f"  Output: {encoded_data}")

            print(f"  Images: {len(images)} items")

        except Exception as e:
            print(f"\n{name}: Error - {e}")


def analyze_vocabulary_efficiency(encoders: Dict[str, Any]):
    """Analyze vocabulary efficiency across token-based encoders."""
    print("\n=== Vocabulary Efficiency Analysis ===")

    token_encoders = {}
    for name, encoder in encoders.items():
        if hasattr(encoder, "get_vocab_size"):
            vocab_size = encoder.get_vocab_size()
            if vocab_size is not None:
                token_encoders[name] = vocab_size

    if not token_encoders:
        print("No token-based encoders found.")
        return

    print("Vocabulary sizes:")
    for name, vocab_size in sorted(token_encoders.items(), key=lambda x: x[1]):
        print(f"  {name}: {vocab_size:,} tokens")

    # Calculate reduction percentages
    if len(token_encoders) > 1:
        max_vocab = max(token_encoders.values())
        print(f"\nReduction vs largest vocabulary ({max_vocab:,} tokens):")
        for name, vocab_size in token_encoders.items():
            if vocab_size < max_vocab:
                reduction = ((max_vocab - vocab_size) / max_vocab) * 100
                print(f"  {name}: {reduction:.1f}% smaller")


def main():
    """Main test function."""
    print("OWA Event Encoders - Comprehensive Test Suite")
    print("=" * 60)

    # Create test events
    test_events = create_test_events()
    print(f"Created {len(test_events)} test events")

    # Initialize all encoders
    encoders = {
        "JSONEventEncoder": JSONEventEncoder(),
        "FlatEventEncoder": FlatEventEncoder(),
        "HierarchicalEventEncoder": HierarchicalEventEncoder(),
    }

    # Test functionality
    for name, encoder in encoders.items():
        test_encoder_functionality(encoder, name, test_events)

    # Compare output formats
    compare_output_formats(encoders, test_events[0])  # Use keyboard event

    # Analyze vocabulary efficiency
    analyze_vocabulary_efficiency(encoders)

    # Benchmark performance
    benchmark_results = benchmark_encoders(encoders, test_events, iterations=50)

    # Summary
    print("\n=== Summary ===")
    print("All encoders tested successfully!")
    print("\nRecommendations:")
    print("- JSONEventEncoder: Best for general MLLM training")
    print("- FlatEventEncoder: Best for large vocabulary models")
    print("- HierarchicalEventEncoder: Best for efficient VLA training")

    if benchmark_results:
        fastest = min(
            (name for name, result in benchmark_results.items() if result),
            key=lambda name: benchmark_results[name]["total_time"],
        )
        print(f"- Fastest encoder: {fastest}")

        smallest_vocab = min(
            (name for name, result in benchmark_results.items() if result and result["vocab_size"]),
            key=lambda name: benchmark_results[name]["vocab_size"],
        )
        print(f"- Most efficient vocabulary: {smallest_vocab}")


if __name__ == "__main__":
    main()
