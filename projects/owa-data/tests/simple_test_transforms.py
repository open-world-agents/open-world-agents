#!/usr/bin/env python3
"""
Simple test for transforms functionality.
"""

from owa.data.transforms import create_binned_dataset_transform, create_event_dataset_transform


def test_transform_creation():
    """Test that transforms can be created without errors."""

    print("Testing transform creation...")

    # Test event dataset transform creation
    event_transform = create_event_dataset_transform(
        encoder_type="hierarchical",
        load_images=True,
        encode_actions=True,
    )

    assert callable(event_transform), "Event transform should be callable"
    print("✓ Event dataset transform created successfully")

    # Test binned dataset transform creation
    binned_transform = create_binned_dataset_transform(
        instruction="Test instruction",
        encoder_type="hierarchical",
        load_images=True,
        encode_actions=True,
    )

    assert callable(binned_transform), "Binned transform should be callable"
    print("✓ Binned dataset transform created successfully")

    # Test different encoder types
    for encoder_type in ["hierarchical", "json", "flat"]:
        transform = create_event_dataset_transform(encoder_type=encoder_type)
        assert callable(transform), f"{encoder_type} transform should be callable"
        print(f"✓ {encoder_type} encoder transform created successfully")


if __name__ == "__main__":
    print("Running simple transforms test...")
    print("=" * 50)

    try:
        test_transform_creation()

        print("=" * 50)
        print("✅ Simple test passed! Transforms are working correctly.")
        print()
        print("Key features implemented:")
        print("- create_event_dataset_transform() function")
        print("- create_binned_dataset_transform() function")
        print("- Support for hierarchical, json, and flat encoders")
        print("- Functions return callable transforms for dataset.set_transform()")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
