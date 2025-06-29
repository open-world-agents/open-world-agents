#!/usr/bin/env python3
"""
Test script for CS:GO to OWAMcap conversion.

This script tests the conversion process on a small sample of the dataset
to ensure everything works correctly before running on the full dataset.
"""

import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from convert_to_owamcap import CSGOActionDecoder, convert_hdf5_to_owamcap, verify_owamcap_file


def test_action_decoder():
    """Test the action decoder with sample data."""
    print("Testing action decoder...")

    decoder = CSGOActionDecoder()

    # Create a sample action vector (51 dimensions)
    import numpy as np

    action_vector = np.zeros(51)

    # Test with definitive structure from actions_to_onehot function
    # Set some keyboard keys
    action_vector[0] = 1  # W key (index 0)
    action_vector[3] = 1  # D key (index 3)

    # Set mouse clicks
    action_vector[11] = 1  # Left click (index 11)
    action_vector[12] = 1  # Right click (index 12)

    # Set mouse movement (using definitive structure)
    action_vector[13 + 12] = 1  # Mouse X = 2 (index 12 in mouse_x_possibles)
    action_vector[36 + 8] = 1  # Mouse Y = 2 (index 8 in mouse_y_possibles)

    # Decode actions
    actions = decoder.decode_actions(action_vector)

    print(f"  Decoded actions: {actions}")

    # Validate results
    assert "w" in actions["keys_pressed"], "W key should be detected"
    assert "d" in actions["keys_pressed"], "D key should be detected"
    assert actions["mouse_left_click"], "Left mouse click should be detected"
    assert actions["mouse_right_click"], "Right mouse click should be detected"
    assert actions["mouse_dx"] != 0 or actions["mouse_dy"] != 0, "Mouse movement should be detected"

    print("  ✓ Action decoder test passed")
    return True


def test_conversion_with_sample_data():
    """Test conversion with synthetic HDF5 data."""
    print("\nTesting conversion with sample data...")

    import h5py
    import numpy as np

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "test_sample.hdf5"
        output_dir = temp_path / "output"
        output_dir.mkdir()

        # Create sample HDF5 file
        print("  Creating sample HDF5 file...")
        with h5py.File(input_file, "w") as f:
            num_frames = 10  # Small sample

            for i in range(num_frames):
                # Create sample frame (150, 280, 3) - RGB image
                frame = np.random.randint(0, 255, (150, 280, 3), dtype=np.uint8)
                f[f"frame_{i}_x"] = frame

                # Create sample action vector (51,)
                action_vector = np.zeros(51, dtype=np.float64)

                # Add some random actions (based on definitive structure)
                if i % 3 == 0:  # Every 3rd frame, press W key
                    action_vector[0] = 1.0  # W key
                if i % 4 == 0:  # Every 4th frame, mouse click
                    action_vector[11] = 1.0  # Left click at index 11
                if i % 2 == 0:  # Every 2nd frame, mouse movement
                    action_vector[13 + (i % 23)] = 1.0  # Random mouse X (23 bins)
                    action_vector[36 + (i % 15)] = 1.0  # Random mouse Y (15 bins)

                f[f"frame_{i}_y"] = action_vector

                # Create helper array [kill_flag, death_flag]
                helper_arr = np.array([0.0, 0.0], dtype=np.float64)
                if i == 5:  # Kill at frame 5
                    helper_arr[0] = 1.0
                f[f"frame_{i}_helperarr"] = helper_arr

        print(f"  Sample HDF5 file created: {input_file}")

        # Test conversion
        print("  Converting to OWAMcap...")
        try:
            mcap_path = convert_hdf5_to_owamcap(input_file, output_dir, storage_mode="external_mkv", max_frames=None)
            print(f"  ✓ Conversion successful: {mcap_path}")

            # Verify the output
            print("  Verifying OWAMcap file...")
            stats = verify_owamcap_file(mcap_path)

            print(f"    File size: {stats['file_size_mb']:.2f} MB")
            print(f"    Messages: {stats['message_count']}")
            print(f"    Frames: {stats['frame_count']}")
            print(f"    Duration: {stats['duration_seconds']:.2f} seconds")
            print(f"    Topics: {list(stats['topics'].keys())}")

            if stats["errors"]:
                print(f"    Errors: {stats['errors']}")
                return False
            else:
                print("    ✓ No errors found")

            # Basic validation
            assert stats["frame_count"] == num_frames, f"Expected {num_frames} frames, got {stats['frame_count']}"
            assert "screen" in stats["topics"], "Screen topic should be present"
            assert "mouse" in stats["topics"], "Mouse topic should be present"
            assert "window" in stats["topics"], "Window topic should be present"
            # Note: "keyboard" topic may not be present if no key press/release events occur

            print("  ✓ Conversion test passed")
            return True

        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_real_dataset_sample():
    """Test with a real dataset file if available."""
    print("\nTesting with real dataset sample...")

    # Look for dataset files
    dataset_paths = [
        Path("/mnt/raid12/datasets/CounterStrike_Deathmatch/dataset_aim_expert"),
        Path("/mnt/raid12/datasets/CounterStrike_Deathmatch/dataset_dm_expert_othermaps"),
    ]

    sample_file = None
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            hdf5_files = list(dataset_path.glob("*.hdf5"))
            if hdf5_files:
                sample_file = hdf5_files[0]
                break

    if not sample_file:
        print("  No real dataset files found, skipping real dataset test")
        return True

    print(f"  Found sample file: {sample_file}")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        try:
            print("  Converting sample file...")
            mcap_path = convert_hdf5_to_owamcap(
                sample_file,
                output_dir,
                storage_mode="embedded",  # Skip video for faster testing
                max_frames=50,  # Limit to 50 frames for testing
            )

            print("  Verifying converted file...")
            stats = verify_owamcap_file(mcap_path)

            print(f"    File size: {stats['file_size_mb']:.2f} MB")
            print(f"    Messages: {stats['message_count']}")
            print(f"    Frames: {stats['frame_count']}")
            print(f"    Topics: {list(stats['topics'].keys())}")

            if stats["errors"]:
                print(f"    Errors found: {len(stats['errors'])}")
                for error in stats["errors"][:3]:
                    print(f"      - {error}")
                return False
            else:
                print("    ✓ No errors found")

            print("  ✓ Real dataset test passed")
            return True

        except Exception as e:
            print(f"  ✗ Real dataset test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("=== CS:GO to OWAMcap Conversion Tests ===\n")

    tests = [
        test_action_decoder,
        test_conversion_with_sample_data,
        test_real_dataset_sample,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result is True:
                passed += 1
            elif result is False:
                failed += 1
            else:
                # If function doesn't return boolean, assume success if no exception
                passed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
