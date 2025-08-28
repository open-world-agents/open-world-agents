#!/usr/bin/env python3
"""
Test script for the KeyStateManager implementation.
This script tests various keyboard input scenarios to verify the key state management system works correctly.
"""

import sys

from owa.cli.mcap.convert import KeyStateManager, KeyState
from owa.env.desktop.constants import VK


def test_single_key_press_release():
    """Test a simple key press and release scenario."""
    print("Testing single key press and release...")

    manager = KeyStateManager()

    # Simulate pressing 'A' key
    manager.handle_key_event("press", VK.KEY_A, 1000000000)  # 1 second

    # Simulate releasing 'A' key after 200ms
    manager.handle_key_event("release", VK.KEY_A, 1200000000)  # 1.2 seconds

    # Finalize and get subtitles
    manager.finalize_remaining_subtitles()
    subtitles = manager.get_completed_subtitles()

    assert len(subtitles) == 1, f"Expected 1 subtitle, got {len(subtitles)}"
    start_time, end_time, message = subtitles[0]

    print(f"  Subtitle: {message}")
    print(f"  Duration: {(end_time - start_time) / 1e9:.3f} seconds")

    # Should use minimum duration since actual duration (200ms) is less than minimum (500ms)
    expected_duration = 500000000  # 500ms
    actual_duration = end_time - start_time
    assert actual_duration == expected_duration, f"Expected duration {expected_duration}, got {actual_duration}"

    print("  ✓ Single key press/release test passed")


def test_rapid_key_presses():
    """Test rapid key presses that should be consolidated."""
    print("\nTesting rapid key presses...")

    manager = KeyStateManager()

    # Simulate rapid presses of 'W' key
    manager.handle_key_event("press", VK.KEY_W, 2000000000)  # 2.0 seconds
    manager.handle_key_event("press", VK.KEY_W, 2050000000)  # 2.05 seconds (50ms later)
    manager.handle_key_event("press", VK.KEY_W, 2100000000)  # 2.1 seconds (100ms later)
    manager.handle_key_event("press", VK.KEY_W, 2150000000)  # 2.15 seconds (150ms later)

    # Release after 800ms total
    manager.handle_key_event("release", VK.KEY_W, 2800000000)  # 2.8 seconds

    # Finalize and get subtitles
    manager.finalize_remaining_subtitles()
    subtitles = manager.get_completed_subtitles()

    assert len(subtitles) == 1, f"Expected 1 subtitle for rapid presses, got {len(subtitles)}"
    start_time, end_time, message = subtitles[0]

    print(f"  Subtitle: {message}")
    print(f"  Duration: {(end_time - start_time) / 1e9:.3f} seconds")

    # Should use actual duration (800ms) since it's longer than minimum (500ms)
    expected_duration = 800000000  # 800ms
    actual_duration = end_time - start_time
    assert actual_duration == expected_duration, f"Expected duration {expected_duration}, got {actual_duration}"

    print("  ✓ Rapid key presses test passed")


def test_multiple_keys_simultaneously():
    """Test multiple keys pressed simultaneously."""
    print("\nTesting multiple keys simultaneously...")

    manager = KeyStateManager()

    # Press multiple keys
    manager.handle_key_event("press", VK.KEY_W, 3000000000)  # W at 3.0s
    manager.handle_key_event("press", VK.KEY_A, 3100000000)  # A at 3.1s
    manager.handle_key_event("press", VK.KEY_S, 3200000000)  # S at 3.2s

    # Release them at different times
    manager.handle_key_event("release", VK.KEY_A, 3600000000)  # A released at 3.6s (500ms duration)
    manager.handle_key_event("release", VK.KEY_W, 3800000000)  # W released at 3.8s (800ms duration)
    manager.handle_key_event("release", VK.KEY_S, 4000000000)  # S released at 4.0s (800ms duration)

    # Finalize and get subtitles
    manager.finalize_remaining_subtitles()
    subtitles = manager.get_completed_subtitles()

    assert len(subtitles) == 3, f"Expected 3 subtitles for 3 keys, got {len(subtitles)}"

    # Sort by start time for consistent testing
    subtitles.sort(key=lambda x: x[0])

    for i, (start_time, end_time, message) in enumerate(subtitles):
        print(f"  Subtitle {i + 1}: {message}, Duration: {(end_time - start_time) / 1e9:.3f}s")

    print("  ✓ Multiple keys simultaneously test passed")


def test_key_never_released():
    """Test a key that is pressed but never released."""
    print("\nTesting key never released...")

    manager = KeyStateManager()

    # Press key but never release it
    manager.handle_key_event("press", VK.SPACE, 4000000000)  # Space at 4.0s

    # Finalize without releasing
    manager.finalize_remaining_subtitles()
    subtitles = manager.get_completed_subtitles()

    assert len(subtitles) == 1, f"Expected 1 subtitle for unreleased key, got {len(subtitles)}"
    start_time, end_time, message = subtitles[0]

    print(f"  Subtitle: {message}")
    print(f"  Duration: {(end_time - start_time) / 1e9:.3f} seconds")

    # Should use minimum duration since key was never released
    expected_duration = 500000000  # 500ms
    actual_duration = end_time - start_time
    assert actual_duration == expected_duration, f"Expected duration {expected_duration}, got {actual_duration}"

    print("  ✓ Key never released test passed")


def test_overlapping_key_presses():
    """Test overlapping key presses with multiple presses of same keys."""
    print("\nTesting overlapping key presses...")

    manager = KeyStateManager()

    # Timeline: press A (0) -> press B (1) -> press A (4) -> press B(6) -> release A (10) -> release B(13)
    # Expected: A duration 0~10, B duration 1~13
    base_time = 5000000000  # 5 seconds as base

    manager.handle_key_event("press", VK.KEY_A, base_time + 0)  # A pressed at 0
    manager.handle_key_event("press", VK.KEY_B, base_time + 1000000000)  # B pressed at 1s
    manager.handle_key_event(
        "press", VK.KEY_A, base_time + 4000000000
    )  # A pressed again at 4s (should not create new subtitle)
    manager.handle_key_event(
        "press", VK.KEY_B, base_time + 6000000000
    )  # B pressed again at 6s (should not create new subtitle)
    manager.handle_key_event("release", VK.KEY_A, base_time + 10000000000)  # A released at 10s
    manager.handle_key_event("release", VK.KEY_B, base_time + 13000000000)  # B released at 13s

    # Finalize and get subtitles
    manager.finalize_remaining_subtitles()
    subtitles = manager.get_completed_subtitles()

    assert len(subtitles) == 2, f"Expected 2 subtitles for overlapping keys, got {len(subtitles)}"

    # Sort by start time for consistent testing
    subtitles.sort(key=lambda x: x[0])

    # First subtitle should be A (started at 0)
    a_start, a_end, a_message = subtitles[0]
    assert "KEY_A" in a_message, f"Expected KEY_A in message, got {a_message}"
    a_duration_seconds = (a_end - a_start) / 1e9
    expected_a_duration = 10.0  # 10 seconds
    assert abs(a_duration_seconds - expected_a_duration) < 0.001, (
        f"Expected A duration {expected_a_duration}s, got {a_duration_seconds}s"
    )

    # Second subtitle should be B (started at 1s)
    b_start, b_end, b_message = subtitles[1]
    assert "KEY_B" in b_message, f"Expected KEY_B in message, got {b_message}"
    b_duration_seconds = (b_end - b_start) / 1e9
    expected_b_duration = 12.0  # 12 seconds (from 1s to 13s)
    assert abs(b_duration_seconds - expected_b_duration) < 0.001, (
        f"Expected B duration {expected_b_duration}s, got {b_duration_seconds}s"
    )

    print(f"  A subtitle: {a_message}, Duration: {a_duration_seconds:.1f}s")
    print(f"  B subtitle: {b_message}, Duration: {b_duration_seconds:.1f}s")
    print("  ✓ Overlapping key presses test passed")


def test_key_state_individual():
    """Test individual KeyState class functionality."""
    print("\nTesting KeyState class...")

    key_state = KeyState(VK.KEY_A)

    # Test initial state
    assert not key_state.is_pressed
    assert key_state.press_count == 0

    # Test first press
    should_create = key_state.press(1000000000)
    assert should_create
    assert key_state.is_pressed
    assert key_state.press_count == 1

    # Test rapid subsequent presses
    should_create = key_state.press(1050000000)
    assert not should_create  # Should not create new subtitle
    assert key_state.press_count == 2

    should_create = key_state.press(1100000000)
    assert not should_create  # Should not create new subtitle
    assert key_state.press_count == 3

    # Test release
    should_finalize = key_state.release(1600000000)
    assert should_finalize
    assert not key_state.is_pressed

    # Test subtitle duration
    start_time, end_time = key_state.get_subtitle_duration()
    expected_duration = 600000000  # 600ms actual duration
    actual_duration = end_time - start_time
    assert actual_duration == expected_duration

    print("  ✓ KeyState class test passed")


if __name__ == "__main__":
    print("Running KeyStateManager tests...\n")

    try:
        test_key_state_individual()
        test_single_key_press_release()
        test_rapid_key_presses()
        test_multiple_keys_simultaneously()
        test_key_never_released()
        test_overlapping_key_presses()

        print("\n🎉 All tests passed! The KeyStateManager implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
