#!/usr/bin/env python3
"""
Convert COMMAND dataset from Convert format to OWAMcap format.

This script converts the COMMAND dataset that has been processed into video.mp4 and joystick.json
format into OWAMcap format using gamepad events instead of keyboard/mouse.

Input structure (Convert dataset):
- video.mp4: Front-view video
- joystick.json: Gamepad control data with timestamps
- metadata.json: Metadata about the conversion

Output structure (OWAMcap):
- .mcap files with gamepad events and screen captures
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
from mcap_owa.highlevel import OWAMcapWriter
from owa.msgs.desktop.gamepad import GamepadEvent
from owa.msgs.desktop.screen import ScreenCaptured, MediaRef
from owa.msgs.desktop.window import WindowInfo


def load_joystick_data(joystick_path: Path) -> List[Dict[str, Any]]:
    """
    Load joystick data from JSON file.

    Args:
        joystick_path: Path to joystick.json file

    Returns:
        List of joystick data dictionaries
    """
    with open(joystick_path, 'r') as f:
        return json.load(f)


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load metadata from JSON file.

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_video_info(video_path: Path) -> tuple[int, int, float, int]:
    """
    Get video information using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return width, height, fps, frame_count


def joystick_to_gamepad_events(joystick_data: List[Dict[str, Any]]) -> List[tuple[int, GamepadEvent]]:
    """
    Convert joystick data to OWA GamepadEvent messages.

    Args:
        joystick_data: List of joystick data dictionaries

    Returns:
        List of (timestamp, GamepadEvent) tuples
    """
    events = []

    # Track previous axis values to generate events only on changes
    prev_axes = [0.0, 0.0, 0.0, 0.0]  # [left_x, left_y, right_x, right_y]

    for joy_data in joystick_data:
        timestamp = joy_data['timestamp']
        axes = joy_data['axes']

        # Generate axis events for changes
        axis_names = [
            "GAMEPAD_AXIS_LEFT_X",    # Index 0: Left stick X (angular control)
            "GAMEPAD_AXIS_LEFT_Y",    # Index 1: Left stick Y (unused)
            "GAMEPAD_AXIS_RIGHT_X",   # Index 2: Right stick X (unused)
            "GAMEPAD_AXIS_RIGHT_Y"    # Index 3: Right stick Y (linear control)
        ]

        for i, (current_value, prev_value) in enumerate(zip(axes, prev_axes)):
            # Only generate event if value changed significantly (deadzone)
            if abs(current_value - prev_value) > 0.01:
                event = GamepadEvent(
                    event_type="axis",
                    gamepad_type="GAMEPAD_TYPE_STANDARD",
                    axis=axis_names[i],
                    value=current_value,
                    timestamp=timestamp
                )
                events.append((timestamp, event))

        prev_axes = axes.copy()

    return events


def convert_single_dataset(input_dir: Path, output_dir: Path, dataset_name: str) -> bool:
    """
    Convert a single dataset directory from Convert format to OWAMcap.

    Args:
        input_dir: Input directory containing video.mp4, joystick.json, etc.
        output_dir: Output directory for .mcap file
        dataset_name: Name for the dataset (used in filenames)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check required files - video.mp4 and metadata.json are required
        video_path = input_dir / "video.mp4"
        joystick_path = input_dir / "joystick.json"
        metadata_path = input_dir / "metadata.json"

        # Video and metadata are required, joystick is optional
        if not video_path.exists():
            print(f"Missing required video.mp4 in {input_dir}")
            return False

        if not metadata_path.exists():
            print(f"Missing required metadata.json in {input_dir}")
            return False

        has_joystick = joystick_path.exists()

        # Load data
        print(f"Loading data for {dataset_name}...")
        if has_joystick:
            joystick_data = load_joystick_data(joystick_path)
            print(f"Loaded {len(joystick_data)} joystick entries")
        else:
            joystick_data = []
            print("No joystick data found - will create video-only OWAMcap")

        # metadata = load_metadata(metadata_path)  # Not currently used

        # Get video information
        width, height, fps, frame_count = get_video_info(video_path)
        print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        mcap_path = output_dir / f"{dataset_name}.mcap"

        # Convert joystick data to gamepad events (if available)
        if joystick_data:
            print(f"Converting {len(joystick_data)} joystick entries to gamepad events...")
            gamepad_events = joystick_to_gamepad_events(joystick_data)
            print(f"Generated {len(gamepad_events)} gamepad events")
        else:
            gamepad_events = []
            print("No gamepad events to generate")

        # Write OWAMcap file
        print(f"Writing OWAMcap file: {mcap_path}")

        with OWAMcapWriter(mcap_path) as writer:
            # Use first timestamp as base time
            if joystick_data:
                base_timestamp = joystick_data[0]['timestamp']
            else:
                base_timestamp = int(time.time() * 1e9)  # Current time in nanoseconds

            # Write window info
            window_event = WindowInfo(
                title=f"COMMAND-{dataset_name}",
                rect=[0, 0, width, height],
                hWnd=-1,
            )
            writer.write_message(window_event, topic="window", timestamp=base_timestamp)

            # Write initial screen capture event
            screen_event = ScreenCaptured(
                utc_ns=base_timestamp,
                source_shape=(width, height),
                shape=(width, height),
                media_ref=MediaRef(uri=str(video_path), pts_ns=base_timestamp),
            )
            writer.write_message(screen_event, topic="screen", timestamp=base_timestamp)

            # Write gamepad events (if any)
            if gamepad_events:
                print(f"Writing {len(gamepad_events)} gamepad events...")
                for timestamp, event in gamepad_events:
                    writer.write_message(event, topic="gamepad", timestamp=timestamp)
            else:
                print("No gamepad events to write")

            # Write screen capture events for video frames
            # Calculate frame timestamps based on video FPS
            frame_interval_ns = int(1e9 / fps)

            for frame_idx in range(frame_count):
                frame_timestamp = base_timestamp + (frame_idx * frame_interval_ns)

                screen_event = ScreenCaptured(
                    utc_ns=frame_timestamp,
                    source_shape=(width, height),
                    shape=(width, height),
                    media_ref=MediaRef(uri=str(video_path), pts_ns=frame_timestamp),
                )
                writer.write_message(screen_event, topic="screen", timestamp=frame_timestamp)

        print(f"Successfully converted {dataset_name}")
        return True

    except Exception as e:
        print(f"Error converting {dataset_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert COMMAND dataset to OWAMcap format")
    parser.add_argument("input_dir", type=Path, help="Input directory containing Convert dataset")
    parser.add_argument("output_dir", type=Path, help="Output directory for OWAMcap files")
    parser.add_argument("--pattern", type=str, default="*", help="Pattern to match dataset directories")
    parser.add_argument("--max-datasets", type=int, help="Maximum number of datasets to process")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    # Find dataset directories
    dataset_dirs = list(args.input_dir.glob(args.pattern))
    dataset_dirs = [d for d in dataset_dirs if d.is_dir()]

    if not dataset_dirs:
        print(f"No dataset directories found in {args.input_dir}")
        return

    if args.max_datasets:
        dataset_dirs = dataset_dirs[:args.max_datasets]

    print(f"Found {len(dataset_dirs)} datasets to convert")

    # Process each dataset
    successful = 0
    failed = 0

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"\nProcessing {dataset_name}...")

        if convert_single_dataset(dataset_dir, args.output_dir, dataset_name):
            successful += 1
        else:
            failed += 1

    print("\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()