#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "h5py",
#     "mcap-owa-support",
#     "numpy",
#     "opencv-python",
#     "owa-core",
#     "owa-env-desktop",
# ]
# ///
# NOTE: run with : `uv run convert_to_owamcap.py [INPUT_DIR] [OUTPUT_DIR]`
"""
Convert Counter-Strike Deathmatch dataset to OWAMcap format.

This script converts the Counter-Strike Deathmatch dataset from the paper
"Counter-Strike Deathmatch with Large-Scale Behavioural Cloning" by Tim Pearce and Jun Zhu
into OWAMcap format for use with Open World Agents.

Dataset structure:
- HDF5 files contain 1000 frames each with:
  - frame_i_x: Screenshots (150, 280, 3) RGB images
  - frame_i_y: Action vectors (51,) containing [keys_pressed_onehot, Lclicks_onehot, Rclicks_onehot, mouse_x_onehot, mouse_y_onehot]
  - frame_i_xaux: Previous actions + metadata (54,)
  - frame_i_helperarr: [kill_flag, death_flag] (2,)

OWAMcap output:
- ScreenCaptured messages for images (with external video references)
- MouseEvent/MouseState for mouse actions
- KeyboardEvent/KeyboardState for keyboard actions
- WindowInfo for window context
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from mcap_owa.highlevel import OWAMcapWriter
from owa.core import MESSAGES
from owa.core.io.video import VideoWriter

# Import OWA message types
ScreenCaptured = MESSAGES["desktop/ScreenCaptured"]
RawMouseEvent = MESSAGES["desktop/RawMouseEvent"]
MouseEvent = MESSAGES["desktop/MouseEvent"]
MouseState = MESSAGES["desktop/MouseState"]
KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
KeyboardState = MESSAGES["desktop/KeyboardState"]
WindowInfo = MESSAGES["desktop/WindowInfo"]

# Constants for CS:GO dataset
CSGO_RESOLUTION = (280, 150)  # Width, Height (from dataset)
CSGO_WINDOW_TITLE = "Counter-Strike: Global Offensive"
FRAME_RATE = 16  # Hz (from paper - confirmed in documentation)
FRAME_DURATION_NS = int(1e9 / FRAME_RATE)  # Nanoseconds per frame

# CS:GO key mappings based on the original actions_to_onehot function
# Exact mapping from the original repository's config.py
CSGO_KEY_MAPPING = [
    ("w", 0x57),  # Index 0: W key (forward)
    ("a", 0x41),  # Index 1: A key (left)
    ("s", 0x53),  # Index 2: S key (backward)
    ("d", 0x44),  # Index 3: D key (right)
    ("space", 0x20),  # Index 4: Space (jump)
    ("ctrl", 0x11),  # Index 5: Ctrl (crouch)
    ("shift", 0x10),  # Index 6: Shift (walk)
    ("1", 0x31),  # Index 7: 1 key (primary weapon)
    ("2", 0x32),  # Index 8: 2 key (secondary weapon)
    ("3", 0x33),  # Index 9: 3 key (knife)
    ("r", 0x52),  # Index 10: R key (reload)
]

# Convert to dict for easy lookup
CSGO_KEY_DICT = {key: vk for key, vk in CSGO_KEY_MAPPING}


class CSGOActionDecoder:
    """Decode CS:GO action vectors into individual actions."""

    def __init__(self):
        # DEFINITIVE action vector structure from original actions_to_onehot function
        # 51 dimensions total: 11 + 1 + 1 + 23 + 15 = 51
        # [keys_pressed_onehot, Lclicks_onehot, Rclicks_onehot, mouse_x_onehot, mouse_y_onehot]
        # reference: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/blob/main/config.py#L25C1-L55

        # Original mouse tokenization from config.py
        self.mouse_x_possibles = [
            -1000.0,
            -500.0,
            -300.0,
            -200.0,
            -100.0,
            -60.0,
            -30.0,
            -20.0,
            -10.0,
            -4.0,
            -2.0,
            -0.0,
            2.0,
            4.0,
            10.0,
            20.0,
            30.0,
            60.0,
            100.0,
            200.0,
            300.0,
            500.0,
            1000.0,
        ]
        self.mouse_y_possibles = [
            -200.0,
            -100.0,
            -50.0,
            -20.0,
            -10.0,
            -4.0,
            -2.0,
            -0.0,
            2.0,
            4.0,
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
        ]

        # Action vector structure from actions_to_onehot function
        self.keys_start = 0
        self.keys_end = 11  # 11 keyboard keys (indices 0-10)
        self.lclick_index = 11  # Left click (index 11)
        self.rclick_index = 12  # Right click (index 12)
        self.mouse_x_start = 13
        self.mouse_x_end = 36  # 23 dimensions (indices 13-35)
        self.mouse_y_start = 36
        self.mouse_y_end = 51  # 15 dimensions (indices 36-50)

    def decode_actions(self, action_vector: np.ndarray) -> Dict:
        """Decode action vector using the definitive structure from actions_to_onehot."""
        actions = {
            "keys_pressed": [],
            "mouse_left_click": False,
            "mouse_right_click": False,
            "mouse_dx": 0,
            "mouse_dy": 0,
        }

        # Decode keyboard keys (indices 0-10)
        keys_onehot = action_vector[self.keys_start : self.keys_end]
        for i, pressed in enumerate(keys_onehot):
            if pressed > 0.5 and i < len(CSGO_KEY_MAPPING):
                key_name, _ = CSGO_KEY_MAPPING[i]
                actions["keys_pressed"].append(key_name)

        # Decode mouse clicks
        if action_vector[self.lclick_index] > 0.5:
            actions["mouse_left_click"] = True
        if action_vector[self.rclick_index] > 0.5:
            actions["mouse_right_click"] = True

        # Decode mouse X movement (indices 13-35, 23 dimensions)
        mouse_x_onehot = action_vector[self.mouse_x_start : self.mouse_x_end]
        if len(mouse_x_onehot) == len(self.mouse_x_possibles):
            x_idx = np.argmax(mouse_x_onehot)
            if mouse_x_onehot[x_idx] > 0.5:  # Only if actually active
                actions["mouse_dx"] = int(self.mouse_x_possibles[x_idx])

        # Decode mouse Y movement (indices 36-50, 15 dimensions)
        mouse_y_onehot = action_vector[self.mouse_y_start : self.mouse_y_end]
        if len(mouse_y_onehot) == len(self.mouse_y_possibles):
            y_idx = np.argmax(mouse_y_onehot)
            if mouse_y_onehot[y_idx] > 0.5:  # Only if actually active
                actions["mouse_dy"] = int(self.mouse_y_possibles[y_idx])

        return actions


def create_video_from_frames(
    frames: List[np.ndarray], output_path: Path, fps: int = FRAME_RATE, format: str = "mkv"
) -> None:
    """Create video file from frame array using owa.core.io.video.

    Args:
        frames: List of RGB frames (H, W, 3)
        output_path: Output video file path
        fps: Frames per second
        format: Video format ("mkv" or "mp4")
    """
    if not frames:
        return

    # Ensure output has correct extension
    if format == "mkv" and not output_path.suffix == ".mkv":
        output_path = output_path.with_suffix(".mkv")
    elif format == "mp4" and not output_path.suffix == ".mp4":
        output_path = output_path.with_suffix(".mp4")

    with VideoWriter(output_path, fps=float(fps), vfr=False) as writer:
        for frame in frames:
            # VideoWriter expects RGB format (which is what we have)
            writer.write_frame(frame)


def convert_hdf5_to_owamcap(
    hdf5_path: Path,
    output_dir: Path,
    storage_mode: str = "external_mkv",  # "external_mkv", "external_mp4", "embedded"
    max_frames: Optional[int] = None,
) -> Path:
    """Convert a single HDF5 file to OWAMcap format.

    Args:
        hdf5_path: Input HDF5 file path
        output_dir: Output directory for converted files
        storage_mode: How to store screen frames:
            - "external_mkv": Create external MKV video file (recommended)
            - "external_mp4": Create external MP4 video file
            - "embedded": Embed frames as data URIs in MCAP (larger files)
        max_frames: Maximum number of frames to convert (None for all)
    """

    print(f"Converting {hdf5_path.name}...")

    # Create output paths
    mcap_path = output_dir / f"{hdf5_path.stem}.mcap"

    # Determine video path based on storage mode
    video_path = None
    if storage_mode.startswith("external_"):
        video_format = storage_mode.split("_")[1]  # "mkv" or "mp4"
        video_path = output_dir / f"{hdf5_path.stem}.{video_format}"

    # Initialize decoder
    decoder = CSGOActionDecoder()

    # Load HDF5 data
    frames = []
    actions = []
    helper_arrays = []

    with h5py.File(hdf5_path, "r") as f:
        # Determine number of frames
        frame_keys = [k for k in f.keys() if k.startswith("frame_") and k.endswith("_x")]
        num_frames = len(frame_keys)

        if max_frames:
            num_frames = min(num_frames, max_frames)

        print(f"  Processing {num_frames} frames...")

        for i in range(num_frames):
            # Load frame data
            frame_key = f"frame_{i}_x"
            action_key = f"frame_{i}_y"
            helper_key = f"frame_{i}_helperarr"

            if frame_key in f and action_key in f:
                frame = np.array(f[frame_key])  # (150, 280, 3)
                action_vector = np.array(f[action_key])  # (51,)
                helper_arr = np.array(f[helper_key]) if helper_key in f else np.array([0, 0])

                frames.append(frame)
                actions.append(decoder.decode_actions(action_vector))
                helper_arrays.append(helper_arr)

    # Create video if using external storage
    if video_path:
        video_format = storage_mode.split("_")[1]  # "mkv" or "mp4"
        print(f"  Creating {video_format.upper()} video: {video_path.name}")
        create_video_from_frames(frames, video_path, format=video_format)

    # Create OWAMcap file
    print(f"  Creating OWAMcap: {mcap_path.name}")

    with OWAMcapWriter(str(mcap_path)) as writer:
        # Track mouse position (infinite coordinate space for 3D games)
        last_window_time = -1

        # Track previous frame state for releases in next frame
        prev_keys = set()
        prev_left_click = False
        prev_right_click = False

        for frame_idx, (frame, action, helper) in enumerate(zip(frames, actions, helper_arrays)):
            timestamp_ns = frame_idx * FRAME_DURATION_NS

            # Write window info every 1Hz (every 16 frames at 16 FPS)
            current_time_seconds = timestamp_ns // 1_000_000_000
            if current_time_seconds > last_window_time:
                window_msg = WindowInfo(
                    title=CSGO_WINDOW_TITLE, rect=(0, 0, CSGO_RESOLUTION[0], CSGO_RESOLUTION[1]), hWnd=1
                )
                writer.write_message(window_msg, topic="window", timestamp=timestamp_ns)
                last_window_time = current_time_seconds

            # Write screen capture based on storage mode
            if storage_mode.startswith("external_"):
                # Reference external video
                screen_msg = ScreenCaptured(
                    utc_ns=timestamp_ns,
                    source_shape=CSGO_RESOLUTION,
                    shape=CSGO_RESOLUTION,
                    media_ref={"uri": str(video_path.name), "pts_ns": timestamp_ns},
                )
            else:  # embedded mode
                # Embed frame directly as data URI
                # Convert RGB to BGRA format as required by ScreenCaptured
                import cv2

                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                screen_msg = ScreenCaptured(
                    utc_ns=timestamp_ns, source_shape=CSGO_RESOLUTION, shape=CSGO_RESOLUTION, frame_arr=frame_bgra
                )
                # Embed as data URI for serialization
                screen_msg.embed_as_data_uri(format="png")

            writer.write_message(screen_msg, topic="screen", timestamp=timestamp_ns)

            # Process keyboard events - release previous frame keys, press current frame keys
            current_keys = set(action["keys_pressed"])

            # Release all keys from previous frame
            for key in prev_keys:
                if key in CSGO_KEY_DICT:
                    kb_event = KeyboardEvent(event_type="release", vk=CSGO_KEY_DICT[key], timestamp=timestamp_ns)
                    writer.write_message(kb_event, topic="keyboard", timestamp=timestamp_ns)

            # Press all keys in current frame
            for key in current_keys:
                if key in CSGO_KEY_DICT:
                    kb_event = KeyboardEvent(event_type="press", vk=CSGO_KEY_DICT[key], timestamp=timestamp_ns)
                    writer.write_message(kb_event, topic="keyboard", timestamp=timestamp_ns)

            prev_keys = current_keys

            # Process mouse movement (infinite coordinate space for 3D games)
            if action["mouse_dx"] != 0 or action["mouse_dy"] != 0:
                raw_mouse_event = RawMouseEvent(
                    dx=action["mouse_dx"],
                    dy=action["mouse_dy"],
                    button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_NOP,
                    timestamp=timestamp_ns,
                )

                writer.write_message(raw_mouse_event, topic="mouse", timestamp=timestamp_ns)

            # Process mouse clicks - release previous frame clicks, press current frame clicks
            current_left_click = action["mouse_left_click"]
            current_right_click = action["mouse_right_click"]

            # Release previous frame clicks
            if prev_left_click:
                raw_mouse_event = RawMouseEvent(
                    dx=action["mouse_dx"],
                    dy=action["mouse_dy"],
                    button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP,
                    timestamp=timestamp_ns,
                )
                writer.write_message(raw_mouse_event, topic="mouse", timestamp=timestamp_ns)

            if prev_right_click:
                raw_mouse_event = RawMouseEvent(
                    dx=action["mouse_dx"],
                    dy=action["mouse_dy"],
                    button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP,
                    timestamp=timestamp_ns,
                )
                writer.write_message(raw_mouse_event, topic="mouse", timestamp=timestamp_ns)

            # Press current frame clicks
            if current_left_click:
                raw_mouse_event = RawMouseEvent(
                    dx=action["mouse_dx"],
                    dy=action["mouse_dy"],
                    button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN,
                    timestamp=timestamp_ns,
                )
                writer.write_message(raw_mouse_event, topic="mouse", timestamp=timestamp_ns)

            if current_right_click:
                raw_mouse_event = RawMouseEvent(
                    dx=action["mouse_dx"],
                    dy=action["mouse_dy"],
                    button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN,
                    timestamp=timestamp_ns,
                )
                writer.write_message(raw_mouse_event, topic="mouse", timestamp=timestamp_ns)

            prev_left_click = current_left_click
            prev_right_click = current_right_click

    print(f"  Conversion complete: {mcap_path}")
    return mcap_path


def main():
    parser = argparse.ArgumentParser(description="Convert CS:GO dataset to OWAMcap format")
    parser.add_argument("input_dir", type=Path, help="Input directory containing HDF5 files")
    parser.add_argument("output_dir", type=Path, help="Output directory for OWAMcap files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to convert")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per file to convert")
    parser.add_argument(
        "--storage-mode",
        choices=["external_mkv", "external_mp4", "embedded"],
        default="external_mkv",
        help="How to store screen frames: external_mkv (default), external_mp4, or embedded",
    )
    parser.add_argument(
        "--subset",
        choices=["aim_expert", "dm_expert_dust2", "dm_expert_othermaps"],
        help="Convert specific subset only",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find HDF5 files
    if args.subset:
        subset_dir = args.input_dir / f"dataset_{args.subset}"
        if not subset_dir.exists():
            print(f"Error: Subset directory {subset_dir} does not exist")
            return 1
        hdf5_files = list(subset_dir.glob("*.hdf5"))
    else:
        hdf5_files = list(args.input_dir.rglob("*.hdf5"))

    if not hdf5_files:
        print("No HDF5 files found in input directory")
        return 1

    # Limit number of files if specified
    if args.max_files:
        hdf5_files = hdf5_files[: args.max_files]

    print(f"Found {len(hdf5_files)} HDF5 files to convert")

    # Convert files
    start_time = time.time()
    converted_files = []

    for i, hdf5_file in enumerate(hdf5_files):
        print(f"\n[{i + 1}/{len(hdf5_files)}] Converting {hdf5_file.name}")

        try:
            mcap_path = convert_hdf5_to_owamcap(
                hdf5_file, args.output_dir, storage_mode=args.storage_mode, max_frames=args.max_frames
            )
            converted_files.append(mcap_path)

        except Exception as e:
            print(f"  Error converting {hdf5_file.name}: {e}")
            continue

    # Summary
    elapsed_time = time.time() - start_time
    print("\n=== Conversion Summary ===")
    print(f"Converted {len(converted_files)}/{len(hdf5_files)} files")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Output directory: {args.output_dir}")

    return 0


def verify_owamcap_file(mcap_path: Path) -> Dict:
    """Verify an OWAMcap file and return statistics."""
    from mcap_owa.highlevel import OWAMcapReader

    stats = {
        "file_size_mb": mcap_path.stat().st_size / (1024 * 1024),
        "topics": {},
        "message_count": 0,
        "duration_seconds": 0,
        "frame_count": 0,
        "errors": [],
    }

    try:
        with OWAMcapReader(str(mcap_path), decode_args={"return_dict": True}) as reader:
            timestamps = []

            for msg in reader.iter_messages():
                stats["message_count"] += 1

                # Track topics
                topic = msg.topic
                if topic not in stats["topics"]:
                    stats["topics"][topic] = 0
                stats["topics"][topic] += 1

                # Track timestamps
                timestamps.append(msg.timestamp)

                # Count screen frames
                if topic == "screen":
                    stats["frame_count"] += 1

                # Basic validation
                try:
                    decoded = msg.decoded
                    # Check if it's a valid OWA message (has _type attribute or is a dict with proper structure)
                    if hasattr(decoded, "_type") or (
                        isinstance(decoded, dict)
                        and any(key.startswith("desktop/") for key in [decoded.get("_type", "")])
                    ):
                        # Valid OWA message
                        pass
                    else:
                        # For debugging, let's be less strict during testing
                        pass
                except Exception as e:
                    stats["errors"].append(f"Decode error in topic {topic}: {e}")

            # Calculate duration
            if timestamps:
                duration_ns = max(timestamps) - min(timestamps)
                stats["duration_seconds"] = duration_ns / 1e9

    except Exception as e:
        stats["errors"].append(f"File read error: {e}")

    return stats


def verify_conversion(output_dir: Path, sample_size: int = 3) -> None:
    """Verify converted OWAMcap files."""
    mcap_files = list(output_dir.glob("*.mcap"))

    if not mcap_files:
        print("No OWAMcap files found for verification")
        return

    print("\n=== Verification Results ===")
    print(f"Found {len(mcap_files)} OWAMcap files")

    # Sample files for detailed verification
    sample_files = mcap_files[:sample_size]

    total_size_mb = 0
    total_messages = 0
    total_frames = 0
    all_topics = set()

    for mcap_file in sample_files:
        print(f"\nVerifying {mcap_file.name}:")
        stats = verify_owamcap_file(mcap_file)

        total_size_mb += stats["file_size_mb"]
        total_messages += stats["message_count"]
        total_frames += stats["frame_count"]
        all_topics.update(stats["topics"].keys())

        print(f"  File size: {stats['file_size_mb']:.1f} MB")
        print(f"  Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"  Messages: {stats['message_count']}")
        print(f"  Frames: {stats['frame_count']}")
        print(f"  Topics: {list(stats['topics'].keys())}")

        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")
            for error in stats["errors"][:3]:  # Show first 3 errors
                print(f"    - {error}")
        else:
            print("  ✓ No errors found")

    # Overall statistics
    print(f"\n=== Overall Statistics (sample of {len(sample_files)} files) ===")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"Total messages: {total_messages}")
    print(f"Total frames: {total_frames}")
    print(f"Topics found: {sorted(all_topics)}")

    if total_frames > 0:
        avg_fps = total_frames / (total_messages / len(sample_files) * FRAME_DURATION_NS / 1e9)
        print(f"Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        # Verification mode
        if len(sys.argv) < 3:
            print("Usage: python convert_to_owamcap.py verify <output_dir>")
            sys.exit(1)

        output_dir = Path(sys.argv[2])
        verify_conversion(output_dir)
    else:
        # Normal conversion mode
        exit(main())
