#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5py",
#   "numpy>=2.2.0",
#   "mcap-owa-support==0.5.5",
#   "owa-core==0.5.5",
#   "owa-msgs==0.5.5",
#   "tqdm",
#   "joblib",
# ]
# [tool.uv]
# exclude-newer = "2025-07-30T12:00:00Z"
# ///

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapWriter
from owa.core import MESSAGES
from owa.core.io.video import VideoWriter

# Alias print to tqdm.write for better progress bar display
print = tqdm.write

# OWA message types
ScreenCaptured = MESSAGES["desktop/ScreenCaptured"]
RawMouseEvent = MESSAGES["desktop/RawMouseEvent"]
KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
WindowInfo = MESSAGES["desktop/WindowInfo"]

# CS:GO dataset constants
CSGO_RESOLUTION = (280, 150)
CSGO_WINDOW_TITLE = "Counter-Strike: Global Offensive"
FRAME_RATE = 16
FRAME_DURATION_NS = int(1e9 / FRAME_RATE)

# CS:GO key mappings from original dataset
KEYS = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "space": 0x20,
    "ctrl": 0x11,
    "shift": 0x10,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "r": 0x52,
}


def decode_actions(action_vector: np.ndarray) -> Dict:
    """Decode 51-dimensional action vector."""
    actions = {
        "keys_pressed": [],
        "mouse_left_click": False,
        "mouse_right_click": False,
        "mouse_dx": 0,
        "mouse_dy": 0,
    }

    # Keys (indices 0-10)
    key_names = ["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
    for i, pressed in enumerate(action_vector[:11]):
        if pressed > 0.5 and i < len(key_names):
            actions["keys_pressed"].append(key_names[i])

    # Mouse clicks (indices 11-12)
    actions["mouse_left_click"] = action_vector[11] > 0.5
    actions["mouse_right_click"] = action_vector[12] > 0.5

    # Mouse movement (indices 13-35 for X, 36-50 for Y)
    mouse_x = [-1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, 0, 2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000]  # fmt: skip
    mouse_y = [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200]  # fmt: skip

    x_idx = np.argmax(action_vector[13:36])
    if action_vector[13 + x_idx] > 0.5:
        actions["mouse_dx"] = int(mouse_x[x_idx])

    y_idx = np.argmax(action_vector[36:51])
    if action_vector[36 + y_idx] > 0.5:
        actions["mouse_dy"] = int(mouse_y[y_idx])

    return actions


def create_video_from_frames(
    frames: List[np.ndarray], output_path: Path, fps: int = FRAME_RATE, video_format: Optional[str] = None
) -> None:
    """Create video file from frames."""
    if not frames:
        raise ValueError("No frames provided")

    # Auto-detect format from file extension if not provided
    if video_format is None:
        video_format = output_path.suffix.lstrip(".").lower()
        if video_format not in ["mkv", "mp4"]:
            video_format = "mkv"  # Default fallback

    # Ensure correct extension
    if video_format == "mkv" and not output_path.suffix == ".mkv":
        output_path = output_path.with_suffix(".mkv")
    elif video_format == "mp4" and not output_path.suffix == ".mp4":
        output_path = output_path.with_suffix(".mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with VideoWriter(output_path, fps=float(fps), vfr=False) as writer:
            for frame in frames:
                writer.write_frame(frame)
    except Exception as e:
        raise RuntimeError(f"Failed to create video {output_path}: {e}")


def convert_hdf5_to_owamcap(
    hdf5_path: Path, output_dir: Path, storage_mode: str = "external_mkv", max_frames: Optional[int] = None
) -> Path:
    """Convert HDF5 file to OWAMcap format."""

    # Validate input
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Input file not found: {hdf5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mcap_path = output_dir / f"{hdf5_path.stem}.mcap"

    # Setup video path if needed
    video_path = None
    if storage_mode.startswith("external_"):
        video_format = storage_mode.split("_")[1]
        video_path = output_dir / f"{hdf5_path.stem}.{video_format}"

    frames, actions = [], []

    # Load HDF5 data
    try:
        with h5py.File(hdf5_path, "r") as f:
            frame_keys = [k for k in f.keys() if k.startswith("frame_") and k.endswith("_x")]
            num_frames = min(len(frame_keys), max_frames) if max_frames else len(frame_keys)

            if num_frames == 0:
                raise ValueError("No frame data found in HDF5 file")

            for i in range(num_frames):
                frame_key, action_key = f"frame_{i}_x", f"frame_{i}_y"

                if frame_key not in f or action_key not in f:
                    raise KeyError(f"Missing data for frame {i}")

                frame = np.array(f[frame_key])
                action_vector = np.array(f[action_key])

                if frame.shape != (150, 280, 3):
                    raise ValueError(f"Invalid frame shape at {i}: {frame.shape}")
                if action_vector.shape != (51,):
                    raise ValueError(f"Invalid action shape at {i}: {action_vector.shape}")

                frames.append(frame)
                actions.append(decode_actions(action_vector))

    except Exception as e:
        raise RuntimeError(f"Failed to read HDF5 file {hdf5_path}: {e}")

    # Create video if needed
    if video_path:
        video_format = storage_mode.split("_")[1]
        create_video_from_frames(frames, video_path, video_format=video_format)

    # Create MCAP file

    try:
        with OWAMcapWriter(str(mcap_path)) as writer:
            last_window_time = -1
            prev_keys = set()
            prev_left_click = prev_right_click = False

            for frame_idx, (frame, action) in enumerate(zip(frames, actions)):
                timestamp_ns = frame_idx * FRAME_DURATION_NS

                # Write window info every second
                current_time_seconds = timestamp_ns // 1_000_000_000
                if current_time_seconds > last_window_time:
                    window_msg = WindowInfo(title=CSGO_WINDOW_TITLE, rect=(0, 0, CSGO_RESOLUTION[0], CSGO_RESOLUTION[1]), hWnd=1)  # fmt: skip
                    writer.write_message(window_msg, topic="window", timestamp=timestamp_ns)
                    last_window_time = current_time_seconds

                # Write screen capture
                if storage_mode.startswith("external_"):
                    screen_msg = ScreenCaptured(utc_ns=timestamp_ns, source_shape=CSGO_RESOLUTION, shape=CSGO_RESOLUTION, media_ref={"uri": str(video_path.name), "pts_ns": timestamp_ns})  # fmt: skip
                else:
                    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                    screen_msg = ScreenCaptured(
                        utc_ns=timestamp_ns, source_shape=CSGO_RESOLUTION, shape=CSGO_RESOLUTION, frame_arr=frame_bgra
                    )
                    screen_msg.embed_as_data_uri(format="png")

                writer.write_message(screen_msg, topic="screen", timestamp=timestamp_ns)

                # Handle keyboard events
                current_keys = set(action["keys_pressed"])

                # Release previous keys
                for key in prev_keys:
                    if key in KEYS:
                        kb_event = KeyboardEvent(event_type="release", vk=KEYS[key], timestamp=timestamp_ns)
                        writer.write_message(kb_event, topic="keyboard", timestamp=timestamp_ns)

                # Press current keys
                for key in current_keys:
                    if key in KEYS:
                        kb_event = KeyboardEvent(event_type="press", vk=KEYS[key], timestamp=timestamp_ns)
                        writer.write_message(kb_event, topic="keyboard", timestamp=timestamp_ns)

                prev_keys = current_keys

                # Handle mouse movement
                if action["mouse_dx"] != 0 or action["mouse_dy"] != 0:
                    raw_mouse_event = RawMouseEvent(
                        last_x=action["mouse_dx"],
                        last_y=action["mouse_dy"],
                        button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_NOP,
                        timestamp=timestamp_ns,
                    )
                    writer.write_message(raw_mouse_event, topic="mouse/raw", timestamp=timestamp_ns)

                # Handle mouse clicks
                current_left_click = action["mouse_left_click"]
                current_right_click = action["mouse_right_click"]

                # Release previous clicks
                if prev_left_click:
                    raw_mouse_event = RawMouseEvent(
                        last_x=action["mouse_dx"],
                        last_y=action["mouse_dy"],
                        button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP,
                        timestamp=timestamp_ns,
                    )
                    writer.write_message(raw_mouse_event, topic="mouse/raw", timestamp=timestamp_ns)

                if prev_right_click:
                    raw_mouse_event = RawMouseEvent(
                        last_x=action["mouse_dx"],
                        last_y=action["mouse_dy"],
                        button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP,
                        timestamp=timestamp_ns,
                    )
                    writer.write_message(raw_mouse_event, topic="mouse/raw", timestamp=timestamp_ns)

                # Press current clicks
                if current_left_click:
                    raw_mouse_event = RawMouseEvent(
                        last_x=action["mouse_dx"],
                        last_y=action["mouse_dy"],
                        button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN,
                        timestamp=timestamp_ns,
                    )
                    writer.write_message(raw_mouse_event, topic="mouse/raw", timestamp=timestamp_ns)

                if current_right_click:
                    raw_mouse_event = RawMouseEvent(
                        last_x=action["mouse_dx"],
                        last_y=action["mouse_dy"],
                        button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN,
                        timestamp=timestamp_ns,
                    )
                    writer.write_message(raw_mouse_event, topic="mouse/raw", timestamp=timestamp_ns)

                prev_left_click = current_left_click
                prev_right_click = current_right_click

    except Exception as e:
        raise RuntimeError(f"Failed to create MCAP file {mcap_path}: {e}")

    return mcap_path


def convert_single_file(args_tuple: Tuple[Path, Path, str, Optional[int]]) -> Tuple[bool, Path, Optional[str]]:
    """
    Wrapper function for parallel processing of a single HDF5 file.

    Args:
        args_tuple: (hdf5_path, output_dir, storage_mode, max_frames)

    Returns:
        (success, result_path, error_message)
    """
    hdf5_path, output_dir, storage_mode, max_frames = args_tuple

    try:
        mcap_path = convert_hdf5_to_owamcap(hdf5_path, output_dir, storage_mode, max_frames)
        return True, mcap_path, None
    except Exception as e:
        # cleanup both mcap and video file
        mcap_path = output_dir / f"{hdf5_path.stem}.mcap"
        video_path = output_dir / f"{hdf5_path.stem}.mp4"
        if mcap_path.exists():
            mcap_path.unlink()
        if video_path.exists():
            video_path.unlink()
        return False, hdf5_path, str(e)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(description="Convert CS:GO dataset to OWAMcap format")

    # Required arguments
    parser.add_argument("input_dir", type=Path, help="Input directory containing HDF5 files")
    parser.add_argument("output_dir", type=Path, help="Output directory for OWAMcap files")

    # Optional arguments
    parser.add_argument("--max-files", type=int, help="Maximum number of files to convert")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per file to convert")
    parser.add_argument(
        "--storage-mode",
        choices=["external_mkv", "external_mp4", "embedded"],
        default="external_mkv",
        help="How to store screen frames",
    )
    parser.add_argument(
        "--subset",
        choices=["dm_july2021", "aim_expert", "dm_expert_dust2", "dm_expert_othermaps"],
        help="Convert specific subset only",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")

    return parser


def find_hdf5_files(input_dir: Path, subset: Optional[str] = None) -> List[Path]:
    """Find HDF5 files in the input directory."""
    if subset:
        subset_dir = input_dir / f"dataset_{subset}"
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory {subset_dir} does not exist")
        return sorted(subset_dir.glob("*.hdf5"))
    else:
        return sorted(input_dir.rglob("*.hdf5"))


def process_files_parallel(
    hdf5_files: List[Path], output_dir: Path, storage_mode: str, max_frames: Optional[int], workers: int
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Process HDF5 files in parallel using joblib.

    Returns:
        (converted_files, failed_files)
    """
    converted_files = []
    failed_files = []

    # Create conversion tasks
    conversion_tasks = [
        delayed(convert_single_file)((hdf5_file, output_dir, storage_mode, max_frames)) for hdf5_file in hdf5_files
    ]

    # Execute tasks in parallel with streaming results
    parallel_executor = Parallel(n_jobs=workers, verbose=0, return_as="generator")
    results_stream = parallel_executor(conversion_tasks)

    # Process results as they complete
    with tqdm(total=len(hdf5_files), desc="Converting files", unit="file") as pbar:
        for (success, result_path, error), hdf5_file in zip(results_stream, hdf5_files):
            pbar.set_postfix_str(f"Completed: {hdf5_file.name}")

            if success:
                converted_files.append(result_path)
            else:
                print(f"ERROR converting {hdf5_file.name}: {error}")
                failed_files.append((hdf5_file, error))

            pbar.update(1)

    return converted_files, failed_files


def main():
    parser = setup_argument_parser()

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"ERROR: Input directory {args.input_dir} does not exist")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find HDF5 files
    try:
        hdf5_files = find_hdf5_files(args.input_dir, args.subset)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    if not hdf5_files:
        print("ERROR: No HDF5 files found in input directory")
        return 1

    # Limit number of files if specified
    if args.max_files:
        hdf5_files = hdf5_files[: args.max_files]

    print(f"Found {len(hdf5_files)} HDF5 files to convert")
    print(f"Using {args.workers} parallel workers")

    # Convert files
    start_time = time.time()

    # Process files in parallel using joblib with streaming results
    converted_files, failed_files = process_files_parallel(
        hdf5_files, args.output_dir, args.storage_mode, args.max_frames, args.workers
    )

    # Summary
    elapsed_time = time.time() - start_time
    print("\n=== Conversion Summary ===")
    print(f"Converted: {len(converted_files)}/{len(hdf5_files)} files")
    print(f"Failed: {len(failed_files)} files")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Output directory: {args.output_dir}")

    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files:
            print(f"  {file_path.name}: {error}")

    return 0


def verify_conversion(output_dir: Path) -> None:
    """Enhanced verification of converted files."""
    mcap_files = list(output_dir.glob("*.mcap"))
    if not mcap_files:
        print("No MCAP files found for verification")
        return

    print("\n=== Verification ===")
    print(f"Found {len(mcap_files)} MCAP files")

    total_size = sum(f.stat().st_size for f in mcap_files)
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

    valid_files = 0
    invalid_files = []

    # Check all files with enhanced validation
    with tqdm(mcap_files, desc="Verifying files", unit="file") as pbar:
        for mcap_file in pbar:
            # Update progress bar to show current file
            pbar.set_postfix_str(f"Checking: {mcap_file.name}")

            try:
                from mcap_owa.highlevel import OWAMcapReader

                with OWAMcapReader(str(mcap_file)) as reader:
                    message_counts = {"screen": 0, "mouse/raw": 0, "keyboard": 0, "window": 0}
                    timestamps = []

                    for message in reader.iter_messages():
                        topic = message.topic
                        if topic in message_counts:
                            message_counts[topic] += 1
                        timestamps.append(message.timestamp)

                    # Calculate duration
                    if timestamps:
                        duration_ns = max(timestamps) - min(timestamps)
                        duration_s = duration_ns / 1e9
                    else:
                        duration_s = 0

                    # Validation criteria
                    screen_mouse_count = message_counts["screen"] + message_counts["mouse/raw"]
                    is_valid = duration_s > 60 and screen_mouse_count > 1000

                    if is_valid:
                        valid_files += 1
                    else:
                        invalid_files.append(
                            {
                                "file": mcap_file.name,
                                "duration": duration_s,
                                "screen_mouse": screen_mouse_count,
                                "messages": message_counts,
                            }
                        )

            except Exception as e:
                invalid_files.append({"file": mcap_file.name, "error": str(e)})

    print("\nValidation Results:")
    print(f"  Valid files: {valid_files}/{len(mcap_files)} ({valid_files / len(mcap_files) * 100:.1f}%)")
    print(f"  Invalid files: {len(invalid_files)}")

    if invalid_files:
        print("\nInvalid files (showing first 10):")
        for item in invalid_files[:10]:
            if "error" in item:
                print(f"  {item['file']}: ERROR - {item['error']}")
            else:
                print(f"  {item['file']}: {item['duration']:.1f}s, {item['screen_mouse']} screen/mouse msgs")

    print("\nCriteria: >60s duration AND >1000 screen/mouse messages")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        if len(sys.argv) < 3:
            print("Usage: python convert_to_owamcap.py verify <output_dir>")
            sys.exit(1)
        verify_conversion(Path(sys.argv[2]))
    else:
        exit(main())
