#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5py",
#   "numpy>=2.2.0",
#   "mcap-owa-support==0.5.6",
#   "owa-core==0.5.6",
#   "owa-msgs==0.5.6",
#   "tqdm",
#   "joblib",
#   "typer",
#   "opencv-python",
# ]
# [tool.uv]
# exclude-newer = "2025-08-06T12:00:00Z"
# ///

import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Set, Tuple

import cv2
import h5py
import numpy as np
import typer
from joblib import Parallel, delayed
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapWriter
from owa.core.io.video import VideoWriter
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import RawMouseEvent
from owa.msgs.desktop.screen import ScreenCaptured
from owa.msgs.desktop.window import WindowInfo

# Alias print to tqdm.write for better progress bar display
print = tqdm.write

# CS:GO dataset constants
CSGO_RESOLUTION = (280, 150)
CSGO_WINDOW_TITLE = "Counter-Strike: Global Offensive"
FRAME_RATE = 16
FRAME_DURATION_NS = int(1e9 / FRAME_RATE)

# CS:GO key mappings
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

# Mouse button mappings
MOUSE_BUTTONS = {
    "left": (RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN, RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP),
    "right": (
        RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN,
        RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP,
    ),
}


def decode_actions(action_vector: np.ndarray) -> Dict:
    """Decode 51-dimensional action vector."""
    # Keys (indices 0-10)
    key_names = ["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
    keys_pressed = [
        key_names[i] for i, pressed in enumerate(action_vector[:11]) if pressed > 0.5 and i < len(key_names)
    ]

    # Mouse clicks (indices 11-12)
    mouse_left_click = action_vector[11] > 0.5
    mouse_right_click = action_vector[12] > 0.5

    # Mouse movement (indices 13-35 for X, 36-50 for Y)
    mouse_x_values = [
        -1000,
        -500,
        -300,
        -200,
        -100,
        -60,
        -30,
        -20,
        -10,
        -4,
        -2,
        0,
        2,
        4,
        10,
        20,
        30,
        60,
        100,
        200,
        300,
        500,
        1000,
    ]
    mouse_y_values = [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200]

    mouse_dx = mouse_dy = 0
    x_idx = np.argmax(action_vector[13:36])
    if action_vector[13 + x_idx] > 0.5:
        mouse_dx = int(mouse_x_values[x_idx])

    y_idx = np.argmax(action_vector[36:51])
    if action_vector[36 + y_idx] > 0.5:
        mouse_dy = int(mouse_y_values[y_idx])

    return {
        "keys_pressed": keys_pressed,
        "mouse_left_click": mouse_left_click,
        "mouse_right_click": mouse_right_click,
        "mouse_dx": mouse_dx,
        "mouse_dy": mouse_dy,
    }


def handle_keyboard_events(writer: OWAMcapWriter, current_keys: List[str], keyboard_state: Set[str], timestamp: int):
    """Handle keyboard press/release events using state-change approach."""
    # Release keys not in current frame
    for key in list(keyboard_state):
        if key not in current_keys:
            keyboard_state.remove(key)
            if key in KEYS:
                event = KeyboardEvent(event_type="release", vk=KEYS[key], timestamp=timestamp)
                writer.write_message(event, topic="keyboard", timestamp=timestamp)

    # Press new keys
    for key in current_keys:
        if key in KEYS and key not in keyboard_state:
            keyboard_state.add(key)
            event = KeyboardEvent(event_type="press", vk=KEYS[key], timestamp=timestamp)
            writer.write_message(event, topic="keyboard", timestamp=timestamp)


def handle_mouse_events(writer: OWAMcapWriter, action: Dict, button_state: Set[str], timestamp: int):
    """Handle mouse movement and button events using state-change approach."""
    # Handle mouse movement
    if action["mouse_dx"] != 0 or action["mouse_dy"] != 0:
        event = RawMouseEvent(
            last_x=action["mouse_dx"],
            last_y=action["mouse_dy"],
            button_flags=RawMouseEvent.ButtonFlags.RI_MOUSE_NOP,
            timestamp=timestamp,
        )
        writer.write_message(event, topic="mouse/raw", timestamp=timestamp)

    # Handle button state changes
    current_buttons = set()
    if action["mouse_left_click"]:
        current_buttons.add("left")
    if action["mouse_right_click"]:
        current_buttons.add("right")

    # Release buttons
    for button in list(button_state):
        if button not in current_buttons and button in MOUSE_BUTTONS:
            button_state.remove(button)
            _, up_flag = MOUSE_BUTTONS[button]
            event = RawMouseEvent(last_x=0, last_y=0, button_flags=up_flag, timestamp=timestamp)
            writer.write_message(event, topic="mouse/raw", timestamp=timestamp)

    # Press buttons
    for button in current_buttons:
        if button not in button_state and button in MOUSE_BUTTONS:
            button_state.add(button)
            down_flag, _ = MOUSE_BUTTONS[button]
            event = RawMouseEvent(last_x=0, last_y=0, button_flags=down_flag, timestamp=timestamp)
            writer.write_message(event, topic="mouse/raw", timestamp=timestamp)


def create_video_from_frames(frames: List[np.ndarray], output_path: Path, video_format: str = "mkv") -> None:
    """Create video file from frames."""
    if not frames:
        raise ValueError("No frames provided")

    # Ensure correct extension
    output_path = output_path.with_suffix(f".{video_format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(output_path, fps=float(FRAME_RATE), vfr=False) as writer:
        for frame in frames:
            writer.write_frame(frame)


def load_hdf5_data(hdf5_path: Path, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[Dict]]:
    """Load frames and actions from HDF5 file."""
    frames, actions = [], []

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

    return frames, actions


def convert_hdf5_to_owamcap(
    hdf5_path: Path, output_dir: Path, storage_mode: str = "external_mkv", max_frames: Optional[int] = None
) -> Path:
    """Convert HDF5 file to OWAMcap format."""
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Input file not found: {hdf5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mcap_path = output_dir / f"{hdf5_path.stem}.mcap"

    # Load data
    frames, actions = load_hdf5_data(hdf5_path, max_frames)

    # Create video if external storage
    video_path = None
    if storage_mode.startswith("external_"):
        video_format = storage_mode.split("_")[1]
        video_path = output_dir / f"{hdf5_path.stem}.{video_format}"
        create_video_from_frames(frames, video_path, video_format)

    # Create MCAP file
    keyboard_state: Set[str] = set()
    button_state: Set[str] = set()
    with OWAMcapWriter(str(mcap_path)) as writer:
        last_window_time = -1

        for frame_idx, (frame, action) in enumerate(zip(frames, actions)):
            timestamp_ns = frame_idx * FRAME_DURATION_NS

            # Write window info every second
            current_time_seconds = timestamp_ns // 1_000_000_000
            if current_time_seconds > last_window_time:
                window_msg = WindowInfo(
                    title=CSGO_WINDOW_TITLE, rect=(0, 0, CSGO_RESOLUTION[0], CSGO_RESOLUTION[1]), hWnd=1
                )
                writer.write_message(window_msg, topic="window", timestamp=timestamp_ns)
                last_window_time = current_time_seconds

            # Write screen capture
            if storage_mode.startswith("external_"):
                screen_msg = ScreenCaptured(
                    utc_ns=timestamp_ns,
                    source_shape=CSGO_RESOLUTION,
                    shape=CSGO_RESOLUTION,
                    media_ref={"uri": str(video_path.name), "pts_ns": timestamp_ns},
                )
            else:
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                screen_msg = ScreenCaptured(
                    utc_ns=timestamp_ns, source_shape=CSGO_RESOLUTION, shape=CSGO_RESOLUTION, frame_arr=frame_bgra
                )
                screen_msg.embed_as_data_uri(format="png")

            writer.write_message(screen_msg, topic="screen", timestamp=timestamp_ns)

            # Handle input events using state-change approach
            handle_keyboard_events(writer, action["keys_pressed"], keyboard_state, timestamp_ns)
            handle_mouse_events(writer, action, button_state, timestamp_ns)

    return mcap_path


def convert_single_file(args_tuple: Tuple[Path, Path, str, Optional[int]]) -> Tuple[bool, Path, Optional[str]]:
    """Wrapper function for parallel processing of a single HDF5 file."""
    hdf5_path, output_dir, storage_mode, max_frames = args_tuple

    try:
        mcap_path = convert_hdf5_to_owamcap(hdf5_path, output_dir, storage_mode, max_frames)
        return True, mcap_path, None
    except Exception as e:
        # Cleanup files on failure
        mcap_path = output_dir / f"{hdf5_path.stem}.mcap"
        video_extensions = [".mp4", ".mkv"]

        if mcap_path.exists():
            mcap_path.unlink()

        for ext in video_extensions:
            video_path = output_dir / f"{hdf5_path.stem}{ext}"
            if video_path.exists():
                video_path.unlink()

        return False, hdf5_path, str(e)


def find_hdf5_files(input_dir: Path, subset: Optional[str] = None) -> List[Path]:
    """Find HDF5 files in the input directory."""
    if subset:
        subset_dir = input_dir / f"dataset_{subset}"
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory {subset_dir} does not exist")
        return sorted(subset_dir.glob("*.hdf5"))
    return sorted(input_dir.rglob("*.hdf5"))


def process_files_parallel(
    hdf5_files: List[Path], output_dir: Path, storage_mode: str, max_frames: Optional[int], workers: int
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Process HDF5 files in parallel using joblib."""
    conversion_tasks = [
        delayed(convert_single_file)((hdf5_file, output_dir, storage_mode, max_frames)) for hdf5_file in hdf5_files
    ]

    parallel_executor = Parallel(n_jobs=workers, verbose=0, return_as="generator")
    results_stream = parallel_executor(conversion_tasks)

    converted_files, failed_files = [], []

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


# Create Typer app
app = typer.Typer(help="Convert CS:GO dataset to OWAMcap format")


class StorageMode(str, Enum):
    external_mkv = "external_mkv"
    external_mp4 = "external_mp4"
    embedded = "embedded"


class Subset(str, Enum):
    dm_july2021 = "dm_july2021"
    aim_expert = "aim_expert"
    dm_expert_dust2 = "dm_expert_dust2"
    dm_expert_othermaps = "dm_expert_othermaps"


@app.command()
def convert(
    input_dir: Annotated[Path, typer.Argument(help="Input directory containing HDF5 files")],
    output_dir: Annotated[Path, typer.Argument(help="Output directory for OWAMcap files")],
    max_files: Annotated[Optional[int], typer.Option(help="Maximum number of files to convert")] = None,
    max_frames: Annotated[Optional[int], typer.Option(help="Maximum frames per file to convert")] = None,
    storage_mode: Annotated[StorageMode, typer.Option(help="How to store screen frames")] = StorageMode.external_mkv,
    subset: Annotated[Optional[Subset], typer.Option(help="Convert specific subset only")] = None,
    workers: Annotated[int, typer.Option(help="Number of parallel workers")] = 4,
):
    """Convert CS:GO HDF5 files to OWAMcap format."""

    # Validate input
    if not input_dir.exists():
        typer.echo(f"ERROR: Input directory {input_dir} does not exist", err=True)
        raise typer.Exit(1)

    # Find files
    try:
        subset_str = subset.value if subset else None
        hdf5_files = find_hdf5_files(input_dir, subset_str)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    if not hdf5_files:
        typer.echo("ERROR: No HDF5 files found", err=True)
        raise typer.Exit(1)

    # Limit files if specified
    if max_files:
        hdf5_files = hdf5_files[:max_files]

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Converting {len(hdf5_files)} files using {workers} workers")

    # Convert files
    start_time = time.time()
    converted_files, failed_files = process_files_parallel(
        hdf5_files, output_dir, storage_mode.value, max_frames, workers
    )

    # Summary
    elapsed_time = time.time() - start_time
    typer.echo(f"\n✅ Converted: {len(converted_files)}/{len(hdf5_files)} files")
    typer.echo(f"❌ Failed: {len(failed_files)} files")
    typer.echo(f"⏱️  Time: {elapsed_time:.1f}s")

    if failed_files:
        typer.echo("\n❌ Failed files:")
        for file_path, error in failed_files[:5]:  # Show first 5 failures
            typer.echo(f"  {file_path.name}: {error}")
        if len(failed_files) > 5:
            typer.echo(f"  ... and {len(failed_files) - 5} more")


@app.command()
def verify(output_dir: Annotated[Path, typer.Argument(help="Output directory containing MCAP files to verify")]):
    """Verify converted MCAP files."""
    mcap_files = list(output_dir.glob("*.mcap"))
    if not mcap_files:
        typer.echo("No MCAP files found for verification")
        return

    typer.echo(f"Verifying {len(mcap_files)} MCAP files...")
    total_size = sum(f.stat().st_size for f in mcap_files) / (1024 * 1024)
    typer.echo(f"Total size: {total_size:.1f} MB")

    valid_files = 0
    invalid_files = []

    with tqdm(mcap_files, desc="Verifying", unit="file") as pbar:
        for mcap_file in pbar:
            try:
                from mcap_owa.highlevel import OWAMcapReader

                with OWAMcapReader(str(mcap_file)) as reader:
                    message_counts = {"screen": 0, "mouse/raw": 0, "keyboard": 0, "window": 0}
                    timestamps = []

                    for message in reader.iter_messages():
                        if message.topic in message_counts:
                            message_counts[message.topic] += 1
                        timestamps.append(message.timestamp)

                    # Validation criteria
                    duration_s = (max(timestamps) - min(timestamps)) / 1e9 if timestamps else 0
                    screen_mouse_count = message_counts["screen"] + message_counts["mouse/raw"]
                    is_valid = duration_s > 60 and screen_mouse_count > 1000

                    if is_valid:
                        valid_files += 1
                    else:
                        invalid_files.append((mcap_file.name, duration_s, screen_mouse_count))

            except Exception as e:
                invalid_files.append((mcap_file.name, f"ERROR: {e}"))

    typer.echo(f"✅ Valid: {valid_files}/{len(mcap_files)} ({valid_files / len(mcap_files) * 100:.1f}%)")

    if invalid_files:
        typer.echo(f"❌ Invalid files ({len(invalid_files)}):")
        for item in invalid_files[:5]:
            if isinstance(item[1], str):  # Error case
                typer.echo(f"  {item[0]}: {item[1]}")
            else:  # Duration/count case
                typer.echo(f"  {item[0]}: {item[1]:.1f}s, {item[2]} msgs")
        if len(invalid_files) > 5:
            typer.echo(f"  ... and {len(invalid_files) - 5} more")


if __name__ == "__main__":
    app()
