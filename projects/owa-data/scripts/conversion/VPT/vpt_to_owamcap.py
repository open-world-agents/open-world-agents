#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcap-owa-support==0.5.6",
#   "owa-core==0.5.6",
#   "owa-msgs==0.5.6",
#   "owa-env-desktop==0.5.6",
#   "tqdm",
#   "rich",
# ]
# [tool.uv]
# exclude-newer = "2025-08-06T12:00:00Z"
# ///

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

from rich import print
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapWriter
from owa.env.desktop.constants import VK
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import RawMouseEvent
from owa.msgs.desktop.screen import MediaRef, ScreenCaptured
from owa.msgs.desktop.window import WindowInfo

# Constants
VPT_INTERVAL_TICK_NS = 50_000_000  # 50ms per tick
VPT_EXPECTED_TICKS = 6000  # 5 minutes
VPT_X_RESOLUTION, VPT_Y_RESOLUTION = 1280, 720

# VK mapping for keyboard events
VPT_KEYBOARD_VK_MAPPING = {
    "key.keyboard.escape": VK.ESCAPE,
    "key.keyboard.s": VK.KEY_S,
    "key.keyboard.q": VK.KEY_Q,
    "key.keyboard.w": VK.KEY_W,
    "key.keyboard.1": VK.KEY_1,
    "key.keyboard.2": VK.KEY_2,
    "key.keyboard.3": VK.KEY_3,
    "key.keyboard.4": VK.KEY_4,
    "key.keyboard.5": VK.KEY_5,
    "key.keyboard.6": VK.KEY_6,
    "key.keyboard.7": VK.KEY_7,
    "key.keyboard.8": VK.KEY_8,
    "key.keyboard.9": VK.KEY_9,
    "key.keyboard.e": VK.KEY_E,
    "key.keyboard.space": VK.SPACE,
    "key.keyboard.a": VK.KEY_A,
    "key.keyboard.d": VK.KEY_D,
    "key.keyboard.left.shift": VK.LSHIFT,
    "key.keyboard.left.control": VK.LCONTROL,
    "key.keyboard.f": VK.KEY_F,
}

# Mouse button mappings
MOUSE_BUTTON_FLAGS = {
    0: (RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN, RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP),
    1: (RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN, RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP),
    2: (RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_DOWN, RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_UP),
}


def vpt_generate_target_list_file(
    vpt_folder_path: Path, vpt_media_ext: str, target_list_file: Union[str, os.PathLike]
):
    """Filter VPT files with valid jsonl files paired with media files and are 5 minutes long."""
    media_stems = {f.stem for f in vpt_folder_path.iterdir() if f.suffix == vpt_media_ext and f.is_file()}
    jsonl_files = sorted(
        [
            (f, f.stat().st_ctime)
            for f in vpt_folder_path.iterdir()
            if f.suffix == ".jsonl" and f.is_file() and f.stem in media_stems
        ],
        key=lambda x: x[1],
    )

    print(f"{len(jsonl_files)} files found in {vpt_folder_path}.")

    target_files = []
    for file_path, _ in tqdm(jsonl_files):
        try:
            if len(file_path.read_text().splitlines()) == VPT_EXPECTED_TICKS:
                target_files.append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Found {len(target_files)} valid target files")
    Path(target_list_file).write_text("\n".join(str(f) for f in target_files) + "\n")


def create_screen_event(media_file_name: str, timestamp: int) -> ScreenCaptured:
    """Create a screen capture event."""
    return ScreenCaptured(
        utc_ns=timestamp,
        source_shape=(VPT_X_RESOLUTION, VPT_Y_RESOLUTION),
        shape=(VPT_X_RESOLUTION, VPT_Y_RESOLUTION),
        media_ref=MediaRef(uri=media_file_name, pts_ns=timestamp),
    )


def create_mouse_event(dx: int, dy: int, button_flags, timestamp: int) -> RawMouseEvent:
    """Create a mouse event."""
    return RawMouseEvent(last_x=dx, last_y=dy, button_flags=button_flags, timestamp=timestamp)


def handle_keyboard_events(writer, current_keys: list, keyboard_state: set, timestamp: int):
    """Handle keyboard press/release events."""
    # Release keys not in current tick
    for key in list(keyboard_state):
        if key not in current_keys:
            keyboard_state.remove(key)
            if key in VPT_KEYBOARD_VK_MAPPING:
                event = KeyboardEvent(event_type="release", vk=VPT_KEYBOARD_VK_MAPPING[key], timestamp=timestamp)
                writer.write_message(event, topic="keyboard", timestamp=timestamp)

    # Press new keys
    for key in current_keys:
        if key in VPT_KEYBOARD_VK_MAPPING and key not in keyboard_state:
            keyboard_state.add(key)
            event = KeyboardEvent(event_type="press", vk=VPT_KEYBOARD_VK_MAPPING[key], timestamp=timestamp)
            writer.write_message(event, topic="keyboard", timestamp=timestamp)


def handle_mouse_events(writer, tick_data: dict, button_state: set, timestamp: int):
    """Handle mouse movement and button events."""
    dx, dy = int(round(tick_data["mouse"]["dx"])), int(round(tick_data["mouse"]["dy"]))

    # Handle mouse movement
    if dx != 0 or dy != 0:
        event = create_mouse_event(dx, dy, RawMouseEvent.ButtonFlags.RI_MOUSE_NOP, timestamp)
        writer.write_message(event, topic="mouse/raw", timestamp=timestamp)

    # Handle button events
    current_buttons = tick_data["mouse"]["buttons"]

    # Release buttons
    for button in list(button_state):
        if button not in current_buttons and button in MOUSE_BUTTON_FLAGS:
            button_state.remove(button)
            _, up_flag = MOUSE_BUTTON_FLAGS[button]
            event = create_mouse_event(0, 0, up_flag, timestamp)
            writer.write_message(event, topic="mouse/raw", timestamp=timestamp)

    # Press buttons
    for button in current_buttons:
        if button not in button_state and button in MOUSE_BUTTON_FLAGS:
            button_state.add(button)
            down_flag, _ = MOUSE_BUTTON_FLAGS[button]
            event = create_mouse_event(0, 0, down_flag, timestamp)
            writer.write_message(event, topic="mouse/raw", timestamp=timestamp)


def process_single_file(jsonl_file_path, vpt_media_ext):
    """Process a single VPT file and convert it to OWAMcap format."""
    mcap_file_path = jsonl_file_path.with_suffix(".mcap")
    media_file_path = jsonl_file_path.with_suffix(vpt_media_ext)

    try:
        lines = jsonl_file_path.read_text().strip().splitlines()
        assert len(lines) == VPT_EXPECTED_TICKS, f"Expected {VPT_EXPECTED_TICKS} lines, got {len(lines)}"
        ticks = [json.loads(line) for line in lines]
    except Exception as e:
        print(f"Error reading {jsonl_file_path}: {e}")
        return

    with OWAMcapWriter(mcap_file_path) as writer:
        unix_epoch_ns = 0

        # Write window info
        window_event = WindowInfo(
            title=f"VPT-{mcap_file_path}",
            rect=(0, 0, VPT_X_RESOLUTION, VPT_Y_RESOLUTION),
            hWnd=-1,
        )
        writer.write_message(window_event, topic="window", timestamp=unix_epoch_ns)

        # Write initial screen event
        screen_event = create_screen_event(media_file_path.name, unix_epoch_ns)
        writer.write_message(screen_event, topic="screen", timestamp=unix_epoch_ns)

        keyboard_state, button_state = set(), set()

        # Process each tick
        for i, tick in enumerate(ticks):
            log_time = unix_epoch_ns + ((i + 1) * VPT_INTERVAL_TICK_NS)

            # Write screen event
            screen_event = create_screen_event(media_file_path.name, log_time)
            writer.write_message(screen_event, topic="screen", timestamp=log_time)

            # Handle keyboard and mouse events
            handle_keyboard_events(writer, tick["keyboard"]["keys"], keyboard_state, log_time)
            handle_mouse_events(writer, tick, button_state, log_time)


def main(vpt_folder_path: Path, vpt_media_ext: str, vpt_target_list_file: str, max_workers: Optional[int] = None):
    """Main function to convert VPT files to OWAMcap format."""
    max_workers = max_workers or 50
    print(f"Using {max_workers} worker processes.")

    # Generate target list if it doesn't exist
    if not Path(vpt_target_list_file).exists():
        print(f"Generating target list file: {vpt_target_list_file}")
        vpt_generate_target_list_file(vpt_folder_path, vpt_media_ext, vpt_target_list_file)

    # Load target files
    target_files = [Path(line.strip()) for line in Path(vpt_target_list_file).read_text().splitlines()]
    print(f"Converting {len(target_files)} VPT files.")

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, file_path, vpt_media_ext): file_path for file_path in target_files
        }

        with tqdm(total=len(target_files), desc="Converting files") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                    print(f"Successfully converted {file_path}")
                except Exception as exc:
                    print(f"Error converting {file_path}: {exc}")
                finally:
                    pbar.update(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert VPT dataset files to OWAMcap format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vpt-folder-path",
        type=Path,
        default=Path("/mnt/raid12/datasets/owa/mcaps/vpt").expanduser(),
        help="Path to VPT data folder containing paired media and jsonl files",
    )
    parser.add_argument(
        "--vpt-media-ext",
        type=str,
        default=".mkv",
        choices=[".mp4", ".mkv"],
        help="Media file extension for VPT dataset",
    )
    parser.add_argument(
        "--vpt-target-list-file",
        type=str,
        default="./vpt_target_files.txt",
        help="File to store the list of target VPT files to convert",
    )
    parser.add_argument(
        "--max-workers", type=int, default=50, help="Maximum number of worker processes for parallel conversion"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.vpt_folder_path, args.vpt_media_ext, args.vpt_target_list_file, args.max_workers)
