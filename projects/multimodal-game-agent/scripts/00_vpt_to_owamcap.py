import tempfile
from pathlib import Path
import time
from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core.message import OWAMessage
from owa.env.desktop.msg import (
    KeyboardEvent,
    KeyboardState,
    MouseEvent,
    MouseState,
    WindowInfo,
)
from owa.env.gst.msg import ScreenEmitted
from owa.env.desktop.constants import VK
from rich import print
from tqdm import tqdm
import json

VPT_FOLDER_PATH = Path("~/data/Video-Pre-Training/data/").expanduser()
VPT_TARGET_LIST_FILE = "./target_files.txt"
VPT_INTERVAL_TICK_NS = 50_000_000  # 50 ms interval per tick
VPT_MOUSE_PIN_NS = 1_000_000  # 1 ms for mouse pin movement
VPT_X_RESOLUTION = 1280
VPT_Y_RESOLUTION = 720
VPT_EXPECTED_TICKS = 6000  # 5 minutes of 50ms ticks

# ref: https://github.com/openai/Video-Pre-Training/blob/4ea1e8e0eddcdd5ae3cc88621a80c434f22b7f3d/run_inverse_dynamics_model.py#L17
VPT_KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

# NOTE: we only convert navigation related keys and mouse movement
VPT_KEYBOARD_VK_MAPPING = {
    "key.keyboard.s": VK.KEY_S,
    "key.keyboard.w": VK.KEY_W,
    "key.keyboard.space": VK.SPACE,
    "key.keyboard.a": VK.LEFT,
    "key.keyboard.d": VK.RIGHT,
    "key.keyboard.left.shift": VK.LSHIFT,
    "key.keyboard.left.control": VK.LCONTROL,
}


def generate_vpt_file_list():
    """
    Filter VPT files that have valid jsonl files, and are 5 minutes long.
    """
    import os
    from pathlib import Path
    import av
    from PIL import Image
    from tqdm import tqdm
    import json

    all_mp4_stems = set(
        [
            f.stem
            for f in VPT_FOLDER_PATH.iterdir()
            if f.suffix == ".mp4" and f.is_file()
        ]
    )

    # Get all files with their full path and creation time
    all_files = [
        (f, f.stat().st_ctime)
        for f in VPT_FOLDER_PATH.iterdir()
        if f.suffix == ".jsonl" and f.is_file() and f.stem in all_mp4_stems
    ]

    # Sort by creation time (oldest first)
    all_files.sort(key=lambda x: x[1])

    print(f"{len(all_files)} files found in {VPT_FOLDER_PATH}.")

    # Get the oldest 10,000 file paths
    # oldest_10000_files = [f[0] for f in all_files[:10000]]

    target_files = []

    for file_name_jsonl, _ in tqdm(all_files):
        try:
            with open(file_name_jsonl, "r") as f:  # jsonl file
                lines = f.readlines()  # Read non-empty lines
                if len(lines) == VPT_EXPECTED_TICKS:
                    target_files.append(file_name_jsonl)
        except Exception as e:
            print(f"Error reading {file_name_jsonl}. Skipping. Error: {e}")

        # file_name_mp4 = file_name_jsonl.with_suffix(".mp4")
        # Open the video file
        # container = av.open(file_name_mp4)

        # frames = []
        # for frame in container.decode(video=0):
        #     # Convert AV frame to PIL Image (RGB)
        #     img = frame.to_image()
        #     frames.append(img)

        # # Now `frames` contains all PIL.Image.Image objects
        # print(f"Extracted {len(frames)} frames.")
        # print(f"First frame: {frames[0]}")

        # assert len(lines) + 1 == len(frames)

    print(f"{len(target_files)}")

    with open(VPT_TARGET_LIST_FILE, "w") as f:
        for file in target_files:
            f.write(f"{file}\n")


def main():
    if not Path(VPT_TARGET_LIST_FILE).exists():
        print(f"{VPT_TARGET_LIST_FILE=} does not exist. Generating it.")
        generate_vpt_file_list()
        print(f"{VPT_TARGET_LIST_FILE=} generated.")

    with open(VPT_TARGET_LIST_FILE, "r") as f:
        vpt_target_list = [Path(line.strip()) for line in f.readlines()]
        print(f"We will convert {len(vpt_target_list)=} VPT files.")

    for vpt_file_path in tqdm(vpt_target_list):
        print(f"Converting {vpt_file_path=} to OWAMcap format.")
        # Convert the file to OWAMcap format

        mcap_file_path = vpt_file_path.with_suffix(".mcap")
        mp4_file_path = vpt_file_path.with_suffix(".mp4")

        # Writing messages to an OWAMcap file

        unix_epoch_ns = 0  # Unix epoch time in nanoseconds (Jan 1, 1970)
        center_x, center_y = (
            VPT_X_RESOLUTION // 2,
            VPT_Y_RESOLUTION // 2,
        )  # x, y coordinates of the center of the screen (1280x720)

        try:
            with open(vpt_file_path, "r") as f:  # jsonl file
                lines = [line.strip() for line in f.readlines()]  # Read non-empty lines
                assert len(lines) == VPT_EXPECTED_TICKS, (
                    f"File {vpt_file_path} does not have {VPT_EXPECTED_TICKS=} lines. It has {len(lines)} lines."
                )
                ticks = [json.loads(line) for line in lines]
        except Exception as e:
            print(f"Error reading {vpt_file_path}. Skipping. Error: {e}")

        with OWAMcapWriter(mcap_file_path) as writer:
            topic = "window"
            event = WindowInfo(
                title=f"VPT-{mcap_file_path}",
                rect=[0, 0, VPT_X_RESOLUTION, VPT_Y_RESOLUTION],
                hWnd=-1,
            )
            writer.write_message(topic, event, log_time=unix_epoch_ns)

            # NOTE: we assume mouse starts from the center of the screen
            topic = "mouse"
            event = MouseEvent(event_type="move", x=center_x, y=center_y)
            writer.write_message(topic, event, log_time=unix_epoch_ns)

            keyboard_state = set()

            ## SCREEN EVENT
            topic = "screen"
            event = ScreenEmitted(path=str(mp4_file_path), pts=unix_epoch_ns)
            writer.write_message(topic, event, log_time=unix_epoch_ns)

            for i, tick in enumerate(ticks):
                # milli_timestamp = tick["milli"] # we don't use this value of VPT since it seems inaccurate
                log_time = unix_epoch_ns + ((i + 1) * VPT_INTERVAL_TICK_NS)

                ## SCREEN EVENT
                topic = "screen"
                event = ScreenEmitted(path=str(mp4_file_path), pts=log_time)
                writer.write_message(topic, event, log_time=log_time)

                ## KEYBOARD EVENT
                current_tick_keys = tick["keyboard"]["keys"]

                # NOTE: we suppose the keys are pressed/released in the fastest observable timing of tick.

                # press keys that are in the current tick, and not already pressed
                for key in current_tick_keys:
                    if key not in VPT_KEYBOARD_VK_MAPPING:
                        continue  # skip keys that are not in the mapping
                    else:
                        if key in keyboard_state:
                            continue  # already pressed
                        else:
                            keyboard_state.add(key)
                            topic = "keyboard"
                            event = KeyboardEvent(
                                event_type="press", vk=VPT_KEYBOARD_VK_MAPPING[key]
                            )
                            writer.write_message(topic, event, log_time=log_time)

                # release keys that are not in the current tick
                for state_key in list(keyboard_state):
                    if state_key not in current_tick_keys:
                        keyboard_state.remove(state_key)
                        topic = "keyboard"
                        event = KeyboardEvent(
                            event_type="release", vk=VPT_KEYBOARD_VK_MAPPING[state_key]
                        )
                        writer.write_message(topic, event, log_time=log_time)

                ## MOUSE EVENT
                dx = tick["mouse"]["dx"]
                dy = tick["mouse"]["dy"]

                # NOTE: we suppose the mouse coordinates are integer values
                dx = round(int(dx))
                dy = round(int(dy))

                if dx != 0 and dy != 0:
                    # NOTE: we suppose the mouse is pinned to the center. it takes VPT_MOUSE_PIN_NS for the program to pin the mouse to the center
                    topic = "mouse"
                    event = MouseEvent(
                        event_type="move", x=center_x + dx, y=center_y + dy
                    )
                    writer.write_message(
                        topic, event, log_time=log_time - VPT_MOUSE_PIN_NS
                    )

                    topic = "mouse"
                    event = MouseEvent(event_type="move", x=center_x, y=center_y)
                    writer.write_message(topic, event, log_time=log_time)

        # Reading messages from an OWAMcap file
        # read_mcap(mcap_file_path)


def read_mcap(file_path="expert.mcap", num_messages=100):
    # Reading messages from an OWAMcap file
    cnt = 0
    with OWAMcapReader(file_path) as reader:
        for topic, timestamp, msg in reader.iter_decoded_messages():
            print(f"Topic: {topic}, Timestamp: {timestamp}, Message: {msg}")
            cnt += 1
            if cnt > num_messages:
                break


if __name__ == "__main__":
    generate_vpt_file_list()
    # main()
    # read_mcap()
