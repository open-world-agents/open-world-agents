import json
import os
import typing
from pathlib import Path

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.env.desktop.constants import VK
from owa.env.desktop.msg import KeyboardEvent, MouseEvent, WindowInfo
from owa.env.gst.msg import ScreenEmitted
from rich import print
from tqdm import tqdm

VPT_FOLDER_PATH = Path(
    "~/data/Video-Pre-Training/data/"
).expanduser()  # NOTE: Change this to your VPT data folder path. We expect paired mp4 and jsonl files for VPT dataset.
VPT_TARGET_LIST_FILE = "./vpt_target_files.txt"
VPT_INTERVAL_TICK_NS = 50_000_000  # 50 ms interval per tick
VPT_EXPECTED_TICKS = 6000  # 5 minutes of 50ms ticks
VPT_MOUSE_PIN_NS = 1_000_000  # We assume 1 ms for mouse pin movement
VPT_X_RESOLUTION = 1280
VPT_Y_RESOLUTION = 720


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
    "key.keyboard.a": VK.KEY_A,
    "key.keyboard.d": VK.KEY_D,
    "key.keyboard.left.shift": VK.LSHIFT,
    "key.keyboard.left.control": VK.LCONTROL,
}


def vpt_generate_target_list_file(
    target_list_file: typing.Union[str, bytes, os.PathLike] = VPT_TARGET_LIST_FILE,
):
    """
    Filter VPT files that have valid jsonl files paired with mp4, and are 5 minutes long.
    The list of valid target files is saved to `target_list_file`.
    """
    from tqdm import tqdm

    all_mp4_stems = set([f.stem for f in VPT_FOLDER_PATH.iterdir() if f.suffix == ".mp4" and f.is_file()])

    # Get all files with their full path and creation time
    all_jsonl_files = [
        (f, f.stat().st_ctime)
        for f in VPT_FOLDER_PATH.iterdir()
        if f.suffix == ".jsonl" and f.is_file() and f.stem in all_mp4_stems
    ]

    # Sort by creation time (oldest first)
    all_jsonl_files.sort(key=lambda x: x[1])

    print(f"{len(all_jsonl_files)} files found in {VPT_FOLDER_PATH}.")

    target_files = []

    for file_name_jsonl, _ in tqdm(all_jsonl_files):
        try:
            with open(file_name_jsonl, "r") as f:  # jsonl file
                lines = f.readlines()  # Read non-empty lines
                if len(lines) == VPT_EXPECTED_TICKS:
                    target_files.append(file_name_jsonl)
        except Exception as e:
            print(f"Error reading {file_name_jsonl}. Skipping. Error: {e}")

    print(f"{len(target_files)=}")

    with open(target_list_file, "w") as f:
        for file in target_files:
            f.write(f"{file}\n")


def main():
    if not Path(VPT_TARGET_LIST_FILE).exists():
        print(f"{VPT_TARGET_LIST_FILE=} does not exist. Generating it.")
        vpt_generate_target_list_file()
        print(f"{VPT_TARGET_LIST_FILE=} generated.")

    with open(VPT_TARGET_LIST_FILE, "r") as f:
        vpt_target_list = [Path(line.strip()) for line in f.readlines()]
        print(f"We will convert {len(vpt_target_list)=} VPT files.")

    for jsonl_file_path in tqdm(vpt_target_list):
        print(f"Converting {jsonl_file_path=} to OWAMcap format.")
        # Convert the file to OWAMcap format

        mcap_file_path = jsonl_file_path.with_suffix(".mcap")
        mp4_file_path = jsonl_file_path.with_suffix(".mp4")

        # Writing messages to an OWAMcap file

        unix_epoch_ns = 0  # Unix epoch time in nanoseconds (Jan 1, 1970)
        center_x, center_y = (
            VPT_X_RESOLUTION // 2,
            VPT_Y_RESOLUTION // 2,
        )  # x, y coordinates of the center of the screen (1280x720)

        try:
            with open(jsonl_file_path, "r") as f:  # jsonl file
                lines = [line.strip() for line in f.readlines()]  # Read non-empty lines
                assert len(lines) == VPT_EXPECTED_TICKS, (
                    f"File {jsonl_file_path} does not have {VPT_EXPECTED_TICKS=} lines. It has {len(lines)} lines."
                )
                ticks = [json.loads(line) for line in lines]
        except Exception as e:
            print(f"Error reading {jsonl_file_path}. Skipping. Error: {e}")
            continue

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
                            event = KeyboardEvent(event_type="press", vk=VPT_KEYBOARD_VK_MAPPING[key])
                            writer.write_message(topic, event, log_time=log_time)

                # release keys that are not in the current tick
                for state_key in list(keyboard_state):
                    if state_key not in current_tick_keys:
                        keyboard_state.remove(state_key)
                        topic = "keyboard"
                        event = KeyboardEvent(event_type="release", vk=VPT_KEYBOARD_VK_MAPPING[state_key])
                        writer.write_message(topic, event, log_time=log_time)

                ## MOUSE EVENT
                dx = tick["mouse"]["dx"]
                dy = tick["mouse"]["dy"]

                # NOTE: we suppose the mouse coordinates are integer values
                dx = round(int(dx))
                dy = round(int(dy))

                if dx != 0 or dy != 0:
                    # NOTE: we suppose the mouse is pinned to the center. it takes VPT_MOUSE_PIN_NS for the program to pin the mouse to the center
                    topic = "mouse"
                    event = MouseEvent(event_type="move", x=center_x + dx, y=center_y + dy)
                    writer.write_message(topic, event, log_time=log_time - VPT_MOUSE_PIN_NS)

                    topic = "mouse"
                    event = MouseEvent(event_type="move", x=center_x, y=center_y)
                    writer.write_message(topic, event, log_time=log_time)

        # Reading messages from an OWAMcap file
        # read_mcap(mcap_file_path)


def read_mcap(file_path="expert.mcap", num_messages=100):
    # Reading messages from an OWAMcap file
    cnt = 0
    with OWAMcapReader(file_path) as reader:
        for mcap_msg in reader.iter_messages():
            print(f"Topic: {mcap_msg.topic}, Timestamp: {mcap_msg.timestamp}, Message: {mcap_msg.decoded}")
            cnt += 1
            if cnt > num_messages:
                break


if __name__ == "__main__":
    main()
    # read_mcap()
