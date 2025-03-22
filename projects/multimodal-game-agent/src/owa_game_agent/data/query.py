from pathlib import Path

import line_profiler
from pydantic import BaseModel

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits

from .sample import OWATrainingSample

ORIGINAL_FRAME_RATE = 60  # TODO: adaptive logic
MIN_SCREEN_NUM = 5
SCREEN_TOLERANCE = 0.05  # 0.05 seconds tolerance for screen state


class OWAMcapQuery(BaseModel):
    file_path: Path | None = None

    anchor_timestamp_ns: int

    past_range_ns: int
    future_range_ns: int

    screen_framerate: int = 20

    @line_profiler.profile  # profile: 0.723s
    def to_sample(self) -> OWATrainingSample:
        """
        Extracts a training sample from the MCAP file.

        TODO: optimization of this method. each `iter_decoded_messages` takes 10ms and whole `to_sample` takes 30ms.
        TODO: separate various ValueError into multiple cases and log part of them.
        """
        with OWAMcapReader(self.file_path) as reader:
            if (
                self.anchor_timestamp_ns - self.past_range_ns < reader.start_time
                or self.anchor_timestamp_ns + self.future_range_ns > reader.end_time
            ):
                raise ValueError("Query timestamp is out of range")

            # ===== Prepare state_keyboard, state_mouse =====

            last_keyboard_state = None
            keyboard_events_from_last_state = []
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard/state", "keyboard"], end_time=self.anchor_timestamp_ns, reverse=True
            ):  # profile: 31.2% (0.451s)
                if topic == "keyboard/state":
                    last_keyboard_state = msg["buttons"]
                    break
                elif topic == "keyboard":
                    keyboard_events_from_last_state.append((timestamp, msg))
            else:
                raise ValueError("No keyboard state found")

            pressed_vks = set(last_keyboard_state)
            for timestamp, msg in keyboard_events_from_last_state[::-1]:
                if msg["event_type"] == "press":
                    pressed_vks.add(msg["vk"])
                elif msg["event_type"] == "release":
                    try:
                        pressed_vks.remove(msg["vk"])
                    except KeyError:
                        raise ValueError(f"Key release event without press event: {msg}")
                else:
                    raise ValueError(f"Invalid event type: {msg['event_type']}")

            state_mouse = {"pressed": {}, "x": 0, "y": 0}
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["mouse"], end_time=self.anchor_timestamp_ns, reverse=True
            ):  # profile: 23.0% (0.332s)
                state_mouse["x"], state_mouse["y"] = msg["x"], msg["y"]
                break
            else:
                # this line is to temporarily support keyboard-only game. TODO: erase this line.
                state_mouse["x"], state_mouse["y"] = (0, 0)
                # raise ValueError("No mouse state found")

            # in Windows, vk 1/2/4 corresponds to left/right/middle mouse button. TODO: multi-os support
            mouse_keys = {1: "left", 2: "right", 4: "middle"}
            state_keyboard = set(pressed_vks) - {1, 2, 4}
            pressed_mks = set(pressed_vks) & {1, 2, 4}
            state_mouse["pressed"] = set([mouse_keys[button] for button in pressed_mks])

            # ===== Prepare state_screen =====

            state_screen = []
            start_time, end_time = (
                self.anchor_timestamp_ns - self.past_range_ns - SCREEN_TOLERANCE * TimeUnits.SECOND,
                self.anchor_timestamp_ns,
            )
            for idx, (topic, timestamp, msg) in enumerate(
                reader.iter_decoded_messages(
                    topics=["screen"],
                    start_time=start_time,
                    end_time=end_time,
                    reverse=True,  # ensure that the latest screen is included
                )
            ):  # profile: 23.0% (0.332s)
                if idx % (ORIGINAL_FRAME_RATE // self.screen_framerate):
                    continue
                # msg.path: ztype.mkv
                # self.file_path: absolute path to the mcap file
                # convert msg.path to absolute path, with relative_to
                msg.path = (self.file_path.parent / msg.path).as_posix()
                state_screen.append((timestamp - self.anchor_timestamp_ns, msg))

            if len(state_screen) < MIN_SCREEN_NUM:
                raise ValueError(
                    f"Not enough screen states found, expected {MIN_SCREEN_NUM} but got {len(state_screen)}"
                )

            # restrict to the last MIN_SCREEN_NUM states
            state_screen = state_screen[:MIN_SCREEN_NUM]

            # convert to oldest-to-latest order
            state_screen = state_screen[::-1]

            # ===== Prepare action_keyboard, action_mouse =====

            action_keyboard = []
            action_mouse = []
            _state_keyboard = state_keyboard.copy()
            start_time, end_time = self.anchor_timestamp_ns, self.anchor_timestamp_ns + self.future_range_ns
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard", "mouse"],
                start_time=start_time,
                end_time=end_time,
            ):  # profile: 22.6% (0.327s)
                if topic == "keyboard":
                    if msg.event_type == "press" and msg.vk in _state_keyboard:
                        continue
                    action_keyboard.append((timestamp - self.anchor_timestamp_ns, msg))
                    if msg.event_type == "press":
                        _state_keyboard.add(msg.vk)
                    elif msg.event_type == "release":
                        try:
                            _state_keyboard.remove(msg.vk)
                        except KeyError:
                            raise ValueError(f"Key release event without press event: {msg}")
                elif topic == "mouse":
                    action_mouse.append((timestamp - self.anchor_timestamp_ns, msg))
                else:
                    raise ValueError(f"Unexpected topic: {topic}")

        return OWATrainingSample(
            state_keyboard=state_keyboard,
            state_mouse=state_mouse,
            state_screen=state_screen,
            action_keyboard=action_keyboard,
            action_mouse=action_mouse,
        )
