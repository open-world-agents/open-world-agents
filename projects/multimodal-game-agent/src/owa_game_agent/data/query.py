from pathlib import Path

from pydantic import BaseModel

from mcap_owa.highlevel import OWAMcapReader

from .sample import OWATrainingSample

ORIGINAL_FRAME_RATE = 60  # TODO: adaptive logic
MIN_SCREEN_NUM = 5


class OWAMcapQuery(BaseModel):
    file_path: Path | None = None

    anchor_timestamp_ns: int

    past_range_ns: int
    future_range_ns: int

    screen_framerate: int = 20

    def to_sample(self) -> OWATrainingSample:
        """
        Extracts a training sample from the MCAP file.

        TODO: optimization of this method. each `iter_decoded_messages` takes 10ms and whole `to_sample` takes 30ms.
        """
        with OWAMcapReader(self.file_path) as reader:
            if (
                self.anchor_timestamp_ns - self.past_range_ns < reader.start_time
                or self.anchor_timestamp_ns + self.future_range_ns > reader.end_time
            ):
                raise ValueError("Query timestamp is out of range")

            pressed_vks = None
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard/state"], end_time=self.anchor_timestamp_ns - self.past_range_ns, reverse=True
            ):
                pressed_vks = msg["pressed_vk_list"]
                break
            else:
                raise ValueError("No keyboard state found")

            state_mouse = {"pressed": {}, "x": 0, "y": 0}
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["mouse"], end_time=self.anchor_timestamp_ns - self.past_range_ns, reverse=True
            ):
                state_mouse["x"], state_mouse["y"] = msg["x"], msg["y"]
                break
            else:
                raise ValueError("No keyboard state found")

            # in Windows, vk 1/2/4 corresponds to left/right/middle mouse button. TODO: multi-os support
            mouse_keys = {1: "left", 2: "right", 4: "middle"}
            state_keyboard = set(pressed_vks) - {1, 2, 4}
            pressed_mks = set(pressed_vks) & {1, 2, 4}
            state_mouse["pressed"] = set([mouse_keys[button] for button in pressed_mks])

            state_screen = []
            start_time, end_time = self.anchor_timestamp_ns - self.past_range_ns, self.anchor_timestamp_ns
            for idx, (topic, timestamp, msg) in enumerate(
                reader.iter_decoded_messages(
                    topics=["screen"],
                    start_time=start_time,
                    end_time=end_time,
                    reverse=True,  # ensure that the latest screen is included
                )
            ):
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

            # convert to oldest-to-latest order
            state_screen = state_screen[::-1]

            action_keyboard = []
            action_mouse = []
            start_time, end_time = self.anchor_timestamp_ns, self.anchor_timestamp_ns + self.future_range_ns
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard", "mouse"],
                start_time=start_time,
                end_time=end_time,
            ):
                if topic == "keyboard":
                    action_keyboard.append((timestamp - self.anchor_timestamp_ns, msg))
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
