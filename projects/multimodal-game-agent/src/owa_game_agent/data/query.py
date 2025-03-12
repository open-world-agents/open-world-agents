from pydantic import BaseModel

from mcap_owa.highlevel import OWAMcapReader

from .sample import OWATrainingSample


class OWAMcapQuery(BaseModel):
    file_path: str | None = None

    anchor_timestamp_ns: int

    past_range_ns: int
    future_range_ns: int

    screen_framerate: int = 5

    def to_sample(self) -> OWATrainingSample:
        """
        Extracts a training sample from the MCAP file.

        TODO: optimization of this method. each `iter_decoded_messages` takes 10ms and whole `to_sample` takes 30ms.
        """
        with OWAMcapReader(self.file_path) as reader:
            state_keyboard = None
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard/state"], end_time=self.anchor_timestamp_ns - self.past_range_ns, reverse=True
            ):
                state_keyboard = msg["pressed_vk_list"]
                break
            else:
                raise ValueError("No keyboard state found")

            state_screen = []
            start_time, end_time = self.anchor_timestamp_ns - self.past_range_ns, self.anchor_timestamp_ns
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["screen"],
                start_time=start_time,
                end_time=end_time,
            ):
                state_screen.append((timestamp - self.anchor_timestamp_ns, msg))

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
            state_screen=state_screen,
            action_keyboard=action_keyboard,
            action_mouse=action_mouse,
        )
