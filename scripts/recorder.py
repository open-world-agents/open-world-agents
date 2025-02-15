import time
from typing import Optional

import orjson
import typer
from pydantic import BaseModel
from typing_extensions import Annotated

from owa.registry import CALLABLES, LISTENERS, RUNNABLES, activate_module


class BagEvent(BaseModel):
    timestamp_ns: int
    event_src: str
    event_data: bytes


def write_event_into_jsonl(event, source=None):
    # you can find where the event is coming from. e.g. where the calling this function
    # frame = inspect.currentframe().f_back

    with open("event.jsonl", "ab") as f:
        if isinstance(event, BaseModel):
            event_data = event.model_dump_json().encode("utf-8")
        else:
            event_data = orjson.dumps(event)
        bag_event = BagEvent(timestamp_ns=time.time_ns(), event_src=source, event_data=event_data)
        f.write(bag_event.model_dump_json().encode("utf-8") + b"\n")


def window_publisher_callback(event):
    write_event_into_jsonl(event, source="window_publisher")


def control_publisher_callback(*event):
    write_event_into_jsonl(event, source="control_publisher")


def configure():
    activate_module("owa_env_desktop")
    activate_module("owa_env_gst")


def main(
    file_location: Annotated[str, typer.Argument(help="The location of the output file, use `.mkv` extension.")],
    *,
    record_audio: Annotated[bool, typer.Option(help="Whether to record audio")] = True,
    record_video: Annotated[bool, typer.Option(help="Whether to record video")] = True,
    record_timestamp: Annotated[bool, typer.Option(help="Whether to record timestamp")] = True,
    window_name: Annotated[
        Optional[str], typer.Option(help="The name of the window to capture, substring of window name is supported")
    ] = None,
    monitor_idx: Annotated[Optional[int], typer.Option(help="The index of the monitor to capture")] = None,
):
    assert file_location.endswith(".mkv"), "The output file must have `.mkv` extension."

    recorder = RUNNABLES["screen/recorder"]()
    keyboard_listener = LISTENERS["keyboard"](control_publisher_callback)
    mouse_listener = LISTENERS["mouse"](control_publisher_callback)
    recorder.configure(
        filesink_location=file_location,
        record_audio=record_audio,
        record_video=record_video,
        record_timestamp=record_timestamp,
        window_name=window_name,
        monitor_idx=monitor_idx,
    )
    keyboard_listener.configure()
    mouse_listener.configure()

    try:
        # TODO?: add `wait` method to Runnable, which waits until the Runnable is ready to operate well.
        recorder.start()
        keyboard_listener.start()
        mouse_listener.start()
        while True:
            active_window = CALLABLES["window.get_active_window"]()
            window_publisher_callback(active_window)
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
        recorder.join()


if __name__ == "__main__":
    configure()
    typer.run(main)
