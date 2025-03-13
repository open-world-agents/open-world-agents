# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "open-world-agents[envs]",
#     "orjson",
#     "typer",
# ]
#
# [tool.uv.sources]
# open-world-agents = { path = "../" }
# ///
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional

import typer
from loguru import logger
from pydantic import BaseModel
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapWriter
from owa.registry import CALLABLES, LISTENERS, activate_module

queue = Queue()
MCAP_LOCATION = None


class BagEvent(BaseModel):
    timestamp_ns: int
    event_src: str
    event_data: bytes


def callback(event, topic=None):
    queue.put((topic, event, time.time_ns()))


def keyboard_publisher_callback(event):
    callback(event, topic="keyboard")


def mouse_publisher_callback(event):
    callback(event, topic="mouse")


def screen_publisher_callback(event):
    global MCAP_LOCATION
    event.path = Path(event.path).relative_to(MCAP_LOCATION.parent).as_posix()
    callback(event, topic="screen")


def publish_window_info():
    while True:
        active_window = CALLABLES["window.get_active_window"]()
        pressed_vk_list = CALLABLES["keyboard.get_pressed_vk_list"]()
        callback(active_window, topic="window")
        callback(pressed_vk_list, topic="keyboard/state")
        time.sleep(1)


def configure():
    activate_module("owa_env_desktop")
    activate_module("owa_env_gst")


USER_INSTRUCTION = """

Since this recorder records all screen/keyboard/mouse/window events, be aware NOT to record sensitive information, such as passwords, credit card numbers, etc.

Press Ctrl+C to stop recording.

"""


def record(
    file_location: Annotated[
        Path,
        typer.Argument(
            help="The location of the output file. If `output.mcap` is given as argument, the output file would be `output.mcap` and `output.mkv`."
        ),
    ],
    *,
    record_audio: Annotated[bool, typer.Option(help="Whether to record audio")] = True,
    record_video: Annotated[bool, typer.Option(help="Whether to record video")] = True,
    record_timestamp: Annotated[bool, typer.Option(help="Whether to record timestamp")] = True,
    show_cursor: Annotated[bool, typer.Option(help="Whether to show the cursor in the capture")] = True,
    window_name: Annotated[
        Optional[str], typer.Option(help="The name of the window to capture, substring of window name is supported")
    ] = None,
    monitor_idx: Annotated[Optional[int], typer.Option(help="The index of the monitor to capture")] = None,
    additional_args: Annotated[
        Optional[str],
        typer.Option(
            help="Additional arguments to pass to the pipeline. For detail, see https://gstreamer.freedesktop.org/documentation/d3d11/d3d11screencapturesrc.html"
        ),
    ] = None,
):
    """Record screen, keyboard, mouse, and window events to an `.mcap` and `.mkv` file."""
    global MCAP_LOCATION
    output_file = file_location.with_suffix(".mcap")
    MCAP_LOCATION = output_file

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created directory {output_file.parent}")

    # delete the file if it exists
    if output_file.exists():
        output_file.unlink()
        logger.warning(f"Deleted existing file {output_file}")

    configure()
    recorder = LISTENERS["owa_env_gst/omnimodal/appsink_recorder"]()
    keyboard_listener = LISTENERS["keyboard"]().configure(callback=keyboard_publisher_callback)
    mouse_listener = LISTENERS["mouse"]().configure(callback=mouse_publisher_callback)

    additional_properties = {}
    if additional_args is not None:
        for arg in additional_args.split(","):
            key, value = arg.split("=")
            additional_properties[key] = value
    recorder.configure(
        filesink_location=file_location.with_suffix(".mkv"),
        record_audio=record_audio,
        record_video=record_video,
        record_timestamp=record_timestamp,
        show_cursor=show_cursor,
        window_name=window_name,
        monitor_idx=monitor_idx,
        additional_properties=additional_properties,
        callback=screen_publisher_callback,
    )
    window_thread = threading.Thread(target=publish_window_info, daemon=True)
    writer = OWAMcapWriter(output_file)

    logger.info(USER_INSTRUCTION)

    try:
        # TODO?: add `wait` method to Runnable, which waits until the Runnable is ready to operate well.
        recorder.start()
        keyboard_listener.start()
        mouse_listener.start()
        window_thread.start()

        while True:
            topic, event, publish_time = queue.get()
            latency = time.time_ns() - publish_time
            # warn if latency is too high, i.e., > 20ms
            if latency / 1e6 > 20:
                logger.warning(f"Event {event} from {topic} is written to the file with latency {latency / 1e6:.2f}ms")
            writer.write_message(topic, event, publish_time=publish_time)

    except KeyboardInterrupt:
        logger.info("Recording stopped by user.")
    finally:
        # resource cleanup
        try:
            writer.finish()
            logger.info(f"Output file saved to {output_file}")
        except Exception as e:
            logger.error(f"Error occurred while saving the output file: {e}")

        try:
            recorder.stop()
            recorder.join(timeout=5)
        except Exception as e:
            logger.error(f"Error occurred while stopping the recorder: {e}")

        try:
            keyboard_listener.stop()
            mouse_listener.stop()
            keyboard_listener.join(timeout=5)
            mouse_listener.join(timeout=5)
        except Exception as e:
            logger.error(f"Error occurred while stopping the listeners: {e}")

        # window_thread.join()


if __name__ == "__main__":
    typer.run(record)
