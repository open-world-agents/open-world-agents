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
from tqdm import tqdm
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapWriter
from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa.core.time import TimeUnits

# how to use loguru with tqdm: https://github.com/Delgan/loguru/issues/135
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

# TODO: apply https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.env.gst")  # suppress pipeline print

queue = Queue()
MCAP_LOCATION = None


class BagEvent(BaseModel):
    timestamp_ns: int
    event_src: str
    event_data: bytes


def callback(event, topic=None):
    queue.put((topic, event, time.time_ns()))


def keyboard_publisher_callback(event):
    # info only for F1-F12 keys
    if 0x70 <= event.vk <= 0x7B and event.event_type == "press":
        logger.info(f"F1-F12 key pressed: F{event.vk - 0x70 + 1}")
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
        keyboard_state = CALLABLES["keyboard.get_state"]()
        mouse_state = CALLABLES["mouse.get_state"]()
        callback(active_window, topic="window")
        callback(keyboard_state, topic="keyboard/state")
        callback(mouse_state, topic="mouse/state")
        time.sleep(1)


def configure():
    activate_module("owa.env.desktop")
    activate_module("owa.env.gst")


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
    if output_file.exists() or output_file.with_suffix(".mkv").exists():
        delete = typer.confirm("The output file already exists. Do you want to delete it?")
        if not delete:
            print("The recording is aborted.")
            raise typer.Abort()
        output_file.unlink()
        logger.warning(f"Deleted existing file {output_file}")

    if window_name is not None:
        logger.warning(
            "⚠️ WINDOW CAPTURE LIMITATION (as of 2025-03-20) ⚠️\n"
            "When capturing a specific window, mouse coordinates cannot be accurately aligned with the window content due to "
            "limitations in the Windows API (WGC).\n\n"
            "RECOMMENDATION:\n"
            "- Use FULL SCREEN capture when you need mouse event tracking\n"
            "- Full screen mode in games works well if the video output matches your monitor resolution (e.g., 1920x1080)\n"
            "- Any non-fullscreen capture will have misaligned mouse coordinates in the recording"
        )

    configure()
    recorder = LISTENERS["owa.env.gst/omnimodal/appsink_recorder"]()
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
    pbar = tqdm(desc="Recording", unit="event", dynamic_ncols=True)

    logger.info(USER_INSTRUCTION)

    try:
        # TODO?: add `wait` method to Runnable, which waits until the Runnable is ready to operate well.
        recorder.start()
        keyboard_listener.start()
        mouse_listener.start()
        window_thread.start()

        while True:
            topic, event, publish_time = queue.get()
            pbar.update()

            latency = time.time_ns() - publish_time
            # warn if latency is too high, i.e., > 20ms
            if latency > 20 * TimeUnits.MSECOND:
                logger.warning(f"High latency: {latency / TimeUnits.MSECOND:.2f}ms while processing {topic} event.")
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
