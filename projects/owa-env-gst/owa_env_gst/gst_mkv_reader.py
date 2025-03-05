# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import gi

gi.require_version("Gst", "1.0")

import time
from pathlib import Path

import cv2
import numpy as np
from gi.repository import Gst
from loguru import logger

from owa_env_gst import GstPipelineRunner
from owa_env_gst.utils import sample_to_ndarray

# Initialize GStreamer
if not Gst.is_initialized():
    Gst.init(None)


def video_callback(sample: Gst.Sample, appsink: Gst.Element):
    """
    Process video frames from the pipeline.

    Args:
        sample: GStreamer sample containing video frame
        appsink: Source appsink element
    """
    arr = sample_to_ndarray(sample)
    logger.info(f"[Video] Received frame of shape {arr.shape} from appsink '{appsink.get_name()}'")
    video_callback.count = getattr(video_callback, "count", 0) + 1
    if video_callback.count < 5:
        cv2.imwrite(f"video_frame_{video_callback.count}.png", arr)


def audio_callback(sample: Gst.Sample, appsink: Gst.Element):
    """
    Process audio samples from the pipeline.

    Args:
        sample: GStreamer sample containing audio data
        appsink: Source appsink element
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    rate = structure.get_value("rate")
    channels = structure.get_value("channels")

    audio_data = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(audio_data, dtype=np.int16)
    logger.info(
        f"[Audio] Received audio data of length {arr.shape[0]} "
        f"(rate={rate}, channels={channels}) from appsink '{appsink.get_name()}'"
    )


def subtitle_callback(sample: Gst.Sample, appsink: Gst.Element):
    """
    Process subtitle data from the pipeline.

    Args:
        sample: GStreamer sample containing subtitle data
        appsink: Source appsink element
    """
    buf = sample.get_buffer()
    subtitle_text = buf.extract_dup(0, buf.get_size()).decode("utf-8", errors="replace")
    logger.info(f"[Subtitle] Received subtitle: '{subtitle_text.strip()}' from appsink '{appsink.get_name()}'")


def sample_callback(sample: Gst.Sample, appsink: Gst.Element, pipeline: Gst.Pipeline):
    """
    Main callback function that routes samples to appropriate handlers.

    Args:
        sample: GStreamer sample
        appsink: Source appsink element
    """
    sink_name = appsink.get_name()

    if sink_name == "video_sink":
        # Process every 20th video frame on average
        if np.random.rand() < 1 / 20:
            video_callback(sample, appsink)
    elif sink_name == "audio_sink":
        # Process every 40th audio sample on average
        if np.random.rand() < 1 / 40:
            audio_callback(sample, appsink)
    elif sink_name == "subtitle_sink":
        subtitle_callback(sample, appsink)
    else:
        logger.warning(f"Unknown appsink name: {sink_name}")


class GstMKVReader(GstPipelineRunner):
    def on_configure(self, mkv_file_path: Path, framerate:str="6/1"):
        pipeline_description = f"""
        filesrc location={mkv_file_path} ! matroskademux name=demux

        demux.video_0 ! queue ! 
            decodebin ! videoconvert ! videorate ! 
            video/x-raw,framerate={framerate} ! videoconvert ! 
            video/x-raw,format=BGRA ! appsink name=video_sink sync=false emit-signals=true

        demux.audio_0 ! queue ! 
            decodebin ! audioconvert ! audioresample quality=4 ! 
            audio/x-raw,rate=44100,channels=2 ! 
            appsink name=audio_sink sync=false emit-signals=true

        demux.subtitle_0 ! queue ! 
            decodebin ! appsink name=subtitle_sink sync=false emit-signals=true
        """
        super().configure(pipeline_description)
        self.register_appsink_callback(sample_callback)

class GstMKVReader

if __name__ == "__main__":
    # Create and configure the pipeline runner
    runner = GstPipelineRunner().configure(pipeline_description)
    runner.register_appsink_callback(sample_callback)
    # runner.enable_fps_display()

    # Optional: Seek to a specific position before starting
    # runner.seek(start_time=0.4, end_time=1.5)

    try:
        # Start the pipeline
        runner.start()

        # Monitor the pipeline
        while runner.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping pipeline.")
        runner.stop()
    finally:
        runner.join()
        logger.info("Pipeline stopped.")
