import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst
from owa_env_gst import gst_factory

Gst.init(None)


def test_recorder():
    pipeline = gst_factory.recorder_pipeline(
        filesink_location="test.mp4",
        record_audio=True,
        record_video=True,
        record_timestamp=True,
        enable_appsink=False,
        enable_fpsdisplaysink=True,
        fps=60,
    )
    pipeline = Gst.parse_launch(pipeline)


def test_screen_capture():
    pipeline = gst_factory.screen_capture_pipeline()
    pipeline = Gst.parse_launch(pipeline)
