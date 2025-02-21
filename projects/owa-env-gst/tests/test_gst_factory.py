import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # noqa: E402

from owa_env_gst import gst_factory  # noqa: E402

Gst.init(None)


def test_recorder():
    pipeline = gst_factory.recorder_pipeline(
        filesink_location="test.mkv",
        record_audio=True,
        record_video=True,
        record_timestamp=True,
        enable_appsink=False,
        enable_fpsdisplaysink=True,
        fps=60,
    )
    assert (
        pipeline
        == "d3d11screencapturesrc show-cursor=true do-timestamp=true ! videorate drop-only=true ! video/x-raw(memory:D3D11Memory),framerate=0/1,max-framerate=60/1 ! tee name=t t. ! queue leaky=downstream ! d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink t. ! queue ! d3d11convert ! mfh264enc ! h264parse ! queue ! mux. wasapi2src do-timestamp=true loopback=true low-latency=true ! audioconvert ! mfaacenc ! queue ! mux. utctimestampsrc interval=1 ! subparse ! queue ! mux. matroskamux name=mux ! filesink location=test.mkv"
    )
    pipeline = Gst.parse_launch(pipeline)


def test_screen_capture():
    pipeline = gst_factory.screen_capture_pipeline()
    assert (
        pipeline
        == "d3d11screencapturesrc show-cursor=true do-timestamp=true ! videorate drop-only=true ! video/x-raw(memory:D3D11Memory),framerate=0/1,max-framerate=60/1 ! tee name=t t. ! queue leaky=downstream ! d3d11download ! videoconvert ! video/x-raw,format=BGRA ! appsink name=appsink sync=true max-buffers=1 drop=true emit-signals=true"
    )
    pipeline = Gst.parse_launch(pipeline)
