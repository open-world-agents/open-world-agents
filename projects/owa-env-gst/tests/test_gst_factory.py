import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst  # noqa: E402

from owa_env_gst import pipeline_builder  # noqa: E402

if not Gst.is_initialized():
    Gst.init(None)


def test_subprocess_recorder():
    pipeline = pipeline_builder.subprocess_recorder_pipeline(
        filesink_location="test.mkv",
        record_audio=True,
        record_video=True,
        record_timestamp=True,
        enable_fpsdisplaysink=True,
        fps=60,
    )
    expected_pipeline = (
        "d3d11screencapturesrc show-cursor=true do-timestamp=true ! "
        "videorate drop-only=true ! video/x-raw(memory:D3D11Memory),framerate=0/1,max-framerate=60/1 ! "
        "tee name=t "
        "t. ! queue leaky=downstream ! d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink "
        "t. ! queue ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! nvd3d11h265enc ! h265parse ! queue ! mux. "
        "wasapi2src do-timestamp=true loopback=true low-latency=true ! audioconvert ! avenc_aac ! queue ! mux. "
        "utctimestampsrc interval=1 ! subparse ! queue ! mux. "
        "matroskamux name=mux ! filesink location=test.mkv"
    )
    assert pipeline == expected_pipeline
    pipeline = Gst.parse_launch(pipeline)


def test_screen_capture():
    pipeline = pipeline_builder.screen_capture_pipeline()
    expected_pipeline = (
        "d3d11screencapturesrc show-cursor=true do-timestamp=true ! "
        "videorate drop-only=true ! video/x-raw(memory:D3D11Memory),framerate=0/1,max-framerate=60/1 ! "
        "queue leaky=downstream ! d3d11download ! videoconvert ! video/x-raw,format=BGRA ! "
        "appsink name=appsink sync=false max-buffers=1 drop=true emit-signals=true wait-on-eos=false"
    )
    assert pipeline == expected_pipeline
    pipeline = Gst.parse_launch(pipeline)
