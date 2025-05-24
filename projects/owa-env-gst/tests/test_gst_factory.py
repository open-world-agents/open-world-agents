import os

import pytest

import owa.env.gst  # must import 'gst' to enable GST_PLUGIN_PATH environment variable setup


@pytest.fixture
def gstreamer():
    """Fixture that imports and initializes GStreamer at runtime."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    if not Gst.is_initialized():
        Gst.init(None)

    return Gst, owa.env.gst.pipeline_builder


def check_plugin(gst, plugin_name):
    """
    Check if a plugin is loaded and provide diagnostic info if not.

    Args:
        gst: Initialized GStreamer module
        plugin_name: Name of the plugin to check

    Returns:
        tuple: (is_loaded, diagnostic_info)
    """
    registry = gst.Registry.get()
    plugin = registry.find_plugin(plugin_name)

    if plugin:
        return True, f"Plugin '{plugin_name}' is available (version: {plugin.get_version()})"

    # If plugin not found, gather diagnostic information
    diagnostic_info = []
    diagnostic_info.append(f"Plugin '{plugin_name}' is NOT available")

    # Get plugin search paths
    paths = os.environ.get("GST_PLUGIN_PATH", "").split(os.pathsep)
    paths.append(os.environ.get("GST_PLUGIN_SYSTEM_PATH", ""))
    diagnostic_info.append(f"GST plugin search paths: {paths}")

    # List all available plugins for comparison
    all_plugins = [p.get_name() for p in registry.get_plugin_list()]
    diagnostic_info.append(f"Total plugins available: {len(all_plugins)}")

    # Similar plugin names that might be confused with the requested one
    similar_plugins = [p for p in all_plugins if plugin_name.lower() in p.lower()]
    if similar_plugins:
        diagnostic_info.append(f"Similar plugins found: {similar_plugins}")

    return False, "\n".join(diagnostic_info)


def check_element(gst, element_name):
    """
    Check if an element is available and provide diagnostic info if not.

    Args:
        gst: Initialized GStreamer module
        element_name: Name of the element to check

    Returns:
        tuple: (is_available, diagnostic_info)
    """
    registry = gst.Registry.get()
    factory = registry.find_feature(element_name, gst.ElementFactory)

    if factory:
        plugin = factory.get_plugin()
        plugin_name = plugin.get_name() if plugin else "unknown"
        return True, f"Element '{element_name}' is available (from plugin: {plugin_name})"

    # If element not found, gather diagnostic information
    diagnostic_info = []
    diagnostic_info.append(f"Element '{element_name}' is NOT available")

    # List similar element names
    all_elements = [f.get_name() for f in registry.get_feature_list(gst.ElementFactory)]
    similar_elements = [e for e in all_elements if any(part in e.lower() for part in element_name.lower().split("_"))]
    if similar_elements:
        diagnostic_info.append(f"Similar elements found: {similar_elements[:10]}")
        if len(similar_elements) > 10:
            diagnostic_info.append(f"...and {len(similar_elements) - 10} more")

    return False, "\n".join(diagnostic_info)


def test_required_plugins(gstreamer):
    """Test to verify all required GStreamer plugins are available."""
    Gst, _ = gstreamer

    required_plugins = ["d3d11", "nvcodec", "wasapi2"]
    missing_plugins = []

    for plugin in required_plugins:
        is_loaded, info = check_plugin(Gst, plugin)
        print(info)  # Print for pytest output
        if not is_loaded:
            missing_plugins.append(plugin)

    required_elements = ["d3d11screencapturesrc", "nvd3d11h265enc", "wasapi2src"]
    missing_elements = []

    for element in required_elements:
        is_available, info = check_element(Gst, element)
        print(info)  # Print for pytest output
        if not is_available:
            missing_elements.append(element)

    if missing_plugins or missing_elements:
        pytest.skip(f"Missing plugins: {missing_plugins}, Missing elements: {missing_elements}")


def test_subprocess_recorder(gstreamer):
    Gst, pipeline_builder = gstreamer

    # Check required elements before running test
    for element in ["d3d11screencapturesrc", "nvd3d11h265enc", "wasapi2src"]:
        is_available, info = check_element(Gst, element)
        if not is_available:
            pytest.skip(info)

    pipeline = pipeline_builder.subprocess_recorder_pipeline(
        filesink_location="test.mkv",
        record_audio=True,
        record_video=True,
        record_timestamp=True,
        enable_fpsdisplaysink=True,
        fps=60,
    )
    expected_pipeline = (
        "d3d11screencapturesrc show-cursor=true do-timestamp=true window-capture-mode=client show-border=True ! "
        "videorate drop-only=true ! d3d11scale ! video/x-raw(memory:D3D11Memory),framerate=60/1 ! "
        "tee name=t "
        "t. ! queue leaky=downstream ! d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink "
        "t. ! queue ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! nvd3d11h265enc ! h265parse ! queue ! mux. "
        "wasapi2src do-timestamp=true loopback=true low-latency=true loopback-mode=include-process-tree ! audioconvert ! avenc_aac ! queue ! mux. "
        "utctimestampsrc interval=1 ! subparse ! queue ! mux. "
        "matroskamux name=mux ! filesink location=test.mkv"
    )
    assert pipeline == expected_pipeline
    pipeline = Gst.parse_launch(pipeline)


def test_screen_capture(gstreamer):
    Gst, pipeline_builder = gstreamer

    # Check required elements before running test
    if not check_element(Gst, "d3d11screencapturesrc")[0]:
        pytest.skip("d3d11screencapturesrc element not available")

    pipeline = pipeline_builder.screen_capture_pipeline()
    expected_pipeline = (
        "d3d11screencapturesrc show-cursor=true do-timestamp=true window-capture-mode=client show-border=True ! "
        "videorate drop-only=true ! d3d11scale ! video/x-raw(memory:D3D11Memory),framerate=60/1 ! "
        "queue leaky=downstream ! d3d11download ! videoconvert ! video/x-raw,format=BGRA ! "
        "appsink name=appsink sync=false max-buffers=1 drop=true emit-signals=true wait-on-eos=false"
    )
    assert pipeline == expected_pipeline
    pipeline = Gst.parse_launch(pipeline)
