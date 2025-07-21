import time

from owa.core import RUNNABLES


def test_screen_capture(tmp_path):
    """Test basic recorder functionality with minimal recording time."""
    output_file = tmp_path / "output.mkv"

    recorder = RUNNABLES["gst/omnimodal.subprocess_recorder"]()
    recorder.configure(filesink_location=str(output_file), window_name="open-world-agents")

    try:
        recorder.start()
        time.sleep(2)  # Minimal recording time
        recorder.stop()
        recorder.join(timeout=10)

        # Basic verification that the process completed
        assert not recorder.is_alive(), "Recorder should be stopped"

    finally:
        if recorder.is_alive():
            recorder.stop()
            recorder.join(timeout=3)
