import tempfile
import time
from pathlib import Path

import pytest

from owa.core.registry import RUNNABLES, activate_module


# Automatically activate the desktop module for all tests in this session.
@pytest.fixture(scope="session", autouse=True)
def activate_owa_desktop():
    activate_module("owa.env.gst")


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_screen_capture(temp_output_dir):
    """Test basic recorder functionality with minimal recording time."""
    output_file = temp_output_dir / "output.mkv"

    recorder = RUNNABLES["owa.env.gst/omnimodal/subprocess_recorder"]()
    recorder.configure(filesink_location=str(output_file), window_name="open-world-agents")

    try:
        recorder.start()
        time.sleep(2)  # Minimal recording time
        recorder.stop()
        recorder.join(timeout=5)

        # Basic verification that the process completed
        assert not recorder.is_alive(), "Recorder should be stopped"

    finally:
        if recorder.is_alive():
            recorder.stop()
            recorder.join(timeout=3)
