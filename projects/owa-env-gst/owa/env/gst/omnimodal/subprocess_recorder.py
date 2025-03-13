from pathlib import Path
from typing import Optional

from loguru import logger

from owa.core.registry import RUNNABLES
from owa.core.runner import SubprocessRunner

from ..pipeline_builder import subprocess_recorder_pipeline


@RUNNABLES.register("owa.env.gst/omnimodal/subprocess_recorder")
class SubprocessRecorder(SubprocessRunner):
    """A ScreenRecorder Runnable that records video and/or audio using a GStreamer pipeline."""

    def on_configure(
        self,
        filesink_location: str,
        record_audio: bool = True,
        record_video: bool = True,
        record_timestamp: bool = True,
        enable_fpsdisplaysink: bool = True,
        show_cursor: bool = True,
        fps: float = 60,
        window_name: Optional[str] = None,
        monitor_idx: Optional[int] = None,
        additional_properties: Optional[dict] = None,
    ):
        """Prepare the GStreamer pipeline command."""

        # if filesink_location does not exist, create it and warn the user
        if not Path(filesink_location).parent.exists():
            Path(filesink_location).parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Output directory {filesink_location} does not exist. Creating it.")

        # convert to posix path. this is required for gstreamer executable.
        filesink_location = Path(filesink_location).as_posix()

        pipeline_description = subprocess_recorder_pipeline(
            filesink_location=filesink_location,
            record_audio=record_audio,
            record_video=record_video,
            record_timestamp=record_timestamp,
            enable_fpsdisplaysink=enable_fpsdisplaysink,
            show_cursor=show_cursor,
            fps=fps,
            window_name=window_name,
            monitor_idx=monitor_idx,
            additional_properties=additional_properties,
        )

        super().on_configure(f"gst-launch-1.0.exe -e -v {pipeline_description}".split())
