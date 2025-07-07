from pathlib import Path
from typing import Optional

from loguru import logger

from owa.core.runner import SubprocessRunner

from ..pipeline_builder import subprocess_recorder_pipeline


class SubprocessRecorder(SubprocessRunner):
    """
    High-performance screen and audio recorder using GStreamer subprocess.

    This recorder runs GStreamer as a subprocess to capture screen content
    and audio, providing excellent performance and stability for long recordings.
    Supports various output formats and hardware acceleration.

    Examples:
        Basic screen recording with audio:

        >>> recorder = SubprocessRecorder()
        >>> recorder.configure(
        ...     filesink_location="recording.mkv",
        ...     record_audio=True,
        ...     record_video=True,
        ...     fps=30
        ... )
        >>> recorder.start()
        >>> # ... recording runs in background ...
        >>> recorder.stop()

        Video-only recording with custom settings:

        >>> recorder.configure(
        ...     filesink_location="video_only.mp4",
        ...     record_audio=False,
        ...     record_video=True,
        ...     fps=60,
        ...     show_cursor=False
        ... )
    """

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
    ) -> None:
        """
        Prepare the GStreamer pipeline command for subprocess recording.

        Args:
            filesink_location: Path where the recording will be saved.
            record_audio: Whether to include audio in the recording.
            record_video: Whether to include video in the recording.
            record_timestamp: Whether to include timestamp information.
            enable_fpsdisplaysink: Whether to enable FPS display during recording.
            show_cursor: Whether to show the cursor in the recording.
            fps: Frames per second for video recording.
            window_name: Specific window to record (optional).
            monitor_idx: Monitor index to record from (optional).
            additional_properties: Additional pipeline properties (optional).

        Returns:
            None: Configuration is stored internally for subprocess execution.
        """

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
