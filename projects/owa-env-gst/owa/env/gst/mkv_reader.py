import platform
from abc import ABC, abstractmethod
from queue import Empty, Full, Queue
from typing import Self

import av

from .gst_runner import GstPipelineRunner
from .utils import framerate_float_to_str, sample_to_ndarray


class MKVReader(ABC):
    @abstractmethod
    def seek(self, start_time: float, end_time: float) -> Self: ...

    @abstractmethod
    def iter_frames(self): ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def __enter__(self): ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb): ...


class GstMKVReader(MKVReader):
    def __init__(self, mkv_file_path: str, stream: str = "video_0", framerate: float = 60.0):
        stream_type = stream.split("_")[0]
        if stream_type not in ["video", "audio"]:
            raise ValueError("stream must be either video or audio")

        assert stream_type == "video", "audio stream not supported yet"
        # """
        # demux.audio_0 ! queue !
        #         decodebin ! audioconvert ! audioresample quality=4 !
        #         audio/x-raw,rate=44100,channels=2 !
        #         appsink name=audio_sink
        # """

        if platform.system() == "Windows":
            pipeline_description = f"""
                filesrc location={mkv_file_path} ! matroskademux name=demux

                demux.video_0 ! queue ! 
                    decodebin ! d3d11convert ! videorate drop-only=true ! 
                    video/x-raw(memory:D3D11Memory),framerate={framerate_float_to_str(framerate)},format=BGRA ! d3d11download ! 
                    appsink name=video_sink sync=false emit-signals=true wait-on-eos=false max-bytes=1000000000 drop=false
            """
        else:
            pipeline_description = f"""
                filesrc location={mkv_file_path} ! matroskademux name=demux

                demux.video_0 ! queue ! 
                    decodebin ! videoconvert ! videorate drop-only=true ! 
                    video/x-raw,framerate={framerate_float_to_str(framerate)},format=BGRA ! 
                    appsink name=video_sink sync=false emit-signals=true wait-on-eos=false max-bytes=1000000000 drop=false
            """
        # ensure drop property is set to false, to ensure ALL frames are emitted
        runner = GstPipelineRunner().configure(pipeline_description, do_not_modify_appsink_properties=True)
        runner.register_appsink_callback(self._sample_callback)
        self.runner = runner  # Assign runner to an instance variable
        self.frame_queue = Queue()  # Initialize a queue to store frames

    def seek(self, start_time: float, end_time: float) -> Self:
        self.runner.seek(start_time=start_time, end_time=end_time)
        return self

    def _sample_callback(self, sample):
        frame_arr = sample_to_ndarray(sample)
        data = {"data": frame_arr, "pts": sample.get_buffer().pts}
        try:
            self.frame_queue.put_nowait(data)  # Use put_nowait to avoid blocking callback if queue is full
        except Full:
            # Handle queue full scenario, e.g., log a warning or drop frame if acceptable
            # For now, we'll let it drop if the queue is full, as appsink also has max-buffers
            pass

    def iter_frames(self):
        self.runner.start()  # Start the runner
        try:
            while self.runner.is_alive():
                try:
                    data = self.frame_queue.get(timeout=1.0)  # Wait up to 1 second for a frame
                    yield data
                except Empty:
                    # Runner is alive, but queue was empty for the timeout duration. Continue waiting.
                    continue

            # Runner is no longer alive (e.g., EOS or error).
            # Drain any remaining frames that were already in the queue.
            while True:
                try:
                    data = self.frame_queue.get_nowait()  # Non-blocking get
                    yield data
                except Empty:
                    # Queue is empty, all frames processed.
                    break
        finally:
            self.runner.stop()  # Ensure runner is stopped
            self.runner.join()  # Wait for the runner thread to finish

    def close(self):
        self.runner.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# BUG: PyAV has "corrupted size vs. prev_size" error when `frame.to_ndarray(format="bgra")` is called for video "expert-jy-1.mkv"
#      This bug does not occur when format does not contain `alpha` channel, e.g. "bgr24"
#      Guessed reason is mismatch of width/height=770/512 and codec_width/codec_height=800/512.
class PyAVMKVReader(MKVReader):
    def __init__(self, mkv_file_path: str, stream: str = "video_0"):
        # Parse the stream type and index
        stream_type, stream_index = stream.split("_")
        stream_index = int(stream_index)

        if stream_type not in ["video", "audio"]:
            raise ValueError("stream must be either video or audio")

        assert stream_type == "video", "audio stream not supported yet"

        # Open the container
        self.container = av.open(mkv_file_path)

        # Get the appropriate stream
        if stream_type == "video":
            self.stream = self.container.streams.video[stream_index]
        else:
            self.stream = self.container.streams.audio[stream_index]

        # Set desired output format
        self.format = "bgra"

        # Initialize time boundaries
        self.start_time = 0
        self.end_time = float("inf")

    def seek(self, start_time: float, end_time: float) -> Self:
        self.start_time = start_time
        self.end_time = end_time

        # Convert to stream's time base
        start_ts = int(start_time * av.time_base)

        # Seek to the start position (with backward flag to ensure we get keyframe)
        self.container.seek(start_ts, backward=True)

        return self

    def iter_frames(self):
        for frame in self.container.decode(video=0):
            # Calculate frame timestamp in seconds
            pts_seconds = frame.pts * float(self.stream.time_base)

            # Skip frames before start_time
            if pts_seconds < self.start_time:
                continue

            # Stop iteration if we've reached end_time
            if pts_seconds > self.end_time:
                break

            # Convert frame to ndarray in BGRA format
            frame_array = frame.to_ndarray(format=self.format)

            yield {"data": frame_array, "pts": frame.pts}

    def close(self):
        if hasattr(self, "container") and self.container:
            self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
