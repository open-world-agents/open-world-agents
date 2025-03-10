from queue import Empty, Queue
from typing import Self

import line_profiler

from .gst_runner import GstPipelineRunner
from .utils import framerate_float_to_str, sample_to_ndarray


class MKVReader:
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

        pipeline_description = f"""
            filesrc location={mkv_file_path} ! matroskademux name=demux

            demux.video_0 ! queue ! 
                decodebin ! d3d11convert ! videorate drop-only=true ! 
                video/x-raw(memory:D3D11Memory),framerate={framerate_float_to_str(framerate)},format=BGRA ! d3d11download ! 
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
        self.frame_queue.put(data)  # Add the frame to the queue
        print("queued", data["pts"])

    def iter_frames(self):
        self.runner.start()  # Start the runner
        while self.runner.is_alive():
            try:
                data = self.frame_queue.get(timeout=1)
                yield data  # Yield frames as they become available
            except Empty:
                continue
        self.runner.stop()  # Stop the runner
        self.runner.join()  # Wait for the runner to finish

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runner.cleanup()
        return False
