# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litserve",
#     "opencv-python>=4.11.0",
#     "owa-core",
# ]
# ///
import base64

import cv2
import litserve as ls

from owa.core.io.video import VideoReader


# TODO: batch decoding with torchcodec/PyNvVideoCodec
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        pass

    def decode_request(self, request):
        return (request["video_path"], request["pts"])

    # def batch(self, inputs): ...

    def predict(self, x):
        is_batch = isinstance(x, list)
        if not is_batch:
            x = [x]
        results = []
        for video_path, pts in x:
            with VideoReader(video_path) as reader:
                frame = reader.read_frame(pts=pts)
                frame_array = frame.to_ndarray(format="rgb24")
                results.append(frame_array)
        return results if is_batch else results[0]

    # def unbatch(self, output): ...

    def encode_response(self, output):
        # send bmp
        frame_bytes = cv2.imencode(".bmp", output)[1].tobytes()
        return {"frame": base64.b64encode(frame_bytes).decode("utf-8")}


if __name__ == "__main__":
    api = SimpleLitAPI(
        max_batch_size=1,  # default: 1
        batch_timeout=0.01,  # default: 0.0
    )
    server = ls.LitServer(
        api,
        accelerator="cpu",  # default: auto
        workers_per_device=1,  # default: 1
    )
    server.run(port=8001, generate_client_file=False, num_api_servers=None)
