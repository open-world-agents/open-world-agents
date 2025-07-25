# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litserve",
#     "opencv-python>=4.11.0",
#     "numpy",
# ]
# ///
import base64

import cv2
import litserve as ls
import numpy as np


def get_frame_pyav(video_path, time_sec):
    """Extract frame using PyAV."""
    import av

    with av.open(video_path) as container:
        container.seek(int(time_sec * av.time_base), any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts * frame.time_base >= time_sec:
                return np.asarray(frame.to_rgb().to_image())
    raise Exception(f"Failed to capture frame at time: {time_sec}")


# TODO: batch decoding with torchcodec/PyNvVideoCodec
# TODO? review https://lightning.ai/docs/litserve/features/async-concurrency
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
            results.append(get_frame_pyav(video_path, pts))
        return results if is_batch else results[0]

    # def unbatch(self, output): ...

    def encode_response(self, output):
        # send bmp
        success, frame_bytes = cv2.imencode(".bmp", output)
        if not success:
            raise RuntimeError("Failed to encode frame as BMP")
        return {"frame": base64.b64encode(frame_bytes.tobytes()).decode("utf-8")}


if __name__ == "__main__":
    api = SimpleLitAPI(
        max_batch_size=1,  # default: 1
        batch_timeout=0.0,  # default: 0.0
    )
    server = ls.LitServer(
        api,
        accelerator="cpu",  # default: auto
        workers_per_device=16,  # default: 1
    )
    server.run(port=8000, generate_client_file=False, num_api_servers=None)
