import line_profiler
from decord import VideoReader, cpu
from tqdm import tqdm

from owa.env.gst.mkv_reader import GstMKVReader, PyAVMKVReader

VIDEO_PATH = "../../tmp/output.mkv"
MAX_FRAMES = 10**9


@line_profiler.profile
def test_gst():
    with GstMKVReader(VIDEO_PATH) as reader:
        for idx, frame in tqdm(enumerate(reader.iter_frames())):
            if idx == MAX_FRAMES:
                break
            print(frame["pts"], frame["data"].shape)


@line_profiler.profile
def test_av():
    with PyAVMKVReader(VIDEO_PATH) as reader:
        for idx, frame in tqdm(enumerate(reader.iter_frames())):
            if idx == MAX_FRAMES:
                break
            print(frame["pts"], frame["data"].shape)


@line_profiler.profile
def test_decord():
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    for idx in tqdm(range(len(vr))):
        if idx == MAX_FRAMES:
            break
        frame = vr[idx]
        print(idx, frame.shape)


"""
on Windows 11 with i7-14700, 4070 Ti Super

  1.14 seconds - projects\owa-env-gst\benchmark_video_reader.py:22 - test_decord
  3.14 seconds - projects\owa-env-gst\benchmark_video_reader.py:15 - test_av
  3.95 seconds - projects\owa-env-gst\benchmark_video_reader.py:8 - test_gst

on DGX H100, cpus=16 gpus=0

  1.89 seconds - projects/owa-env-gst/main.py:22 - test_decord
  5.57 seconds - projects/owa-env-gst/main.py:15 - test_av
  9.09 seconds - projects/owa-env-gst/main.py:8 - test_gst
"""

if __name__ == "__main__":
    # on Windows 11 with i7-14700, 4070 Ti Super
    # 56sec -> 4.39sec after optimization to process much more in d3d11memory
    # on DGX H100 w/o GPU, 9.84sec
    test_gst()

    # on Windows 11 with i7-14700, 4070 Ti Super
    # 3.17sec w/o to_ndarray, 3.69sec w/ to_ndarray
    # on DGX H100, 3.14sec
    test_av()

    # Benchmarking Decord
    # Add any additional notes based on benchmarking results
    test_decord()
