import av
import line_profiler

from owa_env_gst.mkv_reader import GstMKVReader, PyAVMKVReader


@line_profiler.profile
def test_mkv_reader():
    with GstMKVReader("output.mkv") as reader:
        for frame in reader.iter_frames():
            print(frame["pts"], frame["data"].shape)


@line_profiler.profile
def test_av():
    # container = av.open("output.mkv")
    # stream = next(s for s in container.streams if s.type == "video")
    # for frame in container.decode(stream):
    #     print(frame.pts, frame.time_base, frame.to_ndarray().shape)
    # container.close()

    with PyAVMKVReader("output.mkv") as reader:
        for frame in reader.iter_frames():
            print(frame["pts"], frame["data"].shape)


if __name__ == "__main__":
    # on Windows 11 with i7-14700, 4070 Ti Super
    # 56sec -> 4.39sec after optimization to process much more in d3d11memory
    # on DGX H100 w/o GPU, 9.84sec
    test_mkv_reader()
    # on Windows 11 with i7-14700, 4070 Ti Super
    # 3.17sec w/o to_ndarray, 3.69sec w/ to_ndarray
    # on DGX H100, 3.14sec
    test_av()
