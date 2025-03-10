import av
import line_profiler

from owa_env_gst.mkv_reader import MKVReader


@line_profiler.profile
def test_mkv_reader():
    with MKVReader("output.mkv") as reader:
        for data in reader.iter_frames():
            print(data["pts"])


@line_profiler.profile
def test_av():
    container = av.open("output.mkv")
    stream = next(s for s in container.streams if s.type == "video")
    for frame in container.decode(stream):
        print(frame.pts, frame.time_base, frame.to_ndarray().shape)
    container.close()


if __name__ == "__main__":
    test_mkv_reader()  # 56sec -> 4.39sec after optimization to process much more in d3d11memory
    test_av()  # 3.17sec w/o to_ndarray, 3.69sec w/ to_ndarray
