import time

from owa.env.gst.msg import ScreenEmitted
from owa.env.gst.omnimodal import AppsinkRecorder


def main():
    # Create an instance of the AppsinkRecorder
    recorder = AppsinkRecorder()

    # Configure the recorder with a callback function
    def callback(x: ScreenEmitted):
        path, pts, frame_time_ns, before, after = x
        print(f"Received frame with PTS {pts} at time {frame_time_ns} with shape {before} -> {after}")

    recorder.configure("test.mkv", width=2560 // 2, height=1440 // 2, callback=callback)

    with recorder.session:
        time.sleep(2)


if __name__ == "__main__":
    main()
