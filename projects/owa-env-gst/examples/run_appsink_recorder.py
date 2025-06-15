import time

from owa.env.gst.omnimodal import AppsinkRecorder
from owa.msgs.desktop.screen import ScreenCaptured


def main():
    # Create an instance of the AppsinkRecorder
    recorder = AppsinkRecorder()

    # Configure the recorder with a callback function
    def callback(x: ScreenCaptured):
        path, pts, frame_time_ns, before, after = x
        print(f"Received frame with PTS {pts} at time {frame_time_ns} with shape {before} -> {after}")

    recorder.configure("test.mkv", width=2560 // 2, height=1440 // 2, callback=callback)

    with recorder.session:
        time.sleep(2)


if __name__ == "__main__":
    main()
