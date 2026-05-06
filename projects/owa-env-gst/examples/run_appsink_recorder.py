import time

from owa.env.gst.omnimodal import AppsinkRecorder
from owa.msgs.desktop.screen import ScreenCaptured


def main():
    # Create an instance of the AppsinkRecorder
    recorder = AppsinkRecorder()

    # Configure the recorder with a callback function
    def callback(msg: ScreenCaptured):
        print(f"Video frame: PTS {msg.media_ref.pts_ns} at {msg.utc_ns} shape {msg.source_shape} -> {msg.shape}")

    recorder.configure("test.mkv", width=2560 // 2, height=1440 // 2, callback=callback)

    with recorder.session:
        time.sleep(2)


if __name__ == "__main__":
    main()
