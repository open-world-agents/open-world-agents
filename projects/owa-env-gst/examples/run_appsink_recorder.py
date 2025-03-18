import time

from owa.env.gst.omnimodal import AppsinkRecorder


def main():
    # Create an instance of the AppsinkRecorder
    recorder = AppsinkRecorder()

    # Configure the recorder with a callback function
    def callback(x):
        path, pts, frame_time_ns = x
        print(f"Received frame with PTS {pts} at time {frame_time_ns}")

    recorder.configure("test.mkv", callback=callback)

    with recorder.session:
        time.sleep(3)


if __name__ == "__main__":
    main()
