import time

from owa_env_gst.omnimodal import AppsinkRecorder


def main():
    # Create an instance of the AppsinkRecorder
    recorder = AppsinkRecorder()

    # Configure the recorder with a callback function
    def callback(pts, frame_time_ns):
        print(f"Received frame with PTS {pts} at time {frame_time_ns}")

    recorder.configure(callback=callback)

    with recorder.session:
        time.sleep(3)


if __name__ == "__main__":
    main()
