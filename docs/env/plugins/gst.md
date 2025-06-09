To see detailed implementation, skim over [owa_env_gst](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst). API Docs is being written WIP.

## Examples

- example of `gst/screen` listener
    ```python
    from owa.core.registry import LISTENERS
    import cv2
    import numpy as np

    # Components automatically available - no activation needed!

    # Define a callback to process frames
    def process_frame(frame):
        # Display the frame
        cv2.imshow("Screen Capture", frame.frame_arr)
        cv2.waitKey(1)

    # Create and configure the listener
    screen = LISTENERS["gst/screen"]().configure(
        callback=process_frame,
        fps=30,
        show_cursor=True
    )

    # Run the screen capture
    with screen.session:
        input("Press Enter to stop")
    ```

    For performance metrics:
    ```python
    def process_with_metrics(frame, metrics):
        print(f"FPS: {metrics.fps:.2f}, Latency: {metrics.latency*1000:.2f} ms")
        cv2.imshow("Screen", frame.frame_arr)
        cv2.waitKey(1)

    screen.configure(callback=process_with_metrics)
    ```

- example of `gst/screen_capture` runnable
    ```python
    from owa.core.registry import RUNNABLES

    # Components automatically available - no activation needed!
    screen_capture = RUNNABLES["gst/screen_capture"]().configure(fps=60)

    with screen_capture.session:
        for _ in range(10):
            frame = screen_capture.grab()
            print(f"Shape: {frame.frame_arr.shape}")
    ```

## Known Issues

- Currently, we only supports Windows OS. Other OS support is in TODO-list, but it's priority is not high.
- Currently, we only supports device with NVIDIA GPU. This is also in TODO-list, it's priority is higher than multi-OS support.

- When capturing some screen with `WGC`(Windows Graphics Capture API, it's being activate when you specify window handle), and with some desktop(not all), below issues are observed.
    - maximum FPS can't exceed maximum Hz of physical monitor.
    - When capturing `Windows Terminal` and `Discord`, the following case was reported. I also guess this phenomena is because of usage of `WGC`.
        - When there's no change in window, FPS drops to 1-5 frame.
        - When there's change(e.g. mouse movement) in window, FPS straightly recovers to 60+.


## Auto-generated documentation

::: gst
    handler: owa