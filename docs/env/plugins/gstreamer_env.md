
To see detailed implementation, skim over [owa_env_gst](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-gst/owa_env_gst). API Docs is being written WIP.


## Known Issues

- Currently, we only supports Windows OS. Other OS support is in TODO-list, but it's priority is not high.
- Currently, we only supports device with NVIDIA GPU. This is also in TODO-list, it's priority is higher than multi-OS support.

- When capturing some screen with `WGC`(Windows Graphics Capture API, it's being activate when you specify window handle), and with some desktop(not all), above issues are observed.
    - maximum FPS can't exceed maximum Hz of physical monitor.
    - When capturing `Windows Terminal` and `Discord`, the following case was reported. I also guess this phenomena is because of usage of `WGC`.
        - When there's no change in window, FPS drops to 1-5 frame.
        - When there's change(e.g. mouse movement) in window, FPS straightly recovers to 60+.
