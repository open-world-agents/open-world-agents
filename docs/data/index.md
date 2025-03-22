# Open-sourcing Dataset for Multimodal Desktop Agent

As of now (March 22, 2025), there are few datasets available for building multimodal desktop agents.

Even more scarce are datasets that (1) contain high-frequency screen data, (2) have keyboard/mouse information timestamp-aligned with other modalities like screen recordings, and (3) include human demonstrations.

To address this gap, open-world-agents provides the following three solutions:

1. **File Format - `OWAMcap`**: A high-performance, self-contained, flexible container file format for multimodal desktop log data, powered by the open-source container file format [mcap](https://mcap.dev/). [Learn more...](data_format.md)

2. **Desktop Recorder - `owl mcap record your-filename.mcap`**: A powerful, efficient, and easy-to-use desktop recorder that captures keyboard/mouse and high-frequency screen data.
    - Powered by [`owa-env-gst`](../env/plugins/gstreamer_env.md), ensuring superior performance compared to alternatives. [Learn more...](recorder/why.md)

3. **🤗 [Hugging Face](https://huggingface.co/) Integration** (TODO): We'll accelerate the open-sourcing process and improve accessibility of multimodal desktop agent through integration with Huggingface. Stay Tuned!