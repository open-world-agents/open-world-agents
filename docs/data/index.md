# Open-sourcing Dataset for Multimodal Desktop Agent

As of now (March 22, 2025), there are few datasets available for building multimodal desktop agents.

Even more scarce are datasets that (1) contain high-frequency screen data, (2) have keyboard/mouse information timestamp-aligned with other modalities like screen recordings, and (3) include human demonstrations.

To address this gap, open-world-agents provides the following three solutions:

1. **File Format - `OWAMcap`**: A high-performance, self-contained, flexible container file format for multimodal desktop log data, powered by the open-source container file format [mcap](https://mcap.dev/). [Why OWAMcap?](owamcap_introduction.md) | [Technical Guide](data_format.md)

2. **Desktop Recorder - `ocap your-filename.mcap`**: A powerful, efficient, and easy-to-use desktop recorder that captures keyboard/mouse and high-frequency screen data.
    - Powered by [`owa-env-gst`](../env/plugins/gst.md), ensuring superior performance compared to alternatives. [Learn more...](ocap.md)

3. **🤗 [Hugging Face](https://huggingface.co/) Integration & Community Ecosystem**: The largest collection of open-source desktop interaction datasets in OWAMcap format.
    - **Growing Dataset Collection**: Hundreds of community-contributed datasets covering diverse workflows, applications, and interaction patterns
    - **Easy Upload & Sharing**: Upload your `ocap` recordings directly to HuggingFace with one command
    - **Standardized Format**: All datasets use the unified OWAMcap format for seamless integration
    - **Interactive Visualization**: Preview any dataset at [Hugging Face Spaces](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
    - **Browse Available Datasets**: [🤗 datasets?other=OWA](https://huggingface.co/datasets?other=OWA)

> 🚀 **Community Impact**: With OWA's streamlined recording and sharing pipeline, the open-source desktop agent community has rapidly grown from zero to hundreds of publicly available multimodal datasets, democratizing access to high-quality training data.