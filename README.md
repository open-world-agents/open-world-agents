<div align="center">
  <img src="docs/images/owa-logo.jpg" alt="Open World Agents" width="300"/>

  # 🚀 Open World Agents

  **Everything you need to build state-of-the-art foundation multimodal desktop agent, end-to-end.**

  [![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://open-world-agents.github.io/open-world-agents/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![GitHub stars](https://img.shields.io/github/stars/open-world-agents/open-world-agents?style=social)](https://github.com/open-world-agents/open-world-agents/stargazers)

</div>

> **⚠️ Active Development Notice**: This codebase is under active development. APIs and components may change, and some may be moved to separate repositories. Documentation may be incomplete or reference features still in development.

> **📄 Research Paper**: This project was first introduced and developed for the D2E project. For more details, see [D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](https://worv-ai.github.io/d2e/). If you find this work useful, please cite our paper.

## Quick Start

> 💡 This is a conceptual overview. See the [Quick Start Guide](https://open-world-agents.github.io/open-world-agents/quick-start/) for detailed instructions.

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_to_event.py --train-dir ./

# 3. Train your model (coming soon)
$ python train.py --dataset ./event-dataset
```
<!-- END-SYNC: quick-start-3-steps -->

## Installation

```bash
# For video recording, install GStreamer first. Skip if you only need data processing.
$ conda install open-world-agents::gstreamer-bundle

# Install OWA
$ pip install owa
```

## Documentation

| Resource | Description |
|----------|-------------|
| 📖 **[Quick Start Guide](https://open-world-agents.github.io/open-world-agents/quick-start/)** | Complete tutorial: Record → Process → Train |
| 🔍 **[Project Overview](https://open-world-agents.github.io/open-world-agents/overview/)** | Architecture, packages, and benchmarks |
| 🌍 **[Environment Framework](https://open-world-agents.github.io/open-world-agents/env/)** | Real-time desktop automation |
| 📊 **[Data Infrastructure](https://open-world-agents.github.io/open-world-agents/data/)** | OWAMcap format and data pipeline |
| 🤗 **[Community Datasets](https://huggingface.co/datasets?other=OWA)** | Browse and share datasets |

## Core Components

<!-- SYNC-ID: core-components-list -->
- 🌍 **[Environment Framework](https://open-world-agents.github.io/open-world-agents/env/)**: "USB-C of desktop agents" - universal interface for native desktop automation with pre-built plugins for desktop control, high-performance screen capture, and zero-configuration plugin system
- 📊 **[Data Infrastructure](https://open-world-agents.github.io/open-world-agents/data/)**: Complete desktop agent data pipeline from recording to training with `OWAMcap` format - a [universal standard](https://open-world-agents.github.io/open-world-agents/data/getting-started/why-owamcap/) powered by [MCAP](https://mcap.dev/)
- 🛠️ **[CLI Tools](https://open-world-agents.github.io/open-world-agents/cli/)**: Command-line utilities (`owl`) for recording, analyzing, and managing agent data
- 🤖 **[Examples](https://open-world-agents.github.io/open-world-agents/examples/)**: Complete implementations and training pipelines for multimodal agents
<!-- END-SYNC: core-components-list -->

## Contributing

We welcome contributions! See our [Contributing Guide](https://open-world-agents.github.io/open-world-agents/contributing/).

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{choi2025d2e,
  title={D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI},
  author={Choi, Suwhan and Jung, Jaeyoon and Seong, Haebin and Kim, Minchan and Kim, Minyeong and Cho, Yongjun and Kim, Yoonshik and Park, Yubeen and Yu, Youngjae and Lee, Yunsung},
  journal={arXiv preprint arXiv:2510.05684},
  year={2025}
}
```
