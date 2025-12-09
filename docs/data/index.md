# Data in OWA

Desktop AI needs high-quality, synchronized multimodal data: screen captures, mouse/keyboard events, and window context. OWA provides the **complete pipeline** from recording to training.

## Quick Start

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_to_event.py --train-dir ./

# 3. Train your model
$ python train.py --dataset ./event-dataset
```
<!-- END-SYNC: quick-start-3-steps -->

> 📖 **Detailed Guide**: [Complete Quick Start Tutorial](../quick-start.md)

## Documentation

### 🚀 Getting Started

- **[Why OWAMcap?](getting-started/why-owamcap.md)** - Understand the problem and solution
- **[Recording Data](getting-started/recording-data.md)** - Capture desktop interactions with `ocap`
- **[Exploring Data](getting-started/exploring-data.md)** - View and analyze your recordings

### 📚 Technical Reference

- **[OWAMcap Format Guide](technical-reference/format-guide.md)** - Complete technical specification
- **[Data Pipeline](technical-reference/data-pipeline.md)** - Transform recordings to training-ready datasets

### 🛠️ Tools
- **[Data Viewer](viewer.md)** - Web-based visualization tool
- **[Data Conversions](conversions.md)** - Convert existing datasets (VPT, CS:GO) to OWAMcap

## 🤗 Community Datasets

<!-- SYNC-ID: community-datasets -->
**Browse Datasets**: [🤗 HuggingFace](https://huggingface.co/datasets?other=OWA)

- **Standardized Format**: All datasets use OWAMcap for seamless integration
- **Interactive Preview**: [Hugging Face Spaces Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
<!-- END-SYNC: community-datasets -->