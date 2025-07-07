# Data in OWA: Complete Desktop Agent Data Pipeline

Desktop AI needs high-quality, synchronized multimodal data: screen captures, mouse/keyboard events, and window context. OWA provides the **complete pipeline** from recording to training.

## 🚀 Quick Start: Record → Train in 3 Steps

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_events_to_event_dataset.py --train-dir ./

# 3. Train your model
$ python train.py --dataset ./event-dataset
```

> 📖 **Detailed Guide**: [Complete Quick Start Tutorial](../quick-start.md) - Step-by-step walkthrough with examples and troubleshooting
<!-- END-SYNC: quick-start-3-steps -->

## The OWA Data Ecosystem

### 🎯 **Getting Started**
New to OWA data? Start here:

- **[Why OWAMcap?](getting-started/why-owamcap.md)** - Understand the problem and solution
- **[Recording Data](getting-started/recording-data.md)** - Capture desktop interactions with `ocap`
- **[Exploring Data](getting-started/exploring-data.md)** - View and analyze your recordings

### 📚 **Technical Reference**
Deep dive into the format and pipeline:

- **[OWAMcap Format Guide](technical-reference/format-guide.md)** - Complete technical specification
- **[Data Pipeline](technical-reference/data-pipeline.md)** - Transform recordings to training-ready datasets

### 🛠️ **Tools & Ecosystem**
- **[Data Viewer](tools/viewer.md)** - Web-based visualization tool
- **[Comparison with LeRobot](tools/comparison-with-lerobot.md)** - Technical comparison with alternatives

## 🤗 Community Datasets

<!-- SYNC-ID: community-datasets -->
**Browse Available Datasets**: [🤗 datasets?other=OWA](https://huggingface.co/datasets?other=OWA)

- **Growing Collection**: Hundreds of community-contributed datasets
- **Standardized Format**: All use OWAMcap for seamless integration
- **Interactive Preview**: [Hugging Face Spaces Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
- **Easy Sharing**: Upload recordings directly with one command

> 🚀 **Impact**: OWA has democratized desktop agent data, growing from zero to hundreds of public datasets in the unified OWAMcap format.
<!-- END-SYNC: community-datasets -->