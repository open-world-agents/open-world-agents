# Data in OWA: Complete Desktop Agent Data Pipeline

Desktop AI needs high-quality, synchronized multimodal data: screen captures, mouse/keyboard events, and window context. OWA provides the **complete pipeline** from recording to training.

## 🚀 Quick Start: Record → Train in 3 Steps

```bash
# 1. Record desktop interaction
ocap my-session.mcap

# 2. Process to training format
python scripts/01_raw_events_to_event_dataset.py --train-dir ./

# 3. Train your model
python train.py --dataset ./event-dataset
```

## The OWA Data Ecosystem

### 🎯 **Getting Started**
New to OWA data? Start here:

- **[Why OWAMcap?](why_owamcap.md)** - Understand the problem and solution
- **[Recording Data](ocap.md)** - Capture desktop interactions with `ocap`
- **[Exploring Data](how_to_explorer_and_edit.md)** - View and analyze your recordings

### 📚 **Technical Reference**
Deep dive into the format and pipeline:

- **[OWAMcap Format Guide](owamcap_format_guide.md)** - Complete technical specification
- **[Data Pipeline](owa_data_pipeline.md)** - Transform recordings to training-ready datasets

### 🛠️ **Tools & Ecosystem**
- **[Data Viewer](viewer.md)** - Web-based visualization tool
- **[Comparison with LeRobot](comparison_with_lerobot.md)** - Technical comparison with alternatives

## 🤗 Community Datasets

**Browse Available Datasets**: [🤗 datasets?other=OWA](https://huggingface.co/datasets?other=OWA)

- **Growing Collection**: Hundreds of community-contributed datasets
- **Standardized Format**: All use OWAMcap for seamless integration
- **Interactive Preview**: [Hugging Face Spaces Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)
- **Easy Sharing**: Upload recordings directly with one command

> 🚀 **Impact**: OWA has democratized desktop agent data, growing from zero to hundreds of public datasets in the unified OWAMcap format.