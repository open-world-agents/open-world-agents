# OWAMcap Data Viewer

Interactive web-based visualization tool for exploring OWAMcap datasets with synchronized playback of screen recordings and interaction events.

<div align="center">
  <img src="../examples/viewer.png" alt="OWA Dataset Visualizer"/>
</div>

## 🌐 Public Hosted Viewer

**Quick Start**: [https://huggingface.co/spaces/open-world-agents/visualize_dataset](https://huggingface.co/spaces/open-world-agents/visualize_dataset)

### Features
- **Upload Files**: Drag & drop your `.mcap` files (up to 100MB)
- **HuggingFace Integration**: Enter any `repo_id` to view public datasets
- **Synchronized Playback**: Video + events timeline
- **Interactive Controls**: Pause, seek, frame-by-frame navigation

### Usage
1. Visit the viewer URL
2. Either upload your files or enter a HuggingFace dataset ID
3. Explore your data with synchronized video and event timeline

## 🏠 Self-Hosted Setup

For larger files or private datasets, run the viewer locally:

```bash
# Navigate to viewer directory
cd projects/owa-mcap-viewer

# Set data path
export EXPORT_PATH=/path/to/your/mcap-files

# Install dependencies
vuv install

# Start server
uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload
```

Access at `http://localhost:7860`

### Benefits of Self-Hosting
- **No file size limits**
- **Private data stays local**
- **Faster loading for large datasets**
- **Customizable interface**