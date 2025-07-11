# Exploring OWAMcap Data

Practical guide to viewing and analyzing OWAMcap recordings using different tools and workflows.

## 📁 Sample Dataset

Download our example dataset to follow along:

- `example.mcap` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mcap) - 22 KiB metadata file
- `example.mkv` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mkv) - Video recording

??? demo "Preview: example.mkv"
    <video controls>
    <source src="../example.mkv" type="video/mp4">
    </video>

## 🔍 Exploration Workflows

Choose the approach that fits your use case:

### 🌐 Interactive Web Viewer (Recommended)

**Best for**: Visual exploration, beginners, sharing with others

**[OWA Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)** provides synchronized playback of video and events.

<div align="center">
  <img src="../viewer.png" alt="OWA Dataset Visualizer"/>
</div>

**Quick Start**:

1. Visit the visualizer link
2. Upload your `.mcap` file or enter a HuggingFace dataset ID
3. Use timeline controls to explore synchronized data

**Limitations**: 100MB file limit for public hosting → [Self-hosting guide](../tools/viewer.md)

### 💻 Command Line Analysis

**Best for**: Quick inspection, scripting, CI/CD pipelines

The `owl` CLI provides fast analysis without loading video data:

**Common Commands**:

```bash
# Get file overview
owl mcap info example.mcap

# List first 10 messages
owl mcap cat example.mcap --n 10

# Filter by topic
owl mcap cat example.mcap --topics screen --topics mouse

# Convert to subtitle format
owl mcap convert example.mcap  # Creates example.srt
```

**Example Output**:
```
library:   mcap-owa-support 0.5.1
messages:  864 (10.36s duration)
channels:  screen(590), mouse(209), keyboard(32), window(11)
```

### 🎬 Video Player with Subtitles

**Best for**: Visual timeline analysis, understanding user behavior

1. **Generate subtitle file**:
   ```bash
   owl mcap convert example.mcap  # Creates example.srt
   ```

2. **Open in video player**: Use [VLC](https://www.videolan.org/vlc/) or any player that supports subtitles
   - Load `example.mkv`
   - Load `example.srt` as subtitles
   - See events overlaid on video timeline

**Download example**: `example.srt` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.srt)

### 🐍 Python API

**Best for**: Custom analysis, data processing, integration

For programmatic access, see the [OWAMcap Format Guide](../technical-reference/format-guide.md#working-with-owamcap) which covers:

- Reading and writing MCAP files
- Working with media references
- Advanced filtering and processing
- Custom message types

## 🔧 Analysis Workflows

### 📊 Quick Dataset Overview
```bash
# Get basic stats
owl mcap info *.mcap

# Compare multiple files
for file in *.mcap; do
  echo "=== $file ==="
  owl mcap info "$file" | grep -E "(messages|duration|channels)"
done
```

### ⏱️ Event Timeline Analysis
```bash
# Extract events to subtitle format for timeline view
owl mcap convert session.mcap

# View in VLC with subtitles to see event timing
vlc session.mkv --sub-file session.srt
```

### 🎯 Topic-Specific Analysis
```bash
# Focus on user interactions
owl mcap cat session.mcap --topics mouse --topics keyboard

# Screen capture analysis
owl mcap cat session.mcap --topics screen --n 100
```

## 🛠️ Creating and Modifying Files

For programmatic creation and editing of OWAMcap files, see the comprehensive guide in [OWAMcap Format Guide](../technical-reference/format-guide.md#working-with-owamcap), which covers:

- **Writing MCAP files** with Python API
- **Custom message types** and registration
- **Media handling** strategies
- **Advanced usage patterns**

## 📊 Next Steps

- **[Data Pipeline](../technical-reference/data-pipeline.md)** - Transform recordings for ML training
- **[Format Guide](../technical-reference/format-guide.md)** - Complete technical reference
- **[Viewer Setup](../tools/viewer.md)** - Self-host for large datasets