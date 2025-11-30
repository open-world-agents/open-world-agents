# OWA Dataset Visualizer

Browser-based visualizer for OWA recordings. Syncs MCAP input data with MKV video.

## Features

- **HuggingFace Hub integration**: Browse and load datasets directly from HuggingFace
- **Local file support**: Drag & drop or select MCAP + MKV files. No server uploads.
- **Large file support**: Uses MCAP index for seeking. Never loads entire file.
- **Input overlay**: Keyboard (all keys), mouse (L/R/M buttons, scroll wheel), cursor minimap
- **Mouse mode**: Toggle Relative (FPS) / Absolute (2D/RTS). Recenter interval for relative.
- **Seek handling**: Video pauses while loading state, resumes automatically.
- **Info panels**: Active window info, MCAP topic stats

## Prerequisites

Install Node.js via [nvm](https://github.com/nvm-sh/nvm) (recommended):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install --lts
```

## Usage

```bash
npm install
npm run dev
```

Open http://localhost:5173.

**URL modes:**

- Landing page: Browse featured datasets or load local files
- `?repo_id=org/dataset`: Load HuggingFace dataset
- `?base_url=http://localhost:8080`: Local file server mode
- `?mcap=/test.mcap&mkv=/test.mkv`: Direct URL mode

### Local File Server

Serve a local directory for browsing multiple recordings:

```bash
python scripts/serve_local.py /path/to/recordings -p 8080
```

Then open `http://localhost:5173/?base_url=http://localhost:8080`

Features:

- Auto-scans for mcap/video pairs
- HTTP Range support for streaming large videos
- Multi-threaded for concurrent requests

## Structure

```
src/
├── main.js      # Routing, landing page
├── viewer.js    # Viewer logic, render loop
├── hf.js        # HuggingFace API, file tree
├── state.js     # StateManager, message handlers
├── mcap.js      # MCAP loading, TimeSync
├── overlay.js   # Keyboard/mouse canvas drawing
├── ui.js        # Side panel, loading indicator
├── config.js    # Featured datasets
├── constants.js # VK codes, colors, flags
└── styles.css
```

## How Seeking Works

1. Find nearest `keyboard/state` snapshot before target time
2. Replay `keyboard` events from snapshot to target
3. Find nearest `mouse/state` snapshot before target time
4. Replay mouse events from snapshot to target
5. Find latest `window` info

This enables O(snapshot interval) seek instead of O(file size).

## Development

**Message definitions**: Always reference `owa-msgs` for field names and types. Never guess message structure—check the schema source of truth.
