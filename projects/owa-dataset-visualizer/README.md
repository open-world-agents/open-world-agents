# OWA Dataset Visualizer

Browser-based visualizer for OWA recordings. Syncs MCAP input data with MKV video.

## Features

- **Local-only**: All processing in browser. No server uploads.
- **Large file support**: Uses MCAP index for seeking. Never loads entire file.
- **Input overlay**: Keyboard (all keys), mouse (L/R/M buttons, scroll wheel), cursor minimap
- **Mouse mode**: Toggle Relative (FPS) / Absolute (2D/RTS). Recenter interval for relative.
- **Seek handling**: Video pauses while loading state, resumes automatically.
- **Info panels**: Active window info, MCAP topic stats

## Usage

```bash
npm install
npm run dev
```

Open http://localhost:5173, select MCAP + MKV files.

**URL auto-load:** `?mcap=/test.mcap&mkv=/test.mkv`

## Structure

```
src/
├── main.js      # Entry, video events, render loop
├── state.js     # StateManager, message handlers
├── mcap.js      # MCAP loading, TimeSync
├── overlay.js   # Keyboard/mouse canvas drawing
├── ui.js        # Side panel, loading indicator
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

