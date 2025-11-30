# OWA Dataset Visualizer

Browser-based visualizer for OWA recordings. Syncs MCAP input data with MKV video.

## Features

- **100% local processing**: No server uploads. MCAP parsed in-browser via `@mcap/core`
- **Hour+ recordings**: Indexed seeking, never reads entire file
- **Full input visualization**:
  - Keyboard: All keys with Windows VK codes
  - Mouse: Left/right/middle buttons + scroll wheel (up/down)
  - Position: Minimap showing cursor location
- **2D/3D game support**:
  - Relative mode: Accumulates raw deltas (FPS games)
  - Absolute mode: Uses cursor coordinates (RTS/2D games)
  - Recenter interval for relative mode
- **Safe seeking**: Video pauses during MCAP state loading
- **Side panel**: Window info (title, position, size) + MCAP stats (topics, message counts)

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

## MCAP Topics

| Topic | Data |
|-------|------|
| `screen` | `media_ref.pts_ns` for time sync |
| `keyboard` | `{event_type, vk}` |
| `keyboard/state` | `{buttons: [vk...]}` (snapshot) |
| `mouse` | `{event_type, x, y, button, pressed, dy}` |
| `mouse/raw` | `{last_x, last_y, button_flags, button_data}` |
| `mouse/state` | `{x, y, buttons: [...]}` (snapshot) |
| `window` | `{title, rect, hWnd}` |

## How Seeking Works

1. Find nearest `keyboard/state` snapshot before target time
2. Replay `keyboard` events from snapshot to target
3. Find nearest `mouse/state` snapshot before target time
4. Replay mouse events from snapshot to target
5. Find latest `window` info

This enables O(snapshot interval) seek instead of O(file size).

