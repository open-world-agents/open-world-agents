# OWA Dataset Visualizer

Browser-based visualizer for OWA recordings. Syncs MCAP input data with MKV video.

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
├── main.js      # Entry point, video sync, render loop
├── state.js     # Input state management
├── mcap.js      # MCAP loading, time sync
├── overlay.js   # Keyboard/mouse canvas rendering
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
| `mouse` | `{event_type, x, y, button, pressed}` |
| `mouse/raw` | `{last_x, last_y, button_flags, button_data}` |
| `mouse/state` | `{x, y, buttons: [...]}` (snapshot) |
| `window` | `{title, rect, hWnd}` |

## Design

- **Minimal dependencies**: `@mcap/core`, `fzstd`, `vite`
- **No framework**: Vanilla JS, ~800 lines total
- **Indexed reads**: Seeks via MCAP index, not full scan
- **State snapshots**: `*/state` topics enable fast seeking

