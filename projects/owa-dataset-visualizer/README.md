# OWA Dataset Visualizer

Browser-based visualizer for OWA dataset recordings. Plays MKV video with synchronized keyboard/mouse overlay from MCAP data.

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:5173 and select MCAP + MKV files.

## Development Testing

### Setup Test Data

Create symlinks to test files:

```bash
# Use default test data path
npm run setup-test

# Or specify custom paths
./scripts/setup-test-data.sh /path/to/recording.mcap /path/to/recording.mkv
```

### Auto-load via URL

After setup, open with URL parameters:

```
http://localhost:5173/?mcap=/test.mcap&mkv=/test.mkv
```

This is useful for:
- Automated testing (e.g., Playwright)
- Quick iteration during development
- Debugging without manual file selection

### Playwright Testing

With Playwright MCP enabled, you can automate browser testing:

1. Run `npm run setup-test` to create test data symlinks
2. Start dev server: `npm run dev`
3. Use Playwright to navigate to the URL with parameters
4. Inspect page state, take screenshots, etc.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Browser                                    │
│  ┌────────────────────────────────────────┐ │
│  │ <video> plays MKV directly             │ │
│  │ <canvas> overlay (pointer-events:none) │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │ @mcap/core parses MCAP (index only)    │ │
│  │ fzstd handles zstd decompression       │ │
│  │ requestAnimationFrame syncs overlay    │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `src/main.js` | Entry point, file loading, render loop |
| `src/mcap-loader.js` | MCAP parsing with BlobReadable + zstd |
| `src/styles.css` | Dark theme styling |
| `scripts/setup-test-data.sh` | Creates test data symlinks |

## MCAP Data Structure

Expected topics:
- `screen` - Contains `media_ref.pts_ns` for time sync
- `keyboard/state` - `{ buttons: [vkCodes...] }`
- `mouse/state` - `{ x, y, buttons: [] }`

