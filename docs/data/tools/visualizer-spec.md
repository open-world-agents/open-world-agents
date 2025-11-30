# OWA Dataset Visualizer - Technical Specification

> **Status**: Implemented
> **Version**: 2.0
> **Last Updated**: 2025-11-30

## 1. Executive Summary

This document describes the architecture of the OWA Dataset Visualizer (`projects/owa-dataset-visualizer`) - a browser-based tool for inspecting MCAP+MKV recording pairs.

### 1.1 Design Goals

- **No re-encoding**: Play original video files directly
- **Client-side processing**: All MCAP decoding happens in browser
- **Multiple data sources**: Local files, HuggingFace Hub, custom file servers

### 1.2 Architecture Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARCHITECTURE PRINCIPLE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Backend = DUMB FILE SERVER (byte-range support only)          │
│   Frontend = SMART CLIENT (all decoding & visualization)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Move ALL intelligence to the frontend. The backend becomes a trivially simple file server that can be replaced by any static hosting (S3, CDN, local files).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Local Files │  │  HTTP Server │  │  HF Hub CDN  │  │   S3/GCS    │  │
│  │  file://     │  │  http://     │  │  https://    │  │   https://  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  │
│         │                 │                 │                 │         │
│         └─────────────────┴────────┬────────┴─────────────────┘         │
│                                    │                                     │
│                          Byte-Range Requests                             │
│                           (Range: bytes=...)                             │
│                                    │                                     │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Browser)                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     File Fetcher                                 │    │
│  │  • Fetch .mcap with byte-range for windowed loading              │    │
│  │  • Stream .mkv to <video> element                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│              ┌─────────────────────┼─────────────────────┐              │
│              │                     │                     │              │
│              ▼                     ▼                     ▼              │
│  ┌───────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │
│  │   MCAP Decoder    │  │  Video Player   │  │  Overlay Renderer   │   │
│  │   (@mcap/browser) │  │  <video> HTML5  │  │  <canvas> 2D        │   │
│  │                   │  │                 │  │                     │   │
│  │  • Parse messages │  │  • Native codec │  │  • Keyboard viz     │   │
│  │  • Decode schemas │  │  • Seek/play    │  │  • Mouse viz        │   │
│  │  • Time indexing  │  │  • Speed ctrl   │  │  • Sync to video    │   │
│  └───────────────────┘  └─────────────────┘  └─────────────────────┘   │
│              │                     │                     ▲              │
│              │                     │                     │              │
│              └─────────────────────┴─────────────────────┘              │
│                           Time Synchronization                           │
│                        (video.currentTime → mcap timestamp)             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Design Principles

### 3.1 Minimal Backend

The backend is **just a file server**:

```python
# ENTIRE BACKEND - That's it!
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/files", StaticFiles(directory="/data"), name="files")
```

**Requirements:**

- Serve static files with proper MIME types
- Support HTTP Range requests (for video seeking & MCAP windowed loading)
- CORS headers for cross-origin access

**That's all.** No MCAP parsing, no message decoding, no business logic.

### 3.2 Smart Frontend

The frontend handles:

1. **File fetching** - Request MCAP/MKV files via fetch API
2. **MCAP decoding** - Parse binary MCAP format using `@mcap/browser`
3. **Message decoding** - Decode JSON schema messages
4. **Time synchronization** - Map video time to MCAP timestamps
5. **Visualization** - Render keyboard/mouse overlays on canvas

### 3.3 Why This Architecture?

| Aspect             | Old (Backend-heavy)          | New (Frontend-heavy)      |
| ------------------ | ---------------------------- | ------------------------- |
| Backend complexity | High (MCAP parsing, caching) | Minimal (file server)     |
| Scalability        | Limited by server            | Unlimited (client-side)   |
| Offline support    | ❌ Requires server           | ✅ Works with local files |
| CDN compatible     | ❌ Needs compute             | ✅ Pure static files      |
| Development        | Backend + Frontend           | Frontend only             |

---

## 4. Deployment Modes

### 4.1 Mode A: Any Static File Server

```bash
# Python
python -m http.server 8000 --directory /path/to/data

# Node.js
npx serve /path/to/data

# Nginx, Apache, S3, CloudFront, etc.
```

### 4.2 Mode B: HuggingFace Hub Direct

```javascript
// Frontend fetches directly from HF CDN
const mcapUrl = `https://huggingface.co/datasets/${repoId}/resolve/main/${filename}`;
const response = await fetch(mcapUrl);
```

### 4.3 Mode C: Local Files (file://)

```html
<!-- Open index.html directly, reference local files -->
<input type="file" accept=".mcap,.mkv" />
```

---

## 5. Frontend Technical Specification

### 5.1 MCAP Decoding in Browser

Using Foxglove's official TypeScript libraries:

```javascript
// Dependencies
import { McapIndexedReader } from "@mcap/browser";

// Load MCAP file
const response = await fetch(mcapUrl);
const reader = await McapIndexedReader.Initialize({
  readable: new ReadableStream(...),
});

// Read messages in time range
for await (const message of reader.readMessages({
  startTime: { sec: 0, nsec: 0 },
  endTime: { sec: 10, nsec: 0 },
})) {
  const decoded = JSON.parse(new TextDecoder().decode(message.data));
  // Process keyboard/mouse events
}
```

**NPM Packages:**

- `@mcap/core` - Low-level readers/writers
- `@mcap/browser` - Browser-specific utilities (fetch, streams)

### 5.2 Windowed Data Loading

```javascript
class McapDataLoader {
  constructor(mcapUrl) {
    this.mcapUrl = mcapUrl;
    this.reader = null;
    this.cache = new Map(); // timestamp_window -> events[]
  }

  async loadWindow(centerTimeNs, windowSizeNs = 10_000_000_000) {
    const startTime = centerTimeNs - windowSizeNs / 2;
    const endTime = centerTimeNs + windowSizeNs / 2;

    // Check cache first
    const cacheKey = `${startTime}-${endTime}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    // Fetch and decode
    const events = await this.readMessages(startTime, endTime);
    this.cache.set(cacheKey, events);
    return events;
  }
}
```

### 5.3 Time Synchronization

```javascript
// Core synchronization formula
function videoTimeToMcapTimestamp(videoTimeSec, basePtsTime) {
  // basePtsTime = firstScreenEvent.timestamp - firstScreenEvent.media_ref.pts_ns
  return basePtsTime + BigInt(Math.floor(videoTimeSec * 1_000_000_000));
}

// Sync loop using requestAnimationFrame
function syncLoop() {
  const videoTime = video.currentTime;
  const mcapTime = videoTimeToMcapTimestamp(videoTime, basePtsTime);

  // Get events active at this timestamp
  const activeEvents = getEventsAtTime(mcapTime);

  // Render overlays
  renderKeyboard(activeEvents.keyboard);
  renderMouse(activeEvents.mouse);

  requestAnimationFrame(syncLoop);
}
```

### 5.4 Frontend Module Structure

```
frontend/
├── index.html              # Single page application
├── css/
│   └── styles.css          # Minimal styling
└── js/
    ├── main.js             # Entry point, initialization
    ├── mcap-loader.js      # MCAP fetching & decoding
    ├── video-player.js     # Video controls & sync
    ├── overlay-renderer.js # Canvas drawing
    ├── keyboard-viz.js     # Keyboard visualization
    ├── mouse-viz.js        # Mouse visualization
    └── timeline.js         # Timeline & event markers
```

---

## 6. Backend Technical Specification

### 6.1 Minimal File Server

The backend is intentionally minimal - just serve files with Range support:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

# Serve files with automatic Range support
app.mount("/files", StaticFiles(directory="/data"), name="files")
```

### 6.2 Optional: File Listing API

For convenience, a simple listing endpoint:

```python
@app.get("/api/files")
async def list_files(path: str = ""):
    """List .mcap and .mkv files in directory"""
    base = Path("/data") / path
    files = []
    for f in base.glob("*.mcap"):
        mkv = f.with_suffix(".mkv")
        if mkv.exists():
            files.append({
                "basename": f.stem,
                "mcap": str(f.relative_to("/data")),
                "mkv": str(mkv.relative_to("/data")),
                "size": f.stat().st_size + mkv.stat().st_size,
            })
    return files
```

### 6.3 HuggingFace Hub Integration

For HF datasets, the frontend fetches directly from HF CDN:

```javascript
// No backend needed - direct CDN access
const HF_CDN = "https://huggingface.co/datasets";

async function loadFromHuggingFace(repoId, filename) {
  const mcapUrl = `${HF_CDN}/${repoId}/resolve/main/${filename}.mcap`;
  const mkvUrl = `${HF_CDN}/${repoId}/resolve/main/${filename}.mkv`;

  return { mcapUrl, mkvUrl };
}
```

---

## 7. Functional Requirements

### 7.1 Core Features (P0)

| ID   | Requirement                        | Implementation               |
| ---- | ---------------------------------- | ---------------------------- |
| P0-1 | Play video without re-encoding     | HTML5 `<video>` element      |
| P0-2 | Decode MCAP in browser             | `@mcap/browser` library      |
| P0-3 | Keyboard state visualization       | Canvas overlay               |
| P0-4 | Mouse position/click visualization | Canvas overlay               |
| P0-5 | Time synchronization               | `requestAnimationFrame` loop |

### 7.2 Enhanced Features (P1)

| ID   | Requirement                 | Implementation               |
| ---- | --------------------------- | ---------------------------- |
| P1-1 | Visual keyboard layout      | SVG/Canvas keyboard graphic  |
| P1-2 | Mouse trail visualization   | Canvas path drawing          |
| P1-3 | Event timeline with markers | Custom timeline component    |
| P1-4 | Playback speed control      | `video.playbackRate`         |
| P1-5 | Frame-by-frame navigation   | `video.currentTime` stepping |

### 7.3 Advanced Features (P2)

| ID   | Requirement              | Implementation        |
| ---- | ------------------------ | --------------------- |
| P2-1 | Event list panel         | Virtual scroll list   |
| P2-2 | Search/filter events     | Client-side filtering |
| P2-3 | Export events to JSON    | Blob download         |
| P2-4 | Keyboard shortcuts       | Event listeners       |
| P2-5 | Multiple file comparison | Split view            |

---

## 8. Keyboard Visualization Design

### 8.1 Visual Layout

Graphical keyboard layout:

```
┌─────────────────────────────────────────────────────────────────┐
│ ESC │ F1 │ F2 │ F3 │ F4 │ F5 │ F6 │ F7 │ F8 │ F9 │F10│F11│F12│
├─────────────────────────────────────────────────────────────────┤
│  `  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ 0 │ - │ = │ BACK  │
├─────────────────────────────────────────────────────────────────┤
│ TAB │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │ [ │ ] │   \   │
├─────────────────────────────────────────────────────────────────┤
│CAPS │ A │ S │ D │ F │ G │ H │ J │ K │ L │ ; │ ' │  ENTER  │
├─────────────────────────────────────────────────────────────────┤
│ SHIFT │ Z │ X │ C │ V │ B │ N │ M │ , │ . │ / │   SHIFT   │
├─────────────────────────────────────────────────────────────────┤
│CTRL│WIN│ALT│         SPACE              │ALT│WIN│MENU│CTRL│
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Key States

| State         | Color     | Description                  |
| ------------- | --------- | ---------------------------- |
| Idle          | `#6B6B6B` | Key not pressed              |
| Pressed       | `#50B0AB` | Key currently held down      |
| Just Released | `#3D8480` | Released within 200ms (fade) |

---

## 9. Mouse Visualization Design

### 9.1 Cursor Indicator

- Circle following mouse position (scaled to video dimensions)
- Color indicates button state:
  - Default: White outline
  - Left click: Red fill
  - Right click: Blue fill
  - Middle click: Yellow fill

### 9.2 Click Effects

```
Normal:     ○  (white outline, 8px)
Left:       ●  (red fill, 12px) + expanding ring
Right:      ●  (blue fill, 12px) + expanding ring
```

### 9.3 Mouse Path Trail (Optional)

- Fading polyline showing recent cursor movement
- Configurable: last 500ms or last 50 positions

---

## 10. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. User provides data source                                            │
│     • Local files via <input type="file">                               │
│     • URL to MCAP/MKV files                                             │
│     • HuggingFace dataset repo ID                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. Frontend initializes                                                 │
│     • Create McapIndexedReader from @mcap/browser                       │
│     • Set <video> src to MKV URL                                        │
│     • Read MCAP summary (channels, schemas, time range)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. On video play/seek                                                   │
│     • Calculate MCAP timestamp from video.currentTime                   │
│     • Load 10-second window of messages around current time             │
│     • Cache loaded windows for smooth playback                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. requestAnimationFrame render loop                                    │
│     • Get current video time                                            │
│     • Find active keyboard/mouse events at this timestamp               │
│     • Render overlays on canvas                                         │
│     • Update timeline position                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Implementation Roadmap

### Phase 1: Frontend MCAP Decoding (P0)

- [ ] Integrate `@mcap/browser` for client-side MCAP parsing
- [ ] Implement windowed message loading
- [ ] Remove backend MCAP parsing endpoints
- [ ] Test with local files and HF CDN

### Phase 2: Visualization Improvements (P1)

- [ ] Visual keyboard layout (Canvas-based)
- [ ] Fix raw mouse position tracking
- [ ] Mouse trail visualization
- [ ] Click effect animations

### Phase 3: Enhanced UX (P2)

- [ ] Event markers on timeline
- [ ] Scrollable event list with search
- [ ] Keyboard shortcuts (Space, arrows)
- [ ] Playback speed control

### Phase 4: Deployment Flexibility (P3)

- [ ] Static HTML export mode
- [ ] Direct HF CDN loading (no backend)
- [ ] PWA support for offline use

---

## 12. File Structure

```
projects/owa-dataset-visualizer/
├── index.html              # Single page application
├── src/
│   ├── main.js             # Entry point, routing
│   ├── viewer.js           # Viewer logic, render loop
│   ├── hf.js               # HuggingFace API, file tree
│   ├── state.js            # StateManager, message handlers
│   ├── mcap.js             # MCAP loading, TimeSync
│   ├── overlay.js          # Keyboard/mouse canvas drawing
│   ├── ui.js               # Side panel, loading indicator
│   ├── config.js           # Featured datasets
│   ├── constants.js        # VK codes, colors, flags
│   └── styles.css
├── scripts/
│   └── serve_local.py      # Local file server with Range support
├── package.json            # NPM dependencies (@mcap/core)
├── Dockerfile              # HuggingFace Spaces deployment
└── nginx.conf              # Production static file serving
```

---

## 13. Migration Path

### From Current Implementation

1. **Keep existing backend** temporarily for file listing
2. **Add frontend MCAP decoding** alongside existing API
3. **Deprecate** `/api/mcap_data` endpoint
4. **Remove** backend MCAP parsing code
5. **Simplify** backend to pure file server

### Backward Compatibility

- Existing URLs continue to work
- No changes to MCAP/MKV file format
- HuggingFace integration preserved

---

## 14. References

- [MCAP Format Specification](https://mcap.dev/)
- [@mcap/core NPM Package](https://www.npmjs.com/package/@mcap/core)
- [OWAMcap Format Guide](../technical-reference/format-guide.md)
- [Viewer Documentation](./viewer.md)
