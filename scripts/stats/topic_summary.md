# MCAP Topic Message Count Summary

## Overview
- **Total MCAP files**: 221
- **Successfully processed**: 186 files
- **Failed to process**: 35 files (mostly permission issues or corrupted files)
- **Total messages across all files**: 191,129,941
- **Dataset size**: 5.1 GB

## Message Counts by Topic

| Topic | Total Messages | Files Containing Topic | Percentage of Total Messages |
|-------|----------------|------------------------|------------------------------|
| `mouse` | 102,134,462 | 178 files | 53.4% |
| `mouse/raw` | 49,304,108 | 177 files | 25.8% |
| `screen` | 34,972,318 | 177 files | 18.3% |
| `keyboard` | 2,732,781 | 175 files | 1.4% |
| `window` | 662,312 | 179 files | 0.3% |
| `keyboard/state` | 661,980 | 179 files | 0.3% |
| `mouse/state` | 661,977 | 179 files | 0.3% |

## Topic Descriptions

- **`mouse`**: Mouse click and movement events (desktop/MouseEvent)
- **`mouse/raw`**: Raw mouse input events (desktop/RawMouseEvent)  
- **`screen`**: Screen capture frames (desktop/ScreenCaptured)
- **`keyboard`**: Keyboard press and release events (desktop/KeyboardEvent)
- **`window`**: Window focus and application information (desktop/WindowInfo)
- **`keyboard/state`**: Keyboard state snapshots (desktop/KeyboardState)
- **`mouse/state`**: Mouse position and button states (desktop/MouseState)

## Key Insights

1. **Mouse events dominate**: Mouse-related events (`mouse` + `mouse/raw`) account for ~79% of all messages
2. **Screen captures**: Screen capture events make up ~18% of messages
3. **Keyboard activity**: Keyboard events are much less frequent, only ~1.7% of total messages
4. **State snapshots**: State-related topics (`window`, `keyboard/state`, `mouse/state`) have similar low frequencies, likely periodic snapshots
5. **High coverage**: Most topics appear in 175-179 out of 186 successful files, indicating consistent data collection

## Data Quality

- **Success rate**: 84% of files processed successfully (186/221)
- **Main failure causes**: 
  - Permission denied (29 files)
  - File corruption/format issues (3 files)
  - Other errors (3 files)

This analysis shows the OWA dataset contains rich multimodal desktop interaction data with comprehensive coverage of user input events and screen captures.
