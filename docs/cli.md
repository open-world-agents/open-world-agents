# OWA CLI (`owl`) - Command Line Tools

The `owl` command provides comprehensive tools for working with OWA data, environments, and message types.

## Installation

The CLI is included with the `owa-cli` package:

```bash
pip install owa-cli
```

After installation, the `owl` command becomes available in your terminal.

<!-- TODO: apply https://github.com/mkdocs/mkdocs-click -->

## Command Overview

| Command Group | Description |
|---------------|-------------|
| `owl mcap` | MCAP file management and analysis |
| `owl messages` | Message type registry management |
| `owl env` | Environment plugin management |
| `owl video` | Video processing (requires FFmpeg) |
| `owl window` | Window management (Windows only) |

## MCAP Commands (`owl mcap`)

### File Information
```bash
# Show detailed file information
owl mcap info session.mcap

# Example output:
# library:   mcap-owa-support 0.5.1; mcap 1.3.0
# profile:   owa
# messages:  864
# duration:  10.36s
# channels:  6
```

### Message Listing
```bash
# List all messages
owl mcap cat session.mcap

# List first 10 messages
owl mcap cat session.mcap --n 10

# Filter by topics
owl mcap cat session.mcap --topics screen mouse

# Filter by time range
owl mcap cat session.mcap --start-time 1000000000 --end-time 2000000000
```

### File Migration
```bash
# Migrate to latest version
owl mcap migrate run old_file.mcap

# Migrate multiple files
owl mcap migrate run file1.mcap file2.mcap

# Dry run to see what would be migrated
owl mcap migrate run old_file.mcap --dry-run
```

### Frame Extraction
```bash
# Extract all frames to directory
owl mcap extract-frames session.mcap --output frames/

# Extract specific frame range
owl mcap extract-frames session.mcap --start-frame 100 --end-frame 200
```

## Message Commands (`owl messages`)

### List Message Types
```bash
# List all available message types
owl messages list

# Search for specific message types
owl messages list --search keyboard
```

### Show Message Details
```bash
# Show message schema and details
owl messages show desktop/KeyboardEvent

# Show with usage example
owl messages show desktop/KeyboardEvent --example

# Output as JSON schema
owl messages show desktop/KeyboardEvent --format json
```

### Validate Messages
```bash
# Validate all message definitions
owl messages validate

# Validate specific message type
owl messages validate desktop/KeyboardEvent
```

## Environment Commands (`owl env`)

### List Plugins
```bash
# List all plugins
owl env list

# List with details
owl env list --details

# Search plugins
owl env search mouse
```

### Show Plugin Details
```bash
# Show plugin overview
owl env show desktop

# Show with components
owl env show desktop --components

# Show specific component
owl env show desktop/mouse.click
```

### Plugin Validation
```bash
# Validate all plugins
owl env validate

# Validate specific plugin
owl env validate desktop
```

## Video Commands (`owl video`)

!!! note "Requires FFmpeg"
    Video commands require FFmpeg to be installed and available in your PATH.

```bash
# Convert video formats
owl video convert input.mkv output.mp4

# Extract frames from video
owl video extract-frames video.mkv --output frames/

# Get video information
owl video info video.mkv
```

## Window Commands (`owl window`)

!!! note "Windows Only"
    Window commands are only available on Windows with `owa.env.desktop` installed.

```bash
# List all windows
owl window list

# Get active window info
owl window active

# Find windows by title
owl window find "Chrome"
```

## Global Options

All `owl` commands support these global options:

```bash
# Show help for any command
owl --help
owl mcap --help
owl mcap info --help

# Verbose output
owl --verbose mcap info session.mcap

# Quiet mode (minimal output)
owl --quiet mcap cat session.mcap
```

## Common Usage Patterns

### Analyzing MCAP Files
```bash
# Quick file overview
owl mcap info session.mcap

# Check message distribution
owl mcap cat session.mcap --n 100 | grep "Topic:" | sort | uniq -c

# Extract specific time range
owl mcap cat session.mcap --start-time 1000000000 --end-time 2000000000
```

### Working with Messages
```bash
# Find message types for your use case
owl messages list --search screen

# Get schema for integration
owl messages show desktop/ScreenCaptured --format json

# Validate custom messages
owl messages validate
```

### Plugin Development
```bash
# Check plugin structure
owl env show my-plugin --components

# Validate documentation
owl env validate-docs my-plugin

# Get plugin statistics
owl env stats
```

## Getting Help

- Use `--help` with any command for detailed usage information
- Check command exit codes for scripting (0 = success, 1 = error)
- Use `--verbose` for debugging information
- Use `--quiet` for minimal output in scripts
