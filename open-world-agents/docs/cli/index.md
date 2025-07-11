# OWA CLI (`owl`) - Command Line Tools

The `owl` command provides comprehensive tools for working with OWA data, environments, and message types. It's your primary interface for managing MCAP files, environment plugins, message schemas, and video processing.

## Installation

The CLI is included with the `owa-cli` package:

```bash
pip install owa-cli
```

After installation, the `owl` command becomes available in your terminal.

## Quick Start

```bash
# Get help for any command
owl --help
owl mcap --help

# Common workflows
owl mcap info session.mcap              # Inspect MCAP files
owl env list                            # List environment plugins  
owl messages show desktop/MouseEvent    # View message schemas
owl video probe recording.mkv           # Analyze video files
```

## Command Groups

The `owl` CLI is organized into specialized command groups:

### 📁 [MCAP Commands](mcap.md) (`owl mcap`)
Tools for working with MCAP files - the core data format for multimodal desktop recordings.

**Key commands**: `info`, `cat`, `convert`, `migrate`, `sanitize`

```bash
owl mcap info session.mcap              # File information
owl mcap cat session.mcap --n 10        # View messages  
owl mcap convert session.mcap           # Convert to subtitles
```

### 🔌 [Environment Commands](env.md) (`owl env`)
Manage environment plugins that capture desktop data and provide system integration.

**Key commands**: `list`, `search`, `validate`, `stats`, `docs`

```bash
owl env list                            # List all plugins
owl env list desktop                    # Plugin details
owl env search keyboard                 # Find components
owl env stats --namespaces              # Show namespaces
```

### 📋 [Message Commands](messages.md) (`owl messages`)
Inspect and validate message type schemas used in MCAP files.

**Key commands**: `list`, `show`, `validate`

```bash
owl messages list                       # All message types
owl messages show desktop/KeyboardEvent # Schema details
owl messages validate                   # Validate definitions
```

### 🎥 [Video Commands](video.md) (`owl video`)
Process and analyze video files from OWA recordings.

**Key commands**: `probe`, `vfr-to-cfr`

```bash
owl video probe session.mkv            # Analyze video
owl video vfr-to-cfr session.mkv       # Convert frame rate
```

## Complete Command Reference

::: mkdocs-click
    :module: owa.cli.click_compat
    :command: click_command
    :prog_name: owl
    :style: table
    :depth: 2

## Common Workflows

```bash
# Record and analyze data
ocap my-session
owl mcap info my-session.mcap
owl mcap convert my-session.mcap

# Environment management
owl env list
owl env validate desktop

# Data processing
owl mcap info *.mcap
owl video probe session.mkv
```

## Getting Help

```bash
owl --help                              # General help
owl mcap --help                         # Command group help
owl mcap info --help                    # Specific command help
```

## Related Documentation

- **[Installation Guide](../install.md)** - Installing OWA and CLI tools
- **[Recording Data](../data/getting-started/recording-data.md)** - Creating MCAP files with ocap
- **[Exploring Data](../data/getting-started/exploring-data.md)** - Data analysis workflows
- **[Environment Guide](../env/guide.md)** - Understanding environment plugins
- **[Custom Messages](../data/technical-reference/custom-messages.md)** - Creating custom message types
