# Data Conversion Examples

Open World Agents provides conversion scripts to transform existing gaming and interaction datasets into the standardized OWAMcap format. This enables researchers to leverage existing datasets for training multimodal desktop agents.

!!! info "What are Data Conversions?"
    Data conversions transform existing gaming datasets (VPT, CS:GO, etc.) into the standardized OWAMcap format, enabling unified training across different games and interaction types.

## Why Convert to OWAMcap?

OWAMcap (Open World Agents MCAP) is a standardized format that:

- :material-merge: **Unifies multimodal data**: Combines video, keyboard, mouse, and metadata in a single format
- :material-database-sync: **Enables cross-dataset training**: Consistent format across different games and interaction types
- :material-flash: **Supports efficient streaming**: Optimized for real-time agent training and inference
- :material-clock-outline: **Preserves temporal relationships**: Maintains precise timing between visual and input events

## Available Conversions

=== "VPT (Minecraft)"

    ### :material-minecraft: Video PreTraining (VPT) → OWAMcap

    Convert OpenAI's Minecraft VPT dataset for navigation and basic interaction training.

    [**:material-book-open: View VPT Conversion Guide**](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-data/scripts/conversion/VPT/README.md){ .md-button .md-button--primary }

=== "CS:GO (FPS)"

    ### :material-pistol: Counter-Strike Deathmatch → OWAMcap

    Convert expert CS:GO gameplay data for competitive FPS agent training.

    [**:material-book-open: View CS:GO Conversion Guide**](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-data/scripts/conversion/CS_DM/README.md){ .md-button .md-button--primary }

## Getting Started

For detailed installation, usage instructions, and troubleshooting, see the individual conversion guides above.

[**:material-folder-open: Browse All Conversion Scripts**](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-data/scripts/conversion/){ .md-button .md-button--primary }


