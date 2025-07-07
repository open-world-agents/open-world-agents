# Data Conversion Examples

Open World Agents provides conversion scripts to transform existing gaming and interaction datasets into the standardized OWAMcap format. This enables researchers to leverage existing datasets for training multimodal desktop agents.

!!! info "What are Data Conversions?"
    Data conversions transform existing gaming datasets (VPT, CS:GO, etc.) into the standardized OWAMcap format, enabling unified training across different games and interaction types.

## Why Convert to OWAMcap?

OWAMcap (Open World Agents MCAP) is a standardized format with these key features:

<!-- SYNC-ID: owamcap-key-features -->
- :material-database-sync: **Universal Standard**: Unlike fragmented formats, enables seamless dataset combination for large-scale foundation models *(OWAMcap)*
- :material-flash: **High-Performance Multimodal Storage**: Lightweight [MCAP](https://mcap.dev/) container with nanosecond precision for synchronized data streams *(MCAP)*
- :material-merge: **Flexible MediaRef**: Smart references to both external and embedded media (file paths, URLs, data URIs, video frames) with lazy loading - keeps metadata files small while supporting rich media *(OWAMcap)* → [Learn more](https://open-world-agents.github.io/open-world-agents/data/technical-reference/format-guide/#media-handling)
- :material-clock-outline: **Training Pipeline Ready**: Native HuggingFace integration, seamless dataset loading, and direct compatibility with ML frameworks *(Ecosystem)* → [Browse datasets](https://huggingface.co/datasets?other=OWA) | [Data pipeline](https://open-world-agents.github.io/open-world-agents/data/technical-reference/data-pipeline/)
<!-- END-SYNC: owamcap-key-features -->

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


