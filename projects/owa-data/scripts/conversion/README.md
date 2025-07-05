# Data Conversion Scripts

This directory contains scripts to convert various gaming and interaction datasets into the Open World Agents MCAP format (OWAMcap). Each conversion handles the unique data structures and formats of different datasets while producing standardized OWAMcap output suitable for training multimodal desktop agents.

## Available Conversions

### [VPT (Video PreTraining)](./VPT/)
Converts OpenAI's Video PreTraining dataset to OWAMcap format.

- **Source**: [OpenAI VPT Dataset](https://github.com/openai/Video-Pre-Training)
- **Game**: Minecraft
- **Format**: MP4 videos + JSONL action files → OWAMcap

[**→ View VPT Conversion Guide**](./VPT/README.md)

### [CS_DM (Counter-Strike Deathmatch)](./CS_DM/)
Converts the Counter-Strike Deathmatch dataset from behavioral cloning research to OWAMcap format.

- **Source**: [Counter-Strike Deathmatch Dataset](https://arxiv.org/abs/2104.04258)
- **Game**: Counter-Strike: Global Offensive
- **Format**: HDF5 files → OWAMcap + External video

[**→ View CS:GO Conversion Guide**](./CS_DM/README.md)

## Getting Started

Each conversion includes detailed documentation with:

- Installation requirements and setup instructions
- Usage examples and command-line options
- Troubleshooting guides and performance tips
- Output format specifications and verification steps

For complete details, see the individual conversion guides linked above.

## Adding New Conversions

To add a new dataset conversion:

1. **Create Directory**: `mkdir NEW_DATASET_NAME`
2. **Implement Script**: Follow existing patterns for data mapping
3. **Add Documentation**: Create `README.md` with usage instructions
4. **Update This Index**: Add entry to the "Available Conversions" section
5. **Test Thoroughly**: Verify output format and data integrity

## References

- [OWAMcap Format Documentation](https://open-world-agents.github.io/open-world-agents/data/technical-reference/format-guide/)
- [Open World Agents Project](https://github.com/open-world-agents/open-world-agents)
