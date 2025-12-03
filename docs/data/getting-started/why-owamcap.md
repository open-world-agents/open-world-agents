# Why OWAMcap?

Desktop and GUI agent datasets use inconsistent formats, making it hard to combine data from different sources.

## The Fragmentation Problem

Existing datasets each define their own format:

| Dataset | Venue | Domain | Format |
|---------|-------|--------|--------|
| [VPT](https://github.com/openai/Video-Pre-Training) | - | Minecraft | MP4 + JSONL (per-frame action dictionaries) |
| [CS Deathmatch](https://arxiv.org/abs/2104.04258) | CoG '22 | CS:GO | HDF5 (screenshots + action labels) + NPY (metadata) |
| [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) | NeurIPS '23 | Web | Playwright traces + DOM snapshots (JSON) + screenshots (base64 JSON) + HAR |
| [OmniACT](https://huggingface.co/datasets/Writer/omniact) | ECCV '24 | Desktop/Web | PNG screenshots + TXT (task + PyAutoGUI script) + bounding box JSON |


This is similar to how the [Open-X Embodiment](https://robotics-transformer-x.github.io/) project had to manually convert 22 different robotics datasets. OWAMcap addresses this by providing a general desktop message definition based on [MCAP](https://mcap.dev/). To demonstrate this, we provide [conversion scripts](../conversions.md) that transform VPT, CS:GO, and other existing datasets into OWAMcap, allowing them to be combined and used with a unified training pipeline.

## From Recording to Training

OWAMcap integrates with the complete [OWA Data Pipeline](../technical-reference/data-pipeline.md):

<!-- SYNC-ID: quick-start-3-steps -->
```bash
# 1. Record desktop interaction
$ ocap my-session.mcap

# 2. Process to training format
$ python scripts/01_raw_events_to_event_dataset.py --train-dir ./

# 3. Train your model
$ python train.py --dataset ./event-dataset
```
<!-- END-SYNC: quick-start-3-steps -->

> 📖 **Detailed Guide**: [Complete Quick Start Tutorial](../../quick-start.md) - Step-by-step walkthrough with examples and troubleshooting

## Key Features

<!-- SYNC-ID: owamcap-key-features -->
- 🌐 **Universal Standard**: Unlike fragmented formats, enables seamless dataset combination for large-scale foundation models *(OWAMcap)*
- ⚡ **High-Performance Multimodal Storage**: Lightweight [MCAP](https://mcap.dev/) container with nanosecond precision for synchronized data streams *(MCAP)*
- 🔗 **Flexible MediaRef**: Smart references to both external and embedded media (file paths, URLs, data URIs, video frames) with lazy loading - keeps metadata files small while supporting rich media *(OWAMcap)* → [Learn more](../technical-reference/format-guide.md#media-handling)
- 🤗 **Training Pipeline Ready**: Native HuggingFace integration, seamless dataset loading, and direct compatibility with ML frameworks *(Ecosystem)* → [Browse datasets](https://huggingface.co/datasets?other=OWA) | [Data pipeline](../technical-reference/data-pipeline.md)
<!-- END-SYNC: owamcap-key-features -->

## Example

```bash
$ owl mcap info example.mcap
messages:  864 (10.36s of interaction data)
file size: 22 KiB (vs 1+ GB raw)
channels:  screen, mouse, keyboard, window
```

See [OWAMcap Format Guide](../technical-reference/format-guide.md) for technical details.
