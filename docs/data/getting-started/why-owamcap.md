# Why OWAMcap?

**The Problem**: Desktop AI datasets are fragmented. Every research group uses different formats, making it impossible to combine datasets or build large-scale foundation models.

**The Solution**: OWAMcap provides a universal standard that treats all desktop interaction datasets equally.

## The Robotics Lesson

The [Open-X Embodiment](https://robotics-transformer-x.github.io/) project had to manually convert **22 different robotics datasets** - months of work just to combine data. Desktop automation is heading down the same path.

## OWAMcap Changes This

### Before: Data Silos
```
Dataset A (Custom Format) ──┐
Dataset B (Custom Format) ──┼── Manual Conversion ──→ Limited Training Data
Dataset C (Custom Format) ──┘
```

### After: Universal Standard
```
Dataset A (OWAMcap) ──┐
Dataset B (OWAMcap) ──┼── Direct Combination ──→ Large-Scale Foundation Models
Dataset C (OWAMcap) ──┘
```

## From Recording to Training in 3 Commands

OWAMcap integrates with the complete [OWA Data Pipeline](../technical-reference/data-pipeline.md):

```bash
# 1. Record desktop interaction
ocap my-session.mcap

# 2. Process to training format
python scripts/01_raw_events_to_event_dataset.py --train-dir ./

# 3. Train your model
python train.py --dataset ./event-dataset
```

**Result**: Any OWAMcap dataset works with any OWA-compatible training pipeline.

## Technical Advantages

- **91.7× compression** through hybrid storage (metadata + external video)
- **Nanosecond precision** for perfect event synchronization
- **Standard tools** work with video files (VLC, FFmpeg, etc.)
- **Lazy loading** for memory-efficient processing

## Real Impact

```bash
$ owl mcap info example.mcap
messages:  864 (10.36s of interaction data)
file size: 22 KiB (vs 1+ GB raw)
channels:  screen, mouse, keyboard, window
```

**Bottom Line**: OWAMcap transforms desktop interaction data from isolated collections into a unified resource for building the next generation of foundation models.

---

**Ready to get started?** Continue to the [OWAMcap Format Guide](../technical-reference/format-guide.md) for technical details.
