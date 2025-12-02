# Data Pipeline

The data pipeline documentation has moved to the [owa-data project](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-data).

## Quick Links

- **[owa-data README](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-data)** — Full documentation for converting OWAMcap recordings into HuggingFace Datasets for VLA training

## Overview

The pipeline converts raw MCAP files into training-ready datasets:

```
Raw MCAP → Event Dataset → FSL Dataset (recommended, 3x faster)
                         → Binned Dataset (traditional state-action)
```

See the [owa-data README](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-data) for:

- Quick start guide
- Stage-by-stage documentation
- Dataset schemas
- Transform usage
- Training examples
