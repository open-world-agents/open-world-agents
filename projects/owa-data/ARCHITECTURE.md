# OWA Data Pipeline Architecture

This document describes the OWA data pipeline that transforms MCAP recordings into token sequences for LLM training.

## Data Flow Overview

```
MCAP Files
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  build_event_dataset()          [HIGH-LEVEL API]        │
│    ├── IntervalExtractor        (extract valid intervals)│
│    └── EventResampler           (adjust sampling rate)   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Event Dataset (HuggingFace Dataset with binary McapMessages)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  build_fsl_dataset()            [HIGH-LEVEL API]        │
│    ├── EventEncoder             (encode to text tokens)  │
│    └── Tokenization Module      (convert to token IDs)   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
FSL Dataset (Fixed Sequence Length, ready for training)
```

---

# High-Level API

For most use cases, use these two functions to build datasets.

## build_event_dataset()

**Location:** `owa/data/processing/mcap_to_event.py`

Converts MCAP files to a HuggingFace Dataset containing binary McapMessages.

```python
from owa.data.processing import McapToEventConfig, build_event_dataset

config = McapToEventConfig(
    rate_settings={"screen": 2.0, "keyboard": 10.0, "mouse/raw": 20.0},
    keep_topics=["screen", "keyboard", "mouse/raw"],
)

event_dataset = build_event_dataset(episode_paths, config=config)
```

**McapToEventConfig options:**
| Field | Type | Description |
|-------|------|-------------|
| `rate_settings` | `Dict[str, float]` | Topic → desired rate (Hz) for resampling |
| `keep_topics` | `List[str]` | Topics to include (None = all) |
| `num_workers` | `int` | Parallel workers for file processing |
| `interval_extractor_config` | `IntervalExtractorConfig` | Which intervals to extract |

**Output schema:**
| Column | Type | Description |
|--------|------|-------------|
| `episode_path` | string | Source MCAP file path |
| `topic` | string | Event topic (screen, keyboard, mouse/raw) |
| `timestamp_ns` | int64 | Event timestamp in nanoseconds |
| `mcap_message` | binary | Serialized McapMessage |

---

## build_fsl_dataset()

**Location:** `owa/data/processing/event_to_fsl.py`

Converts Event Dataset to FSL (Fixed Sequence Length) format for training.

```python
from owa.data.processing import EventToFSLConfig, build_fsl_dataset

config = EventToFSLConfig(
    tokenizer_name="OpenGVLab/InternVL3-1B-hf",
    encoder_type="factorized",
)

fsl_dataset = build_fsl_dataset(event_dataset, config=config)
```

**EventToFSLConfig options:**
| Field | Type | Description |
|-------|------|-------------|
| `tokenizer_name` | `str` | HuggingFace tokenizer name/path |
| `encoder_type` | `str` | "factorized", "hierarchical", or "json" |
| `image_token_config` | `ImageTokenConfig` | Image token settings (auto-detected if None) |
| `fsl_dataset` | `FSLDatasetConfig` | Sequence length and packing options |
| `num_proc` | `int` | Processes for tokenization |

**Output schema:**
| Column | Type | Description |
|--------|------|-------------|
| `input_ids` | sequence[int32] | Token ID sequence |
| `images` | sequence[binary] | Screen capture images |

---

# Low-Level Components

These components are used internally by the high-level API, but can be used directly for custom pipelines.

## 1. IntervalExtractor

**Location:** `owa/data/interval/selector.py`

Extracts valid time intervals from MCAP files.

**Core Method:**
- `extract_intervals(episode_path: Path) -> Intervals`

**Implementations:**
| Class | Description |
|-------|-------------|
| `All` | Returns entire file as one interval |
| `StartStopKey` | Intervals started/stopped by a key (e.g., F6) |
| `InactivityFilter` | Excludes inactive periods |

**Operators:** `&` (AND), `|` (OR), `-` (SUB)

```python
from owa.data.interval import All, InactivityFilter
extractor = All() & InactivityFilter()
```

---

## 2. EventResampler

**Location:** `owa/data/processing/resampler.py`

Adjusts event sampling rate to reduce token count.

**Core Methods:**
- `add_event(mcap_msg)`: Add event to resampler
- `step(now: int)`: Advance time
- `pop_events()`: Return processed events

**Implementations:**
| Class | Description |
|-------|-------------|
| `PassThroughResampler` | No resampling |
| `DropResampler` | Drop events within minimum interval |
| `MouseAggregationResampler` | Aggregate mouse deltas |
| `KeyboardUniformResampler` | Uniform keyboard distribution |

```python
from owa.data.processing import create_resampler
resampler = create_resampler(topic="mouse/raw", min_interval_ns=100_000_000)
```

---

## 3. EventEncoder

**Location:** `owa/data/encoders/`

Encodes McapMessage to text token strings.

**Core Methods:**
- `encode(mcap_msg) -> Tuple[str, List[ScreenCaptured]]`
- `decode(text, images) -> McapMessage`
- `get_vocab() -> Set[str]`

**Implementations:**
| Class | Description |
|-------|-------------|
| `FactorizedEventEncoder` | Decimal-based factorized encoding (default) |
| `HierarchicalEventEncoder` | Hierarchical position encoding |
| `JSONEventEncoder` | JSON format |

**Encoding Example (Factorized):**
```
Keyboard: <EVENT_START><KEYBOARD><T0_1><T1_2><T2_3><PRESS><KEY_A><EVENT_END>
Mouse:    <EVENT_START><MOUSE><T0_0><T1_5><T2_0><MOVE><SIGN_PLUS><D0_1>...<EVENT_END>
```

```python
from owa.data.encoders import create_encoder
encoder = create_encoder("factorized")
```

---

## 4. Tokenization Module

**Location:** `owa/data/tokenization/`

Converts EventEncoder output to final token IDs.

**Components:**
- `ImageTokenConfig`: Image token configuration (frozen dataclass)
- `EventTokenizationContext`: Immutable dependency container
- `expand_tokenizer_for_events()`: Add event tokens to tokenizer (side-effect)
- `tokenize_event()`: Tokenize single event (pure function)
- `decode_episode()`: Decode tokens back to messages (pure function)

```python
from owa.data.tokenization import (
    ImageTokenConfig, EventTokenizationContext,
    expand_tokenizer_for_events, tokenize_event
)

image_config = ImageTokenConfig(prefix="<img>", token="<IMG_CONTEXT>", length=256, suffix="</img>")
expand_tokenizer_for_events(tokenizer, encoder, image_config)
ctx = EventTokenizationContext(encoder, tokenizer, image_config)

result = tokenize_event(ctx, mcap_msg)  # Returns TokenizedEvent dict
```

**TokenizedEvent format:**
```python
{"text": str, "images": list, "token_ids": list, "total_token_count": int}
```

