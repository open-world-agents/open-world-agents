## AS-IS: Current Event Dataset Format

Executing the following command generates HuggingFace datasets with the specified features:

```bash
python scripts/01_raw_events_to_event_dataset.py \
  -tr /mnt/raid11/datasets/owa/mcaps/super-hexagon \
  -te /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s \
  -o /mnt/raid11/datasets/owa/data/super-hexagon-event
```

**Dataset Schema:**
```python
features = Features(
    {
        "file_path": Value("string"),      # Source MCAP file path
        "topic": Value("string"),          # Event topic (e.g., 'keyboard', 'screen')
        "timestamp_ns": Value("int64"),    # Timestamp in nanoseconds
        "message_type": Value("string"),   # Full message type identifier
        "msg": Value("binary"),            # Serialized message content
    }
)
```

**Example Records:**
```python
# Keyboard event example
{'file_path': '/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-new-jy-1.mcap',
 'topic': 'keyboard',
 'timestamp_ns': 1745362786814673800,
 'message_type': 'owa.env.desktop.msg.KeyboardEvent',
 'msg': b'{"event_type":"press","vk":37}'}

# Screen event example
{'file_path': '/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-3.mcap',
 'topic': 'screen',
 'timestamp_ns': 1743128886688495300,
 'message_type': 'owa.env.gst.msg.ScreenEmitted',
 'msg': b'{"path":"expert-jy-3.mkv","pts":70350000000,"utc_ns":1743128886688495300}'}
```

**Terminology:** We refer to the output of `01_raw_events_to_event_dataset.py` as the "Event Dataset", where each row represents a single raw event.

## TO-BE: Target Format for VLA Training

We need to convert this dataset into a format compatible with MLLM (Multimodal Large Language Model) training suites to train VLA (Vision-Language-Action) models.

**Implementation Requirements:**

### EventEncoder

The EventEncoder must handle two main responsibilities:

1. **Text Serialization**: Convert raw events to natural language format for LLM tokenization
   - Example input: `{'file_path': '...', 'topic': 'keyboard', 'timestamp_ns': 1745362786814673800, 'message_type': 'owa.env.desktop.msg.KeyboardEvent', 'msg': b'{"event_type":"press","vk":37}'}`
   - Must be converted to human-readable text format

2. **Multimodal Handling**: Include visual data for screen events
   - Example input: `{'file_path': '...', 'topic': 'screen', 'timestamp_ns': 1743128886688495300, 'message_type': 'owa.env.gst.msg.ScreenEmitted', 'msg': b'{"path":"expert-jy-3.mkv","pts":70350000000,"utc_ns":1743128886688495300}'}`
   - Must extract and include corresponding images for MLLM image processing

**Core Function**: The EventEncoder processes a SINGLE EVENT and converts it to an encoded format.

#### Serialization Format Options

**Brainstormed-one**
```
<EVENT_START>08.490,KEYBOARD,(press,37)<EVENT_END>
```
Where:
- `08.490` = relative timestamp (8 seconds 490 milliseconds from reference point)
- `KEYBOARD` = event type identifier
- `(press,37)` = message content (action, virtual key code)

**Inefficient, but simplest one**
```
<EVENT_START>{'file_path': '/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-new-jy-1.mcap', 'topic': 'keyboard', 'timestamp_ns': 1745362786814673800, 'message_type': 'owa.env.desktop.msg.KeyboardEvent', 'msg': b'{"event_type":"press","vk":37}'}<EVENT_END>
```

**Implementation Strategy:**
- **Phase 1**: Implement the "simplest" raw serialization method first
- **Phase 2**: Optimize for token efficiency using:
  1. Abbreviations (e.g., "owa.env.desktop.msg.KeyboardEvent" → "keyboard")
  2. Special tokens (e.g., "keyboard" → "<KEYBOARD>" token)

**Rationale**: While the raw format is token-inefficient, it provides a foundation for developing more sophisticated serialization methods. The `<EVENT_START>` and `<EVENT_END>` tokens establish the framework for future optimizations.

#### Multimodal Output Format

For screen events with images, the EventEncoder should return:
```python
Tuple[
    str,  # Serialized text with <IMAGE> placeholders where needed
    List[Union[ScreenEmitted, dict]]  # Corresponding image data (lazy-loaded)
]
```

#### Decoding

The EventEncoder must also implement decoding functionality to convert encoded events back to their original format.

## Implementation Status

**Current Implementation**: `EventEncoder` class in `owa/data/event_encoder.py`
- ✅ Phase 1: Raw format encoding/decoding
- ✅ Multimodal support for ScreenEmitted events
- ✅ Type-safe handling of KeyboardEvent, MouseEvent, ScreenEmitted
- ✅ Integration with existing HuggingFace dataset format
- 🔄 Phase 2: Token-efficient format (planned)

### Serialize whole "Event Dataset" and generate MLLM-training-dataset.

Then, with implemented "EventSerializer", we must write a code that convert "Event Dataset" to MLLM-training dataset. My goal is to make `nanoVLM`-compatiable dataset. Write a new "Dataset" class(which inherits pytorch dataset). Refer to nanoVLM/data/datasets.py.