# EventEncoder Implementation

This document describes the implementation of the `EventEncoder` class for converting raw events to MLLM-compatible format.

## Overview

The `EventEncoder` converts raw event data from the Event Dataset into formats suitable for training Vision-Language-Action (VLA) models. It implements Phase 1 of the serialization strategy using the "simplest" raw format with `<EVENT_START>` and `<EVENT_END>` tokens.

## Features

### ✅ Implemented (Phase 1)
- **Raw Format Encoding**: Converts events to text format with `<EVENT_START>`/`<EVENT_END>` tokens
- **Multimodal Support**: Handles screen events with image data using `<IMAGE>` placeholders
- **Round-trip Consistency**: Perfect encode/decode consistency for all event types
- **Batch Processing**: Efficient batch encoding/decoding capabilities
- **Type Safety**: Proper error handling and type validation
- **Integration Ready**: Works with existing HuggingFace dataset format

### 🔄 Planned (Phase 2)
- Token-efficient format with abbreviations
- Special tokens for common patterns
- LLM tokenizer optimization
- Compressed representation for better performance

## Usage

### Basic Usage

```python
from owa.data.event_encoder import EventEncoder

# Initialize encoder
encoder = EventEncoder()

# Sample raw event
raw_event = {
    'file_path': '/path/to/file.mcap',
    'topic': 'keyboard',
    'timestamp_ns': 1745362786814673800,
    'message_type': 'owa.env.desktop.msg.KeyboardEvent',
    'msg': b'{"event_type":"press","vk":37}'
}

# Encode event
text, images = encoder.encode(raw_event)
print(text)
# Output: <EVENT_START>{'file_path': '/path/to/file.mcap', 'topic': 'keyboard', ...}<EVENT_END>

# Decode back to original
decoded_event = encoder.decode(text, images)
assert decoded_event == raw_event  # Perfect round-trip consistency
```

### Batch Processing

```python
# Encode multiple events
events = [keyboard_event, mouse_event, screen_event]
texts, all_images = encoder.encode_batch(events)

# Decode batch
decoded_events = encoder.decode_batch(texts, all_images)
```

### Screen Events with Images

```python
# Screen events automatically handle image data
screen_event = {
    'file_path': '/path/to/screen.mcap',
    'topic': 'screen',
    'timestamp_ns': 1743128886688495300,
    'message_type': 'owa.env.gst.msg.ScreenEmitted',
    'msg': b'{"path":"video.mkv","pts":70350000000,"utc_ns":1743128886688495300}'
}

text, images = encoder.encode(screen_event)
# text contains <IMAGE> placeholder
# images contains ScreenEmitted object with lazy-loading capability
```

## Implementation Details

### Event Types Supported
- **Keyboard Events**: `owa.env.desktop.msg.KeyboardEvent`
- **Mouse Events**: `owa.env.desktop.msg.MouseEvent`  
- **Screen Events**: `owa.env.gst.msg.ScreenEmitted` (with image data)

### Output Format

#### Text Events (Keyboard/Mouse)
```
<EVENT_START>{'file_path': '...', 'topic': 'keyboard', 'timestamp_ns': 123, 'message_type': '...', 'msg': b'...'}<EVENT_END>
```

#### Screen Events
```
<EVENT_START>{'file_path': '...', 'topic': 'screen', 'timestamp_ns': 123, 'message_type': '...', 'msg': b'<IMAGE>'}<EVENT_END>
```
Plus corresponding `ScreenEmitted` object in the images list.

### Error Handling
- **Invalid Input**: Validates required keys and data types
- **JSON Parsing**: Handles malformed message content gracefully
- **Round-trip Safety**: Ensures decode(encode(x)) == x for all valid inputs
- **Batch Consistency**: Validates batch dimensions and formats

## Files

- **`owa/data/event_encoder.py`**: Main EventEncoder implementation
- **`tests/test_event_encoder.py`**: Comprehensive test suite (18 tests)
- **`example_event_encoder.py`**: Usage demonstration script
- **`owa/data/__init__.py`**: Updated to export EventEncoder

## Testing

Run the test suite:
```bash
cd projects/owa-data
python -m pytest tests/test_event_encoder.py -v
```

All 18 tests pass, covering:
- Individual event encoding/decoding
- Batch processing
- Error handling scenarios
- Round-trip consistency
- Multimodal support

## Integration with Goal Document

The implementation fulfills all requirements specified in `goal.md`:

1. ✅ **EventEncoder Class**: Renamed from "EventSerializer" as requested
2. ✅ **Phase 1 Implementation**: Simple raw format with framework for optimization
3. ✅ **Multimodal Support**: Screen events with image handling
4. ✅ **Bidirectional Operations**: Both encode() and decode() methods
5. ✅ **Single Event Processing**: Core functionality processes one event at a time
6. ✅ **Batch Capabilities**: Added for efficiency
7. ✅ **Token Framework**: `<EVENT_START>`/`<EVENT_END>` tokens ready for Phase 2

## Next Steps

For Phase 2 optimization:

1. **Token Efficiency**: Implement abbreviations for common message types
2. **Special Tokens**: Add `<KEYBOARD>`, `<MOUSE>`, `<SCREEN>` tokens
3. **Time Normalization**: Convert absolute timestamps to relative format
4. **Compression**: Reduce redundant information in serialized format
5. **LLM Integration**: Test with actual tokenizers and optimize accordingly

The current implementation provides a solid foundation for these future enhancements while maintaining backward compatibility.
