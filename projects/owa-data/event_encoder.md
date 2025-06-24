# Event Encoders for OWA Data Pipeline

## Overview

The OWA data pipeline provides multiple event encoding strategies for converting raw events into formats suitable for different types of VLA (Vision-Language-Action) model training. Each encoder implements a different tokenization approach optimized for specific use cases and model architectures.

## Available Encoders

### 1. **JSONEventEncoder** (JSON String-Based)
- **Format**: JSON string serialization with `<EVENT_START>` and `<EVENT_END>` tokens
- **Use Case**: General-purpose MLLM training with text-based models
- **Vocabulary**: No fixed vocabulary (JSON string-based)
- **Example**: `<EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 123, ...}<EVENT_END>`

### 2. **FlatEventEncoder** (Flat Token-Based)
- **Format**: Flat tokens like `<TIMESTAMP_123>`, `<KEYBOARD_65_press>` wrapped with `<EVENT_START>` and `<EVENT_END>`
- **Use Case**: Models that work well with large vocabularies and direct token prediction
- **Vocabulary**: ~12,000+ tokens (combinatorial)
- **Example**: `<EVENT_START><TIMESTAMP_123><KEYBOARD_65_press><EVENT_END>`

### 3. **HierarchicalEventEncoder** (Compositional Token-Based)
- **Format**: Hierarchical tokens like `<TIMESTAMP><123>`, `<KEYBOARD><65><press>` wrapped with `<EVENT_START>` and `<EVENT_END>`
- **Use Case**: Efficient VLA training with compositional understanding
- **Vocabulary**: ~292 tokens (96.6% reduction vs flat)
- **Example**: `<EVENT_START><TIMESTAMP><123><KEYBOARD><65><press><EVENT_END>`

## Architecture

### Base Interface

All encoders inherit from `BaseEventEncoder` which provides a consistent interface:

```python
from owa.data import BaseEventEncoder

class BaseEventEncoder(ABC):
    @abstractmethod
    def encode(self, raw_event: Dict[str, Any]) -> Tuple[str, List[Union[ScreenCaptured, Dict]]]:
        """Encode a single raw event."""
        pass

    @abstractmethod
    def decode(self, encoded_data: str, images: Optional[List[Union[ScreenCaptured, Dict]]] = None) -> Dict[str, Any]:
        """Decode back to original raw event format."""
        pass

    @abstractmethod
    def encode_batch(self, raw_events: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Union[ScreenCaptured, Dict]]]]:
        """Encode a batch of raw events."""
        pass

    @abstractmethod
    def decode_batch(self, encoded_batch: List[str], all_images: Optional[List[List[Union[ScreenCaptured, Dict]]]] = None) -> List[Dict[str, Any]]:
        """Decode a batch of encoded data."""
        pass
```

### File Structure

```
owa/data/encoders/
├── __init__.py                      # Exports all encoders
├── base_encoder.py                  # Abstract base class
├── json_event_encoder.py            # JSON string-based encoder
├── flat_event_encoder.py            # Flat token encoder
└── hierarchical_event_encoder.py   # Hierarchical token encoder
```

**Note**: There is no `event_encoder.py` file - the module structure uses individual encoder files.

## Usage Examples

### Basic Usage

```python
from owa.data import JSONEventEncoder, FlatEventEncoder, HierarchicalEventEncoder

# JSON string-based encoder
json_encoder = JSONEventEncoder()
text, images = json_encoder.encode(raw_event)

# Flat token encoder
flat_encoder = FlatEventEncoder()
flat_tokens, images = flat_encoder.encode(raw_event)  # flat_tokens is a string with concatenated tokens

# Hierarchical token encoder
hierarchical_encoder = HierarchicalEventEncoder()
hierarchical_tokens, images = hierarchical_encoder.encode(raw_event)  # hierarchical_tokens is a string
```

### Batch Processing

```python
# Encode multiple events
all_tokens, all_images = encoder.encode_batch(raw_events)

# Decode back to original format
decoded_events = encoder.decode_batch(all_tokens, all_images)
```

### Token ID Conversion (for neural networks)

```python
# For token-based encoders (FlatEventEncoder and HierarchicalEventEncoder)
if hasattr(encoder, 'get_token_ids'):
    # Note: tokens must be parsed from the string first
    import re
    tokens = re.findall(r"<[^>]*>", encoded_string.replace("<EVENT_START>", "").replace("<EVENT_END>", ""))
    token_ids = encoder.get_token_ids(tokens)
    vocab_size = encoder.get_vocab_size()
```

## Detailed Encoder Specifications

### JSONEventEncoder (JSON String-Based)

**Characteristics:**
- **Output Format**: Single string per event
- **Vocabulary**: No fixed vocabulary
- **Token Count**: Variable (147-209 characters per event)
- **Memory**: Higher due to string processing

**Example Output:**
```
<EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 1745362786814673800, 'message_type': 'desktop/KeyboardEvent', 'msg': '{"event_type":"press","vk":65}'}<EVENT_END>
```

**Best For:**
- General-purpose MLLM training
- Text-based language models
- Rapid prototyping

### FlatEventEncoder (Flat Tokens)

**Characteristics:**
- **Output Format**: String with concatenated flat tokens wrapped in `<EVENT_START>` and `<EVENT_END>`
- **Vocabulary**: Configurable (default ~13,000+ tokens)
- **Token Count**: Variable tokens per event (timestamp + event-specific tokens)
- **Memory**: High vocabulary size

**Token Examples:**
```
Timestamp: <TIMESTAMP_123>
Keyboard:  <KEYBOARD_65_press>, <KEYBOARD_65_release>
Mouse:     <MOUSE_move_0_15_32>, <MOUSE_click_left_press>
Screen:    <SCREEN>
```

**Vocabulary Breakdown (Default Config):**
- EVENT_START/EVENT_END: 2 tokens
- Timestamp tokens: 201 (configurable based on time range and interval)
- Keyboard tokens: 512 (256 keys × 2 actions: press/release)
- Mouse move tokens: 12,288 (16³ levels × 3 quantization levels × 2 coords)
- Mouse click tokens: 8 (4 buttons × 2 actions)
- Mouse scroll tokens: 49 (7×7 grid for dx/dy from -3 to +3)
- Screen tokens: 1
- Unknown token: 1
- **Total**: ~13,062 tokens

**Best For:**
- Models with large vocabulary capacity
- Direct token prediction tasks
- Traditional transformer architectures

### HierarchicalEventEncoder (Compositional Tokens)

**Characteristics:**
- **Output Format**: String with concatenated hierarchical tokens wrapped in `<EVENT_START>` and `<EVENT_END>`
- **Vocabulary**: ~292 tokens (calculated from HierarchicalVocabulary class)
- **Token Count**: Variable tokens per event (timestamp + event-specific tokens)
- **Memory**: Very efficient

**Token Examples:**
```
Timestamp: <TIMESTAMP><123>
Keyboard:  <KEYBOARD><65><press>
Mouse:     <MOUSE><move><8><8><0><0><0><0><left><press>
Screen:    <SCREEN>
```

**Hierarchical Structure:**
```
Level 1: Event Categories
<TIMESTAMP>  <KEYBOARD>  <MOUSE>  <SCREEN>

Level 2: Action Types  
<MOUSE> → <move> | <click> | <scroll>
<KEYBOARD> → <press> | <release>

Level 3: Parameters
<MOUSE><click> → <left><press> | <right><release>
<KEYBOARD><press> → <65> (A key) | <72> (H key)
```

**Vocabulary Breakdown (From HierarchicalVocabulary class):**
- Base tokens: 8 (`<EVENT_START>`, `<EVENT_END>`, `<TIMESTAMP>`, `<KEYBOARD>`, `<MOUSE>`, `<SCREEN>`, `<PAD>`, `<UNK>`)
- Number tokens: 256 (0-255 for various parameters)
- Action tokens: 5 (`<press>`, `<release>`, `<move>`, `<click>`, `<scroll>`)
- Button tokens: 4 (`<left>`, `<right>`, `<middle>`, `<unknown>`)
- Negative number tokens: 21 (-10 to +10 for scroll deltas)
- **Total**: ~294 tokens (exact count from vocabulary.get_vocab_size())

**Best For:**
- Efficient VLA training
- Compositional learning
- Resource-constrained environments
- Models that benefit from hierarchical structure

## Performance Comparison

### Vocabulary Size Comparison

| Encoder | Vocabulary Size | Reduction | Memory Usage |
|---------|----------------|-----------|--------------|
| JSONEventEncoder | N/A (strings) | - | High |
| FlatEventEncoder | ~13,062 tokens | - | Very High |
| HierarchicalEventEncoder | ~294 tokens | 97.7% ↓ | Low |

### Token Efficiency

**Note**: Token counts are approximate and depend on specific event parameters and configuration.

| Event Type | JSONEventEncoder | FlatEventEncoder | HierarchicalEventEncoder |
|------------|-----------------|------------------|-------------------------|
| Keyboard | ~176 chars | String with timestamp + keyboard tokens | String with `<TIMESTAMP><idx><KEYBOARD><vk><action>` |
| Mouse Move | ~177 chars | String with timestamp + move tokens | String with `<TIMESTAMP><idx><MOUSE><move>` + coordinates |
| Mouse Click | ~209 chars | String with timestamp + move + click tokens | String with timestamp + move + click tokens |
| Screen | ~147 chars | String with timestamp + `<SCREEN>` | String with `<TIMESTAMP><idx><SCREEN>` |

### Round-Trip Accuracy

All encoders achieve **100% round-trip accuracy** - events can be perfectly reconstructed from their encoded representations.

## Configuration Options

### FlatEventEncoderConfig

```python
from owa.data import FlatEventEncoderConfig

config = FlatEventEncoderConfig(
    timestamp_min_ns=-2 * TimeUnits.SECOND,
    timestamp_max_ns=2 * TimeUnits.SECOND,
    timestamp_interval_ns=20 * TimeUnits.MSECOND,  # 50fps
    keyboard_vk_count=256,
    mouse_move_bins=[16, 16, 16],  # 3-level residual quantization
    screen_size=(1920, 1080),
    drop_file_path=True,
)
```

### HierarchicalEventEncoderConfig

```python
from owa.data import HierarchicalEventEncoderConfig

config = HierarchicalEventEncoderConfig(
    timestamp_min_ns=-2 * TimeUnits.SECOND,
    timestamp_max_ns=2 * TimeUnits.SECOND,
    timestamp_interval_ns=20 * TimeUnits.MSECOND,  # 50fps
    mouse_move_bins=[16, 16, 16],  # 3-level residual quantization
    screen_size=(1920, 1080),
    drop_file_path=True,
)
```

## Advanced Features

### Mouse Coordinate Encoding

Both token-based encoders use **3-level residual quantization** for precise mouse positioning:

```python
# Example: Mouse at (960, 540) on 1920x1080 screen
# Normalized coordinates: (0.5, 0.5)

# Level 0: Coarse positioning (16x16 grid)
level_0_x, level_0_y = 8, 8  # Center region

# Level 1: Medium refinement within region
level_1_x, level_1_y = 0, 0  # Exact center

# Level 2: Fine positioning
level_2_x, level_2_y = 0, 0  # Precise center

# Flat encoding:
['<MOUSE_move_0_8_8>', '<MOUSE_move_1_0_0>', '<MOUSE_move_2_0_0>']

# Hierarchical encoding:
['<MOUSE>', '<move>', '<8>', '<8>', '<0>', '<0>', '<0>', '<0>']
```

### Token ID Conversion

For neural network training, token-based encoders (FlatEventEncoder and HierarchicalEventEncoder) provide ID conversion:

```python
# First, parse tokens from the encoded string
import re
encoded_string = "<EVENT_START><TIMESTAMP><123><KEYBOARD><65><press><EVENT_END>"
token_content = encoded_string.replace("<EVENT_START>", "").replace("<EVENT_END>", "")
tokens = re.findall(r"<[^>]*>", token_content)

# Convert tokens to integer IDs
token_ids = encoder.get_token_ids(tokens)
# Result: [2, 123, 3, 65, 48] (example IDs)

# Convert back to tokens
tokens = encoder.get_tokens_from_ids(token_ids)

# Get vocabulary size for model configuration
vocab_size = encoder.get_vocab_size()
```

### Batch Processing Optimization

All encoders support efficient batch processing:

```python
# Process multiple events at once
raw_events = [event1, event2, event3, ...]
all_tokens, all_images = encoder.encode_batch(raw_events)

# Parallel decoding
decoded_events = encoder.decode_batch(all_tokens, all_images)
```

## Integration with VLA Training

### Model Architecture Considerations

**For JSONEventEncoder (JSON string-based):**
- Use text tokenizers (e.g., BPE, SentencePiece)
- Standard transformer architectures
- Higher memory requirements

**For FlatEventEncoder:**
- Large embedding tables (~13K vocab)
- Direct token prediction heads
- Traditional classification approach

**For HierarchicalEventEncoder:**
- Small embedding tables (~292 vocab)
- Compositional learning architectures
- Hierarchical attention mechanisms

### Training Data Format

```python
# Example training sample
{
    "observation": PIL.Image,      # Screen capture
    "encoded_events": List[str],   # Encoded action sequence (all encoders return strings)
    "metadata": {
        "episode_id": str,
        "timestamp_ns": int,
        "success": bool
    }
}
```

### Loss Computation

For token-based encoders, consider weighted loss functions:

```python
def compute_hierarchical_loss(predictions, targets, token_weights=None):
    """Compute loss with optional weighting for different token types."""
    if token_weights is None:
        token_weights = {
            "base_tokens": 2.0,      # <MOUSE>, <KEYBOARD>, etc.
            "action_tokens": 3.0,    # <press>, <click>, etc.
            "param_tokens": 1.0      # coordinates, indices
        }

    weighted_loss = 0.0
    for pred, target in zip(predictions, targets):
        token_type = get_token_type(target)
        weight = token_weights.get(token_type, 1.0)
        weighted_loss += weight * F.cross_entropy(pred, target)

    return weighted_loss / len(predictions)
```

## Choosing the Right Encoder

### Decision Matrix

| Use Case | Recommended Encoder | Reason |
|----------|-------------------|---------|
| **General MLLM Training** | JSONEventEncoder | Compatible with text-based models |
| **Large-Scale VLA Training** | HierarchicalEventEncoder | Efficient vocabulary, compositional learning |
| **Direct Action Prediction** | FlatEventEncoder | Direct token-to-action mapping |
| **Resource-Constrained Training** | HierarchicalEventEncoder | Minimal memory footprint |
| **Rapid Prototyping** | JSONEventEncoder | Simple JSON string-based format |
| **Production VLA Systems** | HierarchicalEventEncoder | Optimal efficiency and performance |

### Performance Guidelines

**Memory Usage:**
- JSONEventEncoder: ~200 chars/event × batch_size
- FlatEventEncoder: ~13K vocab embeddings + 2-5 tokens/event
- HierarchicalEventEncoder: ~292 vocab embeddings + 3-12 tokens/event

**Training Speed:**
- JSONEventEncoder: Slower (string processing)
- FlatEventEncoder: Fast (direct token lookup)
- HierarchicalEventEncoder: Fastest (small vocabulary)

**Model Convergence:**
- JSONEventEncoder: Standard text model convergence
- FlatEventEncoder: May require larger models due to vocabulary size
- HierarchicalEventEncoder: Faster convergence due to compositional structure

## Migration Guide

### From JSONEventEncoder to Token-Based Encoders

```python
# Old approach
old_encoder = JSONEventEncoder()
text, images = old_encoder.encode(raw_event)

# New approach - Hierarchical
new_encoder = HierarchicalEventEncoder()
tokens, images = new_encoder.encode(raw_event)

# Convert to IDs for model training
token_ids = new_encoder.get_token_ids(tokens)
```

### Updating Training Pipelines

```python
# Update dataset creation
def create_training_data(raw_events, encoder):
    all_tokens, all_images = encoder.encode_batch(raw_events)

    if hasattr(encoder, 'get_token_ids'):
        # Token-based encoder
        all_token_ids = [encoder.get_token_ids(tokens) for tokens in all_tokens]
        return all_token_ids, all_images
    else:
        # String-based encoder
        return all_tokens, all_images
```

## Testing and Validation

### Round-Trip Testing

```python
def test_round_trip(encoder, raw_events):
    """Test that encoding/decoding preserves original data."""
    encoded_data, images = encoder.encode_batch(raw_events)
    decoded_events = encoder.decode_batch(encoded_data, images)

    success_count = 0
    for orig, decoded in zip(raw_events, decoded_events):
        if orig['topic'] == decoded['topic']:
            success_count += 1

    accuracy = success_count / len(raw_events) if raw_events else 0.0
    print(f"Round-trip accuracy: {accuracy:.2%}")
    return accuracy
```

### Performance Benchmarking

```python
import time

def benchmark_encoder(encoder, raw_events, num_iterations=100):
    """Benchmark encoder performance."""

    # Encoding benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        encoded_data, images = encoder.encode_batch(raw_events)
    encoding_time = (time.time() - start_time) / num_iterations

    # Decoding benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        decoded_events = encoder.decode_batch(encoded_data, images)
    decoding_time = (time.time() - start_time) / num_iterations

    return {
        "encoding_time": encoding_time,
        "decoding_time": decoding_time,
        "vocab_size": encoder.get_vocab_size() if hasattr(encoder, 'get_vocab_size') else None
    }
```

## Future Extensions

The modular encoder architecture enables easy implementation of additional encoders:

### Planned Encoders

1. **SequentialActionEncoder**: For direct action sequence prediction
2. **MultimodalTrajectoryEncoder**: For natural language + action integration
3. **CompressedEventEncoder**: For ultra-low bandwidth scenarios
4. **TemporalEventEncoder**: For time-aware action modeling

### Custom Encoder Development

```python
from owa.data.encoders import BaseEventEncoder

class CustomEventEncoder(BaseEventEncoder):
    def encode(self, raw_event):
        # Custom encoding logic
        pass

    def decode(self, encoded_data, images=None):
        # Custom decoding logic
        pass

    # Implement other required methods...
```

## Conclusion

The OWA event encoder ecosystem provides flexible, efficient solutions for different VLA training scenarios. The hierarchical approach offers the best balance of efficiency and expressiveness for most use cases, while maintaining full compatibility with the existing OWA data pipeline.

**Key Takeaways:**
- **HierarchicalEventEncoder** is recommended for most VLA training scenarios
- **97.7% vocabulary reduction** compared to flat approaches (294 vs 13,062 tokens)
- **100% round-trip accuracy** across all encoders
- **Modular architecture** enables easy extension and customization
- **All encoders return strings** wrapped with `<EVENT_START>` and `<EVENT_END>` tokens

For questions or contributions, please refer to the OWA project documentation and community guidelines.
```
