Goal: pseudo-label keyboard/mouse from given screen only data. Video is longer than context window size.
Method: Implement streaming inference for event generation.

Following is pseudo code for event generation.

```
def generate_single_event():
    """Generate a single event. Constrained decoding prevents (1) generation of screen event (2) generation of invalid events."""


while input_stream.has_next():
    next_event = input_stream.next()
    while last_event_timestamp < next_event_timestamp:
        generate_single_event()
    pop_last_event_if_needed() # pop the last event if it's timestamp is after next_event_timestamp
    insert_new_event(next_event)
    last_event_timestamp = next_event.timestamp
```

Cache must fulfill following requirements:
- It's static cache
- If there's no space to append new token, pop the first event. (Note that 'event' is composed of multiple tokens)

Note that all events fulfill following format:
```
<EVENT_START>(TYPE OF EVENT)(TIMESTAMP OF EVENT)(EVENT DATA)<EVENT_END>
```

Following are files you can refer to.

minimal_generation.py: official example generation code from huggingface transformers.
cache_utils.py: code copied from transformers/cache_utils.py
generate.py: contains how you can get input stream


Related materials
- https://huggingface.co/docs/transformers/main/kv_cache
- https://huggingface.co/docs/transformers/main/cache_explanation
- https://huggingface.co/docs/transformers/main/generation_strategies
- src/transformers/cache_utils.py
- src/transformers/generation/utils.py
- src/transformers/masking_utils.py
- src/transformers/models/internvl/modeling_internvl.py
- src/transformers/models/qwen2/modeling_qwen2.py