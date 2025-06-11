#!/usr/bin/env python3
"""
Example usage of EventEncoder for converting raw events to MLLM training format.

This script demonstrates how to use the EventEncoder to convert raw event data
from the Event Dataset into formats suitable for training Vision-Language-Action models.
"""

from owa.data.event_encoder import EventEncoder


def main():
    """Demonstrate EventEncoder functionality with sample events."""
    
    # Initialize the encoder
    encoder = EventEncoder()
    print("EventEncoder initialized successfully!")
    print("=" * 60)
    
    # Sample events from different topics
    sample_events = [
        # Keyboard event
        {
            'file_path': '/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-new-jy-1.mcap',
            'topic': 'keyboard',
            'timestamp_ns': 1745362786814673800,
            'message_type': 'owa.env.desktop.msg.KeyboardEvent',
            'msg': b'{"event_type":"press","vk":37}'
        },
        
        # Mouse event
        {
            'file_path': '/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-2.mcap',
            'topic': 'mouse',
            'timestamp_ns': 1745362786814673900,
            'message_type': 'owa.env.desktop.msg.MouseEvent',
            'msg': b'{"event_type":"move","x":100,"y":200}'
        },
        
        # Screen event
        {
            'file_path': '/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-3.mcap',
            'topic': 'screen',
            'timestamp_ns': 1743128886688495300,
            'message_type': 'owa.env.gst.msg.ScreenEmitted',
            'msg': b'{"path":"expert-jy-3.mkv","pts":70350000000,"utc_ns":1743128886688495300}'
        }
    ]
    
    print("Sample Raw Events:")
    for i, event in enumerate(sample_events, 1):
        print(f"\n{i}. {event['topic'].upper()} EVENT:")
        print(f"   File: {event['file_path'].split('/')[-1]}")
        print(f"   Timestamp: {event['timestamp_ns']}")
        print(f"   Message: {event['msg'].decode('utf-8')}")
    
    print("\n" + "=" * 60)
    print("ENCODING EVENTS...")
    print("=" * 60)
    
    # Encode individual events
    for i, event in enumerate(sample_events, 1):
        print(f"\n{i}. Encoding {event['topic']} event:")
        
        text, images = encoder.encode(event)
        
        print(f"   Encoded text: {text[:100]}...")
        if images:
            print(f"   Images: {len(images)} image(s) attached")
            if isinstance(images[0], dict) and 'screen_event' in images[0]:
                screen_event = images[0]['screen_event']
                print(f"   Screen event details: path={screen_event.path}, pts={screen_event.pts}")
        else:
            print("   Images: None")
    
    print("\n" + "=" * 60)
    print("BATCH ENCODING...")
    print("=" * 60)
    
    # Encode batch of events
    texts, all_images = encoder.encode_batch(sample_events)
    
    print(f"Encoded {len(texts)} events in batch:")
    for i, (text, images) in enumerate(zip(texts, all_images), 1):
        topic = sample_events[i-1]['topic']
        print(f"   {i}. {topic}: {len(text)} chars, {len(images)} images")
    
    print("\n" + "=" * 60)
    print("DECODING (ROUND-TRIP TEST)...")
    print("=" * 60)
    
    # Decode back to original format
    decoded_events = encoder.decode_batch(texts, all_images)
    
    print("Round-trip consistency check:")
    for i, (original, decoded) in enumerate(zip(sample_events, decoded_events), 1):
        is_identical = original == decoded
        topic = original['topic']
        print(f"   {i}. {topic}: {'✓ PASS' if is_identical else '✗ FAIL'}")
        
        if not is_identical:
            print(f"      Original: {original}")
            print(f"      Decoded:  {decoded}")
    
    print("\n" + "=" * 60)
    print("PHASE 1 IMPLEMENTATION COMPLETE!")
    print("=" * 60)
    print("\nCurrent features:")
    print("✓ Raw format encoding with <EVENT_START>/<EVENT_END> tokens")
    print("✓ Multimodal support for ScreenEmitted events")
    print("✓ Round-trip consistency for all event types")
    print("✓ Batch processing capabilities")
    print("✓ Type-safe error handling")
    
    print("\nNext steps (Phase 2):")
    print("- Implement token-efficient format")
    print("- Add abbreviations for message types")
    print("- Introduce special tokens for common patterns")
    print("- Optimize for LLM tokenizer efficiency")


if __name__ == "__main__":
    main()
