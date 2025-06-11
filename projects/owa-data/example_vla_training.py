#!/usr/bin/env python3
"""
Example: Correct VLA Training Format

This script demonstrates the CORRECTED Vision-Language-Action training format where:
- Input: Screen observation + high-level instruction
- Output: Encoded action events (what the model should predict)
"""

from owa.data.vlm_dataset_builder import VLMDatasetBuilder


def demonstrate_correct_vla_format():
    """Demonstrate the correct VLA training format."""
    print("=" * 60)
    print("CORRECT VLA TRAINING FORMAT")
    print("=" * 60)

    # Sample event sequence: User typing "A"
    event_sequence = [
        {
            "file_path": "/data/session.mcap",
            "topic": "keyboard",
            "timestamp_ns": 1000000000,
            "message_type": "owa.env.desktop.msg.KeyboardEvent",
            "msg": b'{"event_type":"press","vk":65}',  # Press 'A' key
        },
        {
            "file_path": "/data/session.mcap",
            "topic": "keyboard",
            "timestamp_ns": 1100000000,
            "message_type": "owa.env.desktop.msg.KeyboardEvent",
            "msg": b'{"event_type":"release","vk":65}',  # Release 'A' key
        },
    ]

    # Create VLA dataset builder
    builder = VLMDatasetBuilder(drop_file_path=True)

    # Process the sequence
    result = builder.process_event_sequence(
        raw_events=event_sequence,
        instruction="Please type 'A'.",  # High-level instruction
    )

    print("VLA Training Sample:")
    print(f"  Instruction (INPUT): {result['instruction']}")
    print(f"  Images (INPUT): {len(result['images'])} screen captures")
    print(f"  Encoded Events (OUTPUT TARGET): {len(result['encoded_events'])} events")
    print()

    print("What the model learns:")
    print("  INPUT:  [screen_image] + 'Please type 'A'.'")
    print("  OUTPUT: Encoded keyboard events for typing 'A'")
    print()

    print("Encoded events (what model should predict):")
    for i, event in enumerate(result["encoded_events"], 1):
        print(f"  {i}. {event}")

    return result


def demonstrate_conversation_format():
    """Show how this translates to conversation format for training."""
    print("\n" + "=" * 60)
    print("CONVERSATION FORMAT FOR VLA TRAINING")
    print("=" * 60)

    # Get the VLA sample
    builder = VLMDatasetBuilder(drop_file_path=True)

    event_sequence = [
        {
            "file_path": "/data/session.mcap",
            "topic": "mouse",
            "timestamp_ns": 2000000000,
            "message_type": "owa.env.desktop.msg.MouseEvent",
            "msg": b'{"event_type":"click","x":100,"y":200,"button":"left","pressed":true}',
        }
    ]

    result = builder.process_event_sequence(
        raw_events=event_sequence, instruction="Please click at position (100, 200)."
    )

    print("Training conversation format:")
    print()
    print("User (INPUT):")
    print(f"  Role: user")
    print(f"  Content: <image_tokens>{result['instruction']}")
    print()
    print("Assistant (TARGET OUTPUT):")
    print(f"  Role: assistant")
    print(f"  Content: {result['encoded_events'][0]}")
    print()

    print("This teaches the model:")
    print("  Given: Screen + 'Please click at position (100, 200)'")
    print("  Predict: <EVENT_START>{'topic': 'mouse', 'event_type': 'click', ...}<EVENT_END>")


def compare_wrong_vs_correct():
    """Compare the wrong approach vs correct VLA approach."""
    print("\n" + "=" * 60)
    print("WRONG vs CORRECT VLA TRAINING")
    print("=" * 60)

    print("❌ WRONG APPROACH (what we had before):")
    print("  User: [image] + 'What actions were performed?' + encoded_events")
    print("  Assistant: 'User clicked the button'  # Natural language description")
    print("  Problem: Model learns to DESCRIBE actions, not PERFORM them")
    print()

    print("✅ CORRECT VLA APPROACH (what we have now):")
    print("  User: [image] + 'Please click the button'  # High-level goal")
    print("  Assistant: <EVENT_START>{'topic': 'mouse', 'event_type': 'click', ...}<EVENT_END>")
    print("  Benefit: Model learns to GENERATE actions from goals")
    print()

    print("Key difference:")
    print("  Wrong: Events as INPUT → Description as OUTPUT")
    print("  Correct: Goal as INPUT → Events as OUTPUT")


def demonstrate_real_world_usage():
    """Show how this would work in real VLA deployment."""
    print("\n" + "=" * 60)
    print("REAL-WORLD VLA DEPLOYMENT")
    print("=" * 60)

    print("After training, the VLA model can:")
    print()
    print("1. Take a screenshot of current screen")
    print("2. Receive high-level instruction: 'Open the file menu'")
    print("3. Generate action sequence:")
    print("   - <EVENT_START>{'topic': 'keyboard', 'vk': 18}<EVENT_END>  # Alt key")
    print("   - <EVENT_START>{'topic': 'keyboard', 'vk': 70}<EVENT_END>  # F key")
    print("4. Execute the generated events to perform the action")
    print()

    print("This enables:")
    print("  ✅ Autonomous task execution from natural language")
    print("  ✅ Screen-aware action planning")
    print("  ✅ Low-level motor control from high-level goals")
    print("  ✅ Generalization to new interfaces and tasks")


def main():
    """Run all demonstrations."""
    print("VLA Training Format Correction")
    print("Understanding the proper Input/Output relationship")

    # Show correct format
    demonstrate_correct_vla_format()

    # Show conversation format
    demonstrate_conversation_format()

    # Compare approaches
    compare_wrong_vs_correct()

    # Show real-world usage
    demonstrate_real_world_usage()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Fixed VLA training format: Goal → Actions")
    print("✅ Model learns to GENERATE actions, not describe them")
    print("✅ Proper input/output relationship for autonomous agents")
    print("✅ Ready for real-world VLA deployment!")


if __name__ == "__main__":
    main()
