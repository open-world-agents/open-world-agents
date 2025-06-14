#!/usr/bin/env python3
"""
Demonstration of OEP-0006: Dedicated OWA Message Package and OWAMcap Profile Specification

This script demonstrates the key features of OEP-0006:
1. Message registry system with automatic discovery
2. Domain-based message naming
3. Entry point registration
4. Backward compatibility
5. Integration with OWAMcap format
"""

import tempfile
from pathlib import Path

# Import the global message registry
from owa.core import MESSAGES

# Also demonstrate backward compatibility with direct imports
from owa.msgs.desktop.keyboard import KeyboardEvent as DirectKeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent as DirectMouseEvent


def demonstrate_message_registry():
    """Demonstrate the message registry system."""
    print("=== OEP-0006 Message Registry Demonstration ===\n")

    # 1. Automatic message discovery
    print("1. Automatic Message Discovery:")
    print(f"   Available message types: {len(MESSAGES)}")
    for message_type in sorted(MESSAGES.keys()):
        print(f"   - {message_type}")
    print()

    # 2. Registry access patterns
    print("2. Registry Access Patterns:")

    # Access via registry
    KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
    MouseEvent = MESSAGES["desktop/MouseEvent"]

    print(f"   Registry access: {KeyboardEvent}")
    print(f"   Direct import:   {DirectKeyboardEvent}")
    print(f"   Same class?      {KeyboardEvent is DirectKeyboardEvent}")
    print()

    # 3. Message creation and validation
    print("3. Message Creation and Validation:")

    # Create keyboard event
    kb_event = KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)
    print(f"   Keyboard event: {kb_event}")
    print(f"   Event type: {kb_event.event_type}, VK: {kb_event.vk}")

    # Create mouse event
    mouse_event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True, timestamp=1234567890)
    print(f"   Mouse event: {mouse_event}")
    print(f"   Position: ({mouse_event.x}, {mouse_event.y}), Button: {mouse_event.button}")
    print()

    # 4. Schema access
    print("4. Schema Access:")
    kb_schema = KeyboardEvent.get_schema()
    print(f"   KeyboardEvent schema properties: {list(kb_schema['properties'].keys())}")

    mouse_schema = MouseEvent.get_schema()
    print(f"   MouseEvent schema properties: {list(mouse_schema['properties'].keys())}")
    print()


def demonstrate_mcap_integration():
    """Demonstrate integration with OWAMcap format."""
    print("5. OWAMcap Integration:")

    try:
        from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            mcap_path = tmp_file.name

        # Write messages using registry
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        MouseEvent = MESSAGES["desktop/MouseEvent"]

        print(f"   Writing to: {mcap_path}")

        with OWAMcapWriter(mcap_path) as writer:
            # Write keyboard events
            for i in range(3):
                event = KeyboardEvent(
                    event_type="press" if i % 2 == 0 else "release", vk=65 + i, timestamp=1234567890 + i
                )
                writer.write_message("keyboard", event, publish_time=i)

            # Write mouse events
            for i in range(3):
                event = MouseEvent(
                    event_type="click",
                    x=100 + i * 10,
                    y=200 + i * 10,
                    button="left",
                    pressed=True,
                    timestamp=1234567890 + i,
                )
                writer.write_message("mouse", event, publish_time=i)

        # Read messages back
        print("   Reading messages back:")
        with OWAMcapReader(mcap_path) as reader:
            for msg in reader.iter_messages():
                # Use the correct attribute name for schema
                schema_name = getattr(msg, "schema_name", getattr(msg, "schema", {}).get("name", "unknown"))
                print(f"     Topic: {msg.topic}, Schema: {schema_name}")
                print(f"     Message: {msg.decoded}")

        # Clean up
        Path(mcap_path).unlink()
        print("   ✓ OWAMcap integration successful")

    except ImportError:
        print("   ⚠ mcap-owa-support not available, skipping OWAMcap demo")

    print()


def demonstrate_extensibility():
    """Demonstrate how to extend the system with custom messages."""
    print("6. Extensibility Example:")
    print("   To add custom messages, create a package with entry points:")
    print()
    print("   # pyproject.toml")
    print('   [project.entry-points."owa.messages"]')
    print('   "custom/MyMessage" = "my_package.messages:MyMessage"')
    print('   "custom/AnotherMessage" = "my_package.messages:AnotherMessage"')
    print()
    print("   Then access via: MESSAGES['custom/MyMessage']")
    print()


def main():
    """Run the complete demonstration."""
    demonstrate_message_registry()
    demonstrate_mcap_integration()
    demonstrate_extensibility()

    print("=== OEP-0006 Implementation Complete ===")
    print()
    print("Key Benefits:")
    print("• Centralized message definitions in owa-msgs package")
    print("• Automatic discovery via entry points")
    print("• Domain-based naming (desktop/KeyboardEvent)")
    print("• Backward compatibility with direct imports")
    print("• Seamless OWAMcap integration")
    print("• Easy extensibility for custom messages")


if __name__ == "__main__":
    main()
