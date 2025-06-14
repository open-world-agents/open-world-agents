#!/usr/bin/env python3
"""
Example demonstrating schema access in OWAMcapReader.

This example shows how to use the new iter_messages_with_schema() and
iter_decoded_messages_with_schema() methods to access schema.name information
while iterating through MCAP messages.
"""

import tempfile

from owa.core.message import OWAMessage
from owa.msgs.desktop.keyboard import KeyboardEvent

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter


class CustomMessage(OWAMessage):
    """Example custom message type."""

    _type = "example/CustomMessage"
    text: str
    value: int


def create_sample_mcap():
    """Create a sample MCAP file with different message types."""
    with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
        file_path = tmp_file.name

    with OWAMcapWriter(file_path) as writer:
        # Write different types of messages
        keyboard_msg = KeyboardEvent(event_type="press", vk=65)  # 'A' key
        writer.write_message("/keyboard", keyboard_msg, log_time=1000)

        custom_msg = CustomMessage(text="Hello", value=42)
        writer.write_message("/custom", custom_msg, log_time=2000)

        keyboard_msg2 = KeyboardEvent(event_type="release", vk=65)
        writer.write_message("/keyboard", keyboard_msg2, log_time=3000)

    return file_path


def demonstrate_schema_access():
    """Demonstrate the new schema access functionality."""
    file_path = create_sample_mcap()

    print("=== Schema Access Example ===\n")

    print("\n2. New unified approach with rich message objects:")

    # Method 2: New unified approach - single method, rich objects
    with OWAMcapReader(file_path) as reader:
        for msg in reader.iter_messages():
            print(f"   Topic: {msg.topic}, Time: {msg.timestamp}")
            print(f"   Schema: {msg.schema_name}")
            print(f"   Message: {msg.decoded}")  # Lazy evaluation
            print(f"   Raw data: {len(msg.data)} bytes")
            print()

    # Method 3: Filtering by schema name
    print("3. Filtering by schema name:")
    with OWAMcapReader(file_path) as reader:
        keyboard_messages = [
            msg for msg in reader.iter_messages() if msg.schema_name == "owa.env.desktop.msg.KeyboardEvent"
        ]

        print(f"   Found {len(keyboard_messages)} keyboard messages:")
        for msg in keyboard_messages:
            print(f"     {msg.decoded['event_type']} key {msg.decoded['vk']} at {msg.timestamp}")

    # Method 4: Schema-based message routing
    print("\n4. Schema-based message routing:")
    with OWAMcapReader(file_path) as reader:
        message_handlers = {
            "owa.env.desktop.msg.KeyboardEvent": handle_keyboard_event,
            "example/CustomMessage": handle_custom_message,
        }

        for msg in reader.iter_messages():
            handler = message_handlers.get(msg.schema_name)
            if handler:
                handler(msg)
            else:
                print(f"     No handler for schema: {msg.schema_name}")

    # Method 5: Lazy evaluation demonstration
    print("\n5. Lazy evaluation demonstration:")
    with OWAMcapReader(file_path) as reader:
        messages = list(reader.iter_messages())
        print(f"   Loaded {len(messages)} message objects")

        # Access only schema info (no decoding yet)
        for msg in messages:
            print(f"     Schema: {msg.schema_name}, Topic: {msg.topic}")

        # Now decode only the first message
        print(f"   Decoding first message: {messages[0].decoded}")
        print("   (Other messages remain undecoded until accessed)")


def handle_keyboard_event(msg):
    """Handler for keyboard events."""
    print(f"     🎹 Keyboard {msg.decoded['event_type']}: key {msg.decoded['vk']} on {msg.topic}")


def handle_custom_message(msg):
    """Handler for custom messages."""
    print(f"     📝 Custom message: '{msg.decoded['text']}' (value: {msg.decoded['value']}) on {msg.topic}")


if __name__ == "__main__":
    demonstrate_schema_access()
