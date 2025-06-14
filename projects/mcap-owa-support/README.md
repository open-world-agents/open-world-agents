# Copied from https://github.com/foxglove/mcap/tree/main/python/mcap-ros1-support


### Usage Demo

```python
import tempfile

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core.message import OWAMessage
from owa.core import MESSAGES

# Access message types through the global registry
KeyboardEvent = MESSAGES['desktop/KeyboardEvent']

class String(OWAMessage):
    _type = "std_msgs/String"
    data: str


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = tmpdir + "/output.mcap"
        with OWAMcapWriter(file_path) as writer:
            for i in range(0, 10):
                publish_time = i
                if i % 2 == 0:
                    topic = "/chatter"
                    event = String(data="string message")
                else:
                    topic = "/keyboard"
                    event = KeyboardEvent(event_type="press", vk=1)
                writer.write_message(topic, event, publish_time=publish_time)

        with OWAMcapReader(file_path) as reader:
            for mcap_msg in reader.iter_messages():
                print(f"Topic: {mcap_msg.topic}, Timestamp: {mcap_msg.timestamp}, Message: {mcap_msg.decoded}")


if __name__ == "__main__":
    main()
```

For above script, stdout is following:
```
Topic: /chatter, Timestamp: 1741767097157638598, Message: {'data': 'string message'}
Topic: /keyboard, Timestamp: 1741767097157965764, Message: {'event_type': 'press', 'vk': 1}
Topic: /chatter, Timestamp: 1741767097157997762, Message: {'data': 'string message'}
Topic: /keyboard, Timestamp: 1741767097158019602, Message: {'event_type': 'press', 'vk': 1}
Topic: /chatter, Timestamp: 1741767097158036925, Message: {'data': 'string message'}
Topic: /keyboard, Timestamp: 1741767097158051239, Message: {'event_type': 'press', 'vk': 1}
Topic: /chatter, Timestamp: 1741767097158065463, Message: {'data': 'string message'}
Topic: /keyboard, Timestamp: 1741767097158089318, Message: {'event_type': 'press', 'vk': 1}
Topic: /chatter, Timestamp: 1741767097158113250, Message: {'data': 'string message'}
Topic: /keyboard, Timestamp: 1741767097158129738, Message: {'event_type': 'press', 'vk': 1}
```