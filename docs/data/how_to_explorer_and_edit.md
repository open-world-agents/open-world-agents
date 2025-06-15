# Exploring & Editing OWAMcap

## Sample Datasets

Below are sample datasets you can download and explore:

- `example.mcap` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mcap)
- `example.mkv` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mkv)

??? demo "Click here to see `example.mkv`!"
    <video controls>
    <source src="../example.mkv" type="video/mp4">
    </video>

## How to Explore the Dataset

There are multiple ways to explore OWAMcap files. Here are three methods:

### 1. [OWA Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)

<div align="center">
  <img src="../viewer.png" alt="OWA Dataset Visualizer"/>
</div>

Click `Choose File` at `Upload Files`. Note that uploading file is inappropriate for large file. To visualize large file, self-host dataset visualizer by your own. [Learn more...](viewer.md)

### 2. Using the `owl` Command Line Tool

The `owl` (Open World agents cLi) tool provides a convenient way to inspect MCAP files.

#### Getting a Summary

View a summary of the MCAP file:

```bash
$ owl mcap info example.mcap
library:   mcap-owa-support 0.1.0; mcap 1.2.2
profile:   owa
messages:  518
duration:  6.8558623s
start:     2025-03-21T17:06:30.7029335+09:00 (1742544390.702933500)
end:       2025-03-21T17:06:37.5587958+09:00 (1742544397.558795800)
compression:
        zstd: [1/1 chunks] [48.19 KiB/9.42 KiB (80.44%)] [1.37 KiB/sec]
channels:
        (1) window            7 msgs (1.02 Hz)    : desktop/WindowInfo [jsonschema]
        (2) keyboard/state    7 msgs (1.02 Hz)    : desktop/KeyboardState [jsonschema]
        (3) mouse/state       7 msgs (1.02 Hz)    : desktop/MouseState [jsonschema]
        (4) mouse           115 msgs (16.77 Hz)   : desktop/MouseEvent [jsonschema]
        (5) screen          362 msgs (52.80 Hz)   : desktop/ScreenCaptured [jsonschema]
        (6) keyboard         20 msgs (2.92 Hz)    : desktop/KeyboardEvent [jsonschema]
channels: 6
attachments: 0
metadata: 0
```

#### Examining Message Content

Inspect detailed messages (note that the output below is a created example):

```bash
$ owl mcap cat example.mcap --n 8 --no-pretty
Topic: window, Timestamp: 1741628814049712700, Message: {'title': 'ZType – Typing Game - Type to Shoot - Chromium', 'rect': [389, 10, 955, 1022], 'hWnd': 7540094}
Topic: keyboard/state, Timestamp: 1741628814049712700, Message: {'buttons': []}
Topic: mouse/state, Timestamp: 1742544390703436600, Message: {'x': 1594, 'y': 1112, 'buttons': []}
Topic: mouse, Timestamp: 1742544390707441200, Message: {'event_type': 'move', 'x': 1597, 'y': 1112}
Topic: screen, Timestamp: 1741628814057575300, Message: {'path': 'example.mkv', 'pts': 14866666666, 'utc_ns': 1741628814056571100}
Topic: screen, Timestamp: 1741628814073392700, Message: {'path': 'example.mkv', 'pts': 14883333333, 'utc_ns': 1741628814072476900}
Topic: keyboard, Timestamp: 1741628815015522100, Message: {'event_type': 'release', 'vk': 162}
```

### 3. Using `OWAMcapReader` in Python

You can programmatically access the MCAP data using the Python API:

```python
from mcap_owa.highlevel import OWAMcapReader

def main():
    with OWAMcapReader("tmp/example.mcap") as reader:
        # Print available topics and time range
        print(reader.topics)
        print(reader.start_time, reader.end_time)
        
        # Iterate through all messages
        for mcap_msg in reader.iter_messages():
            print(f"Topic: {mcap_msg.topic}, Timestamp: {mcap_msg.timestamp}, Message: {mcap_msg.decoded}")

if __name__ == "__main__":
    main()
```

### 4. Using a Media Player (e.g., VLC)

For visual exploration of the data:

1. **Convert MCAP to SRT subtitle format**:
   ```bash
   # This command converts abcd.mcap into abcd.srt
   owl mcap convert abcd.mcap
   ```

2. **Open the .mkv file with a media player** that supports subtitles. We recommend [VLC media player](https://www.videolan.org/vlc/). You may also check `example.srt` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.srt)

## How to Edit OWAMcap Files

You can create and modify OWAMcap files using the Python API. The example below demonstrates writing and reading messages:

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
        
        # Writing messages to an OWAMcap file
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

        # Reading messages from an OWAMcap file
        with OWAMcapReader(file_path) as reader:
            for mcap_msg in reader.iter_messages():
                print(f"Topic: {mcap_msg.topic}, Timestamp: {mcap_msg.timestamp}, Message: {mcap_msg.decoded}")


if __name__ == "__main__":
    main()
```

Example output:

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