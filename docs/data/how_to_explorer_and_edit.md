# Exploring OWAMcap Data

Learn how to view, analyze, and work with OWAMcap recordings using various tools and methods.

## 📁 Sample Dataset

Download and explore our example dataset:

- `example.mcap` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mcap) - 22 KiB metadata file
- `example.mkv` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mkv) - Video recording

??? demo "Preview: example.mkv"
    <video controls>
    <source src="../example.mkv" type="video/mp4">
    </video>

## 🔍 Exploration Methods

Choose the method that best fits your needs:

### 🌐 Web Visualizer (Recommended for Beginners)

**[OWA Dataset Visualizer](https://huggingface.co/spaces/open-world-agents/visualize_dataset)** - Interactive web-based viewer

<div align="center">
  <img src="../viewer.png" alt="OWA Dataset Visualizer"/>
</div>

- **Quick Start**: Upload files or enter HuggingFace dataset ID
- **File Limit**: 100MB for public hosting
- **Self-Hosting**: Available for larger files → [Setup Guide](viewer.md)

### 💻 Command Line Interface

**`owl` CLI** - Powerful command-line analysis tools

#### Getting a Summary

View a summary of the MCAP file:

```bash
$ owl mcap info example.mcap
library:   mcap-owa-support 0.5.1; mcap 1.3.0
profile:   owa
messages:  864
duration:  10.3574349s
start:     2025-06-27T18:49:52.129876+09:00 (1751017792.129876000)   
end:       2025-06-27T18:50:02.4873109+09:00 (1751017802.487310900)  
compression:
        zstd: [1/1 chunks] [116.46 KiB/16.61 KiB (85.74%)] [1.60 KiB/sec] 
channels:
        (1) window           11 msgs (1.06 Hz)    : desktop/WindowInfo [jsonschema]      
        (2) keyboard/state   11 msgs (1.06 Hz)    : desktop/KeyboardState [jsonschema]   
        (3) mouse/state      11 msgs (1.06 Hz)    : desktop/MouseState [jsonschema]      
        (4) screen          590 msgs (56.96 Hz)   : desktop/ScreenCaptured [jsonschema]
        (5) mouse           209 msgs (20.18 Hz)   : desktop/MouseEvent [jsonschema]
        (6) keyboard         32 msgs (3.09 Hz)    : desktop/KeyboardEvent [jsonschema]
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
Topic: screen, Timestamp: 1741628814057575300, Message: {'utc_ns': 1741628814056571100, 'shape': [1080, 1920], 'media_ref': {'uri': 'example.mkv', 'pts_ns': 14866666666}}
Topic: screen, Timestamp: 1741628814073392700, Message: {'utc_ns': 1741628814072476900, 'shape': [1080, 1920], 'media_ref': {'uri': 'example.mkv', 'pts_ns': 14883333333}}
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
                writer.write_message(event, topic=topic, timestamp=publish_time)

        # Reading messages from an OWAMcap file
        with OWAMcapReader(file_path, decode_args={"return_dict": True}) as reader:
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