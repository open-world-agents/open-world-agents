# Exploring & Editing OWAMCap

- `example.mcap` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mcap)
- `example.mkv` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mkv)

<video controls>
<source src="../example.mkv" type="video/mp4">
</video>

## How to explore the dataset

### 1. By `owl`

Utilizing `owl`(Open World agents cLi), you can inspect the summary of `mcap` file.
```
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
        (1) window            7 msgs (1.02 Hz)    : owa.env.desktop.msg.WindowInfo [jsonschema]
        (2) keyboard/state    7 msgs (1.02 Hz)    : owa.env.desktop.msg.KeyboardState [jsonschema]
        (3) mouse/state       7 msgs (1.02 Hz)    : owa.env.desktop.msg.MouseState [jsonschema]
        (4) mouse           115 msgs (16.77 Hz)   : owa.env.desktop.msg.MouseEvent [jsonschema]
        (5) screen          362 msgs (52.80 Hz)   : owa.env.gst.msg.ScreenEmitted [jsonschema]
        (6) keyboard         20 msgs (2.92 Hz)    : owa.env.desktop.msg.KeyboardEvent [jsonschema]
channels: 6
attachments: 0
metadata: 0
```

Also, you may inspect the detailed messages with by running `owl mcap cat`. (Note that `cat` output is a created example.)
```
$ owl mcap cat example.mcap --n 8 --no-pretty
Topic: window, Timestamp: 1741628814049712700, Message: {'title': 'ZType – Typing Game - Type to Shoot - Chromium', 'rect': [389, 10, 955, 1022], 'hWnd': 7540094}
Topic: keyboard/state, Timestamp: 1741628814049712700, Message: {'pressed_vk_list': []}
Topic: mouse/state, Timestamp: 1742544390703436600, Message: {'x': 1594, 'y': 1112, 'buttons': []}
Topic: mouse, Timestamp: 1742544390707441200, Message: {'event_type': 'move', 'x': 1597, 'y': 1112}
Topic: screen, Timestamp: 1741628814057575300, Message: {'path': 'example.mkv', 'pts': 14866666666, 'utc_ns': 1741628814056571100}
Topic: screen, Timestamp: 1741628814073392700, Message: {'path': 'example.mkv', 'pts': 14883333333, 'utc_ns': 1741628814072476900}
Topic: keyboard, Timestamp: 1741628815015522100, Message: {'event_type': 'release', 'vk': 162}
```

### 2. By `OWAMcapReader`

```python
from mcap_owa.highlevel import OWAMcapReader

def main():
    with Reader("tmp/example.mcap") as reader:
        print(reader.topics)
        print(reader.start_time, reader.end_time)
        for topic, timestamp, msg in reader.iter_decoded_messages():
            print(f"Topic: {topic}, Timestamp: {timestamp}, Message: {msg}")

if __name__ == "__main__":
    main()
```

### 3. By Media Player(e.g. VLC)

1. Convert `mcap` file into `srt` file, which is subtitle format for video.
```sh
# This command converts abcd.mcap into abcd.srt
owl mcap convert abcd.mcap
```

2. Open `.mkv` file with media player, which supports subtitle view. We recommend [VLC media player](https://www.videolan.org/vlc/). 


## How to Edit OWAMCAP Files

Following example script shows how to utilize OWAMcapReader and OWAMcapWriter from `mcap_owa.highlevel`. Utilizing both, you may edit middle of the `.mcap` file.

```python
import tempfile

import pytest
from owa.env.desktop.msg import KeyboardEvent

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter


@pytest.fixture
def temp_mcap_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = tmpdir + "/output.mcap"
        yield file_path


def test_write_and_read_messages(temp_mcap_file):
    file_path = temp_mcap_file
    topic = "/chatter"
    event = KeyboardEvent(event_type="press", vk=1)

    with OWAMcapWriter(file_path) as writer:
        for i in range(0, 10):
            publish_time = i
            writer.write_message(topic, event, log_time=publish_time)

    with OWAMcapReader(file_path) as reader:
        messages = list(reader.iter_decoded_messages())
        assert len(messages) == 10
        for i, (_topic, timestamp, msg) in enumerate(messages):
            assert _topic == topic
            assert msg.event_type == "press"
            assert msg.vk == 1
            assert timestamp == i

```