# Introducing OWAMcap

## Overview

OWAMcap is a high-performance, self-contained, flexible container file format for multimodal desktop log data, powered by the open-source container file format [mcap](https://mcap.dev/). This format is designed for efficiently recording and processing message data in Open World Agents (OWA) applications.

!!! tip "So, what exactly is mcap?"
    Simply put, mcap is a format that allows you to record various types of events such as keyboard events, mouse events, and screen captures along with their timestamps. For more detailed information, please refer to the [OWAMcap Format Specification](#internals-owamcap-format-specification) section.

## Usage Example of OWAMcap - Desktop Recorder

What exactly does the OWAMcap format contain? Let's demonstrate with an example of recorded desktop data. Below are sample datasets that you can download and explore yourself:

- `example.mcap` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mcap)
- `example.mkv` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/data/example.mkv)

??? demo "Click here to see `example.mkv`!"
    <video controls>
    <source src="../example.mkv" type="video/mp4">
    </video>
### Exploring Example Data

Let's examine the contents of an OWAMcap file using the `owl` command-line tool (Open World agents cLi).

#### File Summary with `owl mcap info`

First, we can get an overview of the file structure:

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

**Key observations from this output:**

1. **File Overview**:
    - Contains 518 messages recorded over 6.86 seconds
    - Records from March 21, 2025, with precise start and end timestamps

2. **Compression**:
    - Uses zstd compression, reducing file size by 80.44%

3. **Channels (Topics)**:
    - The file contains 6 different channels (or topics), each tracking a specific type of event:

   | Channel # | Name | Message Count | Frequency | Message Type |
   |-----------|------|--------------|-----------|--------------|
   | 1 | window | 7 msgs | 1.02 Hz | WindowInfo |
   | 2 | keyboard/state | 7 msgs | 1.02 Hz | KeyboardState |
   | 3 | mouse/state | 7 msgs | 1.02 Hz | MouseState |
   | 4 | mouse | 115 msgs | 16.77 Hz | MouseEvent |
   | 5 | screen | 362 msgs | 52.80 Hz | ScreenCaptured |
   | 6 | keyboard | 20 msgs | 2.92 Hz | KeyboardEvent |

For example, looking at channel #5 (screen), we can see:

- The topic name is "screen"
- It contains 362 messages
- Recording frequency is 52.80 Hz (slightly lower than the intended 60 Hz, likely due to the short recording time)
- Messages are of type `desktop/ScreenCaptured`

#### Detailed Message Inspection with `owl mcap cat`

To examine individual messages, we can use the `cat` command:

```
$ owl mcap cat example.mcap --n 8 --no-pretty
Topic: window, Timestamp: 1741628814049712700, Message: {'title': 'ZType – Typing Game - Type to Shoot - Chromium', 'rect': [389, 10, 955, 1022], 'hWnd': 7540094}
Topic: keyboard/state, Timestamp: 1741628814049712700, Message: {'buttons': []}
Topic: mouse/state, Timestamp: 1742544390703436600, Message: {'x': 1594, 'y': 1112, 'buttons': []}
Topic: mouse, Timestamp: 1742544390707441200, Message: {'event_type': 'move', 'x': 1597, 'y': 1112}
Topic: screen, Timestamp: 1741628814057575300, Message: {'path': 'example.mkv', 'pts': 14866666666, 'utc_ns': 1741628814056571100}
Topic: screen, Timestamp: 1741628814073392700, Message: {'path': 'example.mkv', 'pts': 14883333333, 'utc_ns': 1741628814072476900}
Topic: keyboard, Timestamp: 1741628815015522100, Message: {'event_type': 'release', 'vk': 162}
```

**What we can learn from these messages:**

1. **Window messages** - Track active windows
    - Example: `{'title': 'ZType – Typing Game - Type to Shoot - Chromium', 'rect': [389, 10, 955, 1022], 'hWnd': 7540094}`
    - Shows which window was active, its title, position and size

2. **Mouse messages** - Track cursor position and button states
    - Position tracking: `{'x': 1597, 'y': 1112}`
    - Event types include: "move", "click", etc.

3. **Keyboard messages** - Track key presses and releases
    - Example: `{'event_type': 'release', 'vk': 162}`
    - Records which virtual key was pressed or released

4. **Screen messages** - Link to video frames in the MKV file
    - Contains paths, presentation timestamps, and UTC timestamps

#### Using This Data

This structured data allows for powerful analysis and use cases:

- You can filter data based on which window was active at a particular time
- You can synchronize keyboard/mouse events with screen captures
- The timestamps allow for precise reconstruction of user interactions

!!! tip "What's VK(Virtual Key Code)?"
    Operating systems don't directly use the physical keyboard input values (scan codes) but instead use virtualized keys called VKs. OWA's recorder uses VKs to record keyboard-agnostic data. If you're interested in more details, you can refer to the following resources:

    - [Keyboard Input Overview, Microsoft](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-keyboard-input)
    - [Virtual-Key Codes, Microsoft](https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes)


## Internals - OWAMcap Format Specification

!!! note "Note for Users"
    This part is intended for developers who want to utilize the OWAMcap file format for their own applications. **Regular users of the library may not need this information.**

### Technical Specifications

- OWAMcap uses the standard `mcap` format with `json` schema
- The [`mcap-owa-support`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/mcap-owa-support) Python package, which is within the open-world-agents repository, provides decoders, writers, and readers for this format
- All messages must inherit from or implement the [`BaseMessage`](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-core/owa/core/message.py#L7) class from `owa.core.message`

!!! mcap "What's MCAP?"
    MCAP (pronounced "em-cap") is an open-source container file format designed for multimodal log data. It supports multiple channels of timestamped pre-serialized data and is ideal for pub/sub or robotics applications.
    
    Key advantages of MCAP:
    
    - **High Performance**: Efficient storage and retrieval of large event data streams
    - **Flexible & Open**: Works with diverse data types beyond robotics
    - **Self-Describing**: Encodes schema information to ensure compatibility
    
    **[Learn more about MCAP](https://mcap.dev/)**

### Implementation Guide

Any message that implements [`BaseMessage`](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-core/owa/core/message.py#L7) can be recorded in the OWAMcap format. This provides flexibility while maintaining a consistent interface. Following block describes the interface of `BaseMessage`.

```python
class BaseMessage(ABC):
    _type: str

    @abstractmethod
    def serialize(self, buffer: io.BytesIO): ...

    @classmethod
    @abstractmethod
    def deserialize(cls, buffer: io.BytesIO) -> Self: ...

    @classmethod
    @abstractmethod
    def get_schema(cls): ...
```

### File Format Considerations

#### Why Use `.mcap`?

There are very few open-source formats available for heterogeneous timestamped data. ROS's bagfile format is one option, but it heavily depends on the ROS ecosystem and often requires installation of ROS1/2. In comparison, `mcap` is self-contained and efficient, especially for random read (or seeking) operations, which is critical for training VLA (Vision-Language-Action) models.