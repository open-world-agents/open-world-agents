## Data Format

- The main recording is saved as a Matroska (`.mkv`) file. This `.mkv` file includes timestamps (nanoseconds since the [epoch](https://docs.python.org/3/library/time.html#epoch)) as subtitles. These timestamps can be used to align events in the `.mcap` file with frames in the `.mkv`.
- Events such as keyboard, mouse, and window interactions are logged in an `.mcap` file with the same name.

!!! mcap "What's MCAP?"
    MCAP (pronounced "em-cap") is an open-source container file format designed for multimodal log data. It supports multiple channels of timestamped pre-serialized data and is ideal for use in pub/sub or robotics applications.

    **[Learn more about MCAP](https://mcap.dev/)**

### Example Data

- `example.mcap` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/recorder/example.mcap)

```
Topic: window, Timestamp: 1741628814049712700, Message: {'title': 'ZType – Typing Game - Type to Shoot - Chromium', 'rect': [389, 10, 955, 1022], 'hWnd': 7540094}
Topic: keyboard/state, Timestamp: 1741628814049712700, Message: {'pressed_vk_list': []}
Topic: screen, Timestamp: 1741628814057575300, Message: {'path': 'example.mkv', 'pts': 14866666666, 'utc_ns': 1741628814056571100}
Topic: screen, Timestamp: 1741628814073392700, Message: {'path': 'example.mkv', 'pts': 14883333333, 'utc_ns': 1741628814072476900}

... (additional lines omitted for brevity) ...

Topic: screen, Timestamp: 1741628814955961800, Message: {'path': 'example.mkv', 'pts': 15766666666, 'utc_ns': 1741628814955961800}
Topic: keyboard, Timestamp: 1741628814978561600, Message: {'event_type': 'release', 'vk': 67}
Topic: keyboard, Timestamp: 1741628815015522100, Message: {'event_type': 'release', 'vk': 162}
Topic: screen, Timestamp: 1741628815016585400, Message: {'path': 'example.mkv', 'pts': 15783333333, 'utc_ns': 1741628815015522100}
Topic: window, Timestamp: 1741628815050666400, Message: {'title': 'data_format.md - open-world-agents - Visual Studio Code', 'rect': [-8, -8, 1928, 1040], 'hWnd': 133438}
Topic: keyboard/state, Timestamp: 1741628815050666400, Message: {'pressed_vk_list': []}
Topic: screen, Timestamp: 1741628815054648300, Message: {'path': 'example.mkv', 'pts': 15830000000, 'utc_ns': 1741628815054648300}
Topic: screen, Timestamp: 1741628815067474000, Message: {'path': 'example.mkv', 'pts': 15870000000, 'utc_ns': 1741628815067474000}

... (additional lines omitted for brevity) ...

Topic: mouse, Timestamp: 1741628816438561600, Message: {'event_type': 'move', 'x': 950, 'y': 891}
Topic: mouse, Timestamp: 1741628816441655400, Message: {'event_type': 'move', 'x': 950, 'y': 891}
Topic: mouse, Timestamp: 1741628816445662100, Message: {'event_type': 'move', 'x': 949, 'y': 891}
Topic: screen, Timestamp: 1741628816446661600, Message: {'path': 'example.mkv', 'pts': 17250000000, 'utc_ns': 1741628816446661600}
```

- `example.mkv` [[Download]](https://github.com/open-world-agents/open-world-agents/blob/main/docs/recorder/example.mkv)

<video controls>
<source src="../example.mkv" type="video/mp4">
</video>

### Reading OWA MCAP Files

!!! Note "Note" 
    Note that awesome read/writer and dataloader for OWA MCAP file is WIP! Stay tuned.

You can read the `.mcap` file using the following script:

```python
from mcap_owa.highlevel import Reader

def main():
    with Reader("tmp/example.mcap") as reader:
        print(reader.topics)
        print(reader.start_time, reader.end_time)
        for topic, timestamp, msg in reader.iter_decoded_messages():
            print(f"Topic: {topic}, Timestamp: {timestamp}, Message: {msg}")

if __name__ == "__main__":
    main()
```

### 💡 Why Use `.mkv` Instead of `.mp4`?

OWA's Recorder uses **Matroska (`.mkv`)** instead of `.mp4` to ensure reliability in case of crashes or power failures.

- If a recording is unexpectedly interrupted (e.g., power outage, software crash), `.mkv` files remain recoverable.
- `.mp4` files, by contrast, may become corrupted or completely lost if not properly finalized.

For safety and data integrity, `.mkv` is the preferred format—you can always convert it to other formats later if needed.


### 💡 Why Use `.mcap`?

MCAP is a powerful open-source format for logging multimodal data. Unlike traditional log formats, MCAP is optimized for performance, flexibility, and interoperability.

- **High Performance**: Efficient storage and retrieval of large event data streams.  
- **Flexible & Open**: Works with diverse data types beyond robotics.  
- **Self-Describing**: Encodes schema information to ensure compatibility.

To enhance MCAP support for Open World Agents (OWA), we have developed the [`mcap-owa-support`](https://github.com/open-world-agents/open-world-agents/blob/main/projects/mcap-owa-support) package. This package provides custom readers and writers for the **OWA MCAP** format, making it easier to log and process event data seamlessly.