# OWAMcap file format

## Definition of OWAMcap file format

OWAMcap format is composed of two files, `.mkv` and `.mcap`:

- `example.mcap`
    - It's standard [`mcap`](https://mcap.dev/) file with `json` schema. [`mcap-owa-support`](https://github.com/open-world-agents/open-world-agents/tree/main/projects/mcap-owa-support) python package contains decoder, writer and reader for this format.
    - All messages must inherit/implement [`BaseMessage`](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-core/owa/core/message.py#L7) class of `owa.core.message`.
    - Events such as keyboard, mouse, and window interactions are logged in an `.mcap` file with the same name.

!!! mcap "What's MCAP?"
    MCAP (pronounced "em-cap") is an open-source container file format designed for multimodal log data. It supports multiple channels of timestamped pre-serialized data and is ideal for use in pub/sub or robotics applications.

    **[Learn more about MCAP](https://mcap.dev/)**

- `example.mkv`:
    - Contains muxed video/audio/subtitle track.
    - Video/audio track is encoded with video/audio encoder. Default codec is `hevc`, `h265`.
    - Subtitle contains timestamps, which is nanoseconds since the [epoch](https://docs.python.org/3/library/time.html#epoch). These timestamps can be used to align events in the `.mcap` file with frames in the `.mkv`.

### Reader/Writer of OWAMcap file format

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