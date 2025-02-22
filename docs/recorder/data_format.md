## Data Format


- The main recording will be saved as a Matroska (`.mkv`) file. This `.mkv` file contains timestamp, nanoseconds since the [epoch](https://docs.python.org/3/library/time.html#epoch), as subtitle. This timestamp is needed to align timestamp between events in `.jsonl` file and frames in `.mkv`. 
- Events such as keyboard, mouse, and window events will be logged in an `.jsonl` file with same name.


### Example Data

- `example.jsonl`
```
{"timestamp_ns":1740134045272214800,"event_src":"control_publisher","event_data":"[\"mouse.click\",1446,1107,left,true]"}
{"timestamp_ns":1740134045347404600,"event_src":"control_publisher","event_data":"[\"mouse.click\",1446,1107,left,false]"}
{"timestamp_ns":1740134045978417500,"event_src":"window_publisher","event_data":"{\"title\":\"ZType – Typing Game - Type to Shoot - Chromium\",\"rect\":[1211,789,1727,1353],\"hWnd\":265272}"}
{"timestamp_ns":1740134046292540600,"event_src":"control_publisher","event_data":"[\"mouse.move\",1445,1107]"}
{"timestamp_ns":1740134046293541900,"event_src":"control_publisher","event_data":"[\"mouse.move\",1444,1107]"}
{"timestamp_ns":1740134046299730700,"event_src":"control_publisher","event_data":"[\"mouse.move\",1435,1107]"}

# long long mouse moves...

{"timestamp_ns":1740134048033194400,"event_src":"control_publisher","event_data":"[\"mouse.click\",1466,1151,left,true]"}
{"timestamp_ns":1740134048100818900,"event_src":"control_publisher","event_data":"[\"mouse.click\",1466,1151,left,false]"}
{"timestamp_ns":1740134048267817500,"event_src":"control_publisher","event_data":"[\"keyboard.press\",81]"}
{"timestamp_ns":1740134048313781500,"event_src":"control_publisher","event_data":"[\"keyboard.press\",87]"}
{"timestamp_ns":1740134048380686200,"event_src":"control_publisher","event_data":"[\"keyboard.press\",69]"}
{"timestamp_ns":1740134048448819100,"event_src":"control_publisher","event_data":"[\"keyboard.release\",81]"}
{"timestamp_ns":1740134048470371900,"event_src":"control_publisher","event_data":"[\"keyboard.release\",87]"}
{"timestamp_ns":1740134048513648900,"event_src":"control_publisher","event_data":"[\"mouse.move\",1466,1152]"}
{"timestamp_ns":1740134048514651700,"event_src":"control_publisher","event_data":"[\"mouse.move\",1467,1153]"}
{"timestamp_ns":1740134048519005500,"event_src":"control_publisher","event_data":"[\"keyboard.release\",69]"}
```
- `example.mkv`: (width, height) = (502, 557). Note that timestamp is embedded as subtitle.
<video controls>
<source src="../example.mkv" type="video/mp4">
</video>


### How to extract timestamp from video file

With In Progress: reader of OWA data!

```py
import subprocess

video_file = "example.mkv"
subtitle_file = "subtitle.srt"
command = [
    "ffmpeg",
    "-i",
    video_file,
    "-map",
    "0:s:0",  # Change this option to select a different subtitle track, if needed
    subtitle_file,
    "-y",  # Overwrite the output file if it exists
]

subprocess.run(command, check=True)
print("Subtitle extraction complete.")

```

```py
import pysrt

# Open the SRT file
subs = pysrt.open('example.srt', encoding='utf-8')

# Iterate through subtitle entries
for sub in subs:
    print(f"Start: {sub.start}, End: {sub.end}")
    print(f"Text: {sub.text}\n")
```