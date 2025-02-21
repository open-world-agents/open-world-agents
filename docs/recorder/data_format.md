## Data Format


- The main recording will be saved as a Matroska (`.mkv`) file. This `.mkv` file contains timestamp, nanoseconds since the [epoch](https://docs.python.org/3/library/time.html#epoch), as subtitle. This timestamp is needed to align timestamp between events in `.jsonl` file and frames in `.mkv`. 
- Events such as keyboard, mouse, and window events will be logged in an `.jsonl` file with same name.


### Example Data

- `example.jsonl`
```
{"timestamp_ns":1740119250371185500,"event_src":"window_publisher","event_data":"{\"title\":\"index.md - open-world-agents - Visual Studio Code\",\"rect\":[1720,0,3440,1392],\"hWnd\":133820}"}
{"timestamp_ns":1740119251372073500,"event_src":"window_publisher","event_data":"{\"title\":\"index.md - open-world-agents - Visual Studio Code\",\"rect\":[1720,0,3440,1392],\"hWnd\":133820}"}
{"timestamp_ns":1740119252372885600,"event_src":"window_publisher","event_data":"{\"title\":\"index.md - open-world-agents - Visual Studio Code\",\"rect\":[1720,0,3440,1392],\"hWnd\":133820}"}
{"timestamp_ns":1740119253373626300,"event_src":"window_publisher","event_data":"{\"title\":\"index.md - open-world-agents - Visual Studio Code\",\"rect\":[1720,0,3440,1392],\"hWnd\":133820}"}
{"timestamp_ns":1740119254374614300,"event_src":"window_publisher","event_data":"{\"title\":\"index.md - open-world-agents - Visual Studio Code\",\"rect\":[1720,0,3440,1392],\"hWnd\":133820}"}
{"timestamp_ns":1740119255179855200,"event_src":"control_publisher","event_data":"[\"keyboard.press\",162]"}
{"timestamp_ns":1740119255231630800,"event_src":"control_publisher","event_data":"[\"keyboard.press\",67]"}
{"timestamp_ns":1740119329381736500,"event_src":"window_publisher","event_data":"{\"title\":\"recorder.py (Working Tree) (recorder.py) - open-world-agents - Visual Studio Code\",\"rect\":[1720,0,3440,1392],\"hWnd\":133820}"}
{"timestamp_ns":1740119329446280600,"event_src":"control_publisher","event_data":"[\"mouse.move\",2592,1309]"}
{"timestamp_ns":1740119329447390300,"event_src":"control_publisher","event_data":"[\"mouse.move\",2591,1309]"}
{"timestamp_ns":1740119329454390800,"event_src":"control_publisher","event_data":"[\"mouse.move\",2587,1309]"}
{"timestamp_ns":1740119329461473500,"event_src":"control_publisher","event_data":"[\"mouse.move\",2583,1309]"}
{"timestamp_ns":1740119329469521800,"event_src":"control_publisher","event_data":"[\"mouse.move\",2580,1309]"}
{"timestamp_ns":1740119329476521000,"event_src":"control_publisher","event_data":"[\"mouse.move\",2577,1310]"}
{"timestamp_ns":1740119329483605100,"event_src":"control_publisher","event_data":"[\"mouse.move\",2575,1310]"}
{"timestamp_ns":1740119329491665600,"event_src":"control_publisher","event_data":"[\"mouse.move\",2575,1310]"}
{"timestamp_ns":1740119329619434700,"event_src":"control_publisher","event_data":"[\"mouse.scroll\",2575,1310,0,1]"}
```
- `example.mkv`

