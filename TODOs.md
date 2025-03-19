[] since mkv path is hard-coded into mcap file, naive file renaming does not work(path to video is being invalid). Implement renaming command & add to docs.
[] accurate mouse state acquisition method. currently, if user does not move mouse after recording, there's no way to extract mouse state.
[] prepare a method to match mouse position and capturing screen. currently, gstreamer captures a window and there's no way to figure out the accurate position of captured screen with regard to whole screen.
[] figure out mkv size issue. currently, if `ffmpeg -i ztype.mkv -filter:v "fps=60" -c:v libx265 -x265-params "keyint=30:no-scenecut=1:bframes=0" -c:a copy ztype_converted.mkv` is run, the size of video shrink from 340MB to 26MB.
