# Introducing Desktop Recorder

OWA's Recorder is a **powerful, efficient, and easy-to-use** screen recording tool designed for modern workflows. Whether you need precise event tracking, high-performance screen capture, or seamless audio synchronization, it delivers everything in a lightweight yet robust package.  

---

## ✅ Key Features  

- **🔹 Simple & Intuitive** — Start recording with:  
    ```sh
    owl mcap record FILE_LOCATION
    ```  
    Stop with `Ctrl+C`. [Learn more...](install_and_usage.md)  

- **🎥 All-in-One Recording** — Captures **screen, audio, and timestamps** in a single `.mkv` file.  
    - Timestamps are embedded as subtitles.  
    - Logs **keyboard, mouse, and window events** in [mcap format](https://mcap.dev/). For data format, [Learn more...](../data_format.md)  

- **🎯 Flexible Capture Options** — Supports `fps`, `window-name`, `monitor-index`, `show-cursor`, and more. [Learn more...](https://gstreamer.freedesktop.org/documentation/d3d11/d3d11screencapturesrc.html)

- **⚡ Optimized Performance** — Hardware-accelerated pipeline ensures high FPS with low CPU/GPU usage.  
    - Uses Windows APIs (`DXGI/WGC` for screen, `WASAPI` for audio).  

---

## 📊 Feature Comparison  

| **Feature**                           | **OWA's Recorder** | **[wcap](https://github.com/mmozeiko/wcap)** | **[pillow](https://github.com/python-pillow/Pillow)/[mss](https://github.com/BoboTiG/python-mss)** |
|---------------------------------------|--------------------|--------------------------------|----------------------------|
| Timestamp embedding (subtitles)    | ✅ Yes             | ❌ No                          | ❌ No                       |
| Python API support                 | ✅ Yes             | ❌ No                          | ❌ No                       |
| Audio + Window + Keyboard + Mouse  | ✅ Yes             | ❌ No                          | ❌ No                       |
| Supports latest Windows APIs       | ✅ Yes             | ✅ Yes                     | ❌ No (legacy APIs only)    |
| Hardware-accelerated encoder        | ✅ Yes             | ✅ Yes                         | ❌ No                       |
| Optional mouse cursor capture      | ✅ Yes             | ✅ Yes                         | ❌ No                       |


---

## ⚡ Performance Benchmark  

OWA's Recorder significantly **outperforms** other Python screen capture tools:  

| **Library**        | **Avg. Time per Frame** | **Relative Speed**    |
|--------------------|------------------------|-----------------------|
| **OWA Recorder**   | **5.7 ms**              | ⚡ **1× (Fastest)**    |
| `pyscreenshot`    | 33 ms                   | 🚶‍♂️ 5.8× slower       |
| `PIL`             | 34 ms                   | 🚶‍♂️ 6.0× slower       |
| `MSS`             | 37 ms                   | 🚶‍♂️ 6.5× slower       |
| `PyQt5`           | 137 ms                  | 🐢 24× slower         |

📌 **Tested on:** Intel i5-11400, GTX 1650  

Not only does OWA Recorder **achieve higher FPS**, but it also maintains **lower CPU/GPU usage**, making it the ideal choice for screen recording.  

