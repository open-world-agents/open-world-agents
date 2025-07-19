# Video Decoding Server

High-performance video frame extraction service using NVIDIA Triton Inference Server.

## Quick Start

1. Start the server:
   ```bash
   ./launch_server.sh /path/to/video/data
   ```

2. Extract a frame:
   ```bash
   python client.py video.mp4 10.5 -o frame.jpg
   ```

3. Benchmark performance (see [Usage](#usage) for options):
   ```bash
   # Using perf_analyzer (recommended)
   docker run -it --net=host -v .:/workspace nvcr.io/nvidia/tritonserver:25.06-py3-sdk \
       perf_analyzer -m video_decoder --percentile=95 --input-data test_input.json --concurrency-range 1:8

   # Or using custom benchmark script
   python benchmark.py --video-list video1.mp4 video2.mp4
   ```

- [perf_analyzer Guide](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md)

## Options

### Client:
```bash
python client.py VIDEO TIME [OPTIONS]
  VIDEO                            Video file path
  TIME                             Time in seconds
  --server-url URL                 Triton server URL (default: 127.0.0.1:8000)
  --output, -o PATH                Output image path (optional)
```

### Benchmark:
```bash
python benchmark.py [OPTIONS]

  --video-list PATH [PATH ...]     Video files to benchmark (required)
  --server-url URL                 Triton server URL (default: 127.0.0.1:8000)
  --concurrency INT [INT ...]      Concurrency levels (default: [1, 4, 16, 64])
  --duration-seconds FLOAT         Benchmark duration (default: 5.0)
  --use-threading                  Use threading instead of multiprocessing
  --max-processes INT              Max processes (default: CPU count)
```

## Features

- **High throughput**: >20 Gbps with multiprocessing
- **Multiple backends**: cv2, pyav, torchcodec
- **Comprehensive metrics**: P95/P99 latencies, throughput, bitrate
- **Resource management**: Configurable process limits

## References

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [Triton Client Libraries](https://github.com/triton-inference-server/client)
- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
