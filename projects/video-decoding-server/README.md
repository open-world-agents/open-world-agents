# Video Decoding Server

High-performance video frame extraction service using NVIDIA Triton Inference Server with PyAV backend.

**Performance (Multi-Instance count=16, cpus=16, PyAV backend):**

| Concurrency | Throughput | Bitrate | P99 Latency |
|-------------|------------|---------|-------------|
| 1 | 17.6 r/s | 155.7 Mbps | 84.1 ms |
| 4 | 80.6 r/s | 713.1 Mbps | 80.7 ms |
| 16 | 421.6 r/s | 3730.0 Mbps | 69.2 ms |
| 64 | 617.6 r/s | 5464.1 Mbps | 135.8 ms |

> **Note**: Bitrate measures the total size of raw frame data (in megabits per second) that the server outputs. It's calculated from the numpy array bytes of extracted frames (e.g., 1920x1080x3 RGB = ~6MB per frame). Higher bitrate means the server can process and return more frame data per second.

**Theoretical Performance Ceiling**: Network loopback (iperf3) typically achieves ~60 Gbps, indicating substantial headroom beyond current 5.5 Gbps performance. The bottleneck is video decoding computation, not network or memory bandwidth.

## Quick Start

1. Start the server:
   ```bash
   ./launch_server.sh /path/to/video/data
   ```

2. Extract a frame:
   ```bash
   python client.py video.mp4 10.5 -o frame.jpg
   ```

3. Benchmark performance (see [Options](#options) for options):
   ```bash
   # Using perf_analyzer (recommended)
   docker run -it --net=host -v .:/workspace nvcr.io/nvidia/tritonserver:25.06-py3-sdk \
       perf_analyzer -m video_decoder --percentile=95 --input-data test_input.json --concurrency-range 1:8

   # Or using custom benchmark script
   python benchmark.py --video-list video1.mp4 video2.mp4
   ```

- [perf_analyzer Guide](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md)

> **Note**: For optimal performance, match CPU cores with instance group count (1:1 ratio). Set `cpus: '16.0'` in docker-compose.yml and `count: 16` in config.pbtxt. Additional CPUs beyond instance count don't improve throughput.

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

## Video Decoding Backends

This server supports three different video decoding backends, each optimized for different use cases:

### 1. OpenCV (cv2)
- **Best for**: General-purpose video processing, simple deployments
- **Performance**: Good CPU performance with broad codec support
- **Dependencies**: `opencv-python-headless>=4.11.0`
- **Usage**: Set `VIDEO_BACKEND=cv2` environment variable

### 2. PyAV
- **Best for**: Production environments requiring stability and memory efficiency
- **Performance**: Excellent memory management with automatic garbage collection
- **Dependencies**: `av>=14.2.0`, `ffmpeg`
- **Usage**: Set `VIDEO_BACKEND=pyav` environment variable (default)
- **Features**:
  - Automatic memory cleanup to prevent thread explosion
  - Robust seeking and frame extraction
  - Wide codec support through FFmpeg

### 3. TorchCodec
- **Best for**: GPU-accelerated decoding, high-performance scenarios
- **Performance**: Hardware-accelerated decoding with CUDA support
- **Dependencies**:
  - `torch>=2.7.1`, `torchcodec>=0.4.0`
  - **Required system package**: `sudo apt install libnvidia-decode-535-server` (or appropriate version for your NVIDIA driver)
- **Usage**: Set `VIDEO_BACKEND=torchcodec` environment variable
- **Features**:
  - GPU-accelerated frame extraction
  - Optimized for NVIDIA hardware
  - Direct tensor output for ML pipelines
- **Project**: [TorchCodec on GitHub](https://github.com/pytorch/torchcodec)

> **Note**: For TorchCodec backend, ensure you have the appropriate NVIDIA decode libraries installed. The package name may vary based on your NVIDIA driver version (e.g., `libnvidia-decode-470-server`, `libnvidia-decode-535-server`).

## Performance Benchmarks

The server achieves high-performance video decoding with the following benchmark results:

### Triton Inference Server Performance

**Single Instance (count=1)**
```
Concurrency | Requests | Throughput | Bitrate     | P95 Latency | P99 Latency
------------------------------------------------------------------------
          1 |      151 |    30.2 r/s |  267.2 Mbps |     51.9 ms |     59.2 ms
          4 |      262 |    52.4 r/s |  463.6 Mbps |    116.3 ms |    130.4 ms
         16 |      297 |    59.4 r/s |  525.5 Mbps |    399.3 ms |    462.0 ms
         64 |      266 |    53.2 r/s |  470.7 Mbps |   1451.7 ms |   1516.4 ms
```

**Multi-Instance (count=16)**
```
Concurrency | Requests | Throughput | Bitrate     | P95 Latency | P99 Latency
------------------------------------------------------------------------
          1 |       88 |    17.6 r/s |  155.7 Mbps |     81.9 ms |     84.1 ms
          4 |      403 |    80.6 r/s |  713.1 Mbps |     69.0 ms |     80.7 ms
         16 |     2108 |   421.6 r/s | 3730.0 Mbps |     57.7 ms |     69.2 ms
         64 |     3088 |   617.6 r/s | 5464.1 Mbps |    125.9 ms |    135.8 ms
```

### Backend Performance Comparison

**Single Worker Performance (instance_group count=1)**

| Backend | Concurrency=1 | Throughput | Bitrate | P95 Latency | P99 Latency |
|---------|---------------|------------|---------|-------------|-------------|
| PyAV | 121 requests | 24.2 r/s | 214.1 Mbps | 59.5 ms | 77.5 ms |
| OpenCV (cv2) | 60 requests | 12.0 r/s | 106.2 Mbps | 91.1 ms | 160.0 ms |
| TorchCodec (CPU, threads=1) | 41 requests | 8.2 r/s | 72.5 Mbps | 190.5 ms | 205.5 ms |
| TorchCodec (CPU, threads=0) | 26 requests | 5.2 r/s | 46.0 Mbps | 293.7 ms | 298.9 ms |
| TorchCodec (CUDA) | 24 requests | 4.8 r/s | 42.5 Mbps | 333.1 ms | 366.6 ms |

**Detailed Backend Analysis:**

- **PyAV**: Best overall performance with optimal CPU utilization and memory management
- **OpenCV**: 2x slower than PyAV but scales well with sufficient parallelization
- **TorchCodec**:
  - CPU performance significantly lower than PyAV/OpenCV
  - GPU acceleration shows minimal benefit without batch processing
  - Current implementation doesn't leverage GPU efficiently for single-frame extraction
  - May benefit from batch processing optimization in future implementations

**Key Findings:**
- PyAV + Triton delivers 5.5+ Gbps throughput with optimal performance for video decoding
- GPU acceleration (TorchCodec CUDA) doesn't provide benefits for single-frame extraction workloads
- OpenCV offers good compatibility but at reduced performance compared to PyAV
- Scalability: Achieves 617+ requests/second with proper parallelization (20 CPU cores)

### Alternative: LitServe Implementation

A LitServe-based implementation is available in [`scripts/litserve/`](scripts/litserve/) for rapid prototyping:
- **Pros**: Easy setup (`pip install litserve`), single-file implementation, built-in dynamic batching
- **Cons**: 5-6x lower throughput compared to Triton Inference Server
- **Use case**: Suitable for experimentation and scenarios where ~100ms latency is acceptable

## References

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [Triton Client Libraries](https://github.com/triton-inference-server/client)
- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
