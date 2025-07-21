# LitServe Video Decoding Implementation

This directory contains an alternative LitServe-based implementation of the video decoding server.

## Overview

While this implementation offers excellent developer experience and rapid prototyping capabilities, it has significantly lower performance compared to the main Triton Inference Server implementation.

## Performance Comparison

For detailed performance benchmarks and comparison with the main Triton implementation, see the [Performance Benchmarks section](../../README.md#performance-benchmarks) in the main README.

## How to Run

1. Install dependencies:
   ```bash
   # Basic dependencies
   pip install litserve opencv-python-headless av

   # For TorchCodec backend, follow official installation guide:
   # https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec
   ```

2. Start the server:
   ```bash
   # PyAV backend (default)
   python decoding_server.py

   # OpenCV backend
   python decoding_server_cv2.py

   # TorchCodec backend
   python decoding_server_torchcodec.py
   ```

3. Test with client:
   ```bash
   python client.py video.mp4 10.5
   ```

## Usage

The LitServe implementation is recommended for:
- Rapid prototyping and experimentation
- Development environments where ease of setup is prioritized
- Scenarios where ~100ms additional latency is acceptable

For production deployments requiring high throughput, use the main Triton Inference Server implementation.