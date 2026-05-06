# Video Compression Benchmark

This directory contains benchmarking tools for comparing different video compression methods.

## test_media_compression.py

A comprehensive benchmark that compares video compression ratios and encoding times across different formats:

- **Raw frames**: Uncompressed numpy arrays (baseline)
- **JPEG compression**: Multiple quality levels (50, 75, 85, 95)
- **PNG compression**: Lossless compression
- **H.265 video**: Multiple CRF values (18, 23, 28)

### Usage

```bash
# Basic usage with default settings (100 frames)
python scripts/benchmark/test_media_compression.py tmp/example.mkv

# Process fewer frames for faster testing
python scripts/benchmark/test_media_compression.py tmp/example.mkv --max-frames 50

# Process more frames for more accurate results
python scripts/benchmark/test_media_compression.py tmp/example.mkv --max-frames 200

# Custom JSON output location
python scripts/benchmark/test_media_compression.py tmp/example.mkv --output-json results/my_results.json
```

### Example Output

```
COMPRESSION BENCHMARK RESULTS
================================================================================
Format               Size         Ratio    Time (s)   Notes
--------------------------------------------------------------------------------
FULL VIDEO COMPARISON
Raw (estimated)      1.0 GB       1.0x     -          (780 frames)
Original Video       11.3 MB      91.7x    -          (H.265 compressed)
--------------------------------------------------------------------------------
SAMPLE COMPRESSION (50 frames)
--------------------------------------------------------------------------------
h265_crf18           96.8 KB      700.1x   0.18
h265_crf23           96.8 KB      700.1x   0.19
h265_crf28           96.8 KB      700.1x   0.18
jpeg_q50             1.6 MB       41.0x    0.05
jpeg_q75             2.3 MB       28.6x    0.05
jpeg_q85             3.0 MB       21.7x    0.05
jpeg_q95             5.4 MB       12.2x    0.06
png                  22.0 MB      3.0x     0.40
raw_sample           66.2 MB      1.0x     0.00
================================================================================

SUMMARY:
Best compression ratio: h265_crf18 (700.1x)
Fastest encoding: raw_sample (0.00s)

Space savings compared to raw sample (50 frames):
  jpeg_q50: 97.6% smaller
  jpeg_q75: 96.5% smaller
  jpeg_q85: 95.4% smaller
  jpeg_q95: 91.8% smaller
  png: 66.7% smaller
  h265_crf18: 99.9% smaller
  h265_crf23: 99.9% smaller
  h265_crf28: 99.9% smaller

Full video: Original H.265 is 98.9% smaller than estimated raw
```

### Key Insights

- **Raw video is enormous**: ~1GB for 13 seconds of 584x792@60fps video
- **H.265 video compression** provides the best compression ratios (90-700x compression)
- **JPEG compression** offers good balance between size and quality (12-41x compression)
- **PNG compression** is lossless but less efficient for video frames (~3x compression)
- **Original H.265 video** is already highly compressed (98.9% smaller than raw)

### Dependencies

- `owa-core` package for video I/O
- `opencv-python` for image compression
- `numpy` for array operations
- `tqdm` for progress bars

### Output

The tool automatically saves detailed results to `tmp/compression_results.json` (or custom path with `--output-json`) containing:
- Exact file sizes in bytes and human-readable format
- Compression ratios relative to raw baseline
- Processing times for each method
- Frame counts and additional metadata

### Notes

- H.265 encoding requires FFmpeg with libx265 support
- CRF values: lower = higher quality/larger size, higher = lower quality/smaller size
- JPEG quality: higher = better quality/larger size
- Processing time includes only compression, not I/O overhead
- Results show both sample compression (processed frames) and full video estimates
