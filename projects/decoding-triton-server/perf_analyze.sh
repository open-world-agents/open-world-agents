docker run -it --net=host -v .:/workspace nvcr.io/nvidia/tritonserver:25.06-py3-sdk \
    perf_analyzer -m video_frame_extractor --percentile=95 --collect-metrics --input-data test_input.json --concurrency-range 1:8:7

docker run -it --net=host -v .:/workspace nvcr.io/nvidia/tritonserver:25.06-py3-sdk \
    perf_analyzer -m video_frame_extractor --percentile=95 --collect-metrics --input-data test_input.json --concurrency-range 32