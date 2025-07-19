I've tried litserve to construct video decoding server, but since it's destructively poor performance I've switched to Triton Inference Server. However, I'm keeping the code here for reference.

Compared with NVIDIA Triton Inference Server, litserve has much lower throughput and higher latency.
- For worker=1, clicnet concurrency=1 setting, throughput was 5x lower(24 r/s vs 5 r/s) and latency was 3x higher(77 ms vs 251ms)
- For worker=16, client concurrency=64 setting, throughput was 6x lower(112.4 r/s vs 617.6 r/s) and latency was 9x higher(990.5 ms vs 135.8ms)