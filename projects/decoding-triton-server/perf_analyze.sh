docker run -it --net=host nvcr.io/nvidia/tritonserver:25.02-py3-sdk \
    perf_analyzer -m video_frame_extractor --collect-metrics --input-data test_input.json --concurrency-range 1:8:4


"""
 Successfully read data for 1 stream/streams with 4 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: TRITON
  Using "time_windows" mode for stabilization
  Stabilizing using average latency and throughput
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 8 concurrent requests
  Using synchronous calls for inference

Request concurrency: 1
WARNING: Unable to parse 'nv_gpu_utilization' metric.
WARNING: Unable to parse 'nv_gpu_power_usage' metric.
WARNING: Unable to parse 'nv_gpu_memory_used_bytes' metric.
WARNING: Unable to parse 'nv_gpu_memory_total_bytes' metric.
  Client: 
    Request count: 205
    Throughput: 10.7678 infer/sec
    Avg latency: 87817 usec (standard deviation 26050 usec)
    p50 latency: 99741 usec
    p90 latency: 113710 usec
    p95 latency: 119979 usec
    p99 latency: 132233 usec
    Avg HTTP time: 84728 usec (send/recv 5447 usec + response wait 79281 usec)
  Server: 
    Inference count: 205
    Execution count: 205
    Successful request count: 205
    Avg request latency: 78810 usec (overhead 2 usec + queue 131 usec + compute input 30 usec + compute infer 77226 usec + compute output 1420 usec)

  Server Prometheus Metrics: 
    Avg GPU Utilization:
    Avg GPU Power Usage:
    Max GPU Memory Usage:
    Total GPU Memory:
Request concurrency: 5
  Client: 
    Request count: 231
    Throughput: 12.2263 infer/sec
    Avg latency: 401456 usec (standard deviation 61847 usec)
    p50 latency: 410275 usec
    p90 latency: 472282 usec
    p95 latency: 494437 usec
    p99 latency: 528134 usec
    Avg HTTP time: 398668 usec (send/recv 5152 usec + response wait 393516 usec)
  Server: 
    Inference count: 231
    Execution count: 231
    Successful request count: 231
    Avg request latency: 392907 usec (overhead 3 usec + queue 311157 usec + compute input 24 usec + compute infer 80355 usec + compute output 1367 usec)

  Server Prometheus Metrics: 
    Avg GPU Utilization:
    Avg GPU Power Usage:
    Max GPU Memory Usage:
    Total GPU Memory:
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 10.7678 infer/sec, latency 87817 usec
Concurrency: 5, throughput: 12.2263 infer/sec, latency 401456 usec
"""