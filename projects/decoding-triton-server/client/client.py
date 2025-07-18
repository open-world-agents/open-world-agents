import argparse
import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image


def extract_frame(client: httpclient.InferenceServerClient, video_path, time_sec):
    # Prepare inputs
    inputs = [httpclient.InferInput("video_path", [1], "BYTES"), httpclient.InferInput("time_sec", [1], "FP32")]

    # Set input data
    inputs[0].set_data_from_numpy(np.array([video_path.encode()], dtype=np.object_))
    inputs[1].set_data_from_numpy(np.array([time_sec], dtype=np.float32))

    # Request output
    outputs = [httpclient.InferRequestedOutput("frame")]

    # Run inference
    start_time = time.time()
    response = client.infer("video_frame_extractor", inputs=inputs, outputs=outputs)
    inference_time = time.time() - start_time

    # Get results
    frame_array = response.as_numpy("frame")
    frame = Image.fromarray(frame_array)

    print(f"Inference time: {inference_time:.4f} seconds")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--time", required=True, type=float, help="Time in seconds")
    parser.add_argument("--output", default="frame.jpg", help="Output image path")
    args = parser.parse_args()

    # Create Triton client
    client = httpclient.InferenceServerClient(url=args.url)

    # Extract frame
    frame = extract_frame(client, args.video, args.time)

    # Save frame
    frame.save(args.output)
    print(f"Frame saved to {args.output}")


if __name__ == "__main__":
    main()
