#!/usr/bin/env python3
"""
Extract RGB_FRONT video and actions from bagfiles.

This script processes ROS bagfiles to extract:
1. RGB_FRONT camera images as video (MP4 or individual frames)
2. CMD_VEL actions (linear.x, angular.z) as numpy arrays

Features:
- Configurable frame rate and action rate (can be different)
- Support for both simulation and real robot bagfiles
- Automatic detection of bagfile type (sim vs real)
- Parallel processing with Ray
- Resume capability (skip already processed files)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import av
import cv2
import numpy as np
import pandas as pd
import ray
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros1_noetic import (
    sensor_msgs__msg__CompressedImage,
    sensor_msgs__msg__Image,
    geometry_msgs__msg__Twist,
)
from tqdm import tqdm

# Import joystick conversion
try:
    from cmd_vel_to_joystick import IsaacHoundbotCmdVelToJoystick, create_isaac_houndbot_config
    HAS_JOYSTICK_CONVERSION = True
except ImportError:
    HAS_JOYSTICK_CONVERSION = False

# Initialize typestore for ROS message deserialization
typestore = get_typestore(Stores.ROS1_NOETIC)

# Define topic configurations to avoid import issues with Ray
def get_topics_config():
    """Get topic configurations, handling import issues gracefully."""
    try:
        from sketchdrive_ros_common.topics import SimTopics, RealTopics
        return SimTopics, RealTopics
    except ImportError:
        # Fallback topic definitions
        SimTopics = {
            "CMD_VEL": type('Topic', (), {"topic_name": "/cmd_vel"})(),
            "RGB_FRONT": type('Topic', (), {"topic_name": "/rgb_front/compressed"})(),
        }
        RealTopics = {
            "CMD_VEL": type('Topic', (), {"topic_name": "/cmd_vel"})(),
            "RGB_FRONT": type('Topic', (), {"topic_name": "/usb_cam_front/image_raw/compressed"})(),
        }
        return SimTopics, RealTopics


def process_image(msg: sensor_msgs__msg__CompressedImage | sensor_msgs__msg__Image, 
                 target_size: Tuple[int, int] = (672, 378)) -> Image.Image:
    """
    Process ROS image message to PIL Image.
    
    Args:
        msg: ROS image message (compressed or uncompressed)
        target_size: Target size for resizing (width, height)
    
    Returns:
        PIL Image
    """
    # Convert ROS message to OpenCV image
    cv_image = message_to_cvimage(msg)
    
    # Convert BGR to RGB and create PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    
    # Resize to target size
    pil_image = pil_image.resize(target_size)
    
    return pil_image


def detect_bagfile_type(bagfile: Path) -> str:
    """
    Detect if bagfile is from simulation or real robot by checking topics.

    Args:
        bagfile: Path to bagfile

    Returns:
        "sim" or "real"
    """
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        topics = set(reader.topics.keys())

        # Check for simulation-specific topics
        sim_topics = {"/rgb_front/compressed", "/rgb_left/compressed", "/rgb_right/compressed"}
        real_topics = {"/usb_cam_front/image_raw/compressed", "/usb_cam_left/image_raw/compressed"}

        if sim_topics.intersection(topics):
            return "sim"
        elif real_topics.intersection(topics):
            return "real"
        else:
            # Default to sim if unclear
            print(f"Warning: Could not determine bagfile type for {bagfile}, defaulting to sim")
            return "sim"


def extract_cmd_vel_actions(bagfile: Path, topics_config: Dict) -> List[Tuple[int, float, float]]:
    """
    Extract CMD_VEL actions from bagfile.
    
    Args:
        bagfile: Path to bagfile
        topics_config: Topic configuration (SimTopics or RealTopics)
    
    Returns:
        List of (timestamp_ns, linear_x, angular_z) tuples
    """
    cmd_vel_list = []
    
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        # Find CMD_VEL connections
        cmd_vel_topic = topics_config["CMD_VEL"].topic_name
        connections = [x for x in reader.connections if x.topic == cmd_vel_topic]
        
        if len(connections) == 0:
            print(f"Warning: No CMD_VEL topic found in {bagfile}")
            return []
        
        # Extract messages
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg: geometry_msgs__msg__Twist = reader.deserialize(rawdata, connection.msgtype)
            cmd_vel_list.append((timestamp, msg.linear.x, msg.angular.z))
    
    return cmd_vel_list


def extract_rgb_front_images(bagfile: Path, topics_config: Dict,
                           target_fps: float = 10.0) -> List[Tuple[int, Image.Image]]:
    """
    Extract RGB_FRONT images from bagfile at specified frame rate.

    Args:
        bagfile: Path to bagfile
        topics_config: Topic configuration (SimTopics or RealTopics)
        target_fps: Target frame rate for extraction

    Returns:
        List of (timestamp_ns, PIL_Image) tuples
    """
    image_list = []

    with AnyReader([bagfile], default_typestore=typestore) as reader:
        # Try to find RGB_FRONT topic, with fallback options
        rgb_front_topic = None

        # Priority order for RGB front topics
        # candidate_topics = [
        #     topics_config["RGB_FRONT"].topic_name,  # From config
        #     "/rgb_front/compressed",                 # Common simulation
        #     "/usb_cam_front/image_raw/compressed",   # Common real robot
        #     "/rgb_front/image_raw/compressed",       # Alternative
        # ]

        # Also check what's actually available in the bagfile
        available_topics = list(reader.topics.keys())

        # Find image topics (not camera_info)
        image_topics = [t for t in available_topics if
                       ("rgb" in t.lower() or "image" in t.lower() or "camera" in t.lower()) and
                       "camera_info" not in t.lower() and
                       ("compressed" in t.lower() or "image_raw" in t.lower())]

        # Prefer front camera topics
        front_topics = [t for t in image_topics if "front" in t.lower()]

        if front_topics:
            rgb_front_topic = front_topics[0]
        elif image_topics:
            rgb_front_topic = image_topics[0]  # Use any image topic as fallback
        else:
            # Try the configured topic anyway
            rgb_front_topic = topics_config["RGB_FRONT"].topic_name

        connections = [x for x in reader.connections if x.topic == rgb_front_topic]

        if len(connections) == 0:
            print(f"Warning: No RGB_FRONT topic found in {bagfile}. Available image topics: {image_topics}")
            return []

        print(f"Using RGB topic: {rgb_front_topic}")

        # Calculate time interval for target FPS
        target_interval_ns = int(1e9 / target_fps)
        last_extracted_time = 0

        # Extract messages
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=connections),
            desc=f"Extracting images from {bagfile.name}",
            disable=False
        ):
            # Skip if not enough time has passed for target FPS
            if timestamp - last_extracted_time < target_interval_ns:
                continue
            
            msg = reader.deserialize(rawdata, connection.msgtype)
            image = process_image(msg)
            image_list.append((timestamp, image))
            last_extracted_time = timestamp
    
    return image_list


def save_video_mp4(images: List[Tuple[int, Image.Image]], output_path: Path, fps: float = 10.0):
    """
    Save images as MP4 video.
    
    Args:
        images: List of (timestamp, PIL_Image) tuples
        output_path: Output video file path
        fps: Video frame rate
    """
    if not images:
        print("Warning: No images to save as video")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with av.open(str(output_path), "w") as container:
        # Set up video stream
        stream = container.add_stream("libx264", rate=int(fps))
        example_image = images[0][1]
        stream.width, stream.height = example_image.size
        stream.pix_fmt = "yuv420p"

        # Set GOP size for random access
        stream.codec_context.gop_size = 1
        
        # Encode frames
        for timestamp, image in images:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
    
    print(f"Saved video with {len(images)} frames to {output_path}")


def save_video_frames(images: List[Tuple[int, Image.Image]], output_dir: Path):
    """
    Save images as individual frame files.
    
    Args:
        images: List of (timestamp, PIL_Image) tuples
        output_dir: Output directory for frames
    """
    if not images:
        print("Warning: No images to save as frames")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (timestamp, image) in enumerate(images):
        frame_path = output_dir / f"frame_{i:06d}_{timestamp}.png"
        image.save(frame_path)

    print(f"Saved {len(images)} frames to {output_dir}")


@ray.remote
def process_single_bagfile(bagfile: Path, output_dir: Path, 
                          video_fps: float = 10.0, action_fps: float = 20.0,
                          save_as_video: bool = True, resume: bool = True) -> Dict[str, Any]:
    """
    Process a single bagfile to extract video and actions.
    
    Args:
        bagfile: Path to bagfile
        output_dir: Output directory
        video_fps: Frame rate for video extraction
        action_fps: Frame rate for action extraction
        save_as_video: If True, save as MP4; if False, save as individual frames
        resume: If True, skip already processed files
    
    Returns:
        Dictionary with processing results
    """
    bagfile_name = bagfile.stem
    bagfile_output_dir = output_dir / bagfile_name
    
    # Check if already processed
    if resume:
        video_path = bagfile_output_dir / "video.mp4" if save_as_video else bagfile_output_dir / "frames"
        actions_path = bagfile_output_dir / "actions.npy"
        metadata_path = bagfile_output_dir / "metadata.json"
        
        if (video_path.exists() and actions_path.exists() and metadata_path.exists()):
            print(f"Skipping {bagfile_name} (already processed)")
            return {"bagfile": bagfile_name, "status": "skipped", "reason": "already_processed"}
    
    try:
        # Detect bagfile type and get topics config
        bagfile_type = detect_bagfile_type(bagfile)
        SimTopics, RealTopics = get_topics_config()
        topics_config = SimTopics if bagfile_type == "sim" else RealTopics

        print(f"Processing {bagfile_name} (type: {bagfile_type})")
        
        # Extract images
        images = extract_rgb_front_images(bagfile, topics_config, video_fps)
        if not images:
            return {"bagfile": bagfile_name, "status": "failed", "reason": "no_images"}
        
        # Extract actions
        actions = extract_cmd_vel_actions(bagfile, topics_config)
        if not actions:
            print(f"Warning: No actions found in {bagfile_name}")
        
        # Create output directory
        bagfile_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save video
        if save_as_video:
            video_path = bagfile_output_dir / "video.mp4"
            save_video_mp4(images, video_path, video_fps)
        else:
            frames_dir = bagfile_output_dir / "frames"
            save_video_frames(images, frames_dir)
        
        # Save actions
        if actions:
            # Convert to numpy array: [timestamp_ns, linear_x, angular_z]
            actions_array = np.array(actions)
            actions_path = bagfile_output_dir / "actions.npy"
            np.save(actions_path, actions_array)

            # Also save as CSV for easy inspection
            actions_df = pd.DataFrame(actions_array, columns=["timestamp_ns", "linear_x", "angular_z"])
            actions_df.to_csv(bagfile_output_dir / "actions.csv", index=False)

            # Convert to joystick format if available
            if HAS_JOYSTICK_CONVERSION:
                try:
                    # Create joystick converter with Isaac_Houndbot configuration
                    joystick_config = create_isaac_houndbot_config()
                    converter = IsaacHoundbotCmdVelToJoystick(joystick_config)

                    # Convert actions to joystick format (with duplicate removal)
                    joystick_data = converter.convert_batch(actions, remove_duplicates=True)

                    # Save joystick data
                    joystick_path = bagfile_output_dir / "joystick.json"
                    with open(joystick_path, "w") as f:
                        json.dump(joystick_data, f, indent=2)

                    # Save joystick statistics
                    joystick_stats = converter.get_statistics(joystick_data)
                    stats_path = bagfile_output_dir / "joystick_stats.json"
                    with open(stats_path, "w") as f:
                        json.dump(joystick_stats, f, indent=2)

                    print(f"Converted {len(joystick_data)} actions to joystick format")

                except Exception as e:
                    print(f"Warning: Failed to convert to joystick format: {e}")
            else:
                print("Joystick conversion not available (cmd_vel_to_joystick.py not found)")
        
        # Save metadata
        metadata = {
            "bagfile": bagfile_name,
            "bagfile_type": bagfile_type,
            "video_fps": video_fps,
            "action_fps": action_fps,
            "num_frames": len(images),
            "num_actions": len(actions),
            "save_as_video": save_as_video,
            "has_joystick_conversion": HAS_JOYSTICK_CONVERSION and len(actions) > 0,
            "topics_used": {
                "rgb_front": topics_config["RGB_FRONT"].topic_name,
                "cmd_vel": topics_config["CMD_VEL"].topic_name,
            }
        }
        
        with open(bagfile_output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Successfully processed {bagfile_name}: {len(images)} frames, {len(actions)} actions")
        
        return {
            "bagfile": bagfile_name,
            "status": "success",
            "num_frames": len(images),
            "num_actions": len(actions),
            "bagfile_type": bagfile_type
        }
        
    except Exception as e:
        print(f"Error processing {bagfile_name}: {str(e)}")
        return {"bagfile": bagfile_name, "status": "failed", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Extract RGB_FRONT video and actions from bagfiles")
    parser.add_argument("bagfiles_dir", type=Path, help="Directory containing bagfiles")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--video-fps", type=float, default=10.0, help="Video frame rate (default: 10.0)")
    parser.add_argument("--action-fps", type=float, default=20.0, help="Action sampling rate (default: 20.0)")
    parser.add_argument("--save-frames", action="store_true", help="Save as individual frames instead of MP4")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already processed files")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--pattern", type=str, default="*.bag", help="Bagfile pattern (default: *.bag)")

    args = parser.parse_args()

    # Find bagfiles
    bagfiles = list(args.bagfiles_dir.glob(args.pattern))
    if not bagfiles:
        print(f"Error: No bagfiles found in {args.bagfiles_dir} with pattern {args.pattern}")
        return

    print(f"Found {len(bagfiles)} bagfiles to process")

    # Initialize Ray
    ray.init(num_cpus=args.max_workers)

    try:
        # Create processing tasks
        tasks = []
        for bagfile in bagfiles:
            task = process_single_bagfile.remote(
                bagfile=bagfile,
                output_dir=args.output_dir,
                video_fps=args.video_fps,
                action_fps=args.action_fps,
                save_as_video=not args.save_frames,
                resume=not args.no_resume
            )
            tasks.append(task)

        # Process with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing bagfiles") as pbar:
            while tasks:
                finished, tasks = ray.wait(tasks, num_returns=1, timeout=1.0)
                for task in finished:
                    result = ray.get(task)
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix_str(f"Last: {result['bagfile']} ({result['status']})")

        # Summary
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        skipped = [r for r in results if r["status"] == "skipped"]

        print("Processing complete:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Skipped: {len(skipped)}")

        if failed:
            print("Failed bagfiles:")
            for result in failed:
                print(f"  {result['bagfile']}: {result.get('reason', 'unknown')}")

        # Save summary
        summary_path = args.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "total_bagfiles": len(bagfiles),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(skipped),
                "results": results,
                "config": {
                    "video_fps": args.video_fps,
                    "action_fps": args.action_fps,
                    "save_as_video": not args.save_frames,
                    "resume": not args.no_resume,
                }
            }, f, indent=2)

        print(f"Summary saved to {summary_path}")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
