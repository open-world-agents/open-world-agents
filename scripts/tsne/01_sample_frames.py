#!/usr/bin/env python3
"""
Script to sample frames uniformly from .mkv files in the OWA game dataset.
For each game, calculates the total duration of all videos and extracts 1000 frames
uniformly distributed across the entire timeline.
"""

import os
import re
from collections import defaultdict
import cv2

def get_video_info(video_path):
    """Get video information using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video {video_path}")
            return 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()
        return duration, fps, frame_count
    except Exception as e:
        print(f"Error getting video info for {video_path}: {e}")
        return 0, 0, 0

def extract_game_name(filename):
    """Extract game name from filename pattern: recording_YYYYMMDD_HHMMSS_GameName_hash.mkv"""
    match = re.match(r'^recording_(?P<date>\d{8})_(?P<time>\d{6})_(?P<name>.+)_(?P<id>[0-9A-Fa-f]{8})(?:_(?P<suffix>[^_]+))?\.mkv', filename)
    if match:
        game_name = match.group(3).replace('_', ' ')
        return game_name
    return None

def find_mkv_files(dataset_path):
    """Find all .mkv files with recording pattern and group by game."""
    games = defaultdict(list)

    # Special case directories
    special_dirs = {
        'apex_legends': '/mnt/raid12/datasets/owa_game_dataset/milkclouds00@gmail.com/apex_legends/',
        'battlefield6': '/mnt/raid12/datasets/owa_game_dataset/jaeyoonskr@gmail.com/battlefield6/',
        'minecraft_vpt': '/mnt/raid12/datasets/owa/mcaps/vpt/'
    }

    # Handle special cases first
    for game_name, special_path in special_dirs.items():
        print(f"Checking special directory: {game_name} at {special_path}")
        if os.path.exists(special_path):
            print(f"  Directory exists, listing files...")
            files = os.listdir(special_path)
            print(f"  Found {len(files)} files")

            if game_name == 'apex_legends':
                # Handle apex legends files: apex_*.mkv, *.mkv (but not recording_*)
                for file in files:
                    if file.endswith('.mkv') and not file.startswith('recording_'):
                        full_path = os.path.join(special_path, file)
                        games['Apex Legends'].append(full_path)

            elif game_name == 'battlefield6':
                # Handle battlefield6 files: prefer _fixed.mkv versions
                fixed_files = [f for f in files if f.endswith('_fixed.mkv')]
                if fixed_files:
                    for file in fixed_files:
                        full_path = os.path.join(special_path, file)
                        games['Battlefield 6'].append(full_path)
                else:
                    # Fallback to non-fixed versions if no fixed versions exist
                    for file in files:
                        if file.endswith('.mkv') and '_fixed' not in file:
                            full_path = os.path.join(special_path, file)
                            games['Battlefield 6'].append(full_path)

            elif game_name == 'minecraft_vpt':
                # Handle minecraft VPT files: various naming patterns
                # Only use the most recent 100 files since it's a huge directory
                mkv_files = [f for f in files if f.endswith('.mkv')]
                if mkv_files:
                    # Sort by modification time (most recent first)
                    mkv_files_with_time = []
                    for file in mkv_files:
                        full_path = os.path.join(special_path, file)
                        try:
                            mtime = os.path.getmtime(full_path)
                            mkv_files_with_time.append((file, mtime, full_path))
                        except OSError:
                            continue

                    # Sort by modification time (newest first) and take first 100
                    mkv_files_with_time.sort(key=lambda x: x[1], reverse=True)
                    recent_files = mkv_files_with_time[:100]

                    print(f"Found {len(mkv_files)} .mkv files in VPT directory, using {len(recent_files)} most recent files")

                    for file, mtime, full_path in recent_files:
                        games['Minecraft VPT'].append(full_path)

    # Handle regular dataset path for standard recording files
    for root, _, files in os.walk(dataset_path):
        # Skip special directories we already handled
        if any(special_path in root for special_path in special_dirs.values()):
            continue

        for file in files:
            if file.startswith('recording_') and file.endswith('.mkv') and '_invalid' not in file:
                game_name = extract_game_name(file)
                if game_name and game_name.strip():  # Skip empty game names
                    full_path = os.path.join(root, file)
                    games[game_name].append(full_path)

    return games

def get_total_game_duration(video_paths):
    """Calculate total duration and collect video info for all videos in a game."""
    video_info_list = []
    total_duration = 0.0

    for video_path in video_paths:
        duration, fps, frame_count = get_video_info(video_path)
        if duration > 0:
            video_info = {
                'path': video_path,
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'start_time': total_duration,
                'end_time': total_duration + duration
            }
            video_info_list.append(video_info)
            total_duration += duration
        else:
            print(f"Warning: Could not get duration for {video_path}, skipping")

    return video_info_list, total_duration

def extract_frame_at_timestamp(video_path, timestamp_seconds, output_path):
    """Extract a single frame at the specified timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return False

    # Set position to timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    else:
        print(f"Could not read frame at {timestamp_seconds}s from {video_path}")
        cap.release()
        return False

def sample_frames_uniformly_from_game(video_info_list, total_duration, output_dir, game_name, num_frames=1000):
    """Sample frames uniformly across the total duration of all videos for a game."""
    if total_duration <= 0:
        print(f"No valid duration found for game {game_name}")
        return []

    print(f"Total duration for {game_name}: {total_duration:.2f}s across {len(video_info_list)} videos")

    # Create output directory for this game
    game_output_dir = os.path.join(output_dir, game_name.replace(' ', '_').replace('-', '_'))
    os.makedirs(game_output_dir, exist_ok=True)

    # Calculate uniform timestamps across total duration
    if num_frames == 1:
        timestamps = [total_duration / 2]  # Middle timestamp for single frame
    else:
        # Uniform distribution from start to end
        timestamps = [i * total_duration / (num_frames - 1) for i in range(num_frames)]

    extracted_frames = []

    for frame_idx, timestamp in enumerate(timestamps):
        # Find which video this timestamp belongs to
        target_video = None
        for video_info in video_info_list:
            if video_info['start_time'] <= timestamp <= video_info['end_time']:
                target_video = video_info
                break

        if target_video is None:
            print(f"Could not find video for timestamp {timestamp:.2f}s")
            continue

        # Calculate relative timestamp within the target video
        relative_timestamp = timestamp - target_video['start_time']

        # Create filename
        video_basename = os.path.splitext(os.path.basename(target_video['path']))[0]
        frame_filename = f"{game_name.replace(' ', '_')}_{frame_idx:04d}_{video_basename}_t{relative_timestamp:.3f}.jpg"
        frame_path = os.path.join(game_output_dir, frame_filename)

        # Extract frame
        if extract_frame_at_timestamp(target_video['path'], relative_timestamp, frame_path):
            extracted_frames.append(frame_path)

            if (frame_idx + 1) % 100 == 0:
                print(f"Extracted {frame_idx + 1}/{num_frames} frames")
        else:
            print(f"Failed to extract frame {frame_idx} at timestamp {timestamp:.2f}s")

    print(f"Successfully extracted {len(extracted_frames)}/{num_frames} frames for {game_name}")
    return extracted_frames

def main():
    dataset_path = "/mnt/raid12/datasets/owa_game_dataset"
    output_dir = "./sampled_frames"
    frames_per_game = 1000

    print("Finding .mkv files...")
    print("Checking standard dataset path and special case directories...")
    print(f"Dataset path: {dataset_path}")
    games = find_mkv_files(dataset_path)
    print("Finished finding files.")

    print(f"Found {len(games)} games:")
    for game_name, videos in games.items():
        print(f"  {game_name}: {len(videos)} videos")

    # Show total video durations for each game (limit to recent 100 files for performance)
    print("\nCalculating total durations...")
    for game_name, video_paths in games.items():
        if game_name.strip():
            # Limit to most recent 100 files for performance
            limited_paths = video_paths
            if len(video_paths) > 100:
                print(f"  {game_name}: Limiting to 100 most recent files out of {len(video_paths)} total")
                # Sort by modification time (most recent first)
                paths_with_time = []
                for path in video_paths:
                    try:
                        mtime = os.path.getmtime(path)
                        paths_with_time.append((path, mtime))
                    except OSError:
                        continue
                paths_with_time.sort(key=lambda x: x[1], reverse=True)
                limited_paths = [path for path, _ in paths_with_time[:100]]

            video_info_list, total_duration = get_total_game_duration(limited_paths)
            files_used = len(limited_paths)
            print(f"  {game_name}: {total_duration:.2f}s total ({total_duration/60:.1f} minutes) from {files_used} files")

    # Process each game
    for game_name, video_paths in games.items():
        print(f"\n=== Processing game: {game_name} ===")

        # Skip empty game names
        if not game_name.strip():
            print(f"Skipping empty game name")
            continue

        try:
            # Limit to most recent 100 files for performance
            limited_paths = video_paths
            if len(video_paths) > 100:
                print(f"Limiting to 100 most recent files out of {len(video_paths)} total for processing")
                # Sort by modification time (most recent first)
                paths_with_time = []
                for path in video_paths:
                    try:
                        mtime = os.path.getmtime(path)
                        paths_with_time.append((path, mtime))
                    except OSError:
                        continue
                paths_with_time.sort(key=lambda x: x[1], reverse=True)
                limited_paths = [path for path, _ in paths_with_time[:100]]

            # Get total duration and video info for all videos in this game
            video_info_list, total_duration = get_total_game_duration(limited_paths)

            if not video_info_list:
                print(f"No valid videos found for {game_name}")
                continue

            # Extract frames uniformly across total duration
            extracted_frames = sample_frames_uniformly_from_game(
                video_info_list, total_duration, output_dir, game_name, frames_per_game
            )

            print(f"Total frames extracted for {game_name}: {len(extracted_frames)}")

        except Exception as e:
            print(f"Error processing game {game_name}: {e}")
            continue

if __name__ == "__main__":
    main()
