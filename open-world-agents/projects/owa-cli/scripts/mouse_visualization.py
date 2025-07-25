from pathlib import Path

import cv2
import numpy as np
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.core.io.video import VideoWriter
from owa.core.time import TimeUnits

# TODO: this script run too slow, must figure out why


def main(
    mcap_path: Annotated[Path, typer.Argument(help="Path to the input .mcap file")],
    output_path: Annotated[Path, typer.Option(help="Path to the output video file")] = Path("mouse_visualization.mp4"),
    fps: Annotated[float, typer.Option(help="Output video frame rate")] = 30.0,
    start_time: Annotated[float, typer.Option(help="Start time in seconds from beginning")] = 0.0,
    end_time: Annotated[float, typer.Option(help="End time in seconds from beginning (0 = no limit)")] = 0.0,
    start_frame: Annotated[int, typer.Option(help="Start frame number (overrides start_time if > 0)")] = 0,
    end_frame: Annotated[int, typer.Option(help="End frame number (overrides end_time if > 0)")] = 0,
    max_frames: Annotated[int, typer.Option(help="Maximum number of frames to process (0 = no limit)")] = 0,
):
    with OWAMcapReader(mcap_path) as reader:
        # Find the first screen message to get the recording start time
        recording_start_time = None
        for mcap_msg in reader.iter_messages(topics=["screen"]):
            recording_start_time = mcap_msg.timestamp
            break
        else:
            typer.echo("No screen messages found in the .mcap file.")
            raise typer.Exit()

        # Determine start and end conditions
        if start_frame > 0:
            # Frame-based start: skip frames to reach start_frame
            actual_start_time = recording_start_time
            frames_to_skip = start_frame - 1  # 1-indexed
            typer.echo(f"Starting from frame {start_frame}")
        else:
            # Time-based start
            actual_start_time = recording_start_time + int(TimeUnits.SECOND * start_time)
            frames_to_skip = 0
            typer.echo(f"Starting from time {start_time} seconds")

        # Determine end condition
        if end_frame > 0:
            # Frame-based end
            target_frame_count = end_frame - max(start_frame - 1, 0)
            typer.echo(f"Ending at frame {end_frame}")
        elif end_time > 0.0:
            # Time-based end - calculate approximate frame count
            duration = end_time - start_time
            target_frame_count = int(duration * fps) if duration > 0 else 0
            typer.echo(f"Ending at time {end_time} seconds")
        elif max_frames > 0:
            # Max frames limit
            target_frame_count = max_frames
            typer.echo(f"Processing maximum {max_frames} frames")
        else:
            # No limit
            target_frame_count = float("inf")
            typer.echo("Processing all available frames")

        x, y = 0, 0
        frame_count = 0
        skipped_frames = 0

        # Create video writer
        with VideoWriter(output_path, fps=fps, vfr=False) as writer:
            typer.echo(f"Creating video: {output_path}")

            # Setup progress bar
            total_for_progress = target_frame_count if target_frame_count != float("inf") else None
            with tqdm(total=total_for_progress, desc="Processing frames", unit="frame") as pbar:
                for mcap_msg in reader.iter_messages(start_time=actual_start_time):
                    if mcap_msg.topic == "mouse":
                        x, y = mcap_msg.decoded.x, mcap_msg.decoded.y
                    elif mcap_msg.topic == "screen":
                        # Skip frames if needed for frame-based start
                        if skipped_frames < frames_to_skip:
                            skipped_frames += 1
                            continue

                        # Check time-based end condition
                        if end_time > 0.0:
                            current_time_seconds = (mcap_msg.timestamp - recording_start_time) / TimeUnits.SECOND
                            if current_time_seconds >= end_time:
                                break

                        msg = mcap_msg.decoded.resolve_relative_path(mcap_path)
                        image = msg.to_pil_image()

                        # Convert PIL image to numpy array (RGB format)
                        frame = np.array(image)

                        # Draw mouse cursor
                        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # Red circle for mouse
                        cv2.putText(
                            frame, f"Frame {frame_count + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                        )

                        # Write frame to video
                        writer.write_frame(frame)
                        frame_count += 1
                        pbar.update(1)

                        # Check if we've reached the target frame count
                        if frame_count >= target_frame_count:
                            break

        typer.echo(f"Video created successfully: {output_path}")
        typer.echo(f"Total frames: {frame_count}")
        typer.echo(f"Duration: {frame_count / fps:.2f} seconds")


if __name__ == "__main__":
    typer.run(main)
