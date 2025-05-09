from pathlib import Path

import cv2
import numpy as np
import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits
from owa.env.gst.msg import ScreenEmitted


def main(mcap_path: Annotated[Path, typer.Argument(help="Path to the input .mcap file")]):
    with OWAMcapReader(mcap_path) as reader:
        for topic, timestamp, msg in reader.iter_decoded_messages(topics=["screen"]):
            start_time = timestamp
            break
        else:
            typer.echo("No screen messages found in the .mcap file.")
            raise typer.Exit()
        x, y = 0, 0
        for i, (topic, timestamp, msg) in enumerate(
            reader.iter_decoded_messages(start_time=start_time + TimeUnits.SECOND * (3 * 60 + 11)), start=1
        ):
            if topic == "mouse":
                x, y = msg["x"], msg["y"]
            elif topic == "screen":
                msg.path = (mcap_path.parent / msg.path).as_posix()
                msg = ScreenEmitted(**msg)
                image = msg.to_pil_image()
                # convert image to frame
                frame = np.array(image)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 1. Show the frame
                cv2.imshow("Mouse Visualization", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # 2. Save the frame
                if i % 10 == 0:
                    cv2.imwrite(f"scripts/mouse_visualized_{i}.png", frame)

                if i > 240:
                    break


if __name__ == "__main__":
    typer.run(main)
