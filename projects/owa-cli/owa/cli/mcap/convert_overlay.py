import bisect
from pathlib import Path

import cv2
import numpy as np
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.core.io.video import VideoWriter
from owa.core.time import TimeUnits
from owa.env.desktop.constants import VK
from owa.msgs.desktop.mouse import RawMouseEvent

# Format: (row, col, width, label, vk_code, is_arrow)
KEYBOARD_LAYOUT = [
    (0, 0, 1, "ESC", VK.ESCAPE, False),
    (0, 1, 1, "F1", VK.F1, False),
    (0, 2, 1, "F2", VK.F2, False),
    (0, 3, 1, "F3", VK.F3, False),
    (0, 4, 1, "F4", VK.F4, False),
    (0, 5, 1, "F5", VK.F5, False),
    (0, 6, 1, "F6", VK.F6, False),
    (0, 7, 1, "F7", VK.F7, False),
    (0, 8, 1, "F8", VK.F8, False),
    (0, 9, 1, "F9", VK.F9, False),
    (0, 10, 1, "F10", VK.F10, False),
    (0, 11, 1, "F11", VK.F11, False),
    (0, 12, 1, "F12", VK.F12, False),
    (0, 13, 1, "BACK", VK.BACK, False),
    (1, 0, 1, "~", VK.OEM_3, False),
    (1, 1, 1, "1", VK.KEY_1, False),
    (1, 2, 1, "2", VK.KEY_2, False),
    (1, 3, 1, "3", VK.KEY_3, False),
    (1, 4, 1, "4", VK.KEY_4, False),
    (1, 5, 1, "5", VK.KEY_5, False),
    (1, 6, 1, "6", VK.KEY_6, False),
    (1, 7, 1, "7", VK.KEY_7, False),
    (1, 8, 1, "8", VK.KEY_8, False),
    (1, 9, 1, "9", VK.KEY_9, False),
    (1, 10, 1, "0", VK.KEY_0, False),
    (1, 11, 1, "-", VK.OEM_MINUS, False),
    (1, 12, 1, "=", VK.OEM_PLUS, False),
    (1, 13, 1, "\\", VK.OEM_5, False),
    (2, 0, 1, "TAB", VK.TAB, False),
    (2, 1, 1, "Q", VK.KEY_Q, False),
    (2, 2, 1, "W", VK.KEY_W, False),
    (2, 3, 1, "E", VK.KEY_E, False),
    (2, 4, 1, "R", VK.KEY_R, False),
    (2, 5, 1, "T", VK.KEY_T, False),
    (2, 6, 1, "Y", VK.KEY_Y, False),
    (2, 7, 1, "U", VK.KEY_U, False),
    (2, 8, 1, "I", VK.KEY_I, False),
    (2, 9, 1, "O", VK.KEY_O, False),
    (2, 10, 1, "P", VK.KEY_P, False),
    (2, 11, 1, "[", VK.OEM_4, False),
    (2, 12, 1, "]", VK.OEM_6, False),
    (2, 13, 1, "ENTER", VK.RETURN, False),
    (3, 0, 1, "CAPS", VK.CAPITAL, False),
    (3, 1, 1, "A", VK.KEY_A, False),
    (3, 2, 1, "S", VK.KEY_S, False),
    (3, 3, 1, "D", VK.KEY_D, False),
    (3, 4, 1, "F", VK.KEY_F, False),
    (3, 5, 1, "G", VK.KEY_G, False),
    (3, 6, 1, "H", VK.KEY_H, False),
    (3, 7, 1, "J", VK.KEY_J, False),
    (3, 8, 1, "K", VK.KEY_K, False),
    (3, 9, 1, "L", VK.KEY_L, False),
    (3, 10, 1, ";", VK.OEM_1, False),
    (3, 11, 1, "'", VK.OEM_7, False),
    (3, 12, 1, "UP", VK.UP, True),
    (3, 13, 1, "SHIFT", VK.RSHIFT, False),
    (4, 0, 1, "SHIFT", VK.LSHIFT, False),
    (4, 1, 1, "Z", VK.KEY_Z, False),
    (4, 2, 1, "X", VK.KEY_X, False),
    (4, 3, 1, "C", VK.KEY_C, False),
    (4, 4, 1, "V", VK.KEY_V, False),
    (4, 5, 1, "B", VK.KEY_B, False),
    (4, 6, 1, "N", VK.KEY_N, False),
    (4, 7, 1, "M", VK.KEY_M, False),
    (4, 8, 1, ",", VK.OEM_COMMA, False),
    (4, 9, 1, ".", VK.OEM_PERIOD, False),
    (4, 10, 1, "/", VK.OEM_2, False),
    (4, 11, 1, "LEFT", VK.LEFT, True),
    (4, 12, 1, "DOWN", VK.DOWN, True),
    (4, 13, 1, "RIGHT", VK.RIGHT, True),
    (5, 0, 1, "CTRL", VK.LCONTROL, False),
    (5, 1, 1, "WIN", VK.LWIN, False),
    (5, 2, 1, "ALT", VK.LMENU, False),
    (5, 3, 8, "SPACE", VK.SPACE, False),
    (5, 11, 1, "ALT", VK.RMENU, False),
    (5, 12, 1, "WIN", VK.RWIN, False),
    (5, 13, 1, "CTRL", VK.RCONTROL, False),
]

BUTTON_PRESS_FLAGS = {
    RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN: "left",
    RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN: "right",
    RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_DOWN: "middle",
}

BUTTON_RELEASE_FLAGS = {
    RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP: "left",
    RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP: "right",
    RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_UP: "middle",
}

MIN_KEY_DURATION_NS = 500_000_000  # 500ms minimum display duration


class KeyTracker:
    """Track keyboard press/release and output (start, end, vk) tuples."""

    def __init__(self):
        self._pressed: dict[int, int] = {}  # vk -> press_time
        self.events: list[tuple[int, int, int]] = []  # (start, end, vk)

    def handle(self, event_type: str, vk: int, timestamp: int):
        if event_type == "press" and vk not in self._pressed:
            self._pressed[vk] = timestamp
        elif event_type == "release" and vk in self._pressed:
            start = self._pressed.pop(vk)
            end = start + max(timestamp - start, MIN_KEY_DURATION_NS)
            self.events.append((start, end, vk))

    def finalize(self):
        for vk, start in self._pressed.items():
            self.events.append((start, start + MIN_KEY_DURATION_NS, vk))
        self._pressed.clear()


def draw_arrow(frame: np.ndarray, center_x: int, center_y: int, direction: str, color: tuple, size: int = 10) -> None:
    """Draw arrow symbol with anti-aliasing."""
    arrow_points = {
        "UP": [
            [center_x, center_y - size],
            [center_x - size, center_y + size // 2],
            [center_x + size, center_y + size // 2],
        ],
        "DOWN": [
            [center_x, center_y + size],
            [center_x - size, center_y - size // 2],
            [center_x + size, center_y - size // 2],
        ],
        "LEFT": [
            [center_x - size, center_y],
            [center_x + size // 2, center_y - size],
            [center_x + size // 2, center_y + size],
        ],
        "RIGHT": [
            [center_x + size, center_y],
            [center_x - size // 2, center_y - size],
            [center_x - size // 2, center_y + size],
        ],
    }

    if direction not in arrow_points:
        return

    pts = np.array(arrow_points[direction], np.int32)
    cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def _draw_ellipse_quadrant(
    center_x: int, center_y: int, ellipse_a: int, ellipse_b: int, start_angle: float, end_angle: float
) -> list:
    """Generate points for an ellipse quadrant."""
    points = [[center_x, center_y]]
    angle = start_angle
    while angle <= end_angle:
        rad = np.radians(angle)
        px = center_x + int(ellipse_a * np.cos(rad))
        py = center_y + int(ellipse_b * np.sin(rad))
        points.append([px, py])
        angle += 1.0
    points.append([center_x, center_y])
    return points


def draw_mouse_figure(frame: np.ndarray, x: int, y: int, active_buttons: set) -> None:
    """Draw mouse figure with left, middle, and right buttons."""
    mouse_width, mouse_height = 60, 80
    center_x, center_y = x + mouse_width // 2, y + mouse_height // 2
    ellipse_a, ellipse_b = mouse_width // 2, mouse_height // 2

    bg_color = (40, 40, 40)
    border_color = (150, 150, 150)
    inactive_color = (70, 70, 70)
    button_colors = {"left": (255, 0, 0), "right": (0, 0, 255), "middle": (255, 255, 0)}

    # Draw mouse body
    cv2.ellipse(frame, (center_x, center_y), (ellipse_a, ellipse_b), 0, 0, 360, bg_color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (center_x, center_y), (ellipse_a, ellipse_b), 0, 0, 360, border_color, 2, lineType=cv2.LINE_AA)

    # Draw left button (180° to 270°)
    left_color = button_colors["left"] if "left" in active_buttons else inactive_color
    left_pts = np.array(_draw_ellipse_quadrant(center_x, center_y, ellipse_a, ellipse_b, 180.0, 270.0), np.int32)
    cv2.fillPoly(frame, [left_pts], left_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [left_pts], True, border_color, 2, lineType=cv2.LINE_AA)

    # Draw right button (270° to 360°)
    right_color = button_colors["right"] if "right" in active_buttons else inactive_color
    right_pts = np.array(_draw_ellipse_quadrant(center_x, center_y, ellipse_a, ellipse_b, 270.0, 360.0), np.int32)
    cv2.fillPoly(frame, [right_pts], right_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [right_pts], True, border_color, 2, lineType=cv2.LINE_AA)

    # Draw middle button (scroll wheel)
    middle_color = button_colors["middle"] if "middle" in active_buttons else inactive_color
    middle_button_width = int(mouse_width * 0.16)
    middle_height = int(ellipse_b * 0.6)
    middle_x = x + mouse_width // 2 - middle_button_width // 2
    middle_y = y + int(mouse_height * 0.1)
    cv2.rectangle(
        frame,
        (middle_x, middle_y),
        (middle_x + middle_button_width, middle_y + middle_height),
        middle_color,
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(
        frame,
        (middle_x, middle_y),
        (middle_x + middle_button_width, middle_y + middle_height),
        border_color,
        2,
        lineType=cv2.LINE_AA,
    )

    # Add scroll wheel lines
    for i in [1, 2]:
        line_y = middle_y + i * middle_height // 3
        cv2.line(frame, (middle_x + 2, line_y), (middle_x + middle_button_width - 2, line_y), border_color, 1)


def draw_mouse_minimap(
    frame: np.ndarray,
    mouse_x: int,
    mouse_y: int,
    minimap_x: int,
    minimap_y: int,
    minimap_width: int,
    minimap_height: int,
    frame_width: int,
    frame_height: int,
    active_mouse_buttons: set,
) -> None:
    """Draw minimap showing mouse position."""
    # Draw border
    cv2.rectangle(
        frame,
        (minimap_x, minimap_y),
        (minimap_x + minimap_width, minimap_y + minimap_height),
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    # Map mouse position to minimap
    padding = 5
    norm_x, norm_y = mouse_x / frame_width, mouse_y / frame_height
    cursor_x = minimap_x + padding + int(norm_x * (minimap_width - 2 * padding))
    cursor_y = minimap_y + padding + int(norm_y * (minimap_height - 2 * padding))
    cursor_x = max(minimap_x + padding, min(minimap_x + minimap_width - padding, cursor_x))
    cursor_y = max(minimap_y + padding, min(minimap_y + minimap_height - padding, cursor_y))

    # Draw cursor
    cursor_radius = 4
    cv2.circle(frame, (cursor_x, cursor_y), cursor_radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cursor_x, cursor_y), cursor_radius, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # Draw click indicators
    button_colors = {"left": (255, 0, 0), "right": (0, 0, 255), "middle": (255, 255, 0)}
    for button_name in active_mouse_buttons:
        color = button_colors.get(button_name, (255, 255, 0))
        cv2.circle(frame, (cursor_x, cursor_y), cursor_radius + 4, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cursor_x, cursor_y), cursor_radius + 7, color, 1, lineType=cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    active_keys: set,
    active_mouse_buttons: set,
    mouse_x: int,
    mouse_y: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Draw keyboard and mouse overlay on the frame."""
    key_size, key_margin = 20, 2
    start_x, start_y = 10, frame_height + 10

    # Draw keyboard
    for row, col, width, label, vk_code, is_arrow in KEYBOARD_LAYOUT:
        x = start_x + int(col * (key_size + key_margin))
        y = start_y + int(row * (key_size + key_margin))
        w = int(width * (key_size + key_margin) - key_margin)
        h = key_size

        is_pressed = vk_code is not None and vk_code in active_keys
        bg_color = (80, 176, 171) if is_pressed else (107, 107, 107)
        text_color = (255, 255, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)

        if is_arrow:
            draw_arrow(frame, x + w // 2, y + h // 2, label, text_color, size=5)
        else:
            # Dynamic font sizing
            font_scale = {1: 0.35, 2: 0.30}.get(len(label), 0.25 if len(label) <= 4 else 0.20)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            max_width = w - 2
            if text_size[0] > max_width:
                font_scale *= max_width / text_size[0]
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(
                frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
            )

    # Draw mouse figure and minimap
    bg_width = int(14 * (key_size + key_margin))
    mouse_figure_x = start_x + bg_width + 10
    draw_mouse_figure(frame, mouse_figure_x, start_y, active_mouse_buttons)

    minimap_x = mouse_figure_x + 60 + 15  # 60 = mouse_width, 15 = margin
    draw_mouse_minimap(
        frame, mouse_x, mouse_y, minimap_x, start_y, 150, 100, frame_width, frame_height, active_mouse_buttons
    )

    return frame


def convert_overlay(
    input_file: Annotated[Path, typer.Argument(help="Input MCAP file")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output video file (default: <input>.mp4)")
    ] = None,
    fps: Annotated[float | None, typer.Option(help="Output frame rate (default: auto-detect from recording)")] = None,
    duration: Annotated[float | None, typer.Option("--duration", "-d", help="Maximum duration in seconds")] = None,
    source_width: Annotated[int, typer.Option(help="Source screen width for mouse scaling")] = 2560,
    source_height: Annotated[int, typer.Option(help="Source screen height for mouse scaling")] = 1440,
    topics: Annotated[list[str], typer.Option(help="MCAP topics to process")] = ["mouse/raw", "keyboard"],
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing output file")] = False,
):
    """Convert MCAP file to video with keyboard/mouse overlays."""
    output = output or input_file.with_suffix(".mp4")

    # Check if output file exists
    if output.exists() and not force:
        typer.echo(f"Error: {output} already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    with OWAMcapReader(input_file) as reader:
        # Get frame dimensions and timing from screen messages
        recording_start_time = recording_end_time = None
        screen_message_count = 0
        frame_width = frame_height = None

        for mcap_msg in reader.iter_messages(topics=["screen"]):
            if recording_start_time is None:
                recording_start_time = mcap_msg.timestamp
                try:
                    msg = mcap_msg.decoded.resolve_relative_path(input_file)
                    frame_width, frame_height = msg.to_pil_image().size
                    typer.echo(f"Detected frame size: {frame_width}x{frame_height}")
                except Exception as e:
                    typer.echo(f"Warning: Could not detect frame size: {e}")
                    frame_width, frame_height = 854, 480
            recording_end_time = mcap_msg.timestamp
            screen_message_count += 1

        if recording_start_time is None:
            typer.echo("No screen messages found.")
            raise typer.Exit()

        frame_width = frame_width or 854
        frame_height = frame_height or 480

        # Calculate FPS
        if fps is None:
            duration_seconds = (recording_end_time - recording_start_time) / TimeUnits.SECOND
            fps = screen_message_count / duration_seconds if duration_seconds > 0 else 30.0
            typer.echo(f"Using original FPS: {fps:.2f} ({screen_message_count} frames over {duration_seconds:.2f}s)")
        else:
            typer.echo(f"Using specified FPS: {fps:.2f}")

        # Collect events from messages
        all_messages = list(reader.iter_messages(topics=topics + ["screen"], start_time=recording_start_time))
        key_tracker = KeyTracker()
        mouse_clicks: list[tuple[int, str, bool]] = []
        mouse_positions: dict[int, tuple[float, float]] = {}

        center_x, center_y = frame_width // 2, frame_height // 2
        scale_x, scale_y = frame_width / source_width, frame_height / source_height
        abs_x, abs_y = float(center_x), float(center_y)

        typer.echo(f"Mouse scaling: {scale_x:.3f}x width, {scale_y:.3f}x height")

        for msg in tqdm(all_messages, desc="Processing messages", unit="msg"):
            d = msg.decoded
            if msg.topic == "keyboard" and hasattr(d, "event_type") and hasattr(d, "vk"):
                key_tracker.handle(d.event_type, d.vk, msg.timestamp)
            elif msg.topic == "mouse/raw":
                if hasattr(d, "button_flags"):
                    for flag, btn in BUTTON_PRESS_FLAGS.items():
                        if d.button_flags & flag:
                            mouse_clicks.append((msg.timestamp, btn, True))
                            break
                    for flag, btn in BUTTON_RELEASE_FLAGS.items():
                        if d.button_flags & flag:
                            mouse_clicks.append((msg.timestamp, btn, False))
                            break
                if hasattr(d, "last_x") and hasattr(d, "last_y"):
                    abs_x = max(0, min(frame_width - 1, abs_x + d.last_x * scale_x))
                    abs_y = max(0, min(frame_height - 1, abs_y + d.last_y * scale_y))
                    mouse_positions[msg.timestamp] = (abs_x, abs_y)
            elif msg.topic == "mouse":
                if getattr(d, "event_type", None) == "click" and d.button and d.pressed is not None:
                    mouse_clicks.append((msg.timestamp, d.button, d.pressed))
                if hasattr(d, "x") and hasattr(d, "y"):
                    abs_x, abs_y = d.x * scale_x, d.y * scale_y
                    mouse_positions[msg.timestamp] = (abs_x, abs_y)

        key_tracker.finalize()

        # Prepare sorted data for rendering
        screen_messages = [m for m in all_messages if m.topic == "screen"]
        if duration is not None:
            max_ts = recording_start_time + int(duration * 1e9)
            screen_messages = [m for m in screen_messages if m.timestamp <= max_ts]
            typer.echo(f"Limiting to {duration}s ({len(screen_messages)} frames)")

        sorted_positions = sorted(mouse_positions.items())
        pos_timestamps = [ts for ts, _ in sorted_positions]
        sorted_clicks = sorted(mouse_clicks)
        sorted_keys = sorted(key_tracker.events, key=lambda x: x[0])

        # Render loop state
        click_idx, key_idx = 0, 0
        active_buttons: dict[str, int] = {}
        active_keys: list[tuple[int, int, int]] = []  # (start, end, vk)
        mouse_x, mouse_y = float(center_x), float(center_y)

        with VideoWriter(output, fps=fps, vfr=False) as writer:
            typer.echo(f"Creating video: {output}")

            for screen_msg in tqdm(screen_messages, desc="Processing frames", unit="frame"):
                ts = screen_msg.timestamp

                # Mouse position (binary search)
                idx = bisect.bisect_right(pos_timestamps, ts)
                if idx > 0:
                    mouse_x, mouse_y = sorted_positions[idx - 1][1]

                # Mouse buttons (incremental)
                while click_idx < len(sorted_clicks) and sorted_clicks[click_idx][0] <= ts:
                    _, btn, pressed = sorted_clicks[click_idx]
                    click_idx += 1
                    if pressed:
                        active_buttons[btn] = ts
                    elif btn in active_buttons:
                        del active_buttons[btn]

                # Keyboard keys (incremental with expiration)
                while key_idx < len(sorted_keys) and sorted_keys[key_idx][0] <= ts:
                    active_keys.append(sorted_keys[key_idx])
                    key_idx += 1
                active_keys = [(s, e, vk) for s, e, vk in active_keys if ts <= e]
                active_vk = {VK(vk) for _, _, vk in active_keys}

                # Render frame
                frame = np.array(screen_msg.decoded.resolve_relative_path(input_file).to_pil_image())
                expanded = np.zeros((frame_height + 150, frame_width, 3), dtype=np.uint8)
                expanded[:frame_height, :] = frame

                writer.write_frame(
                    draw_overlay(
                        expanded, active_vk, set(active_buttons), int(mouse_x), int(mouse_y), frame_width, frame_height
                    )
                )

        typer.echo(f"Video created: {output}")
        typer.echo(f"Frames: {len(screen_messages)}, Duration: {len(screen_messages) / fps:.2f}s")


if __name__ == "__main__":
    typer.run(convert_overlay)
