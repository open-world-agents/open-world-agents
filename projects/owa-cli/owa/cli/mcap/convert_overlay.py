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
from owa.cli.mcap.convert import KeyStateManager

# Keyboard layout for visualization (full compact keyboard)
# Format: (row, col, width, label, vk_code, is_arrow)
# is_arrow: True if this key should render an arrow symbol instead of text
KEYBOARD_LAYOUT = [
    # Row 0: ESC, F1-F12, BACKSPACE
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
    # Row 1: ~, 1-9, 0, -, =, \
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
    # Row 2: TAB, Q-P, [, ], ENTER
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
    # Row 3: CAPS, A-L, ;, ', UP, SHIFT
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
    # Row 4: SHIFT, Z-M, comma, period, /, LEFT, DOWN, RIGHT
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
    # Row 5: CTRL, WIN, ALT, SPACE (8 units), ALT, WIN, CTRL
    (5, 0, 1, "CTRL", VK.LCONTROL, False),
    (5, 1, 1, "WIN", VK.LWIN, False),
    (5, 2, 1, "ALT", VK.LMENU, False),
    (5, 3, 8, "SPACE", VK.SPACE, False),
    (5, 11, 1, "ALT", VK.RMENU, False),
    (5, 12, 1, "WIN", VK.RWIN, False),
    (5, 13, 1, "CTRL", VK.RCONTROL, False),
]

# Button flag mappings for RawMouseEvent
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


def draw_arrow(frame: np.ndarray, center_x: int, center_y: int, direction: str, color: tuple, size: int = 10) -> None:
    """
    Draw an arrow symbol on the frame with anti-aliasing for smooth edges.

    Args:
        frame: The video frame to draw on
        center_x: X coordinate of the arrow center
        center_y: Y coordinate of the arrow center
        direction: Arrow direction ("UP", "DOWN", "LEFT", "RIGHT")
        color: Arrow color (B, G, R)
        size: Arrow size
    """
    if direction == "UP":
        # Triangle pointing up
        pts = np.array(
            [
                [center_x, center_y - size],
                [center_x - size, center_y + size // 2],
                [center_x + size, center_y + size // 2],
            ],
            np.int32,
        )
    elif direction == "DOWN":
        # Triangle pointing down
        pts = np.array(
            [
                [center_x, center_y + size],
                [center_x - size, center_y - size // 2],
                [center_x + size, center_y - size // 2],
            ],
            np.int32,
        )
    elif direction == "LEFT":
        # Triangle pointing left
        pts = np.array(
            [
                [center_x - size, center_y],
                [center_x + size // 2, center_y - size],
                [center_x + size // 2, center_y + size],
            ],
            np.int32,
        )
    elif direction == "RIGHT":
        # Triangle pointing right
        pts = np.array(
            [
                [center_x + size, center_y],
                [center_x - size // 2, center_y - size],
                [center_x - size // 2, center_y + size],
            ],
            np.int32,
        )
    else:
        return

    # Draw with anti-aliasing for smoother edges
    # First fill the polygon
    cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)
    # Then draw the outline with anti-aliasing for extra smoothness
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def draw_mouse_figure(frame: np.ndarray, x: int, y: int, active_buttons: set) -> None:
    """
    Draw a mini mouse figure with left, middle, and right buttons.

    Args:
        frame: The video frame to draw on
        x: X coordinate of the top-left corner
        y: Y coordinate of the top-left corner
        active_buttons: Set of active mouse button names ("left", "right", "middle")
    """
    # Mouse dimensions
    mouse_width = 60
    mouse_height = 80

    # Colors
    bg_color = (40, 40, 40)  # Dark gray background
    border_color = (150, 150, 150)  # Light gray border
    inactive_color = (70, 70, 70)  # Dark gray for inactive buttons
    left_active_color = (255, 0, 0)  # Red for left button (RGB format)
    right_active_color = (0, 0, 255)  # Blue for right button (RGB format)
    middle_active_color = (255, 255, 0)  # Yellow for middle button (RGB format)

    # Draw mouse body (rounded rectangle)
    center_x = x + mouse_width // 2
    center_y = y + mouse_height // 2

    # Draw filled ellipse for mouse body with anti-aliasing
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (mouse_width // 2, mouse_height // 2),
        0,
        0,
        360,
        bg_color,
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (mouse_width // 2, mouse_height // 2),
        0,
        0,
        360,
        border_color,
        2,
        lineType=cv2.LINE_AA,
    )

    # Button dimensions - quadrant shape that fits the mouse ellipse
    middle_button_width = int(mouse_width * 0.16)  # 16% of width for middle button

    # Calculate ellipse parameters to match the mouse body shape
    ellipse_a = mouse_width // 2  # Semi-major axis (horizontal)
    ellipse_b = mouse_height // 2  # Semi-minor axis (vertical)

    # For buttons, we want them to follow the ellipse curve in the upper half
    # The button radius should match the ellipse dimensions
    # Position the button center at the actual center of the ellipse
    button_center_y = center_y  # Use the ellipse center

    # Left button (quadrant shape - following the ellipse curve)
    left_color = left_active_color if "left" in active_buttons else inactive_color

    # Create points for left button polygon (quarter ellipse from center)
    left_points = []

    # Start from the button center point
    left_points.append([center_x, button_center_y])

    # Add points along the ellipse arc from left (180°) to top (270°)
    # This creates a quarter-ellipse quadrant that curves UPWARD and matches the mouse shape
    angle = 180.0
    while angle <= 270.0:
        rad = np.radians(angle)
        px = center_x + int(ellipse_a * np.cos(rad))
        py = button_center_y + int(ellipse_b * np.sin(rad))
        left_points.append([px, py])
        angle += 1.0

    # Close the shape back to center
    left_points.append([center_x, button_center_y])

    left_pts = np.array(left_points, np.int32)
    cv2.fillPoly(frame, [left_pts], left_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [left_pts], True, border_color, 2, lineType=cv2.LINE_AA)

    # Right button (quadrant shape - following the ellipse curve)
    right_color = right_active_color if "right" in active_buttons else inactive_color

    # Create points for right button polygon (quarter ellipse from center)
    right_points = []

    # Start from the button center point
    right_points.append([center_x, button_center_y])

    # Add points along the ellipse arc from top (270°) to right (360°/0°)
    # This creates a quarter-ellipse quadrant that curves UPWARD and matches the mouse shape
    angle = 270.0
    while angle <= 360.0:
        rad = np.radians(angle)
        px = center_x + int(ellipse_a * np.cos(rad))
        py = button_center_y + int(ellipse_b * np.sin(rad))
        right_points.append([px, py])
        angle += 1.0

    # Close the shape back to center
    right_points.append([center_x, button_center_y])

    right_pts = np.array(right_points, np.int32)
    cv2.fillPoly(frame, [right_pts], right_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [right_pts], True, border_color, 2, lineType=cv2.LINE_AA)

    # Middle button (scroll wheel - smaller and narrower)
    middle_color = middle_active_color if "middle" in active_buttons else inactive_color
    middle_x = x + mouse_width // 2 - middle_button_width // 2
    middle_height = int(ellipse_b * 0.6)  # Proportional to ellipse height
    middle_y = y + int(mouse_height * 0.1)  # Position in upper portion
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
    line_y1 = middle_y + middle_height // 3
    line_y2 = middle_y + 2 * middle_height // 3
    cv2.line(frame, (middle_x + 2, line_y1), (middle_x + middle_button_width - 2, line_y1), border_color, 1)
    cv2.line(frame, (middle_x + 2, line_y2), (middle_x + middle_button_width - 2, line_y2), border_color, 1)


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
    """
    Draw a bounded rectangular minimap showing mouse position.

    Args:
        frame: The video frame to draw on
        mouse_x: Current mouse X position in full frame coordinates
        mouse_y: Current mouse Y position in full frame coordinates
        minimap_x: X coordinate of the minimap's top-left corner
        minimap_y: Y coordinate of the minimap's top-left corner
        minimap_width: Width of the minimap rectangle
        minimap_height: Height of the minimap rectangle
        frame_width: Width of the full video frame
        frame_height: Height of the full video frame
        active_mouse_buttons: Set of active mouse button names
    """
    # Draw minimap border (white border with no background for transparency)
    border_color = (255, 255, 255)  # White border
    border_thickness = 1
    cv2.rectangle(
        frame,
        (minimap_x, minimap_y),
        (minimap_x + minimap_width, minimap_y + minimap_height),
        border_color,
        border_thickness,
        lineType=cv2.LINE_AA,
    )

    # Map mouse position from full frame to minimap coordinates
    # Normalize mouse position to [0, 1] range
    norm_x = mouse_x / frame_width
    norm_y = mouse_y / frame_height

    # Map to minimap coordinates (with padding to keep cursor inside border)
    padding = 5  # Keep cursor away from edges
    minimap_cursor_x = minimap_x + padding + int(norm_x * (minimap_width - 2 * padding))
    minimap_cursor_y = minimap_y + padding + int(norm_y * (minimap_height - 2 * padding))

    # Clamp to minimap bounds
    minimap_cursor_x = max(minimap_x + padding, min(minimap_x + minimap_width - padding, minimap_cursor_x))
    minimap_cursor_y = max(minimap_y + padding, min(minimap_y + minimap_height - padding, minimap_cursor_y))

    # Draw cursor in minimap (smaller than full-screen cursor)
    cursor_radius = 4
    cv2.circle(frame, (minimap_cursor_x, minimap_cursor_y), cursor_radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (minimap_cursor_x, minimap_cursor_y), cursor_radius, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # Draw active mouse clicks in minimap (smaller rings)
    for button_name in active_mouse_buttons:
        if button_name == "left":
            color = (255, 0, 0)  # Red for left click
        elif button_name == "right":
            color = (0, 0, 255)  # Blue for right click
        else:
            color = (255, 255, 0)  # Yellow for middle click

        # Draw smaller ring effect in minimap
        cv2.circle(frame, (minimap_cursor_x, minimap_cursor_y), cursor_radius + 4, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (minimap_cursor_x, minimap_cursor_y), cursor_radius + 7, color, 1, lineType=cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    active_keys: set,
    active_mouse_buttons: set,
    mouse_x: int,
    mouse_y: int,
    frame_width: int,
    frame_height: int,
    overlay_height: int,
) -> np.ndarray:
    """
    Draw a keyboard UI overlay on the frame showing which keys are pressed.

    Args:
        frame: The video frame to draw on (already expanded with black space at bottom)
        active_keys: Set of VK codes that are currently pressed
        active_mouse_buttons: Set of mouse button names that are currently pressed
        mouse_x: Current mouse X position in full frame coordinates
        mouse_y: Current mouse Y position in full frame coordinates
        frame_width: Width of the full video frame
        frame_height: Height of the original video frame (without overlay space)
        overlay_height: Height of the black overlay space at the bottom

    Returns:
        The frame with keyboard overlay drawn
    """
    # Keyboard UI settings
    key_size = 20  # Base size for a 1-unit key (reduced from 30)
    key_margin = 2  # Reduced row margin for more compact layout

    # Background for keyboard UI (full compact layout)
    # 14 columns (0-13) + margins
    bg_width = int(14 * (key_size + key_margin))  # Width to fit 14 columns with 1px margins
    # bg_height = int(6 * (key_size + key_margin))  # Height to fit 6 rows with 1px margins

    # Position overlays in the black space at the bottom
    start_x = 10  # Small left margin
    start_y = frame_height + 10  # Start in the black space below the video

    # No need for transparent overlay - draw directly on the black space
    overlay = frame.copy()

    # Draw each key
    for row, col, width, label, vk_code, is_arrow in KEYBOARD_LAYOUT:
        x = start_x + int(col * (key_size + key_margin))
        y = start_y + int(row * (key_size + key_margin))
        w = int(width * (key_size + key_margin) - key_margin)
        h = key_size

        # Check if this key is pressed
        is_pressed = False
        if vk_code is not None and vk_code in active_keys:
            is_pressed = True

        # Choose colors based on pressed state
        if is_pressed:
            bg_color = (80, 176, 171)  # Teal when pressed
            text_color = (255, 255, 255)  # White text/arrow
        else:
            bg_color = (107, 107, 107)  # Gray when not pressed
            text_color = (255, 255, 255)  # White text/arrow

        # Draw key background on overlay
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
        # cv2.rectangle(overlay, (x, y), (x + w, y + h), border_color, 1)

        # Draw key label or arrow
        if is_arrow:
            # Draw arrow symbol
            center_x = x + w // 2
            center_y = y + h // 2
            draw_arrow(overlay, center_x, center_y, label, text_color, size=5)  # Reduced from 8
        else:
            # Draw text label with dynamic font sizing to fit within key boundaries
            # Start with a base font scale and adjust based on label length
            if len(label) == 1:
                font_scale = 0.35  # Single character (letters, numbers, symbols)
            elif len(label) == 2:
                font_scale = 0.30  # Two characters (F1-F12)
            elif len(label) <= 4:
                font_scale = 0.25  # 3-4 characters (CAPS, BACK, etc.)
            else:
                font_scale = 0.20  # 5+ characters (SHIFT, ENTER, SPACE)

            # Measure text size and ensure it fits within key width
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            # If text is too wide, reduce font scale to fit
            max_width = w - 2  # Leave 1px padding on each side
            if text_size[0] > max_width:
                font_scale = font_scale * (max_width / text_size[0])
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(
                overlay, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
            )

    # Draw mouse figure on the right side of the keyboard
    mouse_figure_x = start_x + bg_width + 10
    mouse_figure_y = start_y
    draw_mouse_figure(overlay, mouse_figure_x, mouse_figure_y, active_mouse_buttons)

    # No blending needed - just draw directly on the black space
    # Copy overlay back to frame (only the overlay area changed)
    frame = overlay

    # Draw mouse minimap adjacent to the mouse figure
    # Position minimap to the right of the mouse figure
    mouse_width = 60  # Same as in draw_mouse_figure
    minimap_margin = 15
    minimap_x = mouse_figure_x + mouse_width + minimap_margin
    minimap_y = start_y
    minimap_width = 150
    minimap_height = 100

    draw_mouse_minimap(
        frame,
        mouse_x,
        mouse_y,
        minimap_x,
        minimap_y,
        minimap_width,
        minimap_height,
        frame_width,
        frame_height,
        active_mouse_buttons,
    )

    return frame


def convert_overlay(
    mcap_path: Annotated[Path, typer.Argument(help="Path to the input .mcap file")],
    topics: Annotated[list[str], typer.Option(help="Comma-separated list of topics to include in the overlay")] = [
        "mouse/raw",
        "keyboard",
    ],
    output_video: Annotated[
        Path | None, typer.Argument(help="Path to the output video file. Defaults to <input_video>.mp4")
    ] = None,
    fps: Annotated[
        float | None, typer.Option(help="Output video frame rate (default: use original MCAP frame rate)")
    ] = None,
    max_duration_seconds: Annotated[
        float | None, typer.Option(help="Maximum duration in seconds to generate (default: entire video)")
    ] = None,
    original_width: Annotated[
        int, typer.Option(help="Original recording width resolution for mouse movement scaling")
    ] = 2560,
    original_height: Annotated[
        int, typer.Option(help="Original recording height resolution for mouse movement scaling")
    ] = 1440,
):
    """
    Convert an `.mcap` file into a video with overlays showing mouse clicks and keyboard presses.
    """
    if output_video is None:
        output_video = mcap_path.with_suffix(".mp4")

    with OWAMcapReader(mcap_path) as reader:
        # Find the first screen message to get the recording start time and pts offset
        recording_start_time = None
        recording_end_time = None
        screen_message_count = 0
        frame_width = None
        frame_height = None

        for mcap_msg in reader.iter_messages(topics=["screen"]):
            if recording_start_time is None:
                recording_start_time = mcap_msg.timestamp

                # Get frame dimensions from the first screen message
                try:
                    msg = mcap_msg.decoded.resolve_relative_path(mcap_path)
                    image = msg.to_pil_image()
                    frame_width, frame_height = image.size
                    typer.echo(f"Detected video frame size: {frame_width}x{frame_height}")
                except Exception as e:
                    typer.echo(f"Warning: Could not determine frame size from first screen message: {e}")
                    typer.echo("Falling back to default size: 854x480")
                    frame_width, frame_height = 854, 480
            recording_end_time = mcap_msg.timestamp
            screen_message_count += 1

        if recording_start_time is None:
            typer.echo("No screen messages found in the .mcap file.")
            raise typer.Exit()

        # Set default frame size if not detected
        if frame_width is None or frame_height is None:
            frame_width, frame_height = 854, 480
            typer.echo(f"Using default frame size: {frame_width}x{frame_height}")

        # Calculate original FPS if not specified
        if fps is None:
            duration_seconds = (recording_end_time - recording_start_time) / TimeUnits.SECOND
            original_fps = screen_message_count / duration_seconds if duration_seconds > 0 else 30.0
            fps = original_fps
            typer.echo(
                f"Using original frame rate: {fps:.2f} FPS ({screen_message_count} frames over {duration_seconds:.2f}s)"
            )
        else:
            typer.echo(f"Using specified frame rate: {fps:.2f} FPS")

        # Collect all input events
        all_messages = list(reader.iter_messages(topics=topics + ["screen"], start_time=recording_start_time))

        # Initialize key state manager
        key_state_manager = KeyStateManager()

        # Build a timeline of mouse button presses with timestamps
        mouse_click_timeline = []  # List of (timestamp, button_name, is_press)

        # Calculate center position based on detected frame size
        center_x = frame_width // 2
        center_y = frame_height // 2

        # Calculate scaling factors for mouse movement using provided original resolution
        scale_x = frame_width / original_width
        scale_y = frame_height / original_height

        typer.echo(
            f"Mouse movement scaling: {scale_x:.3f}x (width {original_width} -> {frame_width}), {scale_y:.3f}x (height {original_height} -> {frame_height})"
        )

        # Build timelines for events and mouse positions in a single pass
        mouse_positions = {}  # timestamp -> (x, y)
        abs_x, abs_y = center_x, center_y  # Start at center

        # Process all messages to build event timeline and mouse positions
        for mcap_msg in all_messages:
            # Handle keyboard events with state management
            if mcap_msg.topic == "keyboard":
                if hasattr(mcap_msg.decoded, "event_type") and hasattr(mcap_msg.decoded, "vk"):
                    key_state_manager.handle_key_event(
                        mcap_msg.decoded.event_type, mcap_msg.decoded.vk, mcap_msg.timestamp
                    )
            elif mcap_msg.topic == "mouse/raw":
                # Handle mouse button events
                if hasattr(mcap_msg.decoded, "button_flags"):
                    button_flags = mcap_msg.decoded.button_flags
                    # Check for button press events
                    for flag, button_name in BUTTON_PRESS_FLAGS.items():
                        if button_flags & flag:
                            mouse_click_timeline.append((mcap_msg.timestamp, button_name, True))
                            break
                    # Check for button release events
                    for flag, button_name in BUTTON_RELEASE_FLAGS.items():
                        if button_flags & flag:
                            mouse_click_timeline.append((mcap_msg.timestamp, button_name, False))
                            break

                # Handle mouse movement (relative)
                if hasattr(mcap_msg.decoded, "last_x") and hasattr(mcap_msg.decoded, "last_y"):
                    # Accumulate relative movements with scaling
                    abs_x += mcap_msg.decoded.last_x * scale_x
                    abs_y += mcap_msg.decoded.last_y * scale_y
                    # Clamp to screen bounds
                    abs_x = max(0, min(frame_width - 1, abs_x))
                    abs_y = max(0, min(frame_height - 1, abs_y))
                    mouse_positions[mcap_msg.timestamp] = (abs_x, abs_y)
            elif mcap_msg.topic == "mouse":
                # Handle mouse click events
                if (
                    getattr(mcap_msg.decoded, "event_type", None) == "click"
                    and mcap_msg.decoded.button is not None
                    and mcap_msg.decoded.pressed is not None
                ):
                    button_name = mcap_msg.decoded.button
                    is_pressed = mcap_msg.decoded.pressed
                    mouse_click_timeline.append((mcap_msg.timestamp, button_name, is_pressed))

                # Handle mouse movement (absolute)
                if hasattr(mcap_msg.decoded, "x") and hasattr(mcap_msg.decoded, "y"):
                    # Absolute position from mouse topic - also needs scaling
                    abs_x = mcap_msg.decoded.x * scale_x
                    abs_y = mcap_msg.decoded.y * scale_y
                    mouse_positions[mcap_msg.timestamp] = (abs_x, abs_y)

        # Finalize keyboard states
        key_state_manager.finalize_remaining_subtitles()
        keyboard_events = key_state_manager.get_completed_subtitles()

        # Now render the video with overlays
        frame_count = 0
        current_mouse_x, current_mouse_y = center_x, center_y  # Start at center

        with VideoWriter(output_video, fps=fps, vfr=False) as writer:
            typer.echo(f"Creating video with overlays: {output_video}")

            # Reset reader to iterate through screen messages
            screen_messages = [msg for msg in all_messages if msg.topic == "screen"]

            # Apply duration limit if specified
            if max_duration_seconds is not None:
                max_duration_ns = int(max_duration_seconds * 1e9)
                max_timestamp = recording_start_time + max_duration_ns
                screen_messages = [msg for msg in screen_messages if msg.timestamp <= max_timestamp]
                typer.echo(f"Limiting video to {max_duration_seconds} seconds ({len(screen_messages)} frames)")

            with tqdm(total=len(screen_messages), desc="Processing frames", unit="frame") as pbar:
                for mcap_msg in screen_messages:
                    current_timestamp = mcap_msg.timestamp

                    # Update mouse position from timeline up to this timestamp
                    for ts, (x, y) in mouse_positions.items():
                        if ts <= current_timestamp:
                            current_mouse_x, current_mouse_y = x, y
                        else:
                            break

                    # Get the screen frame
                    msg = mcap_msg.decoded.resolve_relative_path(mcap_path)
                    image = msg.to_pil_image()
                    frame = np.array(image)

                    # Calculate overlay height needed (keyboard + mouse + minimap + margins)
                    overlay_height = 150  # Height for the black space at bottom

                    # Expand frame with black space at the bottom
                    expanded_frame = np.zeros((frame_height + overlay_height, frame_width, 3), dtype=np.uint8)
                    expanded_frame[:frame_height, :] = frame  # Copy original frame to top

                    # Determine which mouse buttons are currently pressed
                    active_mouse_buttons = {}  # button_name -> press_timestamp
                    for click_ts, button_name, is_press in mouse_click_timeline:
                        if click_ts > current_timestamp:
                            break
                        if is_press:
                            active_mouse_buttons[button_name] = click_ts
                        else:
                            if button_name in active_mouse_buttons:
                                del active_mouse_buttons[button_name]

                    # Update active keyboard presses - track VK codes
                    active_vk_codes = set()
                    for start_time, end_time, key_message in keyboard_events:
                        if start_time <= current_timestamp <= end_time:
                            # Extract VK code from message (format: "press KEY_NAME")
                            if key_message.startswith("press "):
                                key_name = key_message[6:]  # Remove "press " prefix
                                try:
                                    vk_code = VK[key_name]
                                    active_vk_codes.add(vk_code)
                                except (KeyError, ValueError):
                                    pass

                    # Draw keyboard UI overlay with mouse minimap in the black space
                    frame = draw_overlay(
                        expanded_frame,
                        active_vk_codes,
                        set(active_mouse_buttons.keys()),
                        current_mouse_x,
                        current_mouse_y,
                        frame_width,
                        frame_height,
                        overlay_height,
                    )

                    # Write frame
                    writer.write_frame(frame)
                    frame_count += 1
                    pbar.update(1)

        typer.echo(f"Video created successfully: {output_video}")
        typer.echo(f"Total frames: {frame_count}")
        typer.echo(f"Duration: {frame_count / fps:.2f} seconds")


if __name__ == "__main__":
    typer.run(convert_overlay)
