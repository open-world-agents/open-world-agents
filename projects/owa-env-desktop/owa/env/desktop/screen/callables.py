import numpy as np


def capture_screen() -> np.ndarray:
    """
    Capture the current screen as a numpy array.

    Returns:
        numpy.ndarray: Screen capture as BGR image array with shape (height, width, 3).

    Examples:
        >>> screen = capture_screen()
        >>> print(f"Screen dimensions: {screen.shape}")  # e.g., (1080, 1920, 3)
        >>> # Save to file: cv2.imwrite('screenshot.png', screen)
    """
    import bettercam

    camera = bettercam.create()
    return camera.grab()
