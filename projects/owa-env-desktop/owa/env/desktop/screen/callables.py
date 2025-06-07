def capture_screen():
    """
    Capture the screen.
    """
    import bettercam

    camera = bettercam.create()
    return camera.grab()
