from owa.core.registry import CALLABLES


@CALLABLES.register("screen.capture")
def capture_screen():
    """
    Capture the screen.
    """
    import bettercam

    camera = bettercam.create()
    return camera.grab()
