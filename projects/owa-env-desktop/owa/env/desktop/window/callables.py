import platform

from owa.core.registry import CALLABLES

from ..msg import WindowInfo

# === Definition of the `get_active_window` function ===

if platform.system() == "Darwin":
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListOptionOnScreenOnly,
    )

    def get_active_window():
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for window in windows:
            if window.get("kCGWindowLayer", 0) == 0:  # Frontmost window
                bounds = window.get("kCGWindowBounds")
                title = window.get("kCGWindowName", "")
                rect = (
                    int(bounds["X"]),
                    int(bounds["Y"]),
                    int(bounds["X"] + bounds["Width"]),
                    int(bounds["Y"] + bounds["Height"]),
                )
                hWnd = window.get("kCGWindowNumber", 0)
                return WindowInfo(title=title, rect=rect, hWnd=hWnd, window=window)
        return None

elif platform.system() == "Windows":
    import pygetwindow as gw

    def get_active_window():
        active_window = gw.getActiveWindow()
        if active_window is not None:
            rect = active_window._getWindowRect()
            title = active_window.title
            rect_coords = (rect.left, rect.top, rect.right, rect.bottom)
            hWnd = active_window._hWnd
            return WindowInfo(title=title, rect=rect_coords, hWnd=hWnd, window=active_window)
        return None

else:

    def get_active_window():
        raise NotImplementedError(f"Platform {platform.system()} is not supported yet")

# === Definition of the `get_window_by_title` function ===


def get_window_by_title(window_title_substring: str) -> WindowInfo:
    os_name = platform.system()
    if os_name == "Windows":
        import pygetwindow as gw

        windows = gw.getWindowsWithTitle(window_title_substring)
        if not windows:
            raise ValueError(f"No window with title containing '{window_title_substring}' found.")

        # Temporal workaround to deal with `cmd`'s behavior: it setup own title as the command it running.
        # e.g. `owl window find abcd` will always find `cmd` window itself running command.
        if "Conda" in windows[0].title:
            windows.pop(0)

        window = windows[0]  # NOTE: only return the first window matching the title
        rect = window._getWindowRect()
        return WindowInfo(
            title=window.title,
            rect=(rect.left, rect.top, rect.right, rect.bottom),
            hWnd=window._hWnd,
            window=window,
        )
    elif os_name == "Darwin":
        from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowLayer, kCGWindowListOptionOnScreenOnly

        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for window in windows:
            # Skip windows that are not on normal level (like menu bars, etc)
            if window.get(kCGWindowLayer, 0) != 0:
                continue

            # Get window name from either kCGWindowName or kCGWindowOwnerName
            title = window.get("kCGWindowName", "")
            if not title:
                title = window.get("kCGWindowOwnerName", "")

            if title and window_title_substring.lower() in title.lower():
                bounds = window.get("kCGWindowBounds")
                if bounds:
                    return WindowInfo(
                        title=title,
                        rect=(
                            int(bounds["X"]),
                            int(bounds["Y"]),
                            int(bounds["X"] + bounds["Width"]),
                            int(bounds["Y"] + bounds["Height"]),
                        ),
                        hWnd=window.get("kCGWindowNumber", 0),
                        window=window,
                    )

        raise ValueError(f"No window with title containing '{window_title_substring}' found.")
    else:
        # Linux or other OS (not implemented yet)
        raise NotImplementedError("Not implemented for Linux or other OS.")


# === Definition of the `when_active` decorator ===


def when_active(window_title_substring: str):
    """Decorator to run the function when the window with the title containing the substring is active."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_active(window_title_substring):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def is_active(window_title_substring: str):
    """Returns whether the window with the title containing the substring is active."""
    try:
        window = get_window_by_title(window_title_substring)
    except ValueError:
        return False

    return get_active_window().hWnd == window.hWnd


def make_active(window_title_substring: str):
    """Make the window with the title containing the substring active."""

    window: WindowInfo = get_window_by_title(window_title_substring)
    window.window.activate()


CALLABLES.register("window.get_active_window")(get_active_window)
CALLABLES.register("window.get_window_by_title")(get_window_by_title)
CALLABLES.register("window.when_active")(when_active)
CALLABLES.register("window.is_active")(is_active)
CALLABLES.register("window.make_active")(make_active)
