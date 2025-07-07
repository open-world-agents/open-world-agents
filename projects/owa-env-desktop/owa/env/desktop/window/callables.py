import platform
from typing import Callable

from owa.msgs.desktop.window import WindowInfo

# --- Platform utils ---
_PLATFORM = platform.system()
_IS_DARWIN = _PLATFORM == "Darwin"
_IS_WINDOWS = _PLATFORM == "Windows"

# === Active Window Fetcher ===


def get_active_window() -> WindowInfo | None:
    """
    Get information about the currently active window.

    Returns:
        WindowInfo object containing title, position, and handle of the active window,
        or None if no active window is found.

    Examples:
        >>> window = get_active_window()
        >>> if window:
        ...     print(f"Active window: {window.title}")
        ...     print(f"Position: {window.rect}")
    """
    if _IS_DARWIN:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListOptionOnScreenOnly,
        )

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
                return WindowInfo(title=title, rect=rect, hWnd=hWnd)
        return None

    elif _IS_WINDOWS:
        import pygetwindow as gw

        active_window = gw.getActiveWindow()
        if active_window is not None:
            rect = active_window._getWindowRect()
            title = active_window.title
            rect_coords = (rect.left, rect.top, rect.right, rect.bottom)
            hWnd = active_window._hWnd
            return WindowInfo(title=title, rect=rect_coords, hWnd=hWnd)
        return WindowInfo(title="", rect=[0, 0, 0, 0], hWnd=-1)
    else:
        raise NotImplementedError(f"Platform {_PLATFORM} is not supported yet")


# === Window finder by title ===


def get_window_by_title(window_title_substring: str) -> WindowInfo:
    """
    Find a window by searching for a substring in its title.

    Args:
        window_title_substring: Substring to search for in window titles.

    Returns:
        WindowInfo object for the first matching window.

    Raises:
        ValueError: If no window with matching title is found.

    Examples:
        >>> window = get_window_by_title("notepad")
        >>> print(f"Found window: {window.title}")
    """
    if _IS_WINDOWS:
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
        )

    elif _IS_DARWIN:
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
                    )

        raise ValueError(f"No window with title containing '{window_title_substring}' found.")
    else:
        # Linux or other OS (not implemented yet)
        raise NotImplementedError("Not implemented for Linux or other OS.")


# === PID extractor by window title ===


def get_pid_by_title(window_title_substring: str) -> int:
    """
    Get the process ID (PID) of a window by its title.

    Args:
        window_title_substring: Substring to search for in window titles.

    Returns:
        Process ID of the window.

    Examples:
        >>> pid = get_pid_by_title("notepad")
        >>> print(f"Notepad PID: {pid}")
    """
    window = get_window_by_title(window_title_substring)
    if _IS_WINDOWS:
        import win32process

        # win32process.GetWindowThreadProcessId returns (tid, pid)
        _, pid = win32process.GetWindowThreadProcessId(window.hWnd)
        return pid
    else:
        # Implement if needed for other OS
        raise NotImplementedError(f"Getting PID by title not implemented for {_PLATFORM}")


# === Active-window decorator and helper ===


def when_active(window_title_substring: str) -> Callable:
    """
    Decorator to run a function only when a specific window is active.

    Args:
        window_title_substring: Substring to search for in window titles.

    Returns:
        Decorator function that conditionally executes the wrapped function.

    Examples:
        >>> @when_active("notepad")
        ... def do_something():
        ...     print("Notepad is active!")
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_active(window_title_substring):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def is_active(window_title_substring: str) -> bool:
    """
    Check if a window with the specified title substring is currently active.

    Args:
        window_title_substring: Substring to search for in window titles.

    Returns:
        True if the window is active, False otherwise.

    Examples:
        >>> if is_active("notepad"):
        ...     print("Notepad is the active window")
    """
    try:
        window = get_window_by_title(window_title_substring)
    except ValueError:
        return False
    active = get_active_window()
    return active is not None and active.hWnd == window.hWnd


# === Registry ===


def make_active(window_title_substring: str) -> None:
    """
    Bring a window to the foreground and make it active.

    Args:
        window_title_substring: Substring to search for in window titles.

    Raises:
        ValueError: If no window with matching title is found.
        NotImplementedError: If the operation is not supported on the current OS.

    Examples:
        >>> make_active("notepad")  # Brings notepad window to front
    """

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
        window.activate()
    else:
        raise NotImplementedError(f"Activation not implemented for this OS: {os_name}")
