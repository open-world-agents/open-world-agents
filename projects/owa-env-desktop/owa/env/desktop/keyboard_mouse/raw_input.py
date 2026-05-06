"""
Windows raw input capture for high-definition mouse data.

Provides access to unfiltered mouse input directly from the HID stack,
bypassing Windows pointer acceleration and screen resolution limits.
"""

import sys
import threading
import time
from typing import Callable, Optional

from loguru import logger

from owa.msgs.desktop.mouse import RawMouseEvent

# Windows-specific imports - only available on Windows
if sys.platform == "win32":
    import ctypes
    from ctypes import POINTER, Structure, Union, byref, sizeof, wintypes

    # Windows constants
    WM_INPUT = 0x00FF
    RIM_TYPEMOUSE = 0
    RIDEV_INPUTSINK = 0x00000100
    RID_INPUT = 0x10000003

    # HID usage constants
    HID_USAGE_PAGE_GENERIC = 0x01
    HID_USAGE_GENERIC_MOUSE = 0x02

    # Raw mouse button flags
    RAWMOUSE_BUTTON_FLAGS = RawMouseEvent.ButtonFlags

    # Windows structures
    class RAWINPUTDEVICE(Structure):
        _fields_ = [
            ("usUsagePage", wintypes.USHORT),
            ("usUsage", wintypes.USHORT),
            ("dwFlags", wintypes.DWORD),
            ("hwndTarget", wintypes.HWND),
        ]

    class RAWINPUTHEADER(Structure):
        _fields_ = [
            ("dwType", wintypes.DWORD),
            ("dwSize", wintypes.DWORD),
            ("hDevice", wintypes.HANDLE),
            ("wParam", wintypes.WPARAM),
        ]

    class RAWMOUSE_BUTTONS(Structure):
        _fields_ = [
            ("usButtonFlags", wintypes.USHORT),
            ("usButtonData", wintypes.USHORT),
        ]

    class RAWMOUSE_BUTTONS_UNION(Union):
        _fields_ = [
            ("ulButtons", wintypes.ULONG),
            ("Buttons", RAWMOUSE_BUTTONS),
        ]

    class RAWMOUSE(Structure):
        _fields_ = [
            ("usFlags", wintypes.USHORT),
            ("ButtonsUnion", RAWMOUSE_BUTTONS_UNION),
            ("ulRawButtons", wintypes.ULONG),
            ("lLastX", wintypes.LONG),
            ("lLastY", wintypes.LONG),
            ("ulExtraInformation", wintypes.ULONG),
        ]

    class RAWINPUT_DATA(Union):
        _fields_ = [
            ("mouse", RAWMOUSE),
            # We only need mouse data for this implementation
        ]

    class RAWINPUT(Structure):
        _fields_ = [
            ("header", RAWINPUTHEADER),
            ("data", RAWINPUT_DATA),
        ]

    # Define missing Windows types
    LRESULT = ctypes.c_long
    ATOM = wintypes.WORD
    HICON = wintypes.HANDLE
    HCURSOR = wintypes.HANDLE
    HBRUSH = wintypes.HANDLE

    # Windows API functions
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    RegisterRawInputDevices = user32.RegisterRawInputDevices
    RegisterRawInputDevices.argtypes = [POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
    RegisterRawInputDevices.restype = wintypes.BOOL

    GetRawInputData = user32.GetRawInputData
    GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, POINTER(wintypes.UINT), wintypes.UINT]
    GetRawInputData.restype = wintypes.UINT

    CreateWindowExW = user32.CreateWindowExW
    CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.INT,
        wintypes.INT,
        wintypes.INT,
        wintypes.INT,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    CreateWindowExW.restype = wintypes.HWND

    DefWindowProcW = user32.DefWindowProcW
    DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    DefWindowProcW.restype = LRESULT

    GetMessageW = user32.GetMessageW
    GetMessageW.argtypes = [wintypes.LPMSG, wintypes.HWND, wintypes.UINT, wintypes.UINT]
    GetMessageW.restype = wintypes.BOOL

    TranslateMessage = user32.TranslateMessage
    TranslateMessage.argtypes = [wintypes.LPMSG]
    TranslateMessage.restype = wintypes.BOOL

    DispatchMessageW = user32.DispatchMessageW
    DispatchMessageW.argtypes = [wintypes.LPMSG]
    DispatchMessageW.restype = LRESULT

    PostQuitMessage = user32.PostQuitMessage
    PostQuitMessage.argtypes = [wintypes.INT]
    PostQuitMessage.restype = None

    GetModuleHandleW = kernel32.GetModuleHandleW
    GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    GetModuleHandleW.restype = wintypes.HMODULE

    # Define WNDCLASSW structure
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", wintypes.INT),
            ("cbWndExtra", wintypes.INT),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", HICON),
            ("hCursor", HCURSOR),
            ("hbrBackground", HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    RegisterClassW = user32.RegisterClassW
    RegisterClassW.argtypes = [POINTER(WNDCLASSW)]
    RegisterClassW.restype = ATOM

    DestroyWindow = user32.DestroyWindow
    DestroyWindow.argtypes = [wintypes.HWND]
    DestroyWindow.restype = wintypes.BOOL

    class RawInputCapture:
        """
        Windows raw input capture for high-definition mouse data.

        This class handles the Windows-specific implementation of raw input capture,
        including window creation, device registration, and message processing.
        """

        def __init__(self):
            self.hwnd: Optional[int] = None
            self.callback: Optional[Callable[[RawMouseEvent], None]] = None
            self.running = False
            self.thread: Optional[threading.Thread] = None
            self._stop_event = threading.Event()

        def register_callback(self, callback: Callable[[RawMouseEvent], None]) -> None:
            """Register callback function to receive raw mouse events."""
            self.callback = callback

        def start(self) -> bool:
            """
            Start raw input capture.

            Returns:
                True if capture started successfully, False otherwise.
            """
            if sys.platform != "win32":
                logger.warning("Raw input capture is only supported on Windows")
                return False

            if self.running:
                logger.warning("Raw input capture is already running")
                return True

            try:
                self.running = True
                self._stop_event.clear()
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
                logger.debug("Raw input capture started")
                return True
            except Exception as e:
                logger.error(f"Failed to start raw input capture: {e}")
                self.running = False
                return False

        def stop(self) -> None:
            """Stop raw input capture."""
            if not self.running:
                return

            self.running = False
            self._stop_event.set()

            if self.thread and self.thread.is_alive():
                # Post quit message to break message loop
                if self.hwnd:
                    PostQuitMessage(0)
                self.thread.join(timeout=2.0)

            logger.debug("Raw input capture stopped")

        def _capture_loop(self) -> None:
            """Main capture loop running in separate thread."""
            try:
                # Create hidden window for message handling
                if not self._create_window():
                    logger.error("Failed to create window for raw input")
                    return

                # Register raw input device
                if not self._register_raw_input():
                    logger.error("Failed to register raw input device")
                    return

                # Message loop
                self._message_loop()

            except Exception as e:
                logger.error(f"Error in raw input capture loop: {e}")
            finally:
                self._cleanup()

        def _create_window(self) -> bool:
            """Create hidden window for receiving raw input messages."""
            try:
                # Simple window creation for message handling
                hinstance = GetModuleHandleW(None)

                # Create window procedure
                self.wndproc = WNDPROC(self._window_proc)

                # Create a simple window class
                wc = WNDCLASSW()
                wc.style = 0
                wc.lpfnWndProc = self.wndproc
                wc.cbClsExtra = 0
                wc.cbWndExtra = 0
                wc.hInstance = hinstance
                wc.hIcon = None
                wc.hCursor = None
                wc.hbrBackground = None
                wc.lpszMenuName = None
                wc.lpszClassName = "OWARawInputWindow"

                if not RegisterClassW(byref(wc)):
                    logger.error("Failed to register window class")
                    return False

                # Create hidden window
                self.hwnd = CreateWindowExW(
                    0,  # dwExStyle
                    "OWARawInputWindow",  # lpClassName
                    "OWA Raw Input",  # lpWindowName
                    0,  # dwStyle (hidden)
                    0,
                    0,
                    0,
                    0,  # position and size
                    None,  # hWndParent
                    None,  # hMenu
                    hinstance,  # hInstance
                    None,  # lpParam
                )

                if not self.hwnd:
                    logger.error("Failed to create window")
                    return False

                return True
            except Exception as e:
                logger.error(f"Error creating window: {e}")
                return False

        def _register_raw_input(self) -> bool:
            """Register for raw mouse input."""
            try:
                rid = RAWINPUTDEVICE()
                rid.usUsagePage = HID_USAGE_PAGE_GENERIC
                rid.usUsage = HID_USAGE_GENERIC_MOUSE
                rid.dwFlags = RIDEV_INPUTSINK
                rid.hwndTarget = self.hwnd

                result = RegisterRawInputDevices(byref(rid), 1, sizeof(RAWINPUTDEVICE))
                if not result:
                    logger.error("Failed to register raw input device")
                    return False

                logger.debug("Raw input device registered successfully")
                return True
            except Exception as e:
                logger.error(f"Error registering raw input: {e}")
                return False

        def _window_proc(self, hwnd, msg, wparam, lparam):
            """Window procedure to handle raw input messages."""
            if msg == WM_INPUT:
                self._handle_raw_input(lparam)
                return 0
            return DefWindowProcW(hwnd, msg, wparam, lparam)

        def _handle_raw_input(self, lparam) -> None:
            """Process WM_INPUT message and extract mouse data."""
            try:
                # Get size of raw input data
                size = wintypes.UINT(0)
                GetRawInputData(lparam, RID_INPUT, None, byref(size), sizeof(RAWINPUTHEADER))

                if size.value == 0:
                    return

                # Allocate buffer and get raw input data
                buffer = (ctypes.c_byte * size.value)()
                result = GetRawInputData(lparam, RID_INPUT, buffer, byref(size), sizeof(RAWINPUTHEADER))

                if result != size.value:
                    return

                # Cast buffer to RAWINPUT structure
                raw_input = ctypes.cast(buffer, POINTER(RAWINPUT)).contents

                # Process mouse data
                if raw_input.header.dwType == RIM_TYPEMOUSE:
                    self._process_mouse_data(raw_input)

            except Exception as e:
                logger.error(f"Error handling raw input: {e}")

        def _process_mouse_data(self, raw_input) -> None:
            """Process raw mouse data and create RawMouseEvent."""
            try:
                mouse_data = raw_input.data.mouse

                # Create raw mouse event
                event = RawMouseEvent(
                    us_flags=mouse_data.usFlags,
                    last_x=mouse_data.lLastX,
                    last_y=mouse_data.lLastY,
                    button_flags=mouse_data.ButtonsUnion.Buttons.usButtonFlags,
                    button_data=mouse_data.ButtonsUnion.Buttons.usButtonData,
                    device_handle=int(raw_input.header.hDevice) if raw_input.header.hDevice else None,
                    timestamp=time.time_ns(),
                )

                # Call registered callback
                if self.callback:
                    self.callback(event)

            except Exception as e:
                logger.error(f"Error processing mouse data: {e}")

        def _message_loop(self) -> None:
            """Windows message loop."""
            msg = wintypes.MSG()
            while self.running and not self._stop_event.is_set():
                bRet = GetMessageW(byref(msg), None, 0, 0)
                if bRet == 0:  # WM_QUIT
                    break
                elif bRet == -1:  # Error
                    logger.error("Error in message loop")
                    break
                else:
                    TranslateMessage(byref(msg))
                    DispatchMessageW(byref(msg))

        def _cleanup(self) -> None:
            """Clean up resources."""
            if self.hwnd:
                try:
                    DestroyWindow(self.hwnd)
                except Exception as e:
                    logger.error(f"Error destroying window: {e}")
                self.hwnd = None

else:
    # Non-Windows platforms - provide stub implementation
    class RawInputCapture:
        """Stub implementation for non-Windows platforms."""

        def __init__(self):
            pass

        def register_callback(self, callback) -> None:
            pass

        def start(self) -> bool:
            logger.warning("Raw input capture is not supported on this platform")
            return False

        def stop(self) -> None:
            pass
