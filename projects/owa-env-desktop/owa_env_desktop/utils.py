import enum
import platform
from typing import Union

from pynput.keyboard import Key, KeyCode


def key_to_vk(key: Union[Key, KeyCode, None]) -> int:
    """Converts a pynput key to a virtual key code.

    The key parameter passed to callbacks is a `pynput.keyboard.Key` for special keys,
    a `pynput.keyboard.KeyCode` for normal alphanumeric keys, or just None for unknown keys.

    This function handles different operating systems accordingly.
    """
    if key is None:
        return 0

    os_name = platform.system()

    if os_name == "Windows":
        # For unknown key, return -1
        if key is None:
            return -1
        # Converts Key(enum.Enum) which has value as KeyCode to KeyCode
        if isinstance(key, enum.Enum) and getattr(key, "value", None) is not None:
            key = key.value
        # Read over virtual key codes from KeyCode
        vk = getattr(key, "vk", None)
        return vk
    elif os_name == "Darwin":
        # TODO: check whether this is correct

        # Mac OS uses key codes
        if isinstance(key, Key):
            # Map common special keys
            mac_key_map = {
                Key.alt: 58,  # Option key
                Key.alt_l: 58,  # Left Option
                Key.alt_r: 61,  # Right Option
                Key.cmd: 55,  # Command key
                Key.cmd_l: 55,  # Left Command
                Key.cmd_r: 54,  # Right Command
                Key.ctrl: 59,  # Control key
                Key.ctrl_l: 59,  # Left Control
                Key.ctrl_r: 62,  # Right Control
                Key.shift: 56,  # Shift key
                Key.shift_l: 56,  # Left Shift
                Key.shift_r: 60,  # Right Shift
                Key.enter: 36,  # Return
                Key.space: 49,  # Space
                Key.backspace: 51,  # Delete
                Key.delete: 117,  # Forward Delete
                Key.tab: 48,  # Tab
                Key.esc: 53,  # Escape
            }
            return mac_key_map.get(key, 0)
        # For regular keys, use the ASCII value
        return ord(key.char.lower()) if hasattr(key, "char") else 0
    else:
        # For other OS, fallback to ASCII values
        if isinstance(key, Key):
            return key.value.vk if hasattr(key.value, "vk") else 0
        return ord(key.char.lower()) if hasattr(key, "char") else 0


def char_to_vk(char: str) -> int:
    """Converts a character to a virtual key code."""
    if char.isalpha():
        return ord(char.upper())
    elif char.isdigit():
        return ord(char)
    else:
        raise ValueError(f"Unsupported character: {char}")


__all__ = ["key_to_vk", "char_to_vk"]
