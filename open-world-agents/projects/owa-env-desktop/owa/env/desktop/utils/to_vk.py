import enum
from typing import Union

from pynput.keyboard import Key, KeyCode


def key_to_vk(key: Union[Key, KeyCode, None]) -> int:
    """Converts a pynput key to a virtual key code.

    The key parameter passed to callbacks is a `pynput.keyboard.Key` for special keys,
    a `pynput.keyboard.KeyCode` for normal alphanumeric keys, or just None for unknown keys.

    Tested on: Windows 11, MacOS
    """
    # For None(unknown key), return -1
    if key is None:
        return -1
    # For Key, which is enum.Enum with KeyCode as value, converts to KeyCode
    if isinstance(key, enum.Enum) and getattr(key, "value", None) is not None:
        key = key.value
    # For KeyCode, converts to vk
    vk = getattr(key, "vk", None)
    return vk


def char_to_vk(char: str) -> int:
    """Converts a character to a virtual key code."""
    if char.isalpha():
        return ord(char.upper())
    elif char.isdigit():
        return ord(char)
    else:
        raise ValueError(f"Unsupported character: {char}")
