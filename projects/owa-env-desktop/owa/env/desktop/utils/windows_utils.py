from ctypes import c_int, c_long, c_uint, c_wchar_p, create_unicode_buffer, windll

MAPVK_VK_TO_VSC = 0


def get_vk_state() -> set[int]:
    """Get a list of currently pressed virtual key (VK) codes.

    Uses the Windows API function `GetAsyncKeyState` to check the state of all virtual key codes (0 to 255).

    Returns:
        List[int]: A list of VK codes where the most significant bit is set (indicating the key is currently pressed).
    """
    _GetAsyncKeyState = windll.user32.GetAsyncKeyState
    _GetAsyncKeyState.argtypes = (c_int,)

    vks = [vk for vk in range(256) if _GetAsyncKeyState(vk) & 0x8000]
    return vks


def vk_to_name(vk: int) -> str:
    """Convert a virtual key (VK) code to its human-readable name.

    Uses the Windows API functions `MapVirtualKeyExW` and `GetKeyNameTextW` to retrieve the key name.

    Args:
        vk (int): The virtual key code to convert.

    Returns:
        str: The human-readable name of the key, or `'Unknown'` if it cannot be determined.
    """
    _MapVirtualKeyEx = windll.user32.MapVirtualKeyExW
    _MapVirtualKeyEx.argtypes = (c_uint, c_uint, c_int)
    _MapVirtualKeyEx.restype = c_uint

    _GetKeyNameText = windll.user32.GetKeyNameTextW
    _GetKeyNameText.argtypes = (c_long, c_wchar_p, c_int)
    _GetKeyNameText.restype = c_int

    scancode = _MapVirtualKeyEx(vk, MAPVK_VK_TO_VSC, 0)
    lParam = scancode << 16
    buffer = create_unicode_buffer(64)
    result = _GetKeyNameText(lParam, buffer, 64)
    if result > 0:
        return buffer.value
    else:
        return "Unknown"


def main() -> None:
    """Demonstrate the usage of `get_vk_state` and `vk_to_name` functions.

    Retrieves the currently pressed keys and prints their VK codes along with human-readable names.
    """
    vks = get_vk_state()  # 300 microseconds
    print("Currently pressed VK codes:", vks)

    for vk in vks:
        name = vk_to_name(vk)
        print(f"VK code: {vk}, Name: {name}")


if __name__ == "__main__":
    main()
