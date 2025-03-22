import pynput._util.win32_vks as VK
from pynput.keyboard import KeyCode

# Copied from https://github.com/moses-palmer/pynput/blob/2dab434cba790ee92e807594e6bf8f83265dca34/lib/pynput/keyboard/_win32.py#L113
# These keys MUST be handled differently because they are not recognized by `KeyCode.from_vk`.
# See also: https://learn.microsoft.com/en-us/windows/win32/inputdev/about-keyboard-input#extended-key-flag
EXTENDED_KEYS = [
    VK.RMENU,
    VK.RCONTROL,
    VK.DELETE,
    VK.DOWN,
    VK.END,
    VK.HOME,
    VK.LEFT,
    VK.NEXT,
    VK.PRIOR,
    VK.RIGHT,
    VK.UP,
    VK.MEDIA_PLAY_PAUSE,
    VK.MEDIA_STOP,
    VK.VOLUME_MUTE,
    VK.VOLUME_DOWN,
    VK.VOLUME_UP,
    VK.MEDIA_PREV_TRACK,
    VK.MEDIA_NEXT_TRACK,
    VK.INSERT,
    VK.NUMLOCK,
    VK.SNAPSHOT,
]


def vk_to_keycode(vk: int) -> KeyCode:
    if vk in EXTENDED_KEYS:
        return KeyCode._from_ext(vk)
    return KeyCode.from_vk(vk)
