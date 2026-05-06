"""
Press Alt(0x12), IME Hangeul(0x15), Alt_L(0xA4) sequentially.

Conclusion of this script: we must consider extended keys, and I've implemented that as `vk_to_keycode`.
"""

import time

from pynput.keyboard import Controller, KeyCode

keyboard = Controller()

time.sleep(2)
# in here, 0x15 is recognized as AltLeft
keys = [KeyCode.from_vk(0x12), KeyCode.from_vk(0x15), KeyCode.from_vk(0xA4)]

# in here, 0x15 is recognized as AltRight.
keys = [KeyCode._from_ext(0x15)]

for key in keys:
    keyboard.press(key)
    time.sleep(0.1)
    keyboard.release(key)
    time.sleep(0.1)
