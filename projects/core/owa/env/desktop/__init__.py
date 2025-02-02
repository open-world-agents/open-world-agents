# This module contains listeners and callables for desktop environment.
# Components:
#    - screen
#    - keyboard_mouse
#    - window


def activate():
    from . import screen  # noqa
    from . import keyboard_mouse  # noqa
    from . import window  # noqa
