# This module contains listeners and callables for desktop environment.
# Components:
#    - screen
#    - keyboard_mouse
#    - window
from . import gst_factory


def activate():
    from . import screen  # noqa
