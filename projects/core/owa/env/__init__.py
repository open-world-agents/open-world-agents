from .callable_listener import Callable, Listener


def register_base_pkg():
    """
    Register the base package of the environment.
    """
    from . import keyboard_mouse  # noqa
    from . import window  # noqa
