from owa.env import Listener
from owa.registry import LISTENERS


@LISTENERS.register("screen")
class ScreenListener(Listener): ...  # TODO: implement single-image grab Callable
