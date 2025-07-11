from typing import Any

from pydantic import BaseModel


class OWATrainingSample(BaseModel):
    """
    This class represents a training sample for OWA.
    To support versatile data manipulation, the type is not fixed.
    e.g. action.keyboard can be pre-tokenized or not.
    """

    state_keyboard: Any
    state_mouse: Any
    state_screen: Any

    action_keyboard: Any
    action_mouse: Any
