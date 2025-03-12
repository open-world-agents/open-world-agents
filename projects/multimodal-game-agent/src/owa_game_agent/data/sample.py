from typing import Any

from pydantic import BaseModel


class OWATrainingSample(BaseModel):
    state_keyboard: Any
    state_screen: Any

    action_keyboard: Any
    action_mouse: Any
