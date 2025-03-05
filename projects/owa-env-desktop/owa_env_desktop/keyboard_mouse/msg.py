from owa.message import OWAMessage


class KeyboardEvent(OWAMessage):
    _type = "owa_env_desktop.keyboard_mouse.KeyboardEvent"

    event_type: str
    vk: int


class MouseEvent(OWAMessage):
    _type = "owa_env_desktop.keyboard_mouse.MouseEvent"

    event_type: str
    x: int
    y: int
    button: str | None = None
    pressed: bool | None = None
    dx: int | None = None
    dy: int | None = None
