from typing import Annotated, Literal

from annotated_types import Ge, Lt

from owa.core.message import OWAMessage

UInt8 = Annotated[int, Ge(0), Lt(256)]


class KeyboardState(OWAMessage):
    _type = "owa.env.desktop.msg.KeyboardState"
    buttons: set[UInt8]


class MouseState(OWAMessage):
    _type = "owa.env.desktop.msg.MouseState"
    x: int
    y: int
    buttons: set[Literal["left", "middle", "right"]]


class KeyboardEvent(OWAMessage):
    _type = "owa.env.desktop.msg.KeyboardEvent"

    event_type: Literal["press", "release"]
    vk: int


class MouseEvent(OWAMessage):
    _type = "owa.env.desktop.msg.MouseEvent"

    event_type: Literal["move", "click", "scroll"]
    x: int
    y: int
    button: str | None = None
    pressed: bool | None = None
    dx: int | None = None
    dy: int | None = None


class WindowInfo(OWAMessage):
    _type = "owa.env.desktop.msg.WindowInfo"

    title: str
    # rect has (left, top, right, bottom) format
    # normally,
    # 0 <= left < right <= screen_width
    # 0 <= top < bottom <= screen_height
    rect: tuple[int, int, int, int]
    hWnd: int

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]
