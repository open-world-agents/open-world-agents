from owa.message import OWAMessage


class KeyboardEvent(OWAMessage):
    _type = "owa_env_desktop.msg.KeyboardEvent"

    event_type: str
    vk: int


class MouseEvent(OWAMessage):
    _type = "owa_env_desktop.msg.MouseEvent"

    event_type: str
    x: int
    y: int
    button: str | None = None
    pressed: bool | None = None
    dx: int | None = None
    dy: int | None = None


class WindowInfo(OWAMessage):
    _type = "owa_env_desktop.msg.WindowInfo"

    title: str
    rect: tuple[int, int, int, int]
    hWnd: int

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]
