from owa.message import OWAMessage


class WindowInfo(OWAMessage):
    _type = "owa_env_desktop.window.WindowInfo"

    title: str
    rect: tuple[int, int, int, int]
    hWnd: int

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]
