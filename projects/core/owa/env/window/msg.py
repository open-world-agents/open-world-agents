from pydantic import BaseModel


class WindowInfo(BaseModel):
    title: str
    rect: tuple[int, int, int, int]
    hWnd: int

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]
