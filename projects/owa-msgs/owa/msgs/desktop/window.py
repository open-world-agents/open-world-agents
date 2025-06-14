"""
Desktop window message definitions.

This module contains message types for window information and events,
following the domain-based message naming convention for better organization.
"""

from owa.core.message import OWAMessage


class WindowInfo(OWAMessage):
    """
    Represents information about a desktop window.
    
    This message captures window properties including title, position, and handle,
    useful for window management and automation tasks.
    
    Attributes:
        title: Window title text
        rect: Window rectangle as (left, top, right, bottom) coordinates
        hWnd: Window handle (platform-specific identifier)
    """
    _type = "desktop/WindowInfo"
    
    title: str
    # rect has (left, top, right, bottom) format
    # normally,
    # 0 <= left < right <= screen_width
    # 0 <= top < bottom <= screen_height
    rect: tuple[int, int, int, int]
    hWnd: int

    @property
    def width(self) -> int:
        """Get window width from rectangle coordinates."""
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        """Get window height from rectangle coordinates."""
        return self.rect[3] - self.rect[1]
