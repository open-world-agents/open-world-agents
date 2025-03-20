from typing import Optional

from owa.core.registry import CALLABLES, activate_module

from ..utils import framerate_float_to_str
from .element import Element


class ElementFactory:
    @staticmethod
    def matroskamux(**properties):
        return Element("matroskamux", properties)

    @staticmethod
    def tee(**properties):
        return Element("tee", properties)

    @staticmethod
    def capsfilter(**properties):
        """https://gstreamer.freedesktop.org/documentation/coreelements/capsfilter.html"""
        return Element(properties["caps"])

    @staticmethod
    def filesink(**properties):
        return Element("filesink", properties)

    @staticmethod
    def d3d11screencapturesrc(
        *,
        show_cursor: bool = True,
        fps: float = 60.0,
        window_name: Optional[str] = None,
        monitor_idx: Optional[int] = None,
        additional_properties: Optional[dict] = None,
    ):
        properties = {
            "show-cursor": str(show_cursor).lower(),
            "do-timestamp": "true",
            "window-capture-mode": "client",
            # wgc is slower than dxgi in d3d11 implmenetation, but capable of specific window capture
            # TODO: upgrade gstreamer & support d3d12 plugins. low-framerate issue in wgc is resolved in d3d12 version. https://discourse.gstreamer.org/t/d3d11screencapturesrc-vs-d3d12screencapturesrc/2080
            # "capture-api": "wgc",
            # "capture-api": "dxgi",
            "show-border": True,
        }
        if window_name is not None:
            activate_module("owa.env.desktop")
            window = CALLABLES["window.get_window_by_title"](window_name)
            properties["window-handle"] = window.hWnd

        if monitor_idx is not None:
            properties["monitor-index"] = monitor_idx

        if additional_properties is not None:
            properties.update(additional_properties)

        framerate = f"framerate=0/1,max-framerate={framerate_float_to_str(fps)}"

        return (
            Element("d3d11screencapturesrc", properties)
            >> Element("videorate", {"drop-only": "true"})
            >> ElementFactory.capsfilter(caps=f"video/x-raw(memory:D3D11Memory),{framerate}")
        )
