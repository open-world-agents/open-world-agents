from typing import Optional

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
        width: Optional[int] = None,
        height: Optional[int] = None,
        additional_properties: Optional[dict] = None,
    ):
        """https://gstreamer.freedesktop.org/documentation/d3d11/d3d11screencapturesrc.html?gi-language=python"""
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
            from owa.core import get_component

            get_window_by_title = get_component("callables", namespace="desktop", name="window.get_window_by_title")
            window = get_window_by_title(window_name)
            properties["window-handle"] = window.hWnd

        if monitor_idx is not None:
            properties["monitor-index"] = monitor_idx

        if additional_properties is not None:
            properties.update(additional_properties)

        # for VFR(Variable Frame Rate) support, use framerate=0/1,max-framerate=60/1
        framerate = f",framerate={framerate_float_to_str(fps)}"
        size = f",width={width},height={height}" if width and height else ""

        return (
            Element("d3d11screencapturesrc", properties)
            >> Element("videorate", {"drop-only": "true"})
            >> Element("d3d11scale")
            >> ElementFactory.capsfilter(caps="video/x-raw(memory:D3D11Memory)" + framerate + size)
        )

    @staticmethod
    def wasapi2src(*, window_name: Optional[str] = None):
        """https://gstreamer.freedesktop.org/documentation/wasapi2/wasapi2src.html"""
        properties = {
            "do-timestamp": "true",
            "loopback": "true",
            "low-latency": "true",
            "loopback-mode": "include-process-tree",
        }
        if window_name is not None:
            from owa.core import get_component

            get_pid_by_title = get_component("callables", namespace="desktop", name="window.get_pid_by_title")
            pid = get_pid_by_title(window_name)
            properties["loopback-target-pid"] = pid
        return Element("wasapi2src", properties)
