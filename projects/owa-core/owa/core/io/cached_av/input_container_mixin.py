from typing import Any, Iterator, overload

import av.container
from av.audio.frame import AudioFrame
from av.audio.stream import AudioStream
from av.packet import Packet
from av.stream import Stream
from av.subtitles.stream import SubtitleStream
from av.subtitles.subtitle import SubtitleSet
from av.video.frame import VideoFrame
from av.video.stream import VideoStream

from .container_mixin import ContainerMixin


class InputContainerMixin(ContainerMixin):
    """Mixin that provides InputContainer interface by delegating to _container.

    Methods and properties copied from PyAV input container interface:
    https://github.com/PyAV-Org/PyAV/blob/main/av/container/input.pyi#L14-L49
    """

    _container: av.container.InputContainer  # type: ignore[override]

    # InputContainer-specific properties (copied from PyAV input container interface)
    @property
    def start_time(self) -> int:
        return self._container.start_time

    @property
    def duration(self) -> int | None:
        return self._container.duration

    @property
    def bit_rate(self) -> int:
        return self._container.bit_rate

    @property
    def size(self) -> int:
        return self._container.size

    def __enter__(self) -> "InputContainerMixin":
        return self

    def close(self) -> None:
        return self._container.close()

    def demux(self, *args: Any, **kwargs: Any) -> Iterator[Packet]:
        return self._container.demux(*args, **kwargs)

    @overload
    def decode(self, video: int) -> Iterator[VideoFrame]: ...

    @overload
    def decode(self, audio: int) -> Iterator[AudioFrame]: ...

    @overload
    def decode(self, subtitles: int) -> Iterator[SubtitleSet]: ...

    @overload
    def decode(self, *args: VideoStream) -> Iterator[VideoFrame]: ...

    @overload
    def decode(self, *args: AudioStream) -> Iterator[AudioFrame]: ...

    @overload
    def decode(self, *args: SubtitleStream) -> Iterator[SubtitleSet]: ...

    def decode(self, *args: Any, **kwargs: Any) -> Iterator[VideoFrame | AudioFrame | SubtitleSet]:
        return self._container.decode(*args, **kwargs)

    def seek(
        self,
        offset: int,
        *,
        backward: bool = True,
        any_frame: bool = False,
        stream: Stream | VideoStream | AudioStream | None = None,
        unsupported_frame_offset: bool = False,
        unsupported_byte_offset: bool = False,
    ) -> None:
        return self._container.seek(
            offset,
            backward=backward,
            any_frame=any_frame,
            stream=stream,
            unsupported_frame_offset=unsupported_frame_offset,
            unsupported_byte_offset=unsupported_byte_offset,
        )

    def flush_buffers(self) -> None:
        return self._container.flush_buffers()
