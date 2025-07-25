from fractions import Fraction
from types import TracebackType
from typing import Any, Type

import av.container
from av.container.streams import StreamContainer
from av.format import ContainerFormat

Real = int | float | Fraction


class ContainerMixin:
    """Mixin that provides Container interface by delegating to _container.

    Methods and properties copied from PyAV container core interface:
    https://github.com/PyAV-Org/PyAV/blob/main/av/container/core.pyi#L70-L98
    """

    _container: av.container.Container

    # Properties from Container (copied from PyAV container core interface)
    @property
    def writeable(self) -> bool:
        return self._container.writeable

    @property
    def name(self) -> str:
        return self._container.name

    @property
    def metadata_encoding(self) -> str:
        return self._container.metadata_encoding

    @property
    def metadata_errors(self) -> str:
        return self._container.metadata_errors

    @property
    def file(self) -> Any:
        return self._container.file

    @property
    def buffer_size(self) -> int:
        return self._container.buffer_size

    @property
    def input_was_opened(self) -> bool:
        return self._container.input_was_opened

    @property
    def io_open(self) -> Any:
        return self._container.io_open

    @property
    def open_files(self) -> Any:
        return self._container.open_files

    @property
    def format(self) -> ContainerFormat:
        return self._container.format

    @property
    def options(self) -> dict[str, str]:
        return self._container.options

    @property
    def container_options(self) -> dict[str, str]:
        return self._container.container_options

    @property
    def stream_options(self) -> list[dict[str, str]]:
        return self._container.stream_options

    @property
    def streams(self) -> StreamContainer:
        return self._container.streams

    @property
    def metadata(self) -> dict[str, str]:
        return self._container.metadata

    @property
    def open_timeout(self) -> Real | None:
        return self._container.open_timeout

    @property
    def read_timeout(self) -> Real | None:
        return self._container.read_timeout

    @property
    def flags(self) -> int:
        return self._container.flags

    def __enter__(self) -> "ContainerMixin":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return self._container.__exit__(exc_type, exc_val, exc_tb)

    def set_timeout(self, timeout: Real | None) -> None:
        return self._container.set_timeout(timeout)

    def start_timeout(self) -> None:
        return self._container.start_timeout()
