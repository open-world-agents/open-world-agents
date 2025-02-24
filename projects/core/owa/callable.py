"""
Callable implements `__call__` method, which allows the object to be called as a function.
"""

from typing import Callable  # noqa
from abc import ABC, abstractmethod


class CallableMixin(ABC):
    @abstractmethod
    def __call__(self): ...
