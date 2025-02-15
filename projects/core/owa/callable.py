"""
Callable implements `__call__` method, which allows the object to be called as a function.
"""

from typing import Callable


class CallableMixin:
    def __call__(self):
        raise NotImplementedError
