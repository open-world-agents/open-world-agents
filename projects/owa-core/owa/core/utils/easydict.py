from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class EasyDict(dict):
    """
    Dictionary with attribute-style access and Pydantic integration.

    Allows accessing dictionary values as attributes (works recursively).
    Useful for configuration objects, parsed JSON content, and nested data structures.

    Examples:
        >>> config = EasyDict({'database': {'host': 'localhost', 'port': 5432}})
        >>> config.database.host
        'localhost'
        >>> config.database.port
        5432

        >>> data = EasyDict({'servers': [{'name': 'web1', 'ip': '192.168.1.1'}]})
        >>> data.servers[0].name
        'web1'
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x) if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, *args)

    # Newly added for Pydantic integration
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(dict))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
