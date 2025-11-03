import inspect
from collections import UserDict
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


# More reference: https://github.com/makinacorpus/easydict
class EasyDict(UserDict):
    """
    Dictionary with attribute-style access and Pydantic integration.

    Allows accessing dictionary values as attributes (works recursively).
    Contains no heavy operation except the lightweight conversion at assignment time from native types to EasyDict.
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

    @staticmethod
    def _convert_value(value):
        """Recursively convert dictionaries to EasyDict and process sequences."""
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return EasyDict(value)
        elif isinstance(value, (list, tuple)):
            return type(value)(EasyDict._convert_value(item) for item in value)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert_value(value))

    def __setattr__(self, name, value):
        # if setattr is called inside EasyDict, call normal setattr and return
        frame = inspect.currentframe().f_back
        caller_class = frame.f_locals.get("self", None).__class__.__name__
        if caller_class == "EasyDict":
            return super().__setattr__(name, value)
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'EasyDict' object has no attribute '{name}'")

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)
        frame = inspect.currentframe().f_back
        caller_class = frame.f_locals.get("self", None).__class__.__name__
        # if getattribute is called outside EasyDict, hide `.data`
        if caller_class != "EasyDict":
            if name == "data":
                return super().__getitem__(name)
        return super().__getattribute__(name)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'EasyDict' object has no attribute '{name}'")

    # Newly added for Pydantic integration
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(dict))
