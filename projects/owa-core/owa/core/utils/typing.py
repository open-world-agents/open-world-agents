import os
from typing import Union

# NOTE: pathlib.Path is an instance of os.PathLike, so we don't need to include it separately
PathLike = Union[str, bytes, os.PathLike]
