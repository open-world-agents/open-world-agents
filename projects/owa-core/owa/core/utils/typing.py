import os

# NOTE: pathlib.Path is an instance of os.PathLike, so we don't need to include it separately
PathLike = str | bytes | os.PathLike
