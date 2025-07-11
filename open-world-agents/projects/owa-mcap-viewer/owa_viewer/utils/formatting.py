import numpy as np


def format_size_human_readable(size_bytes: int) -> str:
    """
    Convert size in bytes to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "2.50 MiB")
    """
    size_units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size_unit = 0

    if size_bytes > 0:
        size_unit = min(len(size_units) - 1, int(np.floor(np.log2(max(1, size_bytes)) / 10)))
        size = size_bytes / (1024**size_unit)
    else:
        size = 0

    return f"{size:.2f} {size_units[int(size_unit)]}"
