import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import HTTPException


# Exception Classes
class AppError(HTTPException):
    """Base application error class"""

    def __init__(
        self, status_code: int, detail: str, error_code: str = "GENERAL_ERROR", extra: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.extra = extra or {}
        super().__init__(status_code=status_code, detail=detail)


class FileNotFoundError(AppError):
    """Raised when a file is not found"""

    def __init__(self, filename: str):
        super().__init__(
            status_code=404,
            detail=f"File not found: {filename}",
            error_code="FILE_NOT_FOUND",
            extra={"filename": filename},
        )


class FileDownloadError(AppError):
    """Raised when a file download fails"""

    def __init__(self, url: str, reason: str):
        super().__init__(
            status_code=500,
            detail=f"Failed to download file: {reason}",
            error_code="FILE_DOWNLOAD_ERROR",
            extra={"url": url, "reason": reason},
        )


class InvalidDataError(AppError):
    """Raised when data is invalid"""

    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail, error_code="INVALID_DATA")


class McapProcessingError(AppError):
    """Raised when MCAP processing fails"""

    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail, error_code="MCAP_PROCESSING_ERROR")


# Formatting Functions
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


# Path Utility Functions
def safe_join(base_dir: str, *paths: str) -> Optional[Path]:
    """
    Join paths safely, ensuring the result is within the base directory.

    Args:
        base_dir: The base directory
        paths: Path components to join

    Returns:
        Path object if safe, None if path would escape base directory
    """
    base = Path(base_dir).resolve()
    target = (base / Path(*paths)).resolve()

    if not target.is_relative_to(base):
        return None
    return target


def extract_original_filename(filename: str) -> Optional[str]:
    """
    Extract original filename from a UUID-appended filename.

    Args:
        filename: Filename that may contain UUID suffix

    Returns:
        Original filename if UUID pattern found, None otherwise
    """
    uuid_pattern = r"(.+)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    match = re.search(uuid_pattern, filename)
    if match:
        return match.group(1)
    return None
